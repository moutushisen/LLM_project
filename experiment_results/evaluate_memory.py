#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Compression Evaluation Script (Scheme A) - Enhanced with Entity-Aware Testing

This script evaluates whether memory generation & compression preserves key points
using pre-generated QA pairs from the dataset.

EVALUATION FLOW:
1. Load dataset with pre-generated QA pairs (query + answer)
2. Use the QA pair to generate memory (memory compression)
3. Compare generated memory against expected_key_points (gold standard)

EVALUATION MODES:
- Standard: Evaluate standard memory compression
- Entity-Aware: Evaluate entity-aware memory compression (preserves key terms)
- Comparison: Compare both methods side-by-side

PRIMARY METRICS:
- Key-Point Recall (macro-averaged with 95% CI): Measures information preservation
- Entity Preservation (macro-averaged with 95% CI): % of key entities preserved
- Compression Ratio (macro-averaged with 95% CI): % of original size retained (lower = better compression)

SECONDARY METRICS (for reference):
- Phrase Precision/Recall/F1 (macro-averaged with 95% CI): Expected to be low due to compression

USAGE:
  # Standard evaluation
  python evaluate_memory.py --model gemini --samples 10
  
  # Entity-aware evaluation
  python evaluate_memory.py --model gemini --samples 10 --entity-aware
  
  # Comparison mode (both methods)
  python evaluate_memory.py --model gemini --samples 10 --compare-modes

Note: Memory compression naturally results in low phrase-level metrics as the goal is 
      semantic preservation, not literal text matching.
"""

import google.generativeai as genai
import json
import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any
import time
from collections import defaultdict
import numpy as np
import random

# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================
def bootstrap_ci_mean(values: List[float], n_boot: int = 1000, alpha: float = 0.05, rng_seed: int = 42) -> Tuple[float, float]:
    """Compute percentile bootstrap CI for the mean.

    If fewer than 2 samples are provided, returns (mean, mean).
    """
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        m = float(arr.mean())
        return m, m
    rng = np.random.default_rng(rng_seed)
    n = arr.size
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = float(arr[idx].mean())
    lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def bootstrap_ci_compression_ratio(results: List[Dict], n_boot: int = 1000, alpha: float = 0.05, rng_seed: int = 42) -> Tuple[float, float]:
    """Compute percentile bootstrap CI for compression ratio using sum(memory)/sum(answer) method.
    
    This avoids bias from short answers by calculating the ratio of total lengths
    rather than averaging individual ratios.
    
    If fewer than 2 samples are provided, returns (ratio, ratio).
    """
    if not results:
        return 0.0, 0.0
    if len(results) == 1:
        r = results[0]
        ratio = float(r['memory_length'] / r['answer_length']) if r['answer_length'] > 0 else 0.0
        return ratio, ratio
    
    rng = np.random.default_rng(rng_seed)
    n = len(results)
    boot_ratios = np.empty(n_boot, dtype=float)
    
    for i in range(n_boot):
        # Resample with replacement
        idx = rng.integers(0, n, size=n)
        boot_sample = [results[j] for j in idx]
        
        # Calculate compression ratio for this bootstrap sample
        total_answer = sum(r['answer_length'] for r in boot_sample)
        total_memory = sum(r['memory_length'] for r in boot_sample)
        boot_ratios[i] = float(total_memory / total_answer) if total_answer > 0 else 0.0
    
    lower = float(np.percentile(boot_ratios, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_ratios, 100 * (1 - alpha / 2)))
    return lower, upper


# Add parent directory to sys.path to import memory modules
sys.path.insert(0, str(Path.home() / "LLM_project"))
from memory.generator import generate_merged_memory
from memory.rolling import RollingMemoryStorage
from memory.entity_aware_generator import (
    generate_entity_aware_memory,
    extract_key_terms
)

# Try to import NLP libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    NLTK_OK = True
    try:
        nltk.data.find('corpora/stopwords.txt')
    except LookupError:
        nltk.download('stopwords', quiet=True)
except ImportError:
    NLTK_OK = False

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_OK = True
except ImportError:
    SENTENCE_TRANSFORMERS_OK = False


# =============================================================================
# Configuration Loading
# =============================================================================
def load_config():
    """Load API configuration"""
    config_path = Path.home() / ".config" / "llm_project" / ".env"
    if config_path.exists():
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in environment")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash-lite')


# =============================================================================
# Text Preprocessing
# =============================================================================
class TextPreprocessor:
    """Text preprocessing with normalization, lowercasing, punctuation removal"""
    
    def __init__(self):
        if NLTK_OK:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stemmer = None
            # Basic stop words (extended for learning context)
            self.stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
                'has', 'have', 'had', 'be', 'been', 'being', 'this', 'that',
                'these', 'those', 'some', 'any', 'can', 'will', 'would', 'should'
            }
        
        # Domain-specific normalization (programming + learning context terms)
        self.synonyms = {
            # Programming terms
            'postgres': 'postgresql',
            'postgresql': 'postgresql',
            'js': 'javascript',
            'javascript': 'javascript',
            'di': 'dependency injection',
            'dependency injection': 'dependency injection',
            'async': 'asynchronous',
            'await': 'asynchronous',
            
            # User experience level synonyms (for learning partner context)
            'beginner': 'novice',
            'novice': 'novice',
            'new to': 'novice',
            'just starting': 'novice',
            'unfamiliar with': 'novice',
            'learning basics': 'novice',
            'starting to learn': 'novice',
            
            'intermediate': 'intermediate level',
            'has some experience': 'intermediate level',
            'familiar with basics': 'intermediate level',
            'working on': 'intermediate level',
            'practicing': 'intermediate level',
            
            'advanced': 'expert',
            'expert': 'expert',
            'experienced': 'expert',
            'proficient': 'expert',
            'skilled in': 'expert',
            'mastering': 'expert',
            
            # Learning context terms
            'student': 'learner',
            'learner': 'learner',
            'studying': 'learning',
            'learning': 'learning',
            'preparing for': 'preparing',
            'building': 'working on',
            'developing': 'working on',
            'creating': 'working on',
            'debugging': 'troubleshooting',
            'fixing': 'troubleshooting',
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text: lowercase, remove punctuation, normalize whitespace"""
        # Lowercase
        text = text.lower()
        
        # Handle slashes and hyphens (e.g., async/await -> async await)
        text = re.sub(r'[/-]', ' ', text)
        
        # Handle camelCase/PascalCase (e.g., rateLimiter -> rate limiter)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Remove punctuation except periods in numbers
        text = re.sub(r'[^\w\s.]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize and optionally stem"""
        tokens = self.normalize_text(text).split()
        
        # Apply synonyms
        tokens = [self.synonyms.get(t, t) for t in tokens]
        
        # Optional stemming
        if self.stemmer and NLTK_OK:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        return tokens
    
    def extract_tokens_without_stopwords(self, text: str) -> List[str]:
        """Extract tokens without stop words"""
        tokens = self.tokenize(text)
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]


# =============================================================================
# Reference Answer - Now using pre-generated answers from dataset
# =============================================================================
# NOTE: This function is kept for backward compatibility but is no longer used
# in the main evaluation flow. The dataset now includes pre-generated answers.


# =============================================================================
# Anchor Phrase Extraction
# =============================================================================
class AnchorExtractor:
    """Extract anchor phrases from key points"""
    
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
    
    def extract_anchors(self, key_point: str, num_anchors: int = 3) -> List[str]:
        """Extract 2-4 anchor phrases from a key point"""
        # Get tokens without stop words
        tokens = self.preprocessor.extract_tokens_without_stopwords(key_point)
        
        # Extract n-grams (1-3 words)
        anchors = []
        
        # Add significant unigrams
        anchors.extend(tokens[:num_anchors])
        
        # Add bigrams
        for i in range(len(tokens) - 1):
            anchors.append(f"{tokens[i]} {tokens[i+1]}")
        
        # Add trigrams if available
        if len(tokens) >= 3:
            for i in range(min(2, len(tokens) - 2)):
                anchors.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")
        
        # Deduplicate and limit
        unique_anchors = []
        seen = set()
        for anchor in anchors:
            if anchor not in seen:
                unique_anchors.append(anchor)
                seen.add(anchor)
        
        return unique_anchors[:num_anchors + 2]  # Return top 2-5 anchors


# =============================================================================
# Multi-Level Matching
# =============================================================================
class KeyPointMatcher:
    """Multi-level key point matching"""
    
    def __init__(self, preprocessor: TextPreprocessor, anchor_extractor: AnchorExtractor,
                 token_threshold: float = 0.66, similarity_threshold: float = 0.75):
        self.preprocessor = preprocessor
        self.anchor_extractor = anchor_extractor
        self.token_threshold = token_threshold
        self.similarity_threshold = similarity_threshold
        
        # Load sentence transformer model if available
        self.sentence_model = None
        if SENTENCE_TRANSFORMERS_OK:
            try:
                self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            except Exception:
                pass
    
    def level1_exact_phrase_match(self, anchor: str, memory_text: str) -> bool:
        """Level 1: Exact phrase matching after preprocessing"""
        normalized_memory = self.preprocessor.normalize_text(memory_text)
        normalized_anchor = self.preprocessor.normalize_text(anchor)
        return normalized_anchor in normalized_memory
    
    def level2_token_coverage(self, anchor_tokens: Set[str], memory_text: str) -> float:
        """Level 2: Token coverage ratio"""
        memory_tokens = set(self.preprocessor.extract_tokens_without_stopwords(memory_text))
        
        if not anchor_tokens:
            return 0.0
        
        matched = anchor_tokens.intersection(memory_tokens)
        coverage = len(matched) / len(anchor_tokens)
        return coverage
    
    def level3_sentence_similarity(self, key_point: str, memory_text: str) -> float:
        """Level 3: Sentence similarity using sentence transformers"""
        if not self.sentence_model:
            return 0.0
        
        try:
            memory_sentences = [s.strip() for s in re.split(r'[.!?]+', memory_text) if s.strip()]
            if not memory_sentences:
                return 0.0
            
            key_point_embedding = self.sentence_model.encode(key_point, convert_to_tensor=True)
            memory_embeddings = self.sentence_model.encode(memory_sentences, convert_to_tensor=True)
            similarities = util.cos_sim(key_point_embedding, memory_embeddings)
            return float(similarities.max())
        except Exception:
            return 0.0
    
    def match_key_point(self, key_point: str, memory_text: str) -> Tuple[bool, str]:
        """
        Match a key point against memory text using multi-level matching
        Returns: (matched: bool, level: str)
        """
        # Extract anchors from key point
        anchors = self.anchor_extractor.extract_anchors(key_point)
        
        # Level 1: Exact phrase matching
        for anchor in anchors:
            if self.level1_exact_phrase_match(anchor, memory_text):
                return True, "level1_exact_phrase"
        
        # Level 2: Token coverage
        anchor_tokens = set()
        for anchor in anchors:
            anchor_tokens.update(self.preprocessor.extract_tokens_without_stopwords(anchor))
        
        coverage = self.level2_token_coverage(anchor_tokens, memory_text)
        if coverage >= self.token_threshold:
            return True, f"level2_token_coverage_{coverage:.2f}"
        
        # Level 3: Sentence similarity (fallback)
        if self.sentence_model:
            similarity = self.level3_sentence_similarity(key_point, memory_text)
            if similarity >= self.similarity_threshold:
                return True, f"level3_sentence_sim_{similarity:.2f}"
        
        return False, "no_match"


# =============================================================================
# Entity Preservation Evaluation
# =============================================================================
def evaluate_entity_preservation(answer: str, memory_text: str, 
                                  preprocessor: TextPreprocessor) -> Tuple[float, int, int]:
    """
    Evaluate entity preservation rate.
    
    Args:
        answer: Original answer text
        memory_text: Compressed memory text
        preprocessor: Text preprocessor
        
    Returns:
        (preservation_rate, preserved_count, total_count)
    """
    # Extract entities from original answer
    original_entities = extract_key_terms(answer, max_terms=30)
    
    if not original_entities:
        return 1.0, 0, 0  # No entities to preserve
    
    # Check which entities are preserved in memory (flexible matching)
    memory_lower = preprocessor.normalize_text(memory_text)
    preserved_entities = set()
    
    for entity in original_entities:
        entity_norm = preprocessor.normalize_text(entity)
        
        # Direct substring match
        if entity_norm in memory_lower:
            preserved_entities.add(entity)
            continue
        
        # Partial token match (e.g., "gradient descent" vs "gradient")
        entity_tokens = set(entity_norm.split())
        memory_tokens = set(memory_lower.split())
        
        # If significant portion of entity tokens present (>=66%)
        if entity_tokens and len(entity_tokens & memory_tokens) / len(entity_tokens) >= 0.66:
            preserved_entities.add(entity)
    
    preservation_rate = len(preserved_entities) / len(original_entities)
    return preservation_rate, len(preserved_entities), len(original_entities)


# =============================================================================
# Phrase-Level Evaluation
# =============================================================================
def evaluate_phrases(gold_phrases: Set[str], memory_text: str,
                     preprocessor: TextPreprocessor,
                     anchor_extractor: AnchorExtractor) -> Tuple[float, float, float]:
    """Evaluate phrase-level precision, recall, F1"""
    gold_norms = {preprocessor.normalize_text(p) for p in gold_phrases if p}
    
    # Extract predicted phrases from memory
    pred_raw = set()
    sentences = [s.strip() for s in re.split(r"[.!?]+", memory_text) if s.strip()]
    for s in sentences:
        pred_raw.update(anchor_extractor.extract_anchors(s, num_anchors=3))
    pred_raw.update(anchor_extractor.extract_anchors(memory_text, num_anchors=5))
    pred_norms = {preprocessor.normalize_text(p) for p in pred_raw if p}

    if not gold_norms or not pred_norms:
        return 0.0, 0.0, 0.0

    # Greedy substring matching
    match_count = 0
    matched_gold = set()
    for p in pred_norms:
        for g in gold_norms:
            if g not in matched_gold and p and g and ((p in g) or (g in p)):
                matched_gold.add(g)
                match_count += 1
                break

    precision = match_count / len(pred_norms)
    recall = match_count / len(gold_norms)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# =============================================================================
# Single-Round Memory Evaluation
# =============================================================================
def evaluate_single_sample(sample: Dict[str, Any], model, preprocessor: TextPreprocessor,
                          anchor_extractor: AnchorExtractor, matcher: KeyPointMatcher,
                          memory_model: str = "gemini-2.5-flash-lite",
                          max_chars: int = 1200,
                          use_entity_aware: bool = False) -> Dict[str, Any]:
    """
    Evaluate a single sample
    
    Args:
        sample: Test sample from evaluation_dataset.json (now includes pre-generated answer)
        model: Gemini model (kept for compatibility, not used for answer generation)
        preprocessor: Text preprocessor
        anchor_extractor: Anchor phrase extractor
        matcher: Key point matcher
        memory_model: Model name for memory generation
        max_chars: Max characters for memory compression
        use_entity_aware: Whether to use entity-aware memory generation
    
    Returns:
        Evaluation results
    """
    sample_id = sample.get('id', 'unknown')
    query = sample.get('query', '')
    answer = sample.get('answer', '')
    key_points = sample.get('expected_key_points', [])
    
    mode_label = "Entity-Aware" if use_entity_aware else "Standard"
    print(f"\n{'='*80}")
    print(f"Sample: {sample_id} | Mode: {mode_label} | Key points: {len(key_points)}")
    
    # Check if answer exists in dataset
    if not answer:
        print(f"⚠️  Warning: No answer found in dataset for sample {sample_id}")
        return {
            'sample_id': sample_id,
            'error': 'No answer in dataset',
            'key_point_recall': 0.0,
            'entity_preservation': 0.0,
            'phrase_precision': 0.0,
            'phrase_recall': 0.0,
            'phrase_f1': 0.0
        }
    
    # Step 1: Use pre-generated answer from dataset
    # Step 2: Generate memory from QA pair (single-turn)
    chat_pairs = [(query, answer)]
    history_text = ""
    
    try:
        if use_entity_aware:
            merged_memory = generate_entity_aware_memory(
                chat_pairs=chat_pairs,
                history_text=history_text,
                max_chars=max_chars,
                model_name=memory_model,
                verbose=False
            )
        else:
            merged_memory = generate_merged_memory(
                chat_pairs=chat_pairs,
                history_text=history_text,
                max_chars=max_chars,
                model_name=memory_model
            )
    except Exception as e:
        print(f"❌ Error generating memory: {str(e)}")
        return {
            'sample_id': sample_id,
            'error': str(e),
            'key_point_recall': 0.0,
            'entity_preservation': 0.0,
            'phrase_precision': 0.0,
            'phrase_recall': 0.0,
            'phrase_f1': 0.0
        }
    
    # Step 3: Extract gold anchor phrases
    # Filter out user profile/context key points (typically the first one or ones starting with "User is")
    gold_phrases = set()
    technical_key_points = []
    for kp in key_points:
        # Skip user profile/context descriptions
        if kp.lower().startswith('user is') or kp.lower().startswith('user has') or 'user' in kp.lower()[:20]:
            continue
        technical_key_points.append(kp)
        anchors = anchor_extractor.extract_anchors(kp)
        gold_phrases.update(anchors)
    
    # Step 4: Match key points
    matched_points = []
    missed_points = []
    
    for kp in key_points:
        matched, level = matcher.match_key_point(kp, merged_memory)
        if matched:
            matched_points.append({'key_point': kp, 'level': level})
        else:
            missed_points.append(kp)
    
    key_point_recall = len(matched_points) / len(key_points) if key_points else 0.0
    
    # Step 5: Phrase-level evaluation
    phrase_precision, phrase_recall, phrase_f1 = evaluate_phrases(
        gold_phrases, merged_memory, preprocessor, anchor_extractor
    )
    
    # Step 6: Entity preservation evaluation
    entity_preservation, entities_preserved, entities_total = evaluate_entity_preservation(
        answer, merged_memory, preprocessor
    )
    
    # Calculate compression ratio
    compression_ratio = len(merged_memory) / len(answer) if len(answer) > 0 else 0.0
    
    print(f"KP-Recall: {key_point_recall:.2%} | Entity-Pres: {entity_preservation:.2%} ({entities_preserved}/{entities_total}) | Compressed: {compression_ratio:.1%}")
    
    # Return results
    return {
        'sample_id': sample_id,
        'domain': sample.get('domain', 'Unknown'),
        'difficulty_level': sample.get('difficulty_level', 'Unknown'),
        'key_point_recall': key_point_recall,
        'entity_preservation': entity_preservation,
        'entities_preserved': entities_preserved,
        'entities_total': entities_total,
        'compression_ratio': compression_ratio,
        'phrase_precision': phrase_precision,
        'phrase_recall': phrase_recall,
        'phrase_f1': phrase_f1,
        'matched_points': matched_points,
        'missed_points': missed_points,
        'answer_length': len(answer),
        'memory_length': len(merged_memory),
        'num_key_points': len(key_points),
        'num_gold_phrases': len(gold_phrases)
    }


# =============================================================================
# Main Evaluation
# =============================================================================
def load_dataset(dataset_name: str) -> Tuple[List[Dict], str, str]:
    """Load evaluation dataset"""
    datasets = {
        'programming': ("data/evaluation_dataset.json", "Programming-Focused"),
        'lmsys': ("data/lmsys_evaluation_dataset.json", "LMSYS Real-World")
    }
    filename, label = datasets.get(dataset_name, datasets['programming'])
    dataset_path = Path(__file__).parent / filename
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f), str(dataset_path), label


def compute_metrics(valid_results: List[Dict]) -> Dict:
    """Compute metrics from valid results"""
    kp_recalls = [r['key_point_recall'] for r in valid_results]
    entity_preservations = [r['entity_preservation'] for r in valid_results]
    
    total_answer_length = sum(r['answer_length'] for r in valid_results)
    total_memory_length = sum(r['memory_length'] for r in valid_results)
    compression_ratio = float(total_memory_length / total_answer_length) if total_answer_length > 0 else 0.0
    
    return {
        'kp_recall': float(np.mean(kp_recalls)),
        'kp_recall_ci': bootstrap_ci_mean(kp_recalls),
        'entity_pres': float(np.mean(entity_preservations)),
        'entity_pres_ci': bootstrap_ci_mean(entity_preservations),
        'compression': compression_ratio,
        'compression_ci': bootstrap_ci_compression_ratio(valid_results),
        'phrase_precision': float(np.mean([r['phrase_precision'] for r in valid_results])),
        'phrase_precision_ci': bootstrap_ci_mean([r['phrase_precision'] for r in valid_results]),
        'phrase_recall': float(np.mean([r['phrase_recall'] for r in valid_results])),
        'phrase_recall_ci': bootstrap_ci_mean([r['phrase_recall'] for r in valid_results]),
        'phrase_f1': float(np.mean([r['phrase_f1'] for r in valid_results])),
        'phrase_f1_ci': bootstrap_ci_mean([r['phrase_f1'] for r in valid_results])
    }


def print_metrics(metrics: Dict, entity_metrics: Dict = None, dataset_label: str = ""):
    """Print metrics in formatted output"""
    m = metrics
    print(f"\n{'='*80}")
    
    if entity_metrics:
        em = entity_metrics
        print(f"COMPARISON: Standard vs Entity-Aware ({dataset_label})")
        print(f"{'='*80}")
        print(f"\n📊 Standard Method:")
        print(f"  ✓ Key-Point Recall:      {m['kp_recall']:.2%}  (95% CI: {m['kp_recall_ci'][0]:.2%}—{m['kp_recall_ci'][1]:.2%})")
        print(f"  ✓ Entity Preservation:   {m['entity_pres']:.2%}  (95% CI: {m['entity_pres_ci'][0]:.2%}—{m['entity_pres_ci'][1]:.2%})")
        print(f"  ✓ Compression to:        {m['compression']:.2%}  (95% CI: {m['compression_ci'][0]:.2%}—{m['compression_ci'][1]:.2%})")
        print(f"\n🧠 Entity-Aware Method:")
        print(f"  ✓ Key-Point Recall:      {em['kp_recall']:.2%}  (95% CI: {em['kp_recall_ci'][0]:.2%}—{em['kp_recall_ci'][1]:.2%})")
        print(f"  ✓ Entity Preservation:   {em['entity_pres']:.2%}  (95% CI: {em['entity_pres_ci'][0]:.2%}—{em['entity_pres_ci'][1]:.2%})")
        print(f"  ✓ Compression to:        {em['compression']:.2%}  (95% CI: {em['compression_ci'][0]:.2%}—{em['compression_ci'][1]:.2%})")
        print(f"\n📈 Improvement:")
        kp_diff = em['kp_recall'] - m['kp_recall']
        ent_diff = em['entity_pres'] - m['entity_pres']
        print(f"  {'📈' if kp_diff > 0 else '📉'} Key-Point: {kp_diff:+.2%} ({m['kp_recall']:.2%}→{em['kp_recall']:.2%})")
        print(f"  {'📈' if ent_diff > 0 else '📉'} Entity: {ent_diff:+.2%} ({m['entity_pres']:.2%}→{em['entity_pres']:.2%})")
    else:
        print("PRIMARY METRICS")
        print(f"{'='*80}")
        print(f"  ✓ Key-Point Recall:      {m['kp_recall']:.2%}  (95% CI: {m['kp_recall_ci'][0]:.2%}—{m['kp_recall_ci'][1]:.2%})")
        print(f"  ✓ Entity Preservation:   {m['entity_pres']:.2%}  (95% CI: {m['entity_pres_ci'][0]:.2%}—{m['entity_pres_ci'][1]:.2%})")
        print(f"  ✓ Compression to:        {m['compression']:.2%}  (95% CI: {m['compression_ci'][0]:.2%}—{m['compression_ci'][1]:.2%})")
        print(f"\nSECONDARY METRICS (Phrase-level)")
        print(f"{'='*80}")
        print(f"  • Phrase Precision:      {m['phrase_precision']:.2%}  (95% CI: {m['phrase_precision_ci'][0]:.2%}—{m['phrase_precision_ci'][1]:.2%})")
        print(f"  • Phrase Recall:         {m['phrase_recall']:.2%}  (95% CI: {m['phrase_recall_ci'][0]:.2%}—{m['phrase_recall_ci'][1]:.2%})")
        print(f"  • Phrase F1:             {m['phrase_f1']:.2%}  (95% CI: {m['phrase_f1_ci'][0]:.2%}—{m['phrase_f1_ci'][1]:.2%})")


def print_breakdowns(valid_results: List[Dict]):
    """Print domain, difficulty, and source dataset breakdowns"""
    # Source dataset breakdown
    print(f"\n{'='*80}")
    print("Breakdown by Source Dataset:")
    print(f"{'='*80}")
    by_source = defaultdict(list)
    for r in valid_results:
        source = r.get('sample_id', '').split('_')[0]  # Extract from ID (programming/lmsys)
        if not source or source not in ['programming', 'lmsys']:
            # Fallback: check domain patterns
            if r.get('domain') in ['Python', 'JavaScript', 'System Design', 'Databases', 'Algorithms']:
                source = 'programming'
            else:
                source = 'lmsys'
        by_source[source].append(r)
    for source, results in sorted(by_source.items()):
        kp = np.mean([r['key_point_recall'] for r in results])
        print(f"  {source:20s}: KP={kp:.2%} (n={len(results)})")
    
    # Domain breakdown
    print(f"\n{'='*80}")
    print("Breakdown by Domain:")
    print(f"{'='*80}")
    by_domain = defaultdict(list)
    for r in valid_results:
        by_domain[r['domain']].append(r)
    for domain, results in by_domain.items():
        kp = np.mean([r['key_point_recall'] for r in results])
        print(f"  {domain:20s}: KP={kp:.2%} (n={len(results)})")
    
    # Difficulty breakdown
    print(f"\n{'='*80}")
    print("Breakdown by Difficulty:")
    print(f"{'='*80}")
    by_difficulty = defaultdict(list)
    for r in valid_results:
        by_difficulty[r['difficulty_level']].append(r)
    for difficulty, results in by_difficulty.items():
        kp = np.mean([r['key_point_recall'] for r in results])
        print(f"  {difficulty:20s}: KP={kp:.2%} (n={len(results)})")


def build_results_dict(metrics: Dict, entity_metrics: Dict, valid_results: List[Dict],
                       all_results: List[Dict], entity_aware_results: List[Dict],
                       args, memory_model: str, max_chars: int, matcher) -> Dict:
    """Build results dictionary for JSON export"""
    m = metrics
    results_dict = {
        'config': {
            'memory_model': memory_model,
            'model_type': args.model,
            'dataset': 'combined',
            'max_chars': max_chars,
            'token_threshold': matcher.token_threshold,
            'similarity_threshold': matcher.similarity_threshold,
            'samples_evaluated': len(valid_results),
            'comparison_mode': args.compare_modes,
            'entity_aware': args.entity_aware,
            'random_seed': args.seed
        },
        'primary_metrics': {
            'key_point_recall': m['kp_recall'],
            'key_point_recall_ci95': [m['kp_recall_ci'][0], m['kp_recall_ci'][1]],
            'entity_preservation': m['entity_pres'],
            'entity_preservation_ci95': [m['entity_pres_ci'][0], m['entity_pres_ci'][1]],
            'compression_ratio': m['compression'],
            'compression_ratio_ci95': [m['compression_ci'][0], m['compression_ci'][1]]
        },
        'secondary_metrics': {
            'phrase_precision': m['phrase_precision'],
            'phrase_precision_ci95': [m['phrase_precision_ci'][0], m['phrase_precision_ci'][1]],
            'phrase_recall': m['phrase_recall'],
            'phrase_recall_ci95': [m['phrase_recall_ci'][0], m['phrase_recall_ci'][1]],
            'phrase_f1': m['phrase_f1'],
            'phrase_f1_ci95': [m['phrase_f1_ci'][0], m['phrase_f1_ci'][1]]
        },
        'detailed_results': all_results
    }
    
    if entity_metrics:
        em = entity_metrics
        results_dict['entity_aware_metrics'] = {
            'key_point_recall': em['kp_recall'],
            'key_point_recall_ci95': [em['kp_recall_ci'][0], em['kp_recall_ci'][1]],
            'entity_preservation': em['entity_pres'],
            'entity_preservation_ci95': [em['entity_pres_ci'][0], em['entity_pres_ci'][1]],
            'compression_ratio': em['compression'],
            'compression_ratio_ci95': [em['compression_ci'][0], em['compression_ci'][1]]
        }
        results_dict['improvements'] = {
            'key_point_recall_improvement': em['kp_recall'] - m['kp_recall'],
            'entity_preservation_improvement': em['entity_pres'] - m['entity_pres']
        }
        results_dict['entity_aware_detailed_results'] = entity_aware_results
    
    return results_dict


def process_and_save_results(all_results: List[Dict], entity_aware_results: List[Dict], 
                             args, memory_model: str, matcher: KeyPointMatcher, max_chars: int):
    """Process and save results (combined from all datasets)"""
    valid_results = [r for r in all_results if 'error' not in r]
    if not valid_results:
        print("No valid results")
        return None
    
    # Compute standard metrics
    metrics = compute_metrics(valid_results)
    
    # Compute entity-aware metrics if in comparison mode
    entity_metrics = None
    if args.compare_modes and entity_aware_results:
        valid_entity_results = [r for r in entity_aware_results if 'error' not in r]
        if valid_entity_results:
            entity_metrics = compute_metrics(valid_entity_results)
    
    # Print results
    print_metrics(metrics, entity_metrics, "Combined Dataset")
    print_breakdowns(valid_results)
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode_suffix = "comparison" if args.compare_modes else ("entity_aware" if args.entity_aware else "standard")
    
    # Save JSON
    results_to_save = build_results_dict(metrics, entity_metrics, valid_results, all_results, 
                                         entity_aware_results, args, memory_model, 
                                         max_chars, matcher)
    output_path = results_dir / f"evaluation_results_{args.model}_{mode_suffix}_{timestamp}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    
    # Save summary report
    report_path = results_dir / f"evaluation_report_{args.model}_{mode_suffix}_{timestamp}.txt"
    save_text_report(report_path, metrics, entity_metrics, valid_results, args, memory_model, max_chars, matcher)
    print(f"Report saved to: {report_path}")
    
    return {
        'kp_recall': metrics['kp_recall'],
        'kp_recall_ci': metrics['kp_recall_ci'],
        'entity_pres': metrics['entity_pres'],
        'entity_pres_ci': metrics['entity_pres_ci'],
        'compression': metrics['compression'],
        'compression_ci': metrics['compression_ci'],
        'entity_metrics': entity_metrics
    }


def save_text_report(report_path: Path, metrics: Dict, entity_metrics: Dict, valid_results: List[Dict],
                     args, memory_model: str, max_chars: int, matcher):
    """Save text report to file"""
    m = metrics
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write(f"{'='*80}\nMemory Compression Evaluation - Combined Dataset\n{'='*80}\n\n")
        f.write(f"Config: {args.model} | {memory_model} | {max_chars} chars | {len(valid_results)} samples | seed={args.seed}\n\n")
        
        # Metrics
        if entity_metrics:
            em = entity_metrics
            f.write(f"{'='*80}\nCOMPARISON: Standard vs Entity-Aware\n{'='*80}\n\n")
            f.write(f"Standard:\n")
            f.write(f"  KP-Recall: {m['kp_recall']:.2%} (CI: {m['kp_recall_ci'][0]:.2%}—{m['kp_recall_ci'][1]:.2%})\n")
            f.write(f"  Entity-Pres: {m['entity_pres']:.2%} (CI: {m['entity_pres_ci'][0]:.2%}—{m['entity_pres_ci'][1]:.2%})\n")
            f.write(f"  Compression: {m['compression']:.2%} (CI: {m['compression_ci'][0]:.2%}—{m['compression_ci'][1]:.2%})\n\n")
            f.write(f"Entity-Aware:\n")
            f.write(f"  KP-Recall: {em['kp_recall']:.2%} (CI: {em['kp_recall_ci'][0]:.2%}—{em['kp_recall_ci'][1]:.2%})\n")
            f.write(f"  Entity-Pres: {em['entity_pres']:.2%} (CI: {em['entity_pres_ci'][0]:.2%}—{em['entity_pres_ci'][1]:.2%})\n")
            f.write(f"  Compression: {em['compression']:.2%} (CI: {em['compression_ci'][0]:.2%}—{em['compression_ci'][1]:.2%})\n\n")
            f.write(f"Improvement: KP {em['kp_recall']-m['kp_recall']:+.2%} | Entity {em['entity_pres']-m['entity_pres']:+.2%}\n\n")
        else:
            f.write(f"{'='*80}\nPRIMARY METRICS\n{'='*80}\n")
            f.write(f"  ✓ Key-Point Recall:    {m['kp_recall']:.2%}  (CI: {m['kp_recall_ci'][0]:.2%}—{m['kp_recall_ci'][1]:.2%})\n")
            f.write(f"  ✓ Entity Preservation: {m['entity_pres']:.2%}  (CI: {m['entity_pres_ci'][0]:.2%}—{m['entity_pres_ci'][1]:.2%})\n")
            f.write(f"  ✓ Compression to:      {m['compression']:.2%}  (CI: {m['compression_ci'][0]:.2%}—{m['compression_ci'][1]:.2%})\n")
            avg_orig = int(np.mean([r['answer_length'] for r in valid_results]))
            avg_comp = int(np.mean([r['memory_length'] for r in valid_results]))
            f.write(f"     (Original→Compressed: avg {avg_orig}→{avg_comp} chars)\n\n")
            f.write(f"{'='*80}\nSECONDARY METRICS\n{'='*80}\n")
            f.write(f"  Phrase Precision: {m['phrase_precision']:.2%}  (CI: {m['phrase_precision_ci'][0]:.2%}—{m['phrase_precision_ci'][1]:.2%})\n")
            f.write(f"  Phrase Recall:    {m['phrase_recall']:.2%}  (CI: {m['phrase_recall_ci'][0]:.2%}—{m['phrase_recall_ci'][1]:.2%})\n")
            f.write(f"  Phrase F1:        {m['phrase_f1']:.2%}  (CI: {m['phrase_f1_ci'][0]:.2%}—{m['phrase_f1_ci'][1]:.2%})\n\n")
        
        # Breakdowns
        f.write(f"{'='*80}\nBreakdowns\n{'='*80}\n\n")
        
        # Source dataset
        f.write("By Source Dataset:\n")
        by_source = defaultdict(list)
        for r in valid_results:
            source = r.get('sample_id', '').split('_')[0]
            if not source or source not in ['programming', 'lmsys']:
                if r.get('domain') in ['Python', 'JavaScript', 'System Design', 'Databases', 'Algorithms']:
                    source = 'programming'
                else:
                    source = 'lmsys'
            by_source[source].append(r)
        for source, results in sorted(by_source.items()):
            kp = np.mean([r['key_point_recall'] for r in results])
            f.write(f"  {source:15s}: KP={kp:.2%} (n={len(results)})\n")
        
        # Domain
        f.write("\nBy Domain:\n")
        by_domain = defaultdict(list)
        for r in valid_results:
            by_domain[r['domain']].append(r)
        for domain, results in by_domain.items():
            kp = np.mean([r['key_point_recall'] for r in results])
            f.write(f"  {domain:20s}: KP={kp:.2%} (n={len(results)})\n")
        
        # Difficulty
        f.write("\nBy Difficulty:\n")
        by_diff = defaultdict(list)
        for r in valid_results:
            by_diff[r['difficulty_level']].append(r)
        for diff, results in by_diff.items():
            kp = np.mean([r['key_point_recall'] for r in results])
            f.write(f"  {diff:15s}: KP={kp:.2%} (n={len(results)})\n")
        f.write(f"\n{'='*80}\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Memory Compression Evaluation Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemini',
        choices=['gemini', 'phi3'],
        help='Model to use for memory generation (gemini or phi3)'
    )
    parser.add_argument(
        '--max-chars',
        type=int,
        default=1200,
        help='Maximum characters for memory compression'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of samples to evaluate (0 for all)'
    )
    parser.add_argument(
        '--token-threshold',
        type=float,
        default=0.66,
        help='Token coverage threshold for Level 2 matching'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.75,
        help='Sentence similarity threshold for Level 3 matching'
    )
    parser.add_argument(
        '--compare-modes',
        action='store_true',
        help='Compare standard vs entity-aware memory generation'
    )
    parser.add_argument(
        '--entity-aware',
        action='store_true',
        help='Use entity-aware memory generation (only when not comparing)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling datasets'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("="*80)
    print("Memory Compression Evaluation - Combined Dataset")
    print("="*80)
    
    # Set model
    memory_model = "gemini-2.5-flash-lite" if args.model == 'gemini' else "phi3:mini"
    print(f"\nModel: {memory_model}")
    print(f"Random Seed: {args.seed}")
    
    model = load_config()
    
    # Load and merge both datasets
    print("\nLoading datasets...")
    prog_data, _, prog_label = load_dataset('programming')
    lmsys_data, _, lmsys_label = load_dataset('lmsys')
    
    # Add source label to each sample
    for sample in prog_data:
        sample['source_dataset'] = 'programming'
    for sample in lmsys_data:
        sample['source_dataset'] = 'lmsys'
    
    # Merge and shuffle
    combined_dataset = prog_data + lmsys_data
    random.seed(args.seed)
    random.shuffle(combined_dataset)
    
    print(f"✓ Loaded {len(prog_data)} samples from {prog_label}")
    print(f"✓ Loaded {len(lmsys_data)} samples from {lmsys_label}")
    print(f"✓ Combined & shuffled: {len(combined_dataset)} total samples")
    
    # Initialize
    preprocessor = TextPreprocessor()
    anchor_extractor = AnchorExtractor(preprocessor)
    matcher = KeyPointMatcher(preprocessor, anchor_extractor, 
                              args.token_threshold, args.similarity_threshold)
    
    max_chars = args.max_chars
    
    # Determine sample size
    test_sample_size = len(combined_dataset) if args.samples == 0 else min(args.samples, len(combined_dataset))
    dataset_samples = combined_dataset[:test_sample_size]
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {len(dataset_samples)} samples (shuffled from both datasets)")
    print(f"{'='*80}")
    
    # Evaluate each sample
    all_results = []
    entity_aware_results = []
    
    # Run evaluations
    for i, sample in enumerate(dataset_samples, 1):
        print(f"\n{'='*80}")
        source = sample.get('source_dataset', 'unknown')
        print(f"Progress: {i}/{len(dataset_samples)} | Source: {source}")
        
        if args.compare_modes:
            # Comparison mode: evaluate both
            for mode_name, use_ea in [("Standard", False), ("Entity-Aware", True)]:
                print(f"\n--- {mode_name} ---")
                result = evaluate_single_sample(sample, model, preprocessor, anchor_extractor, 
                                               matcher, memory_model, max_chars, use_ea)
                (entity_aware_results if use_ea else all_results).append(result)
                time.sleep(2)
        else:
            # Single mode
            result = evaluate_single_sample(sample, model, preprocessor, anchor_extractor,
                                          matcher, memory_model, max_chars, args.entity_aware)
            all_results.append(result)
            time.sleep(2)
    
    # Process and display results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    summary = process_and_save_results(
        all_results=all_results,
        entity_aware_results=entity_aware_results,
        args=args,
        memory_model=memory_model,
        matcher=matcher,
        max_chars=max_chars
    )
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
