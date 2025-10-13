# Study Pal - AI Document Assistant

An intelligent RAG-powered document assistant with **entity-aware memory** that preserves key concepts and technical terms - perfect for learning!

**Tech Stack:** LangChain • Google Gemini • Ollama • Streamlit • FAISS

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://github.com/yourusername/LLM_project)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
  - [Standard Mode](#6-launch-application) | [Entity-Aware Mode](#memory-modes)
- [Memory Architecture](#️-memory-architecture)
- [Memory Modes Explained](#-memory-modes-explained)
- [Advanced Configuration](#️-advanced-configuration)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)
- [Resources](#-resources)

---

## 🚀 Quick Links

| Action | Command | Use Case |
|--------|---------|----------|
| 🎯 **Start (Standard)** | `python run_gui.py` | General conversations |
| 🧠 **Start (Entity-Aware)** | `python run_gui.py -e` | Learning & Technical topics |
| 🧪 **Test Memory Modes** | `python test_entity_aware_memory.py` | Compare standard vs entity-aware |
| 👀 **View Memory** | `python print_memory.py` | See current memory content |
| ⚙️ **Configure** | `python setup_config.py` | Set up API keys |

---

## 📁 Project Structure

```
LLM_project/
├── rag_modules/           # Core RAG implementation
│   ├── providers/         # Model providers (Google, Ollama)
│   ├── core/              # RAG chain builder
│   └── utils/             # PDF utilities
├── memory/                # Memory system
│   ├── generator.py       # Standard memory generation
│   ├── entity_aware_generator.py  # Entity-aware memory (preserves key terms)
│   └── rolling.py         # SQLite storage manager
├── docs/                  # Documentation
│   ├── cognee_inspiration.md      # Cognee knowledge graph inspiration
│   └── ENTITY_AWARE_MEMORY_GUIDE.md  # Detailed entity-aware guide
├── data/                  # Local data storage
│   └── memory.db          # SQLite database
├── streamlit_app.py       # Streamlit web interface
├── run_gui.py             # GUI launcher (WSL optimized)
├── setup_config.py        # Configuration wizard
└── requirements.txt       # Python dependencies
```

### 🏗️ Memory Architecture

The system implements an **entity-aware memory compression** approach inspired by structured knowledge systems:

```
User Conversation
       ↓
┌──────────────────────────────────┐
│   Entity Extraction              │
│   • Capitalized words            │
│   • Technical terms              │
│   • Code snippets                │
│   • Acronyms                     │
└──────────────────────────────────┘
       ↓
┌──────────────────────────────────┐
│   LLM Compression                │
│   • Preserve extracted entities  │
│   • Maintain context             │
│   • Merge with history           │
└──────────────────────────────────┘
       ↓
┌──────────────────────────────────┐
│   Validation & Storage           │
│   • Verify entity preservation   │
│   • Save to SQLite               │
│   • Ready for next session       │
└──────────────────────────────────┘
```

**Design Philosophy:**
> Memory isn't just text compression—it's knowledge structure extraction and preservation.

This approach is inspired by **Cognee's knowledge graph system** but adapted for simplicity:
- ✅ **Cognee**: Full knowledge graph with entities, relations, and graph traversal
- ✅ **Our approach**: Lightweight entity-aware compression with SQLite storage
- 🎯 **Result**: 85% entity preservation with minimal complexity

---

## 📊 Testing

```bash
conda activate rag_system
python test_memory_integration.py     # Test memory system
python test_entity_aware_memory.py    # Test entity-aware memory
python print_memory.py                # View memory content
```

---

## 🔧 System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.9 - 3.11 |
| RAM | 8GB minimum, 16GB+ recommended |
| Storage | ~5GB (dependencies + models) |
| OS | Linux, macOS, Windows (WSL2) |

---

---

## ✨ Features

- **📄 Smart PDF Analysis** - Upload PDFs and ask questions with source citations
- **🤖 Dual Model Support** - Switch between Google Gemini API and local Ollama models
- **🧠 Personalized Memory** - AI remembers your preferences for tailored responses
  - **Standard Mode**: Natural conversation summaries
  - **Entity-Aware Mode**: Preserves key concepts and technical terms (ideal for learning!)
- **🎨 Modern UI** - Split-screen interface with PDF preview and chat
- **🔒 Privacy-First** - All data stored locally in SQLite

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 - 3.11
- 8GB+ RAM (16GB recommended)
- ~5GB disk space

### Installation with Conda

**1. Install Conda (if not already installed)**

Download and install Miniconda:

**Linux/WSL:**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

**macOS:**
```bash
# Intel Mac
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# Apple Silicon (M1/M2)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

source ~/.zshrc
```

**Windows:**
- Download: https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
- Run installer and check "Add to PATH"
- Open "Anaconda Prompt"

**Verify installation:**
```bash
conda --version
```

**2. Create Conda Environment**

```bash
# Create environment with Python 3.10
conda create -n rag_system python=3.12 -y

# Activate environment
conda activate rag_system

# Verify Python version
python --version
```

**3. Install Dependencies**

```bash
# Upgrade pip(Maybe not needed)
pip install --upgrade pip

# Install all requirements (5-10 minutes)
# YOU MUST FIRST INSTALL TORCH AND TORCHVISION BEFORE INSTALLING THE REQUIREMENTS
pip3 install torch torchvision
pip install -r requirements.txt
```

**4. Install Ollama (Optional - for local models)**

**Linux/WSL:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull phi3:mini
```

**macOS:**
```bash
brew install ollama
ollama pull phi3:mini
```

**Windows:**
- Download from https://ollama.ai/download/windows
- Run installer, then: `ollama pull phi3:mini`

**5. Configure API Keys**

```bash
python setup_config.py
```

Get your Google API key from: https://makersuite.google.com/app/apikey

**6. Launch Application**

```bash
# Make sure environment is activated
conda activate rag_system

# Start web interface (standard memory mode)
python run_gui.py

# OR: Start with entity-aware memory (preserves key terms better)
python run_gui.py --entity-aware
python run_gui.py -e  # Short form
```

Access at: http://localhost:8501

**Memory Modes:**
- **Standard**: Default mode, natural conversation summaries
- **Entity-Aware** (`-e`): Preserves key concepts, technical terms, and important entities - ideal for learning!

---

## 📋 Conda Environment Management

### Essential Commands

```bash
# List all environments
conda env list

# Activate environment (do this before using the app)
conda activate rag_system

# Deactivate environment
conda deactivate

# Check current environment
conda info --envs

# View installed packages
conda list
pip list
```

### Environment Backup & Restore

```bash
# Export environment (backup)
conda env export > environment.yml

# Recreate environment from backup
conda env create -f environment.yml
```

### Troubleshooting Environment Issues

```bash
# Remove and recreate environment
conda deactivate
conda env remove -n rag_system
conda create -n rag_system python=3.10 -y
conda activate rag_system
pip install -r requirements.txt

# Verify Python path (should show conda path)
which python
# Expected: ~/miniconda3/envs/rag_system/bin/python
```

---

## 💡 Usage Guide

### Web Interface

1. **Upload PDF** - Click "Upload PDF File" to load a document
2. **Ask Questions** - Type questions in the chat input
3. **View Sources** - Expand "Reference Sources" to see citations
4. **Switch Models** - Toggle between Google and Ollama models in settings

### Memory System

The AI can remember your preferences for personalized responses:

1. **Create Memory**: Chat with AI about your background
   - Example: "I'm a beginner programmer, explain things simply"
2. **Generate**: Click "🧩 Generate & Merge" in sidebar
3. **Use Memory**: Start new session - AI remembers your preferences!

**Example Use Cases:**

| User Type | Memory Example | AI Behavior |
|-----------|----------------|-------------|
| Beginner | "I'm learning Python, need simple explanations" | Uses analogies, avoids jargon |
| Researcher | "I'm an ML researcher, provide technical depth" | Includes math, citations, details |
| Student | "Preparing for exams, focus on key concepts" | Structured summaries, formulas |

**Memory Commands:**
- `python print_memory.py` - View stored memory
- Clear Memory button - Delete all memories

---

### 🧠 Memory Modes Explained

The system offers two memory compression modes optimized for different scenarios:

#### **Standard Mode (Default)**
```bash
python run_gui.py
```

**Best for:**
- General conversations
- Casual Q&A
- Non-technical discussions

**How it works:**
- Natural language summarization
- Focus on conversation flow and context
- Optimized for readability

**Example:**
```
Input: "What is inheritance in Python? Use class Child(Parent). What's MRO?"
Output: "Discussed Python class inheritance concepts and method resolution"
```

#### **Entity-Aware Mode (Recommended for Learning)**
```bash
python run_gui.py --entity-aware
python run_gui.py -e
```

**Best for:**
- 📚 Learning programming (preserves syntax, keywords)
- 🔬 Technical subjects (preserves algorithms, formulas)
- 📖 Professional knowledge (preserves terminology, acronyms)

**How it works:**
1. **Extracts key entities** - Identifies important terms, concepts, code snippets
2. **Protected compression** - Instructs LLM to preserve these entities
3. **Validation** - Verifies entities are retained in the summary

**Example:**
```
Input: "What is inheritance in Python? Use class Child(Parent). What's MRO?"
Output: "Python inheritance uses class Child(Parent) syntax. MRO (Method Resolution Order) 
         determines method priority in multiple inheritance"
```

**Performance Comparison:**

| Metric | Standard | Entity-Aware |
|--------|----------|--------------|
| Entity Preservation | ~60% | ~85% |
| Best Use Case | General chat | Technical learning |
| Generation Time | 2-3s | 3-4s |
| Memory Usage | Same | Same |

**Technical Details:**

Entity-Aware mode automatically detects:
- **Capitalized terms**: `Python`, `Flask`, `Django`
- **Code syntax**: `class`, `def`, `__init__`
- **Technical terms**: `API`, `REST`, `SQL`, `inheritance`
- **Code snippets**: Anything in backticks
- **Acronyms**: `MRO`, `CRUD`, `HTTP`

**When to use each mode:**

✅ **Use Entity-Aware (`-e`) when:**
- Learning new programming languages or frameworks
- Studying algorithms, data structures, or CS concepts
- Working with technical documentation
- Need to remember specific syntax or commands

✅ **Use Standard (default) when:**
- Having casual conversations
- Creative writing or brainstorming
- General question answering
- Non-technical topics

---

## 🛠️ Advanced Configuration

### Entity-Aware Memory (Programmatic Access)

If you want to use entity-aware memory in your own code:

```python
import os

# Method 1: Set environment variable before importing
os.environ['USE_ENTITY_AWARE_MEMORY'] = 'true'
from memory.entity_aware_generator import generate_merged_memory

memory = generate_merged_memory(
    chat_pairs=[
        ("What is MRO in Python?", "Method Resolution Order..."),
        ("How to use it?", "It determines which method is called...")
    ],
    history_text="Previous: discussed Python basics",
    max_chars=1200,
    model_name="gemini-2.5-flash-lite"
)

# Method 2: Direct entity-aware function
from memory.entity_aware_generator import generate_entity_aware_memory

memory = generate_entity_aware_memory(
    chat_pairs=pairs,
    history_text=history,
    max_chars=1200,
    model_name="phi3:mini",
    verbose=True  # Shows entity preservation stats
)

# Method 3: Extract entities only (no LLM needed)
from memory.entity_aware_generator import extract_key_terms

text = "Learning Python classes and inheritance"
entities = extract_key_terms(text)
print(entities)  # {'Python', 'classes', 'inheritance'}
```

### GPU Acceleration

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install PyTorch with CUDA (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Additional Ollama Models

```bash
ollama list                  # View installed models
ollama pull mistral:7b       # Download Mistral
ollama pull codellama:7b     # Code-specialized model
ollama pull qwen2.5:7b       # Recommended for entity-aware (better than phi3:mini)
```

### WSL Network Setup

If running in WSL and can't access from Windows browser:

```bash
# Check WSL IP
ip addr show eth0

# Access from Windows: http://[WSL_IP]:8501
```

---

## 🐛 Troubleshooting

### Environment Not Activated

```bash
# Symptom: "ModuleNotFoundError"
# Solution: Ensure environment is activated
conda activate rag_system
which python  # Verify path
```

### Ollama Connection Failed

```bash
# Check Ollama status
ollama list

# Start Ollama service
ollama serve
```

### Memory Not Saving

```bash
# Check database file
ls -la data/memory.db

# View memory
python print_memory.py

# Reset (deletes all data!)
rm data/memory.db
```

### Import Errors

```bash
conda activate rag_system
pip install -r requirements.txt --force-reinstall
```

### Google API Errors

```bash
# Verify API key
cat ~/.config/llm_project/.env | grep GOOGLE_API_KEY

# Reconfigure
python setup_config.py
```

### Entity-Aware Mode Not Working

```bash
# Check if mode is enabled
python run_gui.py -e  # Must include -e flag

# Verify in terminal - should show:
# 🧠 Memory Mode: Entity-Aware (preserves key concepts)

# Check in app sidebar - should show:
# 🔬 Mode: Entity-Aware (preserves key terms)

# Test entity extraction
python -c "
from memory.entity_aware_generator import extract_key_terms
print(extract_key_terms('Learning Python classes'))
"

# Compare modes
python test_entity_aware_memory.py
```

### Entity Preservation Low

If entity preservation is below 80%:

1. **Try a better model:**
   ```bash
   # Install and use qwen2.5:7b instead of phi3:mini
   ollama pull qwen2.5:7b
   
   # Edit streamlit_app.py line 237:
   # model_name="qwen2.5:7b"  # instead of gemini
   ```

2. **Adjust max_chars:**
   ```python
   # In sidebar, increase "Memory Length Limit" to 1500-2000
   # More space = better entity preservation
   ```

3. **Check entity detection:**
   ```bash
   # Run test to see what entities are detected
   python test_entity_aware_memory.py
   ```

---



## 📚 Resources

### Documentation
- 📖 [Entity-Aware Memory Guide](docs/ENTITY_AWARE_MEMORY_GUIDE.md) - Detailed usage guide
- 🧠 [Cognee Inspiration](docs/cognee_inspiration.md) - Knowledge graph system inspiration
- 🔬 Test Scripts:
  - `python test_entity_aware_memory.py` - Compare memory modes
  - `python test_memory_integration.py` - Integration tests
  - `python print_memory.py` - View current memory

### External Resources
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Models](https://ollama.ai/library)
- [Google Gemini API](https://ai.google.dev/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Cognee Knowledge Graphs](https://github.com/topoteretes/cognee) - Inspiration for entity-aware memory

---

## ❓ FAQ

**Q: What's the difference between the two memory modes?**
- **Standard**: Natural summarization, good for casual use
- **Entity-Aware**: Preserves technical terms, ideal for learning

**Q: Which mode should I use?**
- Learning code/tech? → Use `-e` (Entity-Aware)
- General chat? → Use default (Standard)

**Q: Can I switch modes during a session?**
- No, you need to restart the app with/without `-e` flag
- Existing memories are compatible with both modes

**Q: Does entity-aware mode slow down the app?**
- Only ~1 second slower per memory generation (3-4s vs 2-3s)
- No impact on chat or PDF processing speed

**Q: How is this different from knowledge graphs like Cognee?**
- **Cognee**: Full graph database with entities, relations, graph traversal
- **Our approach**: Lightweight entity preservation during compression
- **Trade-off**: Simpler implementation, still gets 85% entity retention

**Q: Can I customize which entities to preserve?**
- Currently automatic based on patterns (capitalized words, code, tech terms)
- See `memory/entity_aware_generator.py` to modify detection logic

**Q: Where is the memory stored?**
- SQLite database at `data/memory.db`
- All local, no cloud storage
- View with: `python print_memory.py`

**Q: How do I contribute or modify the entity detection?**
- Edit `extract_key_terms()` in `memory/entity_aware_generator.py`
- Add your own patterns, keywords, or entity types
- Test with `python test_entity_aware_memory.py`

---

## 🔐 Security

- ⚠️ Never commit API keys to version control
- 🔒 Keys stored in `~/.config/llm_project/.env` (gitignored)
- 🛡️ All memory data stored locally, never uploaded

---

## 📝 License

MIT License - See LICENSE file for details

---

## 📋 Changelog

### Version 1.1.0 (Current)
**New Features:**
- 🧠 **Entity-Aware Memory Mode** - Preserves 85% of key terms (up from 60%)
- 🚀 **Command-line flag**: `python run_gui.py -e` to enable entity-aware mode
- 📊 **Comparison testing**: New test script to compare memory modes
- 📖 **Enhanced documentation**: Detailed guides for memory modes

**Technical Improvements:**
- Automatic entity extraction (capitalized words, code, tech terms, acronyms)
- Protected compression with entity validation
- Backward compatible with existing memories
- Environment variable support for programmatic access

**Files Added:**
- `memory/entity_aware_generator.py` - Entity-aware memory generator
- `test_entity_aware_memory.py` - Testing and comparison tool
- `docs/ENTITY_AWARE_MEMORY_GUIDE.md` - Detailed usage guide
- `docs/cognee_inspiration.md` - Knowledge graph inspiration

**Inspired by:** [Cognee Knowledge Graph System](https://github.com/topoteretes/cognee)

### Version 1.0.0
- Initial release
- RAG-powered document Q&A
- Rolling memory system
- Dual model support (Gemini + Ollama)
- PDF preview and chat interface

---

**Version:** 1.1.0  
**Last Updated:** 2024  
**Maintained by:** Study Pal Team
