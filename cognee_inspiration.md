# 🧠 Cognee 结构化记忆系统 - 对 LLM_project 的启发

## 📊 Cognee 核心原理

### **工作流程**
```
文本输入
    ↓
1. 实体提取 (Entity Extraction)
   - 使用 LLM 识别实体：人名、地点、概念等
   - 自动分类实体类型
    ↓
2. 关系抽取 (Relationship Extraction)  
   - 识别实体间的关系
   - 构建知识三元组 (主体-关系-客体)
    ↓
3. 知识图谱存储 (Knowledge Graph)
   - 图数据库: 存储节点和边
   - 向量数据库: LanceDB (语义检索)
    ↓
4. 智能检索 (Hybrid Retrieval)
   - 图遍历: 找相关实体
   - 向量搜索: 语义相似度
    ↓
返回结构化知识
```

---

## 🔬 Cognee vs 你的当前实现

### **你的 LLM_project (当前)**

```python
# memory/generator.py
def generate_merged_memory(chat_pairs, history_text, max_chars=1200):
    """
    压缩式记忆：
    1. 文本摘要压缩
    2. 去重合并
    3. 保持在固定长度内
    """
    # ✅ 简单高效
    # ✅ 保留上下文连贯性
    # ❌ 无结构化信息
    # ❌ 难以精确检索特定事实
```

**存储方式:**
- SQLite 存储文本字符串
- 纯文本压缩，无结构

**检索方式:**
- 直接读取完整记忆
- 无法针对性检索

---

### **Cognee 的实现**

```python
# agent.py line 500-542
def _handle_cognee_agent(message, memorizing, ...):
    if memorizing:
        # 1. 添加文本到知识库
        asyncio.run(cognee.add(formatted_message, dataset_name))
        
        # 2. 构建知识图谱 (cognify)
        asyncio.run(cognee.cognify(
            datasets=[dataset_name], 
            chunk_size=self.chunk_size
        ))
        # 内部做了什么：
        # - 提取实体 (Entity Extraction)
        # - 识别关系 (Relationship Extraction)
        # - 构建图结构 (Graph Construction)
        
    else:
        # 3. 图搜索 + 向量检索
        results = asyncio.run(cognee.search(
            query_text=message,
            top_k=self.retrieve_num,
            datasets=[dataset_name]
        ))
```

**存储方式:**
- **图数据库**: 实体和关系（节点和边）
- **向量数据库**: LanceDB (语义嵌入)

**检索方式:**
- 图遍历找相关实体
- 向量搜索找语义相似内容
- 混合结果返回

---

## 💡 对你的项目的启发

### **启发 1: 实体感知的记忆压缩**

**当前问题:**
```python
# 你的压缩可能丢失关键实体
"Einstein developed relativity theory" 
    ↓ [压缩]
"Theory development discussed"  # 丢失了 Einstein!
```

**Cognee 启发:**
```python
# 先提取实体，再压缩
entities = extract_entities(text)  # [Einstein, relativity theory]
relations = extract_relations(text)  # [(Einstein, developed, relativity)]

# 压缩时保护关键实体
compressed = compress_with_entity_protection(text, entities)
```

**实现建议:**
```python
# memory/entity_aware_generator.py
def generate_entity_aware_memory(chat_pairs, history_text, max_chars=1200):
    """实体感知的记忆生成"""
    
    # 1. 提取关键实体
    entities = extract_key_entities(chat_pairs)
    # 例如: ["Python", "类", "继承", "多态"]
    
    # 2. 构建压缩提示词，强调保留实体
    prompt = f"""
    压缩以下对话，必须保留这些关键术语: {', '.join(entities)}
    
    对话: {chat_pairs}
    
    压缩记忆:
    """
    
    # 3. 生成记忆
    memory = llm.invoke(prompt)
    
    # 4. 验证实体完整性
    missing = [e for e in entities if e not in memory]
    if missing:
        memory += f"\n关键词: {', '.join(missing)}"
    
    return memory
```

---

### **启发 2: 结构化记忆存储**

**当前存储 (rolling.py):**
```sql
CREATE TABLE rolling_memory (
    id TEXT PRIMARY KEY,
    updated_at TEXT,
    content TEXT  -- 纯文本字符串
)
```

**Cognee 启发的改进:**
```sql
-- 实体表
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    name TEXT,
    type TEXT,  -- PERSON, CONCEPT, LOCATION等
    description TEXT,
    created_at TEXT
);

-- 关系表  
CREATE TABLE relations (
    id TEXT PRIMARY KEY,
    source_entity_id TEXT,
    target_entity_id TEXT,
    relation_type TEXT,  -- "is_a", "related_to", "example_of"
    context TEXT
);

-- 记忆片段表
CREATE TABLE memory_chunks (
    id TEXT PRIMARY KEY,
    content TEXT,
    entities JSON,  -- 相关实体列表
    created_at TEXT
);
```

**使用示例:**
```python
# 存储
store_memory(
    content="Python的类支持继承",
    entities=["Python", "类", "继承"],
    relations=[
        ("Python", "has_feature", "类"),
        ("类", "supports", "继承")
    ]
)

# 检索
results = search_by_entity("继承")
# 返回: 所有与"继承"相关的记忆片段和关系
```

---

### **启发 3: 混合检索策略**

**当前 (你只有一种方式):**
```python
def get_text():
    # 返回完整记忆文本
    return self.rolling_memory.get_text()
```

**Cognee 的混合检索:**
```python
def hybrid_search(query, top_k=5):
    # 1. 实体匹配
    entity_results = find_by_entity_name(query)
    
    # 2. 关系遍历
    related_entities = traverse_relations(entity_results)
    
    # 3. 向量语义搜索 (如果有向量数据库)
    vector_results = semantic_search(query, top_k)
    
    # 4. 合并结果
    return merge_and_rank([entity_results, related_entities, vector_results])
```

**你可以实现:**
```python
# memory/structured_retriever.py
class StructuredMemoryRetriever:
    def retrieve(self, query: str, mode: str = "hybrid"):
        if mode == "entity":
            # 实体精确匹配
            return self._entity_search(query)
        elif mode == "relation":
            # 关系图遍历
            return self._relation_search(query)
        elif mode == "hybrid":
            # 混合策略
            entities = self._entity_search(query)
            related = self._expand_by_relations(entities)
            return entities + related
```

---

## 🚀 渐进式升级路径

### **阶段 1: 最小改动（立即可做）**

```python
# memory/generator.py 添加实体提取
def build_summary_prompt_with_entities(chat_pairs, history_text, max_chars):
    # 1. 提取关键实体
    entities = _extract_simple_entities(chat_pairs)
    
    # 2. 修改提示词
    return f"""
    请压缩以下对话，重点保留这些关键术语: {entities}
    
    历史记忆: {history_text}
    新对话: {chat_pairs}
    
    ⚠️ 必须包含所有关键术语！
    
    压缩记忆:
    """

def _extract_simple_entities(chat_pairs):
    """简单的实体提取（无需额外模型）"""
    # 使用规则提取：大写单词、专有名词等
    import re
    text = str(chat_pairs)
    
    # 提取大写开头的词
    capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
    
    # 提取常见概念词
    concepts = re.findall(r'\b(class|function|method|algorithm|concept)\b', text, re.I)
    
    return list(set(capitalized + concepts))
```

---

### **阶段 2: 添加结构化存储（中等改动）**

```python
# memory/structured_storage.py
class StructuredMemoryDB:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                type TEXT,
                count INTEGER DEFAULT 1
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id TEXT,
                entity_id INTEGER,
                FOREIGN KEY(entity_id) REFERENCES entities(id)
            )
        ''')
        conn.commit()
    
    def add_memory_with_entities(self, memory_text, entities):
        """存储记忆并关联实体"""
        memory_id = str(uuid.uuid4())
        
        for entity in entities:
            # 存储实体
            entity_id = self._get_or_create_entity(entity)
            # 关联记忆和实体
            self._link_memory_entity(memory_id, entity_id)
        
        return memory_id
    
    def search_by_entity(self, entity_name):
        """通过实体检索记忆"""
        # SQL: JOIN entities and memories
        pass
```

---

### **阶段 3: 完整知识图谱（大改动，可选）**

直接集成 Cognee 或类似库:

```python
# memory/graph_memory.py
import cognee

class GraphMemorySystem:
    async def add_knowledge(self, text, session_id):
        """添加知识到图谱"""
        await cognee.add(text, dataset_name=f"session_{session_id}")
        await cognee.cognify(datasets=[f"session_{session_id}"])
    
    async def query_graph(self, question, session_id, top_k=5):
        """图谱查询"""
        results = await cognee.search(
            query_text=question,
            top_k=top_k,
            datasets=[f"session_{session_id}"]
        )
        return results
```

---

## 📈 效果对比

### **场景: 学习 Python 类继承**

**原始对话:**
```
User: Python的类怎么继承？
AI: 使用class ChildClass(ParentClass)语法
User: 多重继承呢？
AI: class Child(Parent1, Parent2)
User: MRO是什么？
AI: Method Resolution Order，方法解析顺序
```

**当前压缩结果:**
```
"讨论了Python类的继承语法，包括单继承和多重继承，以及方法解析顺序的概念"
```
❌ 问题: 丢失了具体语法细节

**实体感知压缩:**
```
关键实体: Python, class, 继承, MRO
记忆: Python类使用class Child(Parent)实现继承，多重继承用class Child(P1,P2)，
     MRO是方法解析顺序
```
✅ 改进: 保留了关键术语和语法

**知识图谱存储:**
```
实体:
- Python (语言)
- class (概念)
- 继承 (概念)
- MRO (概念)

关系:
- Python → has_feature → class
- class → supports → 继承
- 继承 → has_mechanism → MRO
- MRO → full_name → "Method Resolution Order"

检索 "继承" 时:
→ 找到实体 "继承"
→ 遍历关系找到 class, MRO, Python
→ 返回完整的知识网络
```
✅✅ 最佳: 完整的知识关联

---

## 🎯 推荐的实施方案

### **短期 (1-2周): 实体感知压缩**
1. 修改 `memory/generator.py` 添加简单实体提取
2. 在提示词中强制保留关键实体
3. 测试压缩质量是否提升

### **中期 (1个月): 结构化存储**
1. 扩展 `memory/rolling.py` 添加实体表
2. 实现基于实体的检索
3. 添加统计功能（哪些概念最常讨论）

### **长期 (可选): 知识图谱**
1. 评估是否真的需要（小项目可能不需要）
2. 如需要，集成 Cognee 或自建轻量图谱
3. 实现可视化（显示知识关联）

---

## 💻 示例代码模板

### **立即可用的实体感知版本**

```python
# memory/entity_aware_generator.py
import re
from typing import List, Tuple, Set

def extract_key_terms(text: str) -> Set[str]:
    """提取关键术语（简单版本）"""
    # 1. 大写单词
    capitalized = set(re.findall(r'\b[A-Z][a-z]+\b', text))
    
    # 2. 专业术语（可扩展）
    tech_terms = set(re.findall(
        r'\b(class|function|method|algorithm|API|database|'
        r'machine learning|neural network|regression|'
        r'inheritance|polymorphism|encapsulation)\b', 
        text, re.IGNORECASE
    ))
    
    # 3. 代码关键字
    code_keywords = set(re.findall(r'`([^`]+)`', text))
    
    return capitalized | tech_terms | code_keywords

def build_entity_aware_prompt(
    chat_pairs: List[Tuple[str, str]], 
    history_text: str, 
    max_chars: int
) -> str:
    """构建实体感知的压缩提示词"""
    
    # 提取关键实体
    all_text = "\n".join([f"{q} {a}" for q, a in chat_pairs])
    entities = extract_key_terms(all_text + " " + history_text)
    
    entities_str = ", ".join(sorted(entities)[:20])  # 限制20个
    
    return f"""You are creating a memory summary for a learning assistant.

CRITICAL: You MUST preserve these key terms/entities in the summary:
{entities_str}

Previous Memory:
{history_text[:max_chars]}

New Conversation:
{all_text[:1000]}

Create a compressed memory (max {max_chars} chars) that:
1. Includes ALL key terms listed above
2. Maintains context and relationships
3. Stays concise and readable

Compressed Memory:"""

# 在 generator.py 中使用
def generate_merged_memory(
    chat_pairs: List[Tuple[str, str]],
    history_text: str,
    max_chars: int = 1200,
    model_name: str = None,
    entity_aware: bool = True  # 新参数
) -> str:
    # ... existing code ...
    
    if entity_aware:
        prompt = build_entity_aware_prompt(chat_pairs, history_text, max_chars)
    else:
        prompt = build_summary_prompt(chat_pairs, history_text, max_chars)
    
    # ... rest of code ...
```

---

## 🔗 相关资源

- **Cognee GitHub**: https://github.com/topoteretes/cognee
- **知识图谱入门**: https://www.ibm.com/topics/knowledge-graph
- **LangChain Graph**: https://python.langchain.com/docs/use_cases/graph/
- **Neo4j (图数据库)**: https://neo4j.com/

---

## 📝 总结

**Cognee 给你的核心启发:**

1. ✅ **实体保护**: 压缩时不要丢失关键术语
2. ✅ **结构化存储**: 不只存文本，存实体和关系
3. ✅ **智能检索**: 不只返回记忆，返回相关知识网络
4. ✅ **可扩展性**: 从简单文本到复杂图谱，渐进升级

**最重要的一句话:**
> 记忆不只是压缩文本，而是提取和组织知识结构！

立即开始第一步：**添加实体感知压缩**，你会看到明显的改进！🚀

