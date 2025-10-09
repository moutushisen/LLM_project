# Study Pal - AI Study Assistant

An intelligent document assistant powered by RAG (Retrieval-Augmented Generation) technology that remembers your learning preferences and background to provide personalized study support.

**Powered by:** LangChain • Google Gemini • Ollama • Streamlit

---

## ✨ Core Features

- **📄 PDF Document Analysis**: Smart PDF parsing with content-based Q&A and source citations
- **🤖 Dual Model Support**: Switch freely between Google Gemini API and local Ollama models
- **🧠 Personalized Memory**: AI remembers your learning style, background, and preferences for customized responses
- **🎨 Modern Interface**: Streamlit web UI with split-screen PDF preview and chat
- **💾 Privacy-First**: Memory data stored locally in SQLite for maximum privacy

## 🚀 Quick Start

### One-Line Installation (Recommended)

```bash
# 1. Create environment
conda create -n rag_system python=3.10 -y
conda activate rag_system

# 2. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install Ollama (for local models, optional)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull phi3:mini

# 4. Configure API keys
python setup_config.py

# 5. Launch application
python run_gui.py
```

### Configuration Guide

Run `python setup_config.py` and follow the prompts:
- **Google API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Default Models**: Choose Google or Ollama models
- **Memory System**: Enable personalized memory features

Config file location: `~/.config/llm_project/.env`

---

## 💡 User Guide

### Web Interface (Recommended)

```bash
python run_gui.py
# Access http://localhost:8501 (WSL users see WSL configuration below)
```

**Interface Features:**
- 📤 Upload PDF documents
- 💬 Chat with AI to analyze documents
- 🔄 Switch between Google/Ollama models
- 🧠 Manage personalized memory

### Personalized Memory System

Let AI remember your learning preferences for customized assistance:

**How to Use:**
1. Chat with AI and tell it about your background (e.g., "I'm a beginner, need simple explanations")
2. Click **"🧩 Generate & Merge"** in the sidebar
3. Refresh the page to start a new session
4. Upload a PDF and ask questions - AI will provide personalized responses based on your memory!

**Example Scenarios:**

| Scenario | Memory Settings | AI Response Style |
|----------|----------------|-------------------|
| 🎓 Beginner | "I'm a programming beginner, need simple explanations" | Uses analogies, avoids jargon, step-by-step approach |
| 🔬 Researcher | "I'm an ML researcher, need technical depth and citations" | Provides technical details, math formulas, research references |
| 📝 Exam Prep | "I'm preparing for exams, focus on formulas and key concepts" | Structured summaries, key formulas, concept highlights |

**Memory Controls:**
- 🧩 Generate & Merge: Create/update memory from conversation
- 🗑️ Clear Memory: Delete all saved memory
- 🔄 New Session: Reset chat and reload memory
- View Memory: Run `python print_memory.py`

---

## 📁 Project Structure

```
LLM_project/
├── rag_modules/           # Core RAG modules
│   ├── providers/         # Model providers (Google, Ollama)
│   ├── utils/             # Utility functions (PDF processing, etc.)
│   └── core/              # RAG chain builder
├── memory/                # Memory system
│   ├── generator.py       # Memory generation
│   └── rolling.py         # Rolling memory manager
├── data/                  # Data storage
│   └── memory.db          # SQLite database
├── streamlit_app.py       # Web application
├── run_gui.py             # GUI launcher (WSL optimized)
├── setup_config.py        # Configuration wizard
└── requirements.txt       # Python dependencies
```

---

## 🛠️ System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.9 - 3.11 |
| RAM | Minimum 8GB (16GB+ recommended) |
| OS | Linux / macOS / Windows (WSL2) |
| Disk Space | ~5GB (dependencies + models) |

---

## 🔧 Advanced Configuration

### Model Selection

**Google Models:**
```bash
# Available models: gemini-1.5-pro, gemini-1.5-flash
# Modify in .env file or switch via GUI interface
```

**Local Ollama Models:**
```bash
ollama list                    # List installed models
ollama pull mistral:7b         # Download additional models
ollama pull codellama:7b       # Code-specialized model
```

### GPU Acceleration (Optional)

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### WSL Network Configuration

If running in WSL and Windows browser cannot access `localhost:8501`:

```bash
# Option 1: Use optimized launcher
python run_gui.py

# Option 2: Check WSL IP
ip addr show eth0
# Access from Windows browser: http://[WSL_IP]:8501
```

---

## 🐛 Common Issues

<details>
<summary><b>Ollama Connection Failed</b></summary>

```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve
```
</details>

<details>
<summary><b>Import Errors</b></summary>

```bash
# Ensure environment is activated
conda activate rag_system

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```
</details>

<details>
<summary><b>Memory Not Saving</b></summary>

```bash
# Check database file
ls -la data/memory.db

# View memory content
python print_memory.py

# Reset database (deletes all memories!)
rm data/memory.db
```
</details>

<details>
<summary><b>Google API Errors</b></summary>

```bash
# Verify API key configuration
cat ~/.config/llm_project/.env | grep GOOGLE_API_KEY

# Reconfigure
python setup_config.py
```
</details>

---

## 📊 Testing Tools

```bash
python test_models.py              # Test model connections
python test_memory_integration.py  # Test memory system
python print_memory.py             # View memory content
```

---

## 📚 Documentation

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Model Library](https://ollama.ai/library)
- [Google Gemini API](https://ai.google.dev/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🔐 Security Notes

- ⚠️ Never commit API keys to the repository
- 🔒 Keys are stored in local `.env` file (ignored by version control)
- 🛡️ Memory data is stored locally only, never uploaded to the cloud

---

**Version:** v1.0  
**License:** MIT  
**Author:** Study Pal Team