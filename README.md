# Study Pal - AI Document Assistant

An intelligent RAG-powered document assistant with personalized memory that remembers your learning preferences and provides customized study support.

**Tech Stack:** LangChain • Google Gemini • Ollama • Streamlit • FAISS

## 📁 Project Structure

```
LLM_project/
├── rag_modules/           # Core RAG implementation
│   ├── providers/         # Model providers (Google, Ollama)
│   ├── core/              # RAG chain builder
│   └── utils/             # PDF utilities
├── memory/                # Memory system
│   ├── generator.py       # Memory generation logic
│   └── rolling.py         # SQLite storage manager
├── data/                  # Local data storage
│   └── memory.db          # SQLite database
├── streamlit_app.py       # Streamlit web interface
├── run_gui.py             # GUI launcher (WSL optimized)
├── setup_config.py        # Configuration wizard
└── requirements.txt       # Python dependencies
```

---

## 📊 Testing

```bash
conda activate rag_system
python test_memory_integration.py  # Test memory system
python print_memory.py             # View memory content
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
conda create -n rag_system python=3.10 -y

# Activate environment
conda activate rag_system

# Verify Python version
python --version
```

**3. Install Dependencies**

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements (5-10 minutes)
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

# Start web interface
python run_gui.py
```

Access at: http://localhost:8501

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

## 🛠️ Advanced Configuration

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

---



## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Models](https://ollama.ai/library)
- [Google Gemini API](https://ai.google.dev/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

## 🔐 Security

- ⚠️ Never commit API keys to version control
- 🔒 Keys stored in `~/.config/llm_project/.env` (gitignored)
- 🛡️ All memory data stored locally, never uploaded

---

## 📝 License

MIT License - See LICENSE file for details

**Version:** 1.0  
**Maintained by:** Study Pal Team
