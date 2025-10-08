# RAG Q&A System

A modular Retrieval-Augmented Generation (RAG) Q&A system built with LangChain. Supports both Google Gemini API and local Ollama models with automatic fallback to chat-only mode.

## ✨ Features

- **Dual Providers**: Google Gemini API or local Ollama models
- **RAG Mode**: PDF document analysis with source citations
- **Chat Mode**: Direct conversation when no PDF is loaded
- **Web GUI**: Modern Streamlit interface with PDF preview
- **CLI Interface**: Command-line option for terminal users
- **WSL Optimized**: Network configuration for Windows Subsystem for Linux
- **Memory System**: Rolling memory with automatic summarization and merging using LLM
  - Generate and merge conversation memories on-demand
  - Automatic compression when memory exceeds limit
  - SQLite-based persistent storage
  - Model-based intelligent summarization

## 📋 Prerequisites

- **Python**: 3.9 - 3.11 (recommended)
- **Operating System**: Linux, macOS, or Windows (WSL2 recommended for Windows)
- **RAM**: Minimum 8GB (16GB+ recommended for local models)
- **Disk Space**: ~5GB for dependencies and models

## 🚀 Complete Installation Guide

### Step 1: Create Conda Environment

**Option A: Using Conda (Recommended)**
```bash
# Create a new conda environment with Python 3.10
conda create -n rag_system python=3.10 -y

# Activate the environment
conda activate rag_system
```

**Option B: Using Python venv**
```bash
# Create virtual environment
python3 -m venv rag_env

# Activate on Linux/macOS
source rag_env/bin/activate

# Activate on Windows
.\rag_env\Scripts\activate
```

### Step 2: Install Python Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list | grep -E "langchain|streamlit|torch"
```

**Note**: Installation may take 5-10 minutes depending on your internet connection.

### Step 3: Install Ollama (For Local Models)

**Linux / WSL2:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Verify installation
ollama --version

# Pull recommended models
ollama pull phi3:mini        # Fast, efficient (2.3GB)
ollama pull gemma:2b          # Lightweight (1.4GB)
ollama pull llama2:7b         # Balanced performance (3.8GB)
```

**macOS:**
```bash
# Install via Homebrew
brew install ollama

# Or download from https://ollama.ai/download

# Pull models (same as Linux)
ollama pull phi3:mini
```

**Windows:**
```bash
# Download installer from https://ollama.ai/download
# Then open PowerShell and pull models:
ollama pull phi3:mini
```

### Step 4: Configure API Keys

```bash
# Run the configuration wizard
python setup_config.py

# Follow the prompts to:
# 1. Enter your Google API key (get from https://makersuite.google.com/app/apikey)
# 2. Select default models
# 3. Configure memory settings
```

**Manual Configuration (Alternative):**
```bash
# Create config directory
mkdir -p ~/.config/llm_project

# Create .env file
cat > ~/.config/llm_project/.env << EOF
GOOGLE_API_KEY=your_api_key_here
DEFAULT_GOOGLE_MODEL=gemini-1.5-flash
DEFAULT_LOCAL_MODEL=phi3:mini
MEMORY_ENABLED=true
MEMORY_DB_PATH=/home/mihoyohb/LLM_project/data/memory.db
EOF
```

### Step 5: Verify Installation

```bash
# Test model providers
python test_models.py

# Test GUI dependencies
python test_gui_deps.py

# For WSL users, test network setup
python test_wsl_network.py
```

### Step 6: Launch Application

**Web GUI (Recommended):**
```bash
python run_gui.py
```

**CLI Interface:**
```bash
python -m rag_modules.app
```

**Direct Streamlit:**
```bash
streamlit run streamlit_app.py
```

## 🎯 Quick Start Summary

```bash
# Complete setup in one go (copy and paste):

# 1. Create environment
conda create -n rag_system python=3.10 -y
conda activate rag_system

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install Ollama (Linux/WSL)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull phi3:mini

# 4. Configure
python setup_config.py

# 5. Launch
python run_gui.py
```

## 📁 Project Structure

```
LLM_project/
├── rag_modules/            # Core RAG functionality
│   ├── providers/          # Model providers (Google, Ollama)
│   │   ├── google_provider.py
│   │   └── local_provider.py
│   ├── utils/              # Utility functions
│   │   └── pdf_utils.py
│   ├── core/               # Core RAG chain
│   │   └── chain_builder.py
│   └── app.py              # CLI application entry point
├── memory/                 # Memory management system
│   ├── generator.py        # LLM-based memory generation
│   ├── rolling.py          # Rolling memory manager
│   └── storage.py          # SQLite storage layer
├── data/                   # Data storage
│   └── memory.db           # SQLite database (auto-created)
├── streamlit_app.py        # Web GUI application
├── run_gui.py              # GUI launcher (WSL optimized)
├── setup_config.py         # Configuration wizard
├── print_memory.py         # Memory inspection tool
├── test_models.py          # Model provider tests
├── test_gui_deps.py        # GUI dependency tests
├── test_wsl_network.py     # WSL network diagnostics
├── requirements.txt        # Python dependencies
├── TRANSLATION_SUMMARY.md  # Translation documentation
└── README.md               # This file
```

## 🖥️ User Interfaces

### Web GUI (Streamlit)
- Split-screen layout with PDF preview and chat
- Page navigation and document viewer
- Real-time AI conversation with source citations
- Easy model switching between Google and local providers
- Sidebar memory center for manual memory actions and optional injection

### Memory System

The system includes an intelligent rolling memory feature that helps maintain conversation context:

**Features:**
- **On-Demand Generation**: Click "Generate & Merge" to summarize current conversation
- **Intelligent Compression**: Automatically merges and compresses memories when length exceeds limit
- **Model-Based Summarization**: Uses LLM (Google Gemini or local model) to generate natural, human-like memory summaries
- **Persistent Storage**: SQLite database at `/home/mihoyohb/LLM_project/data/memory.db`
- **Rolling Window**: Maintains a single, continuously updated memory text that evolves with conversations
- **Privacy First**: All data stored locally, no external transmission

**How to Use:**
1. Open the sidebar "🧠 Memory (Rolling Text)"
2. Set your preferred memory length limit (200-5000 characters)
3. Have conversations with the assistant
4. Click "Generate & Merge" to create/update memory
5. View current memory in the read-only text area
6. Clear memory anytime with "Clear Memory" button

**Technical Details:**
- Memory generation uses the same model as your chat (Google or local)
- Prompts are designed for natural, concise summarization
- Automatic deduplication prevents repetitive content
- Read-back verification ensures data integrity

### CLI Interface
- Terminal-based interaction
- Interactive commands for PDF loading
- Text output with source references

## 🐛 WSL Troubleshooting

### GUI Access Issues in Windows Browser

**Problem**: Cannot access `localhost:8501` from Windows browser when running in WSL.

**Solution**:

1. **Use the optimized launcher**:
   ```bash
   python run_gui.py
   ```

2. **Network diagnostics**:
   ```bash
   python test_wsl_network.py
   ```

3. **Manual IP detection**:
   ```bash
   ip addr show eth0
   # Look for inet 192.168.x.x format
   ```

4. **Access via WSL IP**:
   ```
   http://[WSL_IP]:8501
   # Example: http://192.168.50.2:8501
   ```

5. **If still failing**:
   - Check Windows firewall settings
   - Run `wsl --shutdown` in PowerShell
   - Restart WSL and retry

### Dependency Installation Issues

If package installation fails:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

## 🔐 Security Notes

- **Never commit API keys** to the repository
- Store keys securely in local `.env` file (ignored by Git)
- Use `setup_config.py` for secure configuration
 - Memory config written by setup: `MEMORY_ENABLED=true`, `MEMORY_DB_PATH=/home/mihoyohb/LLM_project/data/memory.db`

## 🛠️ Advanced Configuration

### GPU Acceleration (Optional)

For faster embeddings and local model inference:

```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Install PyTorch with CUDA support (if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is being used
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Environment Variables

All configuration is stored in `~/.config/llm_project/.env`:

```bash
# Google API Configuration
GOOGLE_API_KEY=your_api_key_here

# Default Models
DEFAULT_GOOGLE_MODEL=gemini-1.5-flash
DEFAULT_LOCAL_MODEL=phi3:mini

# Memory System
MEMORY_ENABLED=true
MEMORY_DB_PATH=/home/mihoyohb/LLM_project/data/memory.db
```

### Custom Model Configuration

**For Google Models:**
```bash
# Available models: gemini-1.5-pro, gemini-1.5-flash, gemini-1.0-pro
# Edit in .env file or use GUI model switcher
```

**For Local Models:**
```bash
# Browse available models
ollama list

# Pull additional models
ollama pull mistral:7b
ollama pull codellama:7b

# Update DEFAULT_LOCAL_MODEL in .env
```

## 🧹 Maintenance

### Clean Python Cache
```bash
find . -name "__pycache__" -type d -exec rm -rf {} + -o -name "*.pyc" -delete
```

### Inspect Memory Database
```bash
# View current memory
python print_memory.py

# Or use SQLite directly
sqlite3 data/memory.db "SELECT * FROM rolling_memory;"
```

### Reset Configuration
```bash
# Remove existing config
rm -rf ~/.config/llm_project/.env

# Run setup again
python setup_config.py
```

### Update Dependencies
```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade langchain langchain-google-genai
```

## 📊 Testing

Test different components:
```bash
# Test model providers
python test_models.py

# Test GUI dependencies
python test_gui_deps.py

# Test WSL network setup (WSL only)
python test_wsl_network.py

# View memory content
python print_memory.py
```

## 🐛 Common Issues and Solutions

### Issue 1: Ollama Connection Failed
```bash
# Check if Ollama is running
ollama list

# If not running, start it
ollama serve

# Verify with
curl http://localhost:11434/api/tags
```

### Issue 2: Import Errors
```bash
# Ensure environment is activated
conda activate rag_system

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue 3: Google API Errors
```bash
# Verify API key is set
cat ~/.config/llm_project/.env | grep GOOGLE_API_KEY

# Test API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(os.path.expanduser('~/.config/llm_project/.env')); print(os.getenv('GOOGLE_API_KEY')[:10] + '...')"
```

### Issue 4: Memory Not Saving
```bash
# Check database file exists and is writable
ls -la data/memory.db

# Inspect database content
python print_memory.py

# Reset database (will delete all memories!)
rm data/memory.db
# Memory will be recreated on next use
```

## 📚 Additional Resources

- **LangChain Documentation**: https://python.langchain.com/
- **Ollama Models**: https://ollama.ai/library
- **Google Gemini API**: https://ai.google.dev/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **FAISS Documentation**: https://faiss.ai/