# RAG Q&A System

A modular Retrieval-Augmented Generation (RAG) Q&A system built with LangChain. Supports both Google Gemini API and local Ollama models with automatic fallback to chat-only mode.

## ✨ Features

- **Dual Providers**: Google Gemini API or local Ollama models
- **RAG Mode**: PDF document analysis with source citations
- **Chat Mode**: Direct conversation when no PDF is loaded
- **Web GUI**: Modern Streamlit interface with PDF preview
- **CLI Interface**: Command-line option for terminal users
- **WSL Optimized**: Network configuration for Windows Subsystem for Linux

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

For local Ollama support:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull phi3:mini
```

### 2. Configure API Key
```bash
python setup_config.py
```

### 3. Launch Application

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

## 📁 Project Structure

```
LLM_project/
├── rag_modules/
│   ├── providers/          # Model providers
│   ├── utils/              # Utilities
│   ├── core/               # Core functionality
│   └── app.py              # CLI application
├── streamlit_app.py        # Web GUI
├── run_gui.py              # GUI launcher (WSL optimized)
├── setup_config.py         # Configuration wizard
├── test_*.py               # Test scripts
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 🖥️ User Interfaces

### Web GUI (Streamlit)
- Split-screen layout with PDF preview and chat
- Page navigation and document viewer
- Real-time AI conversation with source citations
- Easy model switching between Google and local providers

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

## 🧹 Maintenance

Python bytecode caches are ignored by `.gitignore`. Manual cleanup if needed:
```bash
find . -name "__pycache__" -type d -exec rm -rf {} + -o -name "*.pyc" -delete
```

## 📊 Testing

Test different components:
```bash
python test_models.py      # Test model providers
python test_gui_deps.py    # Test GUI dependencies
python test_wsl_network.py # Test WSL network setup
```