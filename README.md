## RAG Q\&A System Overview

This is a modular Retrieval-Augmented Generation (RAG) Q\&A system built with LangChain. It supports both the **Google Gemini API** and **local Ollama models**. The system automatically switches to **Chat-only mode** when no PDF is loaded.


### Core Features

  * **Dual Providers**: Use either Google Gemini or a local Ollama model.
  * **RAG Mode**: After loading a PDF, the system will **retrieve and generate** answers, providing **source citations**.
  * **Chat-only Mode**: You can chat directly with the model when no PDF is loaded.
  * **CLI Interface**: A simple, interactive command-line interface.
  * **Modular Design**: The code is structured for easy extension and maintenance.

### Important API Key Notes

  * **Security**: **Never commit your API keys to the repository**. Store them securely in a local `.env` file, which is ignored by Git.
  * **Configuration**: Run `python setup_config.py` the first time to securely set up your Google API key.

### Quick Start

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    Note: For Ollama support, install ollama separately:
    ```bash
    curl -fsSL https://ollama.ai/install.sh | sh
    ollama pull phi3:mini
    ```
2.  **Configure API key**:
    ```bash
    python setup_config.py
    ```
3.  **Run the app**:
    ```bash
    python interactive_rag_simplified.py
    ```
## 📁 Project Structure

```
LLM_project/
├── rag_modules/
│   ├── providers/
│   │   ├── google_provider.py   # Google API provider
│   │   └── local_provider.py    # Local (Ollama + HF embeddings)
│   ├── utils/
│   │   └── pdf_utils.py         # PDF loading and splitting
│   ├── core/
│   │   └── chain_builder.py     # Vector store + retrieval chain
│   └── app.py                   # Orchestrator + CLI
├── interactive_rag_simplified.py # Launcher script
├── setup_config.py              # Configuration wizard
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🧹 Git Hygiene
Python bytecode caches are ignored by `.gitignore`:
- `__pycache__/`, `*.pyc` will not be committed

Cleanup caches manually if needed:
```bash
find . -name "__pycache__" -type d -exec rm -rf {} + -o -name "*.pyc" -delete
```