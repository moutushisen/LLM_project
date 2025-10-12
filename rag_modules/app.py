import os
import time
import platform
from typing import Optional, List

from dotenv import load_dotenv
from pathlib import Path

from .providers.google_provider import query_google_models, get_google_providers
from .providers.local_provider import get_local_providers
from .utils import pdf_utils
from .core.chain_builder import create_rag_chain


class SimpleRAGApp:
    def __init__(self):
        # Load from user HOME config first (cross-platform), fallback to project .env
        system = platform.system()
        if system == "Windows":
            home_env = Path.home() / "AppData" / "Roaming" / "llm_project" / ".env"
        else:
            home_env = Path.home() / ".config" / "llm_project" / ".env"
        
        if home_env.exists():
            load_dotenv(dotenv_path=str(home_env))
        else:
            load_dotenv()
        self.vectorstore = None
        self.retrieval_chain = None
        self.llm = None
        self.current_model: Optional[str] = None
        self.model_type: Optional[str] = None
        self.available_google_models: List[str] = []
        self.splits = None
        self.current_pdf: Optional[str] = None
        self.chat_only: bool = False

    def print_header(self):
        print("\n" + "="*60)
        print("📚 Study Pal - Your Reading Helper")
        print("AI-Powered Document Assistant with Memory")
        print("="*60)

    def query_google_models(self) -> List[str]:
        models = query_google_models()
        self.available_google_models = models
        return models

    def setup_google_model(self, model_name: str = "gemini-1.5-flash", memory_context: Optional[str] = None) -> bool:
        try:
            providers = get_google_providers(model_name)
            if not providers:
                raise ValueError("Google AI not available or failed to initialize")
            
            embeddings, llm = providers
            self._configure_model(embeddings, llm, model_name, "google", memory_context)
            return True
        except Exception as e:
            print(f"Failed to setup Google model: {e}")
            return False

    def setup_local_model(self, model_name: str = "phi3:mini", memory_context: Optional[str] = None) -> bool:
        try:
            providers = get_local_providers(model_name)
            if not providers:
                raise ValueError("Local model not available or failed to initialize")
            
            embeddings, llm = providers
            self._configure_model(embeddings, llm, model_name, "local", memory_context)
            return True
        except Exception as e:
            print(f"Failed to setup local model: {e}")
            return False

    def _configure_model(self, embeddings, llm, model_name: str, model_type: str, memory_context: Optional[str] = None):
        """Helper method to configure model settings and avoid code duplication
        
        Args:
            embeddings: Embedding model
            llm: Language model
            model_name: Name of the model
            model_type: Type of model (google/local)
            memory_context: Optional memory context to personalize the assistant
        """
        # Always keep a reference to the LLM for auxiliary tasks (e.g., summarization)
        self.llm = llm
        if self.splits:
            self.vectorstore, self.retrieval_chain = create_rag_chain(self.splits, embeddings, llm, memory_context)
            self.chat_only = False
        else:
            # Chat-only mode
            self.vectorstore = None
            self.retrieval_chain = None
            self.chat_only = True
            print("Chat-only mode enabled (no PDF loaded)")
        
        self.current_model = model_name
        self.model_type = model_type
        print("RAG system ready!")

    def show_commands(self):
        print("\nAvailable Commands:")
        print("  /files      - View all PDF files")
        print("  /load       - Select and load specific PDF file")
        print("  /models     - Query Google API available models")
        print("  /switch     - Switch model")
        print("  /local      - Switch to local model")
        print("  /info       - Display current system information")
        print("  /help       - Display help")
        print("  /quit       - Exit system")
        print("\nEnter questions directly to start conversation!")

    def show_info(self):
        print(f"\nSystem Information:")
        print(f"  Current model: {self.current_model}")
        print(f"  Model type: {self.model_type}")
        print(f"  Current PDF: {os.path.basename(self.current_pdf) if self.current_pdf else 'Not loaded'}")
        print(f"  Document chunks: {len(self.splits) if self.splits else 0}")
        print(f"  System status: {'Ready' if self.retrieval_chain else 'Not ready'}")
        if self.current_pdf:
            print(f"  PDF path: {self.current_pdf}")

    def ask(self, question: str):
        try:
            if self.retrieval_chain:
                self._ask_rag_mode(question)
            elif self.llm:
                self._ask_chat_mode(question)
            else:
                print("Model not initialized")
                print(f"\nQuestion: {question}")
        except Exception as e:
            print(f"Error processing question: {e}")
    
    def _ask_rag_mode(self, question: str):
        """Handle RAG mode question answering"""
        print(f"\n🤔 Question: {question}")
        start_time = time.time()
        response = self.retrieval_chain.invoke({"input": question})
        end_time = time.time()

        print(f"\n🤖 Answer:")
        print("-" * 50)
        print(response.get("answer", ""))
        print("-" * 50)

        self._show_sources(response.get("context", []))
        print(f"Response time: {end_time - start_time:.2f}s | Model: {self.current_model}")
    
    def _ask_chat_mode(self, question: str):
        """Handle chat-only mode question answering"""
        print(f"\nQuestion: {question}")
        start_time = time.time()
        llm_response = self.llm.invoke(question)
        end_time = time.time()
        
        content = getattr(llm_response, 'content', None)
        if content is None:
            content = str(llm_response)
        
        print(f"\nAnswer:")
        print("-" * 50)
        print(content)
        print("-" * 50)
        print(f"Response time: {end_time - start_time:.2f}s | Model: {self.current_model}")
    
    def _show_sources(self, source_documents):
        """Helper method to display source documents"""
        if not source_documents:
            return
            
        print("Reference sources:")
        unique_sources = set()
        for doc in source_documents:
            page_num = doc.metadata.get('page', doc.metadata.get('page_number', 'N/A'))
            if page_num != 'N/A':
                page_num += 1
            source_info = f"  - Source: {os.path.basename(doc.metadata.get('source', 'Unknown file'))}, Page: {page_num}"
            if source_info not in unique_sources:
                unique_sources.add(source_info)
                print(source_info)
                content_preview = doc.page_content.strip().replace('\n', ' ')
                print(f"    \"...{content_preview[:120]}...\"")

    def switch_model(self):
        print("\nModel Switch:")
        print("1. Google API model")
        print("2. Local Ollama model")
        
        choice = input("Select model type (1/2): ").strip()
        
        try:
            if choice == "1":
                self._switch_to_google()
            elif choice == "2":
                self._switch_to_local()
            else:
                print("Invalid selection")
        except Exception as e:
            print(f"Switch failed: {e}")
    
    def _switch_to_google(self):
        """Handle switching to Google model"""
        if not self.available_google_models:
            print("Querying available models...")
            self.query_google_models()
        
        if not self.available_google_models:
            raise ValueError("No available Google models")
        
        print("\nAvailable Google models:")
        for i, model in enumerate(self.available_google_models[:10], 1):
            print(f"  {i}. {model}")
        
        model_choice = int(input("Select model (enter number): ")) - 1
        if not (0 <= model_choice < len(self.available_google_models)):
            raise ValueError("Invalid selection")
        
        selected_model = self.available_google_models[model_choice]
        if self.setup_google_model(selected_model):
            print(f"Switched to: {selected_model}")
        else:
            raise ValueError("Switch failed")
    
    def _switch_to_local(self):
        """Handle switching to local model"""
        if self.setup_local_model():
            print("Switched to local model")
        else:
            raise ValueError("Switch failed")

    def run(self):
        self.print_header()
        
        # Check for API key
        if not os.getenv("GOOGLE_API_KEY"):
            print("No Google API key found!")
            print("Run 'python setup_config.py' to configure your API key")
            print("   Or set GOOGLE_API_KEY environment variable")
            print()
        
        # Start in chat-only mode. Use /load command to add a PDF.
        print("Starting in chat-only mode. Use /load command to add a PDF for RAG functionality.")

        # Default provider
        if os.getenv("GOOGLE_API_KEY"):
            default_model = os.getenv("DEFAULT_GOOGLE_MODEL", "gemini-1.5-flash")
            self.setup_google_model(default_model)
        else:
            default_model = os.getenv("DEFAULT_LOCAL_MODEL", "phi3:mini")
            self.setup_local_model(default_model)

        self.show_commands()

        while True:
            try:
                user_input = input("\nEnter a question or command: ").strip()
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                else:
                    self.ask(user_input)
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _handle_command(self, user_input: str):
        """Handle command processing with simplified error handling"""
        cmd = user_input.lower()
        
        if cmd in ['/quit', '/q', '/exit']:
            print("Goodbye!")
            exit()
        elif cmd == '/models':
            self._handle_models_command()
        elif cmd == '/switch':
            self.switch_model()
        elif cmd == '/files':
            pdf_utils.show_pdf_files()
        elif cmd == '/load':
            self._handle_load_command()
        elif cmd == '/local':
            self._handle_local_command()
        elif cmd == '/info':
            self.show_info()
        elif cmd == '/help':
            self.show_commands()
        else:
            print("Unknown command, type /help for help")
    
    def _handle_models_command(self):
        """Handle /models command"""
        models = self.query_google_models()
        if models:
            print(f"\nFound {len(models)} Google models:")
            for i, model in enumerate(models[:10], 1):
                print(f"  {i}. {model}")
        else:
            print("No available models found")
    
    def _handle_load_command(self):
        """Handle /load command"""
        selected_pdf = pdf_utils.select_pdf_interactive()
        if not selected_pdf:
            return
            
        splits, pdf_path = pdf_utils.load_pdf(selected_pdf)
        if not splits:
            print("Load failed")
            return
            
        self.splits = splits
        self.current_pdf = pdf_path
        
        # Reinitialize model with new PDF
        if self.model_type == "google":
            self.setup_google_model(self.current_model or "gemini-1.5-flash")
        else:
            self.setup_local_model(self.current_model or "phi3:mini")
        
        print("PDF file loaded and RAG system updated")
    
    def _handle_local_command(self):
        """Handle /local command"""
        if self.setup_local_model():
            print("Switched to local model")
        else:
            print("Switch failed")


def main():
    app = SimpleRAGApp()
    app.run()


