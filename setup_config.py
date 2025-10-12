#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import getpass
import platform
from pathlib import Path


def setup_google_api():
    """Setup Google API key"""
    print("\n🔑 Google API Setup")
    print("Get your API key: https://makersuite.google.com/app/apikey")
    print("⚠️  Never share your API key with anyone!\n")
    
    api_key = getpass.getpass("Enter your Google API key (hidden): ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return False
    
    if not api_key.startswith('AIza'):
        print("⚠️  API keys usually start with 'AIza'")
        confirm = input("Continue anyway? (y/N): ").strip().lower()
        if confirm != 'y':
            return False
    
    return api_key


def setup_models():
    """Setup default models"""
    print("\n🤖 Model Configuration")
    
    # Google model
    google_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
    print("Google models: 1) gemini-1.5-pro  2) gemini-1.5-flash  3) gemini-1.0-pro")
    
    choice = input("Select (1-3, default=2): ").strip()
    try:
        google_model = google_models[int(choice) - 1] if choice else google_models[1]
    except (ValueError, IndexError):
        google_model = google_models[1]
    
    # Local model
    local_model = input("Local model (default=phi3:mini): ").strip() or "phi3:mini"
    
    return google_model, local_model


def create_env_file(api_key, google_model, local_model):
    """Create .env file in user's home config directory"""
    project_root = Path.cwd()
    db_path = project_root / "data" / "memory.db"
    
    env_content = f"""# Google API Configuration
GOOGLE_API_KEY={api_key}

# Default models
DEFAULT_GOOGLE_MODEL={google_model}
DEFAULT_LOCAL_MODEL={local_model}

# Memory feature
MEMORY_ENABLED=true
MEMORY_DB_PATH={db_path.resolve()}
"""

    # Cross-platform config directory selection
    system = platform.system()
    if system == "Windows":
        # Windows: use AppData\Roaming
        config_dir = Path.home() / "AppData" / "Roaming" / "llm_project"
    else:
        # Linux/WSL/macOS: use .config
        config_dir = Path.home() / ".config" / "llm_project"
    
    config_dir.mkdir(parents=True, exist_ok=True)
    env_path = config_dir / ".env"
    
    with open(env_path, "w") as f:
        f.write(env_content)

    print(f"\n✅ Configuration saved to: {env_path}")
    return True


def main():
    """Main setup wizard"""
    print("\n📚 Study Pal - Configuration Setup")
    print("=" * 50)
    
    # Check if config already exists (cross-platform)
    system = platform.system()
    if system == "Windows":
        home_env = Path.home() / "AppData" / "Roaming" / "llm_project" / ".env"
    else:
        home_env = Path.home() / ".config" / "llm_project" / ".env"
    
    if home_env.exists():
        print(f"⚠️  Configuration already exists: {home_env}")
        overwrite = input("Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("❌ Setup cancelled")
            return
    
    # Setup Google API
    api_key = setup_google_api()
    if not api_key:
        print("❌ Setup cancelled - API key required")
        return
    
    # Setup models
    google_model, local_model = setup_models()
    
    # Create .env file
    if create_env_file(api_key, google_model, local_model):
        print("\n🎉 Setup complete!")
        print("\n📌 Next steps:")
        print("   python run_gui.py")
        print("\n🔒 Keep your API key secure!")


if __name__ == "__main__":
    main()
