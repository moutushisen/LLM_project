#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import getpass
from pathlib import Path


def setup_google_api():
    """Setup Google API key"""
    print("\n🔑 Google API Setup")
    print("=" * 40)
    print("1. Go to: https://makersuite.google.com/app/apikey")
    print("2. Create a new API key")
    print("3. Copy the API key and paste it below")
    print("⚠️  Warning: Don't share your API key with anyone!!!!")
    print("⚠️  Warning: Don't share your API key with anyone!!!!")
    print("⚠️  Warning: Don't share your API key with anyone!!!!")

    print()
    
    api_key = getpass.getpass("Enter your Google API key (hidden): ").strip()
    
    if not api_key:
        print("No API key provided")
        return False
    
    if not api_key.startswith('AIza'):
        print("Warning: Google API keys usually start with 'AIza'")
        confirm = input("Continue anyway? (y/N): ").strip().lower()
        if confirm != 'y':
            return False
    
    return api_key


def setup_models():
    """Setup default models"""
    print("\n🤖 Model Configuration")
    print("=" * 40)
    
    # Google model
    google_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
    print("Available Google models:")
    for i, model in enumerate(google_models, 1):
        print(f"  {i}. {model}")
    
    choice = input(f"\nSelect default Google model (1-{len(google_models)}, default=2): ").strip()
    try:
        if choice:
            google_model = google_models[int(choice) - 1]
        else:
            google_model = google_models[1]  # gemini-1.5-flash
    except (ValueError, IndexError):
        google_model = google_models[1]
    
    # Local model
    local_model = input("Enter default local model (default=phi3:mini): ").strip()
    if not local_model:
        local_model = "phi3:mini"
    
    return google_model, local_model


def create_env_file(api_key, google_model, local_model):
    """Create .env file under user's home config directory (~/.config/llm_project/.env)"""
    env_content = f"""# Google API Configuration
GOOGLE_API_KEY={api_key}

# Default models
DEFAULT_GOOGLE_MODEL={google_model}
DEFAULT_LOCAL_MODEL={local_model}
"""

    config_dir = Path.home() / ".config" / "llm_project"
    config_dir.mkdir(parents=True, exist_ok=True)
    env_path = config_dir / ".env"
    with open(env_path, "w") as f:
        f.write(env_content)

    print(f"\nConfiguration saved to {env_path}")
    print("Your API key is stored in your HOME directory, not the project repo.")
    return True


def main():
    """Main setup wizard"""
    print("RAG System Configuration Wizard")
    print("=" * 50)
    
    # Check if HOME config .env already exists
    home_env = Path.home() / ".config" / "llm_project" / ".env"
    if home_env.exists():
        print("⚠️  ~/.config/llm_project/.env already exists!")
        overwrite = input("Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("Setup cancelled")
            return
    
    # Setup Google API
    api_key = setup_google_api()
    if not api_key:
        print("Setup cancelled - Google API key required")
        return
    
    # Setup models
    google_model, local_model = setup_models()
    
    # Create .env file
    if create_env_file(api_key, google_model, local_model):
        print("\nSetup complete!")
        print("\nNext steps:")
        print("1. Run: python interactive_rag_simplified.py")
        print("2. Or run: python -c \"from rag_modules.app import main; main()\"")
        print("\nWarning⚠️!!!!!!!!!! Keep your .env file secure and never share it!!!!!!!!!!")


if __name__ == "__main__":
    main()
