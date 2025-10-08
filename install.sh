#!/bin/bash
# =============================================================================
# RAG Q&A System - Quick Installation Script
# =============================================================================
# This script automates the installation process for the RAG Q&A System
# Usage: bash install.sh
# =============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main installation process
main() {
    print_header "RAG Q&A System - Installation Script"
    
    # Step 1: Check prerequisites
    print_info "Step 1/6: Checking prerequisites..."
    
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.9-3.11 first."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Python version: $PYTHON_VERSION"
    
    if ! command_exists pip3; then
        print_error "pip3 is not installed. Please install pip first."
        exit 1
    fi
    
    print_success "pip3 is installed"
    
    # Step 2: Check/Create Conda environment (optional)
    print_header "Step 2/6: Python Environment Setup"
    
    if command_exists conda; then
        print_info "Conda detected. Would you like to create a new conda environment?"
        read -p "Create conda environment 'rag_system'? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Creating conda environment..."
            conda create -n rag_system python=3.10 -y
            print_success "Conda environment 'rag_system' created"
            print_warning "Please activate it manually: conda activate rag_system"
            print_warning "Then run this script again, or continue with pip install below"
        else
            print_info "Skipping conda environment creation"
        fi
    else
        print_info "Conda not detected. Using system Python."
        print_warning "Consider using conda or venv for isolated environment"
    fi
    
    # Step 3: Upgrade pip
    print_header "Step 3/6: Upgrading pip"
    print_info "Upgrading pip to latest version..."
    python3 -m pip install --upgrade pip
    print_success "pip upgraded successfully"
    
    # Step 4: Install Python dependencies
    print_header "Step 4/6: Installing Python Dependencies"
    print_info "This may take 5-10 minutes..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_success "Python dependencies installed successfully"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    # Step 5: Install Ollama
    print_header "Step 5/6: Installing Ollama (Local Model Support)"
    
    if command_exists ollama; then
        print_success "Ollama is already installed"
        OLLAMA_VERSION=$(ollama --version 2>&1 | head -n1)
        print_info "Version: $OLLAMA_VERSION"
    else
        print_info "Ollama not detected. Would you like to install it?"
        read -p "Install Ollama? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Installing Ollama..."
            curl -fsSL https://ollama.ai/install.sh | sh
            print_success "Ollama installed successfully"
        else
            print_warning "Skipping Ollama installation (you can use Google API only)"
        fi
    fi
    
    # Step 5.1: Pull Ollama models
    if command_exists ollama; then
        print_info "Would you like to pull recommended models?"
        print_info "Options: 1) phi3:mini (2.3GB), 2) gemma:2b (1.4GB), 3) Both, 4) Skip"
        read -p "Enter choice (1-4): " -n 1 -r
        echo
        case $REPLY in
            1)
                print_info "Pulling phi3:mini..."
                ollama pull phi3:mini
                print_success "phi3:mini downloaded"
                ;;
            2)
                print_info "Pulling gemma:2b..."
                ollama pull gemma:2b
                print_success "gemma:2b downloaded"
                ;;
            3)
                print_info "Pulling both models..."
                ollama pull phi3:mini
                ollama pull gemma:2b
                print_success "Both models downloaded"
                ;;
            *)
                print_info "Skipping model download"
                ;;
        esac
    fi
    
    # Step 6: Run configuration wizard
    print_header "Step 6/6: Configuration"
    
    CONFIG_FILE="$HOME/.config/llm_project/.env"
    if [ -f "$CONFIG_FILE" ]; then
        print_warning "Configuration file already exists at: $CONFIG_FILE"
        read -p "Run configuration wizard to reconfigure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python3 setup_config.py
        else
            print_info "Skipping configuration"
        fi
    else
        print_info "Running configuration wizard..."
        python3 setup_config.py
    fi
    
    # Final summary
    print_header "Installation Complete!"
    
    print_success "✅ All components installed successfully"
    echo ""
    echo "Next steps:"
    echo "1. If using conda, activate environment: ${GREEN}conda activate rag_system${NC}"
    echo "2. Launch web GUI: ${GREEN}python run_gui.py${NC}"
    echo "3. Or launch CLI: ${GREEN}python -m rag_modules.app${NC}"
    echo ""
    echo "For troubleshooting, run tests:"
    echo "  ${BLUE}python test_models.py${NC}      - Test model providers"
    echo "  ${BLUE}python test_gui_deps.py${NC}    - Test GUI dependencies"
    echo "  ${BLUE}python test_wsl_network.py${NC} - Test network (WSL only)"
    echo ""
    print_info "For detailed documentation, see README.md"
    echo ""
}

# Run main function
main
