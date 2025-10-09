#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Study Pal - GUI Launcher - With WSL Network Optimization
"""

import subprocess
import sys
import os
import socket
from pathlib import Path

def get_wsl_ip():
    """Get WSL IP address"""
    try:
        # Get WSL IP address
        result = subprocess.run(['ip', 'route', 'show', 'default'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            # Extract WSL IP from default route
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'default via' in line:
                    parts = line.split()
                    gateway_ip = parts[2]
                    # WSL IP is usually gateway IP with last octet +1
                    ip_parts = gateway_ip.split('.')
                    wsl_ip = '.'.join(ip_parts[:-1]) + '.2'
                    return wsl_ip
        
        # Fallback method: get via hostname
        hostname = socket.gethostname()
        wsl_ip = socket.gethostbyname(hostname)
        return wsl_ip
    except:
        return None

def get_windows_ip():
    """Get Windows host IP"""
    try:
        # Get Windows IP via default gateway
        result = subprocess.run(['ip', 'route', 'show', 'default'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'default via' in line:
                    parts = line.split()
                    return parts[2]  # Gateway IP is Windows IP
        return None
    except:
        return None

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'streamlit_chat', 
        'fitz',  # PyMuPDF
        'PIL'    # Pillow
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == 'fitz':
                missing_packages.append('PyMuPDF')
            elif package == 'PIL':
                missing_packages.append('Pillow')
            else:
                missing_packages.append(package)
    
    return missing_packages

def main():
    """Main function"""
    print("📚 Study Pal - Your Reading Helper")
    print("=" * 60)
    
    # Network diagnostics
    print("🔍 Network diagnostics...")
    wsl_ip = get_wsl_ip()
    windows_ip = get_windows_ip()
    
    print(f"WSL IP Address: {wsl_ip or 'Not detected'}")
    print(f"Windows Host IP: {windows_ip or 'Not detected'}")
    print()
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print("❌ Missing the following dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n📦 Please install dependencies first:")
        print("pip install streamlit streamlit-chat PyMuPDF Pillow")
        print("\nOr:")
        print("pip install -r requirements.txt")
        return
    
    # Check configuration
    home_env = Path.home() / ".config" / "llm_project" / ".env"
    if not home_env.exists() and not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  API configuration not found!")
        print("Please run first: python setup_config.py")
        print("Or set GOOGLE_API_KEY environment variable")
        print()
    
    # Launch Streamlit
    print("🌐 Starting Web interface...")
    print("=" * 60)
    
    if wsl_ip:
        print(f"✅ Access in Windows browser: http://{wsl_ip}:8501")
        print(f"✅ Access within WSL: http://localhost:8501")
    else:
        print("⚠️  Cannot auto-detect WSL IP, please find manually")
        print("   Run 'ip addr show eth0' to view WSL IP address")
        print("   Then access in Windows browser: http://[WSL_IP]:8501")
    
    print("📍 If unable to access, check Windows firewall settings")
    print("📍 Or try running in Windows: wsl --shutdown then restart WSL")
    print("Press Ctrl+C to stop service")
    print("=" * 60)
    print()
    
    try:
        # Launch Streamlit, bind to all interfaces
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.address", "0.0.0.0",  # Bind to all IPs
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--server.headless", "true",  # Don't auto-open browser
            "--logger.level", "info"
        ])
    except KeyboardInterrupt:
        print("\n👋 GUI interface closed")

if __name__ == "__main__":
    main()
