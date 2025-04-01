#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_header(message):
    """Print a formatted header message."""
    print("\n" + "=" * 50)
    print(f" {message} ".center(50))
    print("=" * 50)

def check_python_version():
    """Check if Python version is 3.8 or newer."""
    if sys.version_info < (3, 8):
        print("Error: This application requires Python 3.8 or newer.")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def install_dependencies():
    """Install required dependencies from requirements.txt."""
    print_header("Installing dependencies")
    requirements_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("✅ All dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies.")
        sys.exit(1)

def create_directories():
    """Create necessary directories for the application."""
    print_header("Creating directories")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    
    # Create main directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("✅ Directories created!")

def create_desktop_shortcut():
    """Create a desktop shortcut to launch the application."""
    print_header("Creating desktop shortcut")
    
    home_dir = str(Path.home())
    desktop_dir = os.path.join(home_dir, "Desktop")
    
    if not os.path.exists(desktop_dir):
        print("⚠️ Desktop directory not found. Skipping shortcut creation.")
        return
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_script = os.path.join(base_dir, "app", "start_app.py")
    
    # Create shortcut based on OS
    if sys.platform == "win32":
        # Windows
        shortcut_path = os.path.join(desktop_dir, "STEM Emotion Recognition.bat")
        with open(shortcut_path, "w") as f:
            f.write(f'@echo off\n"{sys.executable}" "{app_script}"\n')
        print(f"✅ Created shortcut at: {shortcut_path}")
    elif sys.platform == "darwin":
        # macOS
        shortcut_path = os.path.join(desktop_dir, "STEM Emotion Recognition.command")
        with open(shortcut_path, "w") as f:
            f.write(f'#!/bin/bash\n"{sys.executable}" "{app_script}"\n')
        os.chmod(shortcut_path, 0o755)  # Make executable
        print(f"✅ Created shortcut at: {shortcut_path}")
    else:
        # Linux
        shortcut_path = os.path.join(desktop_dir, "STEM Emotion Recognition.desktop")
        with open(shortcut_path, "w") as f:
            f.write(f"""[Desktop Entry]
Name=STEM Emotion Recognition
Exec="{sys.executable}" "{app_script}"
Type=Application
Terminal=false
""")
        os.chmod(shortcut_path, 0o755)  # Make executable
        print(f"✅ Created shortcut at: {shortcut_path}")

def print_setup_complete():
    """Print setup complete message."""
    print_header("Setup Complete!")
    print("The Emotion Recognition application is now ready to use.")
    print("To start the application, either:")
    print("  1. Double-click the desktop shortcut")
    print("  2. Run 'python app/start_app.py' from the application directory")
    print("\nEnjoy your STEM Day activity!\n")

def main():
    """Main function to run setup."""
    print_header("STEM Day Emotion Recognition Setup")
    
    check_python_version()
    install_dependencies()
    create_directories()
    create_desktop_shortcut()
    print_setup_complete()

if __name__ == "__main__":
    main() 