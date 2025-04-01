import os
import sys
import subprocess

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "tensorflow",
        "opencv-python",
        "pillow",
        "numpy",
        "matplotlib"
    ]
    
    try:
        import pip
        installed_packages = [pkg.key for pkg in pip.get_installed_distributions()]
    except:
        # Newer version of pip
        import pkg_resources
        installed_packages = [pkg.key for pkg in pkg_resources.working_set]
    
    missing_packages = [pkg for pkg in required_packages if pkg.lower() not in installed_packages]
    
    if missing_packages:
        print("Some required packages are missing. Installing...")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("All dependencies installed successfully!")
    else:
        print("All dependencies are already installed.")

def start_application():
    """Start the Emotion Recognition application."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the main application script
    app_script = os.path.join(script_dir, "emotion_recognition.py")
    
    # Start the application
    subprocess.call([sys.executable, app_script])

if __name__ == "__main__":
    print("===== STEM Day Emotion Recognition Activity =====")
    print("Checking dependencies...")
    check_dependencies()
    print("Starting application...")
    start_application() 