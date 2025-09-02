import sys
import os
import platform
from pathlib import Path

def check_environment():
    # Add the project directory to Python path
    project_dir = str(Path(__file__).parent.absolute())
    if project_dir not in sys.path:
        sys.path.append(project_dir)
    
    print("=== Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"Project directory: {project_dir}")
    print(f"Python path: {sys.path}")
    
    # Check environment variables
    print("\n=== Environment Variables ===")
    for var in ['DJANGO_SETTINGS_MODULE', 'PATH', 'PYTHONPATH']:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    
    # Check if we can import Django
    try:
        import django
        print(f"\n✓ Django version: {django.get_version()}")
        
        # Try to import project modules
        try:
            from config import settings
            print("✓ Project settings imported successfully")
            
            # Check installed apps
            print("\nInstalled apps:")
            for app in settings.INSTALLED_APPS:
                print(f"- {app}")
                
        except Exception as e:
            print(f"\n✗ Error importing project settings: {e}")
            
    except ImportError:
        print("\n✗ Django is not installed or not in PYTHONPATH")
    
    # Check for required packages
    print("\n=== Checking Required Packages ===")
    required_packages = [
        'django',
        'djangorestframework',
        'sentence-transformers',
        'faiss-cpu',
        'numpy',
        'torch',
        'transformers',
        'requests',
        'tqdm',
        'scikit-learn',
        'nltk',
        'pandas'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            version = __import__(package).__version__
            print(f"✓ {package}: {version}")
        except ImportError:
            print(f"✗ {package}: Not installed")
        except Exception as e:
            print(f"✗ {package}: Error - {str(e)}")

if __name__ == "__main__":
    check_environment()
