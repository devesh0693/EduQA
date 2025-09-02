import os
import sys
import traceback
from pathlib import Path

def check_environment():
    print("=== Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # Check environment variables
    print("\n=== Environment Variables ===")
    for var in ['SECRET_KEY', 'DEBUG', 'DJANGO_SETTINGS_MODULE']:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

def check_imports():
    print("\n=== Checking Imports ===")
    imports = [
        'django',
        'rest_framework',
        'sentence_transformers',
        'faiss',
        'numpy',
        'torch'
    ]
    
    for module in imports:
        try:
            __import__(module)
            print(f"✓ {module} imported successfully")
        except ImportError as e:
            print(f"✗ {module} failed to import: {e}")

def check_django():
    print("\n=== Django Setup ===")
    try:
        import django
        from django.conf import settings
        
        print(f"Django version: {django.get_version()}")
        print(f"Django settings module: {os.environ.get('DJANGO_SETTINGS_MODULE')}")
        
        # Try to access settings
        try:
            print(f"DEBUG mode: {settings.DEBUG}")
            print(f"Installed apps: {len(settings.INSTALLED_APPS)} apps")
            print(f"Database: {settings.DATABASES.get('default', {}).get('ENGINE', 'Not configured')}")
        except Exception as e:
            print(f"Error accessing Django settings: {e}")
            
    except Exception as e:
        print(f"Error setting up Django: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    # Add backend to Python path
    backend_dir = str(Path(__file__).parent / 'backend')
    if backend_dir not in sys.path:
        sys.path.append(backend_dir)
    
    # Set Django settings module
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    
    check_environment()
    check_imports()
    check_django()
