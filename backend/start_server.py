#!/usr/bin/env python
"""
Django server startup script for Educational QA System
"""

import os
import sys
import django
import subprocess
import time
import requests
from pathlib import Path

def setup_django():
    """Setup Django environment"""
    # Set Django settings module
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    # Setup Django
    django.setup()

def test_model_loading():
    """Test that the ML model loads correctly"""
    try:
        from apps.core.model_utils import get_or_create_model
        print("🧠 Testing model loading...")
        model = get_or_create_model('all-mpnet-base-v2', 'cpu')
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def check_database():
    """Check database status"""
    try:
        from django.core.management import execute_from_command_line
        print("🗄️ Checking database...")
        execute_from_command_line(['manage.py', 'check', '--database', 'default'])
        print("✅ Database is ready")
        return True
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False

def run_migrations():
    """Run any pending migrations"""
    try:
        from django.core.management import execute_from_command_line
        print("🔄 Running migrations...")
        execute_from_command_line(['manage.py', 'migrate', '--verbosity=1'])
        print("✅ Migrations completed")
        return True
    except Exception as e:
        print(f"❌ Migrations failed: {e}")
        return False

def test_endpoints():
    """Test API endpoints"""
    base_url = "http://127.0.0.1:8000"
    endpoints = [
        "/api/qa/health/",
        "/api/qa/popular-questions/",
    ]
    
    print("🔍 Testing API endpoints...")
    for endpoint in endpoints:
        try:
            url = base_url + endpoint
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint} - OK")
            else:
                print(f"⚠️ {endpoint} - Status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ {endpoint} - Connection refused")
        except Exception as e:
            print(f"❌ {endpoint} - Error: {e}")

def start_server():
    """Start the Django development server"""
    try:
        print("🚀 Starting Django development server...")
        print("📊 Server will be available at: http://127.0.0.1:8000")
        print("📖 API Documentation available at: http://127.0.0.1:8000/api/")
        print("🛑 Press Ctrl+C to stop the server")
        print("-" * 60)
        
        # Start the server using Django's management command
        from django.core.management import execute_from_command_line
        execute_from_command_line([
            'manage.py', 
            'runserver', 
            '127.0.0.1:8000',
            '--verbosity=1'
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")

def main():
    """Main function"""
    print("🎓 Educational QA System - Django Server")
    print("=" * 50)
    
    # Setup Django
    setup_django()
    
    # Check database
    if not check_database():
        print("❌ Database check failed. Exiting.")
        sys.exit(1)
    
    # Run migrations
    if not run_migrations():
        print("❌ Migration failed. Exiting.")
        sys.exit(1)
    
    # Test model loading
    if not test_model_loading():
        print("❌ Model loading failed. Exiting.")
        sys.exit(1)
    
    print("=" * 50)
    print("✅ All checks passed! Starting server...")
    print()
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()
