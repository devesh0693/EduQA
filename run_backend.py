#!/usr/bin/env python3
"""
Simple backend startup script for the QA System
This script sets up and runs the Django backend without Docker
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def setup_backend():
    """Setup the Django backend"""
    print("🚀 Setting up Django Backend...")
    
    # Change to backend directory
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return False
    
    os.chdir(backend_dir)
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        success, stdout, stderr = run_command("python -m venv venv")
        if not success:
            print(f"❌ Failed to create virtual environment: {stderr}")
            return False
        print("✅ Virtual environment created")
    
    # Activate virtual environment and install requirements
    print("📦 Installing dependencies...")
    
    # Windows activation
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate && pip install -r requirements.txt"
    else:  # Unix/Linux/Mac
        activate_cmd = "source venv/bin/activate && pip install -r requirements.txt"
    
    success, stdout, stderr = run_command(activate_cmd)
    if not success:
        print(f"❌ Failed to install dependencies: {stderr}")
        return False
    print("✅ Dependencies installed")
    
    # Run migrations
    print("🗄️ Running database migrations...")
    if os.name == 'nt':  # Windows
        migrate_cmd = "venv\\Scripts\\activate && python manage.py migrate"
    else:
        migrate_cmd = "source venv/bin/activate && python manage.py migrate"
    
    success, stdout, stderr = run_command(migrate_cmd)
    if not success:
        print(f"❌ Failed to run migrations: {stderr}")
        return False
    print("✅ Migrations completed")
    
    # Setup initial data
    print("📊 Setting up initial data...")
    if os.name == 'nt':  # Windows
        setup_cmd = "venv\\Scripts\\activate && python manage.py setup_data"
    else:
        setup_cmd = "source venv/bin/activate && python manage.py setup_data"
    
    success, stdout, stderr = run_command(setup_cmd)
    if not success:
        print(f"⚠️ Warning: Failed to setup data: {stderr}")
    else:
        print("✅ Initial data setup completed")
    
    return True

def start_backend_server():
    """Start the Django development server"""
    print("🌐 Starting Django development server...")
    
    backend_dir = Path("backend")
    os.chdir(backend_dir)
    
    # Start the server
    if os.name == 'nt':  # Windows
        server_cmd = "venv\\Scripts\\activate && python manage.py runserver 0.0.0.0:8000"
    else:
        server_cmd = "source venv/bin/activate && python manage.py runserver 0.0.0.0:8000"
    
    print("✅ Backend server starting at http://localhost:8000")
    print("📝 Press Ctrl+C to stop the server")
    
    # Run the server (this will block)
    try:
        subprocess.run(server_cmd, shell=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

def main():
    """Main function"""
    print("🎓 Educational QA System - Backend Setup")
    print("=" * 50)
    
    # Setup backend
    if not setup_backend():
        print("❌ Backend setup failed!")
        return
    
    # Start server
    start_backend_server()

if __name__ == "__main__":
    main()
