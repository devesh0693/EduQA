@echo off
REM QA System Startup Script for Windows
REM This script sets up and starts the entire QA system

echo 🚀 Starting Educational QA System...
echo =====================================

REM Check if Docker is running
echo [INFO] Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running! Please start Docker Desktop and try again.
    pause
    exit /b 1
)
echo [SUCCESS] Docker is running

REM Check if Docker Compose is available
echo [INFO] Checking Docker Compose...
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Compose is not installed!
    pause
    exit /b 1
)
echo [SUCCESS] Docker Compose is available

REM Create necessary directories
echo [INFO] Creating necessary directories...
if not exist "backend\logs" mkdir backend\logs
if not exist "backend\media" mkdir backend\media
if not exist "backend\static" mkdir backend\static
if not exist "data\datasets" mkdir data\datasets
if not exist "ml_models" mkdir ml_models
echo [SUCCESS] Directories created

REM Setup environment files
echo [INFO] Setting up environment files...
if not exist ".env.development" (
    copy env.development.example .env.development >nul
    echo [SUCCESS] Created .env.development
)

if not exist ".env.production" (
    copy env.production.example .env.production >nul
    echo [WARNING] Created .env.production (update with your settings for production)
)
echo [SUCCESS] Environment files ready

REM Build and start services
echo [INFO] Building and starting services...
docker-compose down --remove-orphans
docker-compose up -d --build
echo [SUCCESS] Services started

REM Wait for services to be ready
echo [INFO] Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Run database migrations
echo [INFO] Running database migrations...
docker-compose exec -T backend python manage.py migrate
echo [SUCCESS] Migrations completed

REM Setup initial data
echo [INFO] Setting up initial data...
docker-compose exec -T backend python manage.py setup_data
echo [SUCCESS] Initial data setup completed

REM Collect static files
echo [INFO] Collecting static files...
docker-compose exec -T backend python manage.py collectstatic --noinput
echo [SUCCESS] Static files collected

REM Test the system
echo [INFO] Testing the system...
python test_connection.py
if errorlevel 1 (
    echo [WARNING] Some tests failed, but continuing...
)

REM Show final status
echo.
echo 🎉 QA System Startup Complete!
echo ==============================
echo.
echo 📊 Service Status:
docker-compose ps
echo.
echo 🌐 Access Points:
echo    Frontend:     http://localhost:3000
echo    Backend API:  http://localhost:8000
echo    Admin Panel:  http://localhost:8000/admin
echo    Health Check: http://localhost:8000/api/qa/health/
echo.
echo 🔑 Default Credentials:
echo    Admin: admin/admin123
echo.
echo 📝 Next Steps:
echo    1. Open http://localhost:3000 in your browser
echo    2. Try asking questions about machine learning
echo    3. Check the admin panel to upload documents
echo.
echo 🔧 Useful Commands:
echo    View logs:     docker-compose logs -f
echo    Stop services: docker-compose down
echo    Restart:       docker-compose restart
echo    Update:        git pull ^&^& docker-compose up -d --build
echo.
pause
