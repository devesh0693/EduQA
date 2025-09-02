@echo off
REM QA System Deployment Script for Windows
REM Usage: deploy.bat [development|production]

setlocal enabledelayedexpansion

set ENVIRONMENT=%1
if "%ENVIRONMENT%"=="" set ENVIRONMENT=development

echo Deploying QA System in %ENVIRONMENT% mode...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo Error: docker-compose is not installed!
    pause
    exit /b 1
)

REM Load environment variables
if "%ENVIRONMENT%"=="production" (
    if exist .env.production (
        for /f "tokens=1,2 delims==" %%a in (.env.production) do (
            if not "%%a"=="" if not "%%a:~0,1%"=="#" set %%a=%%b
        )
    ) else (
        echo Error: .env.production file not found!
        echo Please copy env.production.example to .env.production and update the values.
        pause
        exit /b 1
    )
) else (
    if exist .env.development (
        for /f "tokens=1,2 delims==" %%a in (.env.development) do (
            if not "%%a"=="" if not "%%a:~0,1%"=="#" set %%a=%%b
        )
    )
)

echo Building and starting services...

REM Stop existing services
docker-compose down

REM Build and start services
if "%ENVIRONMENT%"=="production" (
    REM Production deployment with monitoring
    docker-compose --profile production --profile monitoring up -d --build
) else (
    REM Development deployment
    docker-compose up -d --build
)

REM Wait for services to be ready
echo Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Run post-deployment tasks
echo Running database migrations...
docker-compose exec backend python manage.py migrate

echo Collecting static files...
docker-compose exec backend python manage.py collectstatic --noinput

if "%ENVIRONMENT%"=="development" (
    echo Loading sample data...
    docker-compose exec backend python manage.py setup_data
)

REM Check health
echo Checking system health...
docker-compose ps

echo.
echo 🚀 QA System Deployment Complete!
echo ==================================
echo Environment: %ENVIRONMENT%
echo Backend API: http://localhost:8000
echo Frontend: http://localhost:3000
echo Admin Panel: http://localhost:8000/admin

if "%ENVIRONMENT%"=="production" (
    echo.
    echo 📊 Monitoring:
    echo Prometheus: http://localhost:9090
    echo Grafana: http://localhost:3001 (admin/admin)
)

echo.
echo 📝 Next Steps:
echo 1. Access the application at http://localhost:3000
echo 2. Create an admin user if needed
echo 3. Upload documents through the admin panel
echo 4. Start asking questions!
echo.
pause
