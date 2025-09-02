#!/bin/bash

# QA System Startup Script
# This script sets up and starts the entire QA system

set -e

echo "🚀 Starting Educational QA System..."
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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

# Check if Docker is running
check_docker() {
    print_status "Checking Docker..."
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running! Please start Docker Desktop and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed!"
        exit 1
    fi
    print_success "Docker Compose is available"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p backend/logs
    mkdir -p backend/media
    mkdir -p backend/static
    mkdir -p data/datasets
    mkdir -p ml_models
    
    print_success "Directories created"
}

# Setup environment files
setup_environment() {
    print_status "Setting up environment files..."
    
    # Create development environment if it doesn't exist
    if [ ! -f .env.development ]; then
        cp env.development.example .env.development
        print_success "Created .env.development"
    fi
    
    # Create production environment if it doesn't exist
    if [ ! -f .env.production ]; then
        cp env.production.example .env.production
        print_warning "Created .env.production (update with your settings for production)"
    fi
    
    print_success "Environment files ready"
}

# Build and start services
start_services() {
    print_status "Building and starting services..."
    
    # Stop any existing services
    docker-compose down --remove-orphans
    
    # Build and start services
    docker-compose up -d --build
    
    print_success "Services started"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for database
    print_status "Waiting for database..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
            print_success "Database is ready"
            break
        fi
        sleep 1
        timeout=$((timeout - 1))
    done
    
    if [ $timeout -eq 0 ]; then
        print_error "Database failed to start within 60 seconds"
        exit 1
    fi
    
    # Wait for backend
    print_status "Waiting for backend..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:8000/api/qa/health/ > /dev/null 2>&1; then
            print_success "Backend is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -eq 0 ]; then
        print_warning "Backend health check failed, but continuing..."
    fi
    
    # Wait for frontend
    print_status "Waiting for frontend..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:3000 > /dev/null 2>&1; then
            print_success "Frontend is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -eq 0 ]; then
        print_warning "Frontend health check failed, but continuing..."
    fi
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    docker-compose exec -T backend python manage.py migrate
    
    print_success "Migrations completed"
}

# Setup initial data
setup_data() {
    print_status "Setting up initial data..."
    
    docker-compose exec -T backend python manage.py setup_data
    
    print_success "Initial data setup completed"
}

# Collect static files
collect_static() {
    print_status "Collecting static files..."
    
    docker-compose exec -T backend python manage.py collectstatic --noinput
    
    print_success "Static files collected"
}

# Test the system
test_system() {
    print_status "Testing the system..."
    
    if command -v python3 &> /dev/null; then
        python3 test_connection.py
    else
        print_warning "Python3 not found, skipping automated tests"
    fi
}

# Show final status
show_status() {
    echo ""
    echo "🎉 QA System Startup Complete!"
    echo "=============================="
    echo ""
    echo "📊 Service Status:"
    docker-compose ps
    echo ""
    echo "🌐 Access Points:"
    echo "   Frontend:     http://localhost:3000"
    echo "   Backend API:  http://localhost:8000"
    echo "   Admin Panel:  http://localhost:8000/admin"
    echo "   Health Check: http://localhost:8000/api/qa/health/"
    echo ""
    echo "🔑 Default Credentials:"
    echo "   Admin: admin/admin123"
    echo ""
    echo "📝 Next Steps:"
    echo "   1. Open http://localhost:3000 in your browser"
    echo "   2. Try asking questions about machine learning"
    echo "   3. Check the admin panel to upload documents"
    echo ""
    echo "🔧 Useful Commands:"
    echo "   View logs:     docker-compose logs -f"
    echo "   Stop services: docker-compose down"
    echo "   Restart:       docker-compose restart"
    echo "   Update:        git pull && docker-compose up -d --build"
    echo ""
}

# Main execution
main() {
    echo "Starting QA System setup..."
    echo ""
    
    check_docker
    check_docker_compose
    create_directories
    setup_environment
    start_services
    wait_for_services
    run_migrations
    setup_data
    collect_static
    test_system
    show_status
}

# Run main function
main "$@"
