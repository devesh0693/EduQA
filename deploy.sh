#!/bin/bash

# QA System Deployment Script
# Usage: ./deploy.sh [development|production]

set -e

ENVIRONMENT=${1:-development}
echo "Deploying QA System in $ENVIRONMENT mode..."

# Load environment variables
if [ "$ENVIRONMENT" = "production" ]; then
    if [ -f .env.production ]; then
        export $(cat .env.production | grep -v '^#' | xargs)
    else
        echo "Error: .env.production file not found!"
        echo "Please copy env.production.example to .env.production and update the values."
        exit 1
    fi
else
    if [ -f .env.development ]; then
        export $(cat .env.development | grep -v '^#' | xargs)
    fi
fi

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "Error: Docker is not running!"
        exit 1
    fi
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        echo "Error: docker-compose is not installed!"
        exit 1
    fi
}

# Function to backup database
backup_database() {
    if [ "$ENVIRONMENT" = "production" ]; then
        echo "Creating database backup..."
        docker-compose exec -T postgres pg_dump -U $POSTGRES_USER $POSTGRES_DB > backup_$(date +%Y%m%d_%H%M%S).sql
        echo "Backup created successfully."
    fi
}

# Function to run database migrations
run_migrations() {
    echo "Running database migrations..."
    docker-compose exec backend python manage.py migrate
    echo "Migrations completed."
}

# Function to collect static files
collect_static() {
    echo "Collecting static files..."
    docker-compose exec backend python manage.py collectstatic --noinput
    echo "Static files collected."
}

# Function to create superuser (interactive)
create_superuser() {
    echo "Do you want to create a superuser? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        docker-compose exec backend python manage.py createsuperuser
    fi
}

# Function to load sample data
load_sample_data() {
    if [ "$ENVIRONMENT" = "development" ]; then
        echo "Loading sample data..."
        docker-compose exec backend python manage.py setup_data
        echo "Sample data loaded."
    fi
}

# Function to check system health
check_health() {
    echo "Checking system health..."
    
    # Check if all services are running
    docker-compose ps
    
    # Check backend health
    if curl -f http://localhost:8000/api/qa/health/ > /dev/null 2>&1; then
        echo "✅ Backend is healthy"
    else
        echo "❌ Backend health check failed"
        return 1
    fi
    
    # Check frontend
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ Frontend is accessible"
    else
        echo "❌ Frontend is not accessible"
        return 1
    fi
    
    echo "🎉 All systems are healthy!"
}

# Function to show deployment info
show_info() {
    echo ""
    echo "🚀 QA System Deployment Complete!"
    echo "=================================="
    echo "Environment: $ENVIRONMENT"
    echo "Backend API: http://localhost:8000"
    echo "Frontend: http://localhost:3000"
    echo "Admin Panel: http://localhost:8000/admin"
    
    if [ "$ENVIRONMENT" = "production" ]; then
        echo ""
        echo "📊 Monitoring:"
        echo "Prometheus: http://localhost:9090"
        echo "Grafana: http://localhost:3001 (admin/admin)"
        
        echo ""
        echo "🔧 Production Commands:"
        echo "View logs: docker-compose logs -f"
        echo "Restart services: docker-compose restart"
        echo "Scale backend: docker-compose up -d --scale backend=3"
        echo "Backup database: docker-compose exec postgres pg_dump -U $POSTGRES_USER $POSTGRES_DB > backup.sql"
    fi
    
    echo ""
    echo "📝 Next Steps:"
    echo "1. Access the application at http://localhost:3000"
    echo "2. Create an admin user if needed"
    echo "3. Upload documents through the admin panel"
    echo "4. Start asking questions!"
}

# Main deployment process
main() {
    echo "🔍 Checking prerequisites..."
    check_docker
    check_docker_compose
    
    echo "📦 Building and starting services..."
    
    # Stop existing services
    docker-compose down
    
    # Build and start services
    if [ "$ENVIRONMENT" = "production" ]; then
        # Production deployment with monitoring
        docker-compose --profile production --profile monitoring up -d --build
    else
        # Development deployment
        docker-compose up -d --build
    fi
    
    # Wait for services to be ready
    echo "⏳ Waiting for services to be ready..."
    sleep 30
    
    # Run post-deployment tasks
    run_migrations
    collect_static
    load_sample_data
    
    # Check health
    if check_health; then
        show_info
    else
        echo "❌ Deployment completed but health checks failed."
        echo "Check the logs with: docker-compose logs"
        exit 1
    fi
}

# Run main function
main "$@"
