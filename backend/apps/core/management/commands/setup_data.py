# backend/apps/core/management/commands/setup_data.py
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from apps.qa.document_processor import document_processor
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Set up initial data for the QA system'

    def handle(self, *args, **options):
        self.stdout.write('Setting up QA system data...')
        
        try:
            # Create superuser if it doesn't exist
            if not User.objects.filter(username='admin').exists():
                User.objects.create_superuser(
                    username='admin',
                    email='admin@example.com',
                    password='admin123'
                )
                self.stdout.write(
                    self.style.SUCCESS('Created superuser: admin/admin123')
                )
            else:
                self.stdout.write('Superuser already exists')
            
            # Create a default user for document processing
            default_user, created = User.objects.get_or_create(
                username='system',
                defaults={
                    'email': 'system@example.com',
                    'first_name': 'System',
                    'last_name': 'User',
                    'is_staff': True
                }
            )
            
            if created:
                default_user.set_password('system123')
                default_user.save()
                self.stdout.write('Created system user')
            
            # Process and store documents
            self.stdout.write('Processing documents...')
            count = document_processor.process_and_store_documents(default_user)
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully processed {count} documents')
            )
            
            self.stdout.write(
                self.style.SUCCESS('QA system setup completed successfully!')
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error setting up data: {str(e)}')
            )
            logger.error(f'Setup error: {str(e)}')