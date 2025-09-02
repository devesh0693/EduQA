from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from apps.qa.document_processor import DocumentProcessor
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Process and load documents into the database'

    def handle(self, *args, **options):
        # Get or create a system user
        User = get_user_model()
        user, _ = User.objects.get_or_create(
            username='system',
            defaults={
                'email': 'system@example.com',
                'is_active': False,
                'is_staff': True
            }
        )
        
        # Initialize document processor
        processor = DocumentProcessor()
        
        # Process and store documents
        try:
            count = processor.process_and_store_documents(user)
            self.stdout.write(self.style.SUCCESS(f'Successfully processed {count} documents'))
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            self.stderr.write(self.style.ERROR(f'Error processing documents: {str(e)}'))
