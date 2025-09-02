from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from apps.qa.document_processor import document_processor
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Clear existing documents and reprocess all data sources'

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
        
        # Clear existing documents and embeddings
        from apps.core.models import Document, DocumentEmbedding
        deleted_docs, _ = Document.objects.all().delete()
        deleted_embeddings, _ = DocumentEmbedding.objects.all().delete()
        
        self.stdout.write(
            self.style.SUCCESS(f'Deleted {deleted_docs} documents and {deleted_embeddings} embeddings')
        )
        
        # Process all documents
        self.stdout.write('Starting document processing...')
        try:
            count = document_processor.process_and_store_documents(user)
            self.stdout.write(
                self.style.SUCCESS(f'Successfully processed {count} documents')
            )
        except Exception as e:
            logger.exception("Error processing documents")
            self.stderr.write(
                self.style.ERROR(f'Error processing documents: {str(e)}')
            )
