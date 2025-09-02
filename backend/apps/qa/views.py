from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.middleware.csrf import get_token, get_token as csrf_get_token
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.core.exceptions import SuspiciousFileOperation
from django.views.decorators.cache import cache_page
from functools import wraps
import logging
import json
import os
import numpy as np
from django.conf import settings
from django.db.models import Q, F, Count
from django.core.cache import cache
from django.utils import timezone
from datetime import timedelta
from django.views import View
from django.http import HttpRequest, JsonResponse
import openai
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

logger = logging.getLogger(__name__)

@csrf_exempt
@require_http_methods(['POST'])
def ask_question(request):
    """
    Handle question submission and return an AI-generated response.
    """
    try:
        # Parse the request data
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        
        if not question:
            return JsonResponse(
                {'error': 'Please enter a question'}, 
                status=400
            )
        
        logger.info(f"Processing question: {question}")
        
        # Generate a meaningful response based on the question
        response = generate_ai_response(question)
        
        return JsonResponse({
            'answer': response['answer'],
            'sources': response.get('sources', []),
            'confidence': response.get('confidence', 0.8),
            'context_used': response.get('context', 'general knowledge'),
            'answer_id': f'ans-{timezone.now().timestamp()}',
            'is_fallback': response.get('is_fallback', False)
        })
        
    except json.JSONDecodeError:
        return JsonResponse(
            {'error': 'Invalid JSON data'}, 
            status=400
        )
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}", exc_info=True)
        return JsonResponse(
            {
                'answer': "I'm having trouble processing your question right now. Please try again later.",
                'sources': [],
                'confidence': 0.0,
                'is_fallback': True,
                'error': str(e)
            },
            status=500
        )

def generate_ai_response(question):
    """
    Generate a response to the given question using available knowledge.
    
    Args:
        question (str): The user's question
        
    Returns:
        dict: Response containing answer and metadata
    """
    try:
        # Convert question to lowercase for easier matching
        question_lower = question.lower()
        
        # Knowledge base for common educational questions
        knowledge_base = {
            'what is machine learning': {
                'answer': """
                Machine learning is a branch of artificial intelligence that focuses on building systems 
                that learn from data. Instead of being explicitly programmed to perform a task, 
                these systems use algorithms to analyze and learn from data, improving their performance 
                over time as they are exposed to more data.
                """,
                'sources': [{
                    'title': 'Introduction to Machine Learning',
                    'content': 'Basic concepts and overview of machine learning',
                    'document_id': 'ml-intro-101'
                }],
                'confidence': 0.95
            },
            'how does a neural network work': {
                'answer': """
                A neural network is a computing system inspired by the biological neural networks in animal brains. 
                It consists of layers of interconnected nodes (neurons) that process information. 
                Each connection can transmit a signal to other neurons, and the receiving neuron processes it and signals 
                to other neurons connected to it. During training, the network adjusts the strength of these connections 
                to improve its performance on specific tasks.
                """,
                'sources': [{
                    'title': 'Neural Networks Explained',
                    'content': 'Detailed explanation of neural network architecture and training',
                    'document_id': 'nn-explained-202'
                }],
                'confidence': 0.90
            },
            'what is the difference between ai and ml': {
                'answer': """
                Artificial Intelligence (AI) is the broader concept of machines being able to carry out tasks 
                in a way that we would consider "smart". Machine Learning (ML) is a current application of AI 
                based on the idea that we can give machines access to data and let them learn for themselves.
                
                In other words:
                - AI is the broader concept of enabling a machine to simulate human behavior
                - ML is an application of AI that allows machines to learn from data without being explicitly programmed
                """,
                'sources': [{
                    'title': 'AI vs ML: Key Differences',
                    'content': 'Comparative analysis of Artificial Intelligence and Machine Learning',
                    'document_id': 'ai-vs-ml-303'
                }],
                'confidence': 0.92
            }
        }
        
        # Check if we have a direct match in our knowledge base
        if question_lower in knowledge_base:
            return knowledge_base[question_lower]
            
        # Check for partial matches or keywords
        for q, response in knowledge_base.items():
            if any(word in question_lower for word in q.split()):
                return response
        
        # If no specific match found, provide a general response
        return {
            'answer': f"""
            I understand you're asking about "{question}". While I don't have a specific answer for this question, 
            I can tell you that this appears to be an educational query. Would you like me to help you find 
            more information about this topic?
            """,
            'sources': [],
            'confidence': 0.5,
            'is_fallback': True
        }
        
    except Exception as e:
        logger.error(f"Error in generate_ai_response: {str(e)}", exc_info=True)
        return {
            'answer': "I'm sorry, I encountered an error while processing your question.",
            'sources': [],
            'confidence': 0.0,
            'is_fallback': True
        }

# Initialize the sentence transformer model
model = None
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Successfully loaded sentence transformer model")
except Exception as e:
    logger.error(f"Error loading sentence transformer model: {str(e)}")

class DebugView(View):
    """
    A debug endpoint that provides system information and status.
    """
    def get(self, request: HttpRequest) -> JsonResponse:
        """
        Return system debug information.
        
        Returns:
            JsonResponse: System information and status
        """
        try:
            import platform
            import sys
            import django
            
            return JsonResponse({
                'status': 'operational',
                'system': {
                    'python_version': sys.version,
                    'platform': platform.platform(),
                    'django_version': django.get_version(),
                    'time': timezone.now().isoformat(),
                    'model_loaded': model is not None,
                },
                'endpoints': [
                    '/api/qa/ask/',
                    '/api/qa/sessions/',
                    '/api/qa/documents/',
                    '/api/qa/popular-questions/',
                ]
            })
            
        except Exception as e:
            logger.error(f"Error in debug view: {str(e)}")
            return JsonResponse(
                {'error': 'An error occurred while retrieving debug information'}, 
                status=500
            )

@cache_page(60 * 15)  # Cache for 15 minutes
@require_http_methods(['GET'])
def get_popular_questions(request):
    """
    Retrieve a list of popular or frequently asked questions.
    
    Returns:
        JsonResponse: List of popular questions with their metadata
    """
    try:
        # TODO: Replace this with actual popular questions retrieval logic
        # For now, return a placeholder list of questions
        popular_questions = [
            {
                'id': 1,
                'question': 'What is machine learning?',
                'ask_count': 42,
                'last_asked': (timezone.now() - timedelta(days=1)).isoformat()
            },
            {
                'id': 2,
                'question': 'How does a neural network work?',
                'ask_count': 35,
                'last_asked': (timezone.now() - timedelta(hours=3)).isoformat()
            },
            {
                'id': 3,
                'question': 'What is the difference between AI and ML?',
                'ask_count': 28,
                'last_asked': (timezone.now() - timedelta(days=2)).isoformat()
            }
        ]
        
        return JsonResponse({
            'status': 'success',
            'count': len(popular_questions),
            'results': popular_questions
        })
        
    except Exception as e:
        logger.error(f"Error retrieving popular questions: {str(e)}")
        return JsonResponse(
            {'error': 'An error occurred while retrieving popular questions'}, 
            status=500
        )

@require_GET
def get_document(request, document_id):
    """
    Retrieve a specific document by its ID.
    
    Args:
        request: The HTTP request
        document_id: The UUID of the document to retrieve
        
    Returns:
        JsonResponse: The document data or an error message
    """
    try:
        # TODO: Replace this with actual document retrieval logic
        # For now, return a placeholder response
        return JsonResponse({
            'id': str(document_id),
            'title': 'Document Title',
            'content': 'Document content would be here',
            'created_at': timezone.now().isoformat(),
            'status': 'processed'  # or 'processing', 'failed', etc.
        })
        
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {str(e)}")
        return JsonResponse(
            {'error': 'An error occurred while retrieving the document'}, 
            status=500
        )

@csrf_exempt
@require_http_methods(['POST'])
def process_documents(request):
    """
    Handle document uploads for processing.
    
    Expected multipart form data with 'file' field containing the document.
    """
    try:
        if 'file' not in request.FILES:
            return JsonResponse(
                {'error': 'No file provided'}, 
                status=400
            )
            
        uploaded_file = request.FILES['file']
        file_name = default_storage.save(uploaded_file.name, ContentFile(uploaded_file.read()))
        
        # TODO: Add actual document processing logic here
        # For now, just log and return a success response
        logger.info(f"Received file: {file_name}")
        
        return JsonResponse({
            'status': 'success',
            'message': 'Document received and queued for processing',
            'file_name': file_name
        })
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return JsonResponse(
            {'error': 'An error occurred while processing the document'}, 
            status=500
        )

@csrf_exempt
@require_http_methods(['POST'])
def submit_feedback(request):
    """
    Handle submission of user feedback.
    
    Expected JSON payload:
    {
        'question': 'The original question',
        'answer': 'The answer provided',
        'feedback': 'positive/negative/neutral',
        'comments': 'Optional user comments',
        'session_id': 'Optional session ID'
    }
    """
    try:
        data = json.loads(request.body)
        
        # Log the feedback (in a real app, you'd save this to a database)
        logger.info(f"Received feedback: {data}")
        
        return JsonResponse({
            'status': 'success',
            'message': 'Feedback received. Thank you!'
        })
        
    except json.JSONDecodeError:
        return JsonResponse(
            {'error': 'Invalid JSON data'}, 
            status=400
        )
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        return JsonResponse(
            {'error': 'An error occurred while processing your feedback'}, 
            status=500
        )

@require_GET
def get_session_history(request, session_id):
    """
    Retrieve the chat history for a specific session.
    
    Args:
        request: The HTTP request
        session_id: The ID of the session to retrieve history for
        
    Returns:
        JsonResponse: The chat history for the session
    """
    try:
        # TODO: Replace this with actual session history retrieval logic
        # For now, return an empty list as a placeholder
        return JsonResponse({
            'session_id': session_id,
            'history': []
        })
        
    except Exception as e:
        logger.error(f"Error retrieving session history: {str(e)}")
        return JsonResponse(
            {'error': 'An error occurred while retrieving session history'}, 
            status=500
        )

# ... [rest of the imports and code] ...

@csrf_exempt
@require_http_methods(['GET'])
@ensure_csrf_cookie
def get_csrf_token(request):
    """Get CSRF token for the current session.
    
    This endpoint sets the CSRF token in a cookie and returns it in the response.
    """
    return JsonResponse({
        'status': 'success',
        'csrftoken': get_token(request)
    })

# ... [rest of the views] ...
