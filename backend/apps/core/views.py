from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import json
import logging
import uuid
import numpy as np
import hashlib
from datetime import datetime
from django.utils import timezone
from django.core.cache import cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .models import Document, DocumentEmbedding
from .serializers import DocumentSearchSerializer, DocumentSerializer
from .model_utils import get_or_create_model

# Set up logging
logger = logging.getLogger(__name__)

# In-memory storage for sessions (replace with database in production)
SESSIONS = {}
CHAT_HISTORY = {}

class QAView(View):
    """Main QA view for handling question-answering requests"""
    
    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)
    
    def post(self, request):
        """Handle QA requests"""
        try:
            data = json.loads(request.body)
            question = data.get('question', '').strip()
            session_id = data.get('session_id', str(uuid.uuid4()))
            
            if not question:
                return JsonResponse({
                    'error': 'Question is required',
                    'status': 'error'
                }, status=400)
            
            # Initialize session if it doesn't exist
            if session_id not in SESSIONS:
                SESSIONS[session_id] = {
                    'created_at': datetime.now().isoformat(),
                    'last_activity': datetime.now().isoformat()
                }
                CHAT_HISTORY[session_id] = []
            
            # Update last activity
            SESSIONS[session_id]['last_activity'] = datetime.now().isoformat()
            
            # TODO: Replace this with your actual QA logic
            # This is where you'd integrate your sentence transformer and FAISS search
            answer = self.get_answer(question)
            confidence = 0.85
            
            # Store in chat history
            chat_entry = {
                'question': question,
                'answer': answer,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            CHAT_HISTORY[session_id].append(chat_entry)
            
            return JsonResponse({
                'session_id': session_id,
                'question': question,
                'answer': answer,
                'confidence': confidence,
                'status': 'success'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'error': 'Invalid JSON data',
                'status': 'error'
            }, status=400)
        except Exception as e:
            logger.error(f"Error in QA processing: {str(e)}")
            return JsonResponse({
                'error': 'Internal server error',
                'status': 'error'
            }, status=500)
    
    def get(self, request):
        """Handle GET requests - return API info"""
        return JsonResponse({
            'message': 'QA API is running',
            'endpoints': {
                'ask': 'POST /api/qa/ask/',
                'sessions': 'GET /api/qa/sessions/',
                'session_history': 'GET /api/qa/sessions/<session_id>/history/'
            },
            'status': 'active'
        })
    
    def get_answer(self, question):
        """
        Get answer for a question using your ML models
        TODO: Integrate with your sentence transformer and FAISS search
        """
        # This is a placeholder - replace with your actual QA logic
        question_lower = question.lower()
        
        # Simple keyword-based responses for demo
        responses = {
            'machine learning': 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.',
            'neural network': 'Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information.',
            'deep learning': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns in data.',
            'python': 'Python is a high-level programming language known for its simplicity and readability, widely used in data science and AI.',
            'ai': 'Artificial Intelligence (AI) refers to the simulation of human intelligence in machines programmed to think and learn.',
            'algorithm': 'An algorithm is a step-by-step procedure or formula for solving a problem or completing a task.'
        }
        
        for keyword, response in responses.items():
            if keyword in question_lower:
                return response
        
        return f"I understand you're asking about '{question}'. This is an educational question that would benefit from more context or specific details."

@csrf_exempt
@require_http_methods(["POST"])
def ask_question(request):
    """Alternative function-based view for asking questions"""
    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not question:
            return JsonResponse({
                'error': 'Question is required',
                'status': 'error'
            }, status=400)
        
        # Initialize session if it doesn't exist
        if session_id not in SESSIONS:
            SESSIONS[session_id] = {
                'created_at': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat()
            }
            CHAT_HISTORY[session_id] = []
        
        # Update last activity
        SESSIONS[session_id]['last_activity'] = datetime.now().isoformat()
        
        # Get answer using simple keyword matching (replace with your ML logic)
        answer = get_simple_answer(question)
        confidence = 0.75
        
        # Store in chat history
        chat_entry = {
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        CHAT_HISTORY[session_id].append(chat_entry)
        
        return JsonResponse({
            'session_id': session_id,
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'status': 'success'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON data',
            'status': 'error'
        }, status=400)
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        return JsonResponse({
            'error': 'Internal server error',
            'status': 'error'
        }, status=500)

def get_simple_answer(question):
    """Simple answer generation - replace with your ML model"""
    return f"Thank you for asking: '{question}'. This is a placeholder response that you can replace with your actual QA system."

@require_http_methods(["GET"])
def get_session_history(request, session_id):
    """
    Retrieve the chat history for a specific session.
    GET /api/qa/sessions/<session_id>/history/
    """
    try:
        if session_id not in SESSIONS:
            return JsonResponse({
                'error': 'Session not found',
                'status': 'error'
            }, status=404)
        
        session_info = SESSIONS[session_id]
        chat_history = CHAT_HISTORY.get(session_id, [])
        
        return JsonResponse({
            'session_id': session_id,
            'session_info': session_info,
            'history': chat_history,
            'total_messages': len(chat_history),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error retrieving session history: {str(e)}")
        return JsonResponse({
            'error': 'Failed to retrieve session history',
            'status': 'error'
        }, status=500)

@require_http_methods(["GET"])
def list_sessions(request):
    """
    List all active sessions.
    GET /api/qa/sessions/
    """
    try:
        sessions_list = []
        for session_id, session_info in SESSIONS.items():
            chat_count = len(CHAT_HISTORY.get(session_id, []))
            sessions_list.append({
                'session_id': session_id,
                'created_at': session_info['created_at'],
                'last_activity': session_info['last_activity'],
                'message_count': chat_count
            })
        
        # Sort by last activity (most recent first)
        sessions_list.sort(key=lambda x: x['last_activity'], reverse=True)
        
        return JsonResponse({
            'sessions': sessions_list,
            'total_sessions': len(sessions_list),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        return JsonResponse({
            'error': 'Failed to retrieve sessions',
            'status': 'error'
        }, status=500)

@csrf_exempt
@require_http_methods(["DELETE"])
def clear_session(request, session_id):
    """
    Clear a specific session's history.
    DELETE /api/qa/sessions/<session_id>/
    """
    try:
        if session_id not in SESSIONS:
            return JsonResponse({
                'error': 'Session not found',
                'status': 'error'
            }, status=404)
        
        # Clear the session
        del SESSIONS[session_id]
        if session_id in CHAT_HISTORY:
            del CHAT_HISTORY[session_id]
        
        return JsonResponse({
            'message': f'Session {session_id} cleared successfully',
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}")
        return JsonResponse({
            'error': 'Failed to clear session',
            'status': 'error'
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def create_session(request):
    """
    Create a new chat session.
    POST /api/qa/sessions/create/
    """
    try:
        session_id = str(uuid.uuid4())
        
        SESSIONS[session_id] = {
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat()
        }
        CHAT_HISTORY[session_id] = []
        
        return JsonResponse({
            'session_id': session_id,
            'created_at': SESSIONS[session_id]['created_at'],
            'message': 'Session created successfully',
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        return JsonResponse({
            'error': 'Failed to create session',
            'status': 'error'
        }, status=500)

# Health check endpoint
@require_http_methods(["GET"])
def health_check(request):
    """
    Health check endpoint for the QA service.
    GET /api/qa/health/
    """
    return JsonResponse({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'service': 'QA Service',
        'version': '1.0.0'
    })

class DocumentSearchView(APIView):
    """
    Advanced semantic search with Redis caching and improved accuracy.
    POST /api/core/documents/search/
    """
    model = None
    model_loaded = False
    embedding_dimension = 768
    
    @classmethod
    def load_model(cls):
        """Load SentenceTransformer model using robust utility"""
        if cls.model is None and not cls.model_loaded:
            try:
                logger.info("Loading sentence transformer model for semantic search...")
                
                # Use the robust model loader utility
                cls.model = get_or_create_model('all-mpnet-base-v2', 'cpu')
                cls.model_loaded = True
                
                logger.info(f"Successfully loaded sentence transformer model. Embedding dimension: {cls.embedding_dimension}")
                
                # Test the model
                test_embedding = cls.model.encode('test', convert_to_tensor=False, normalize_embeddings=True)
                logger.info(f"Model test successful. Test embedding shape: {test_embedding.shape}")
                    
            except Exception as e:
                error_msg = f"Error loading sentence transformer model: {str(e)}"
                logger.error(error_msg, exc_info=True)
                cls.model_loaded = False
                cls.model = None
                raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self.load_model()
        except Exception as e:
            logger.error(f"Failed to initialize model in DocumentSearchView: {str(e)}")
    
    def generate_cache_key(self, query, limit):
        """Generate a cache key for the search query"""
        query_hash = hashlib.md5(f"{query}_{limit}".encode()).hexdigest()
        return f"search_results_{query_hash}"
    
    def get_cached_results(self, cache_key):
        """Get cached search results"""
        try:
            return cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
            return None
    
    def cache_results(self, cache_key, results, timeout=300):
        """Cache search results for 5 minutes"""
        try:
            cache.set(cache_key, results, timeout)
        except Exception as e:
            logger.warning(f"Cache storage failed: {str(e)}")
    
    def normalize_query(self, query):
        """Normalize and clean the search query"""
        # Remove extra whitespace and convert to lowercase
        query = ' '.join(query.lower().split())
        # Remove common stop words that don't add meaning
        stop_words = {'what', 'is', 'are', 'how', 'why', 'when', 'where', 'the', 'a', 'an'}
        words = [word for word in query.split() if word not in stop_words or len(query.split()) <= 2]
        return ' '.join(words) if words else query
    
    def calculate_semantic_similarity(self, query_embedding, doc_embeddings_batch):
        """Calculate semantic similarity between query and document embeddings"""
        similarities = cosine_similarity(query_embedding, doc_embeddings_batch)
        return similarities[0]  # Return similarities for the single query
    
    def post(self, request):
        try:
            logger.info(f"Received search request: {request.data}")
            
            # Ensure model is loaded
            if not self.__class__.model_loaded or not self.__class__.model:
                try:
                    self.__class__.load_model()
                except Exception as e:
                    logger.error(f"Failed to load model: {str(e)}", exc_info=True)
                    return Response({
                        'error': 'Search service temporarily unavailable',
                        'details': 'Please try again in a moment',
                        'status': 'service_unavailable'
                    }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
            # Validate request
            serializer = DocumentSearchSerializer(data=request.data)
            if not serializer.is_valid():
                return Response({
                    'error': 'Invalid request',
                    'details': serializer.errors
                }, status=status.HTTP_400_BAD_REQUEST)
            
            original_query = serializer.validated_data['query'].strip()
            limit = min(serializer.validated_data.get('limit', 10), 20)
            
            if not original_query:
                return Response({
                    'error': 'Query cannot be empty',
                    'status': 'bad_request'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Normalize query for better matching
            normalized_query = self.normalize_query(original_query)
            logger.info(f"Normalized query: '{original_query}' -> '{normalized_query}'")
            
            # Check cache first
            cache_key = self.generate_cache_key(normalized_query, limit)
            cached_results = self.get_cached_results(cache_key)
            if cached_results:
                logger.info(f"Returning cached results for query: '{normalized_query}'")
                return Response(cached_results)
            
            # Generate query embedding (don't normalize to match stored embeddings)
            start_time = timezone.now()
            query_embedding = self.__class__.model.encode(
                normalized_query,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=False  # Match the normalization of stored embeddings
            )
            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            # Get all document embeddings
            document_embeddings = list(
                DocumentEmbedding.objects
                .select_related('document')
                .filter(document__is_active=True)
                .order_by('id')
            )
            
            if not document_embeddings:
                return Response({
                    'query': original_query,
                    'results': [],
                    'message': 'No documents available for search',
                    'count': 0
                }, status=status.HTTP_200_OK)
            
            logger.info(f"Processing {len(document_embeddings)} document chunks for query: '{normalized_query}'")
            
            # Prepare embeddings for batch processing
            doc_embeddings_matrix = []
            doc_metadata = []
            
            for doc_embedding in document_embeddings:
                if doc_embedding.embedding_vector and isinstance(doc_embedding.embedding_vector, list):
                    if len(doc_embedding.embedding_vector) == self.embedding_dimension:
                        doc_embeddings_matrix.append(doc_embedding.embedding_vector)
                        doc_metadata.append({
                            'doc_embedding': doc_embedding,
                            'document': doc_embedding.document,
                            'chunk_text': doc_embedding.chunk_text,
                            'title': doc_embedding.document.title
                        })
            
            if not doc_embeddings_matrix:
                return Response({
                    'query': original_query,
                    'results': [],
                    'message': 'No valid embeddings found for search',
                    'count': 0
                }, status=status.HTTP_200_OK)
            
            # Convert to numpy array for batch processing
            doc_embeddings_matrix = np.array(doc_embeddings_matrix, dtype=np.float32)
            
            # Calculate similarities in batch (much faster)
            similarities = self.calculate_semantic_similarity(query_embedding, doc_embeddings_matrix)
            
            # Create results with similarity scores
            scored_results = []
            for i, similarity in enumerate(similarities):
                metadata = doc_metadata[i]
                
                # Apply query-specific boosting
                boosted_similarity = self.apply_query_boosting(
                    similarity, 
                    normalized_query, 
                    metadata['title'], 
                    metadata['chunk_text']
                )
                
                # Only include results with meaningful similarity (adjusted for unnormalized embeddings)
                if boosted_similarity >= 0.1:  # Lower threshold since embeddings aren't normalized
                    scored_results.append({
                        'similarity': float(boosted_similarity),
                        'document': metadata['document'],
                        'chunk': metadata['doc_embedding'],
                        'chunk_text': metadata['chunk_text']
                    })
            
            # Sort by similarity (highest first)
            scored_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Group by document and create final results
            documents = {}
            for result in scored_results:
                doc_id = result['document'].id
                doc_title = result['document'].title
                
                if doc_id not in documents:
                    documents[doc_id] = {
                        'id': str(doc_id),
                        'title': doc_title,
                        'content': result['document'].content[:200] + '...' if len(result['document'].content) > 200 else result['document'].content,
                        'score': result['similarity'],
                        'chunks': [],
                        'relevance_category': self.get_relevance_category(result['similarity'])
                    }
                
                # Add chunk with highlighting
                if len(documents[doc_id]['chunks']) < 3:
                    highlighted_text = self.highlight_query_terms(result['chunk_text'], original_query)
                    documents[doc_id]['chunks'].append({
                        'text': highlighted_text,
                        'similarity': result['similarity'],
                        'preview': result['chunk_text'][:150] + '...' if len(result['chunk_text']) > 150 else result['chunk_text']
                    })
                    
                    # Update document score to highest chunk similarity
                    documents[doc_id]['score'] = max(documents[doc_id]['score'], result['similarity'])
                    documents[doc_id]['relevance_category'] = self.get_relevance_category(documents[doc_id]['score'])
            
            # Convert to sorted list
            final_results = sorted(
                documents.values(), 
                key=lambda x: x['score'], 
                reverse=True
            )[:limit]
            
            # Calculate processing time
            processing_time = (timezone.now() - start_time).total_seconds()
            
            response_data = {
                'query': original_query,
                'normalized_query': normalized_query,
                'results': final_results,
                'count': len(final_results),
                'total_chunks_processed': len(document_embeddings),
                'processing_time': round(processing_time, 3),
                'timestamp': timezone.now().isoformat(),
                'status': 'success'
            }
            
            # Cache the results
            self.cache_results(cache_key, response_data)
            
            logger.info(f"Search completed. Found {len(final_results)} relevant documents for '{normalized_query}' in {processing_time:.3f}s")
            return Response(response_data)
            
        except Exception as e:
            error_msg = f"Error in document search: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Response({
                'error': 'Search failed',
                'details': 'An error occurred while processing your search',
                'status': 'error',
                'debug': str(e) if settings.DEBUG else None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def apply_query_boosting(self, base_similarity, query, title, content):
        """Apply boosting based on query-document relevance"""
        boosted_score = base_similarity
        
        query_words = set(query.lower().split())
        title_words = set(title.lower().split())
        content_words = set(content.lower().split())
        
        # Title match boost
        title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
        if title_overlap > 0.5:
            boosted_score += 0.15  # Strong title match
        elif title_overlap > 0.25:
            boosted_score += 0.08  # Partial title match
        
        # Content density boost
        content_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
        if content_overlap > 0.7:
            boosted_score += 0.1   # High content relevance
        elif content_overlap > 0.4:
            boosted_score += 0.05  # Moderate content relevance
        
        # Query length consideration
        if len(query_words) <= 2 and title_overlap > 0:
            boosted_score += 0.05  # Boost short, specific queries
        
        return min(1.0, boosted_score)  # Cap at 1.0
    
    def get_relevance_category(self, score):
        """Categorize relevance based on similarity score (adjusted for unnormalized embeddings)"""
        if score >= 0.25:
            return 'Highly Relevant'
        elif score >= 0.20:
            return 'Very Relevant'
        elif score >= 0.15:
            return 'Moderately Relevant'
        else:
            return 'Low Relevance'
    
    def highlight_query_terms(self, text, query):
        """Highlight query terms in the text"""
        if not text or not query:
            return text
        
        # Simple highlighting - replace with more sophisticated method if needed
        highlighted_text = text
        query_words = query.lower().split()
        
        for word in query_words:
            if len(word) > 2:  # Only highlight meaningful words
                # Case-insensitive replacement with highlighting
                import re
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                highlighted_text = pattern.sub(f'**{word}**', highlighted_text)
        
        return highlighted_text
