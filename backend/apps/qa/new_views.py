from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.middleware.csrf import get_token, get_token as csrf_get_token
from django.views.decorators.csrf import ensure_csrf_cookie
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
import openai
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Initialize the sentence transformer model
model = None
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Successfully loaded sentence transformer model")
except Exception as e:
    logger.error(f"Error loading sentence transformer model: {str(e)}")

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
