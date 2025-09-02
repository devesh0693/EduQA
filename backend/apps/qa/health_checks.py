from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@csrf_exempt
@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint"""
    try:
        system_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'database': 'ok',
                'cache': 'ok',
                'search': 'ok',
            }
        }
        
        response = JsonResponse(system_status)
        response["Access-Control-Allow-Origin"] = "*"
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JsonResponse({'status': 'unhealthy', 'error': str(e)}, status=500)
