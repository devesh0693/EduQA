from django.urls import path
from . import views
from .health_checks import health_check

urlpatterns = [
    path('ask/', views.ask_question, name='ask_question'),
    path('sessions/<str:session_id>/history/', views.get_session_history, name='session_history'),
    path('feedback/', views.submit_feedback, name='submit_feedback'),
    path('documents/process/', views.process_documents, name='process_documents'),
    path('documents/<uuid:document_id>/', views.get_document, name='get_document'),
    path('health/', health_check, name='health_check'),
    path('popular-questions/', views.get_popular_questions, name='popular_questions'),
    path('debug/', views.DebugView.as_view(), name='debug_info'),
    path('csrf-token/', views.get_csrf_token, name='get_csrf_token'),
]
