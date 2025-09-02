from django.urls import path
from . import views
from .views import DocumentSearchView

app_name = 'qa'

urlpatterns = [
    # Main QA endpoint (class-based view)
    path('', views.QAView.as_view(), name='qa_main'),
    
    # Alternative function-based endpoint for asking questions
    path('ask/', views.ask_question, name='ask_question'),
    
    # Session management
    path('sessions/', views.list_sessions, name='list_sessions'),
    path('sessions/create/', views.create_session, name='create_session'),
    path('sessions/<str:session_id>/history/', views.get_session_history, name='session_history'),
    path('sessions/<str:session_id>/', views.clear_session, name='clear_session'),
    
    # Health check
    path('health/', views.health_check, name='health_check'),
    
    # Document search
    path('documents/search/', DocumentSearchView.as_view(), name='document_search'),
]