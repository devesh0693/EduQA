# backend/apps/core/models.py
from django.db import models
from django.contrib.auth.models import User
import json

class Document(models.Model):
    title = models.CharField(max_length=255)
    content = models.TextField()
    file_path = models.CharField(max_length=500, blank=True)
    file_type = models.CharField(max_length=50)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return self.title

class DocumentEmbedding(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='embeddings')
    chunk_text = models.TextField()
    chunk_index = models.IntegerField()
    embedding_vector = models.JSONField()  # Store as JSON array
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['document', 'chunk_index']

class QASession(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

class Question(models.Model):
    session = models.ForeignKey(QASession, on_delete=models.CASCADE, related_name='questions')
    question_text = models.TextField()
    asked_at = models.DateTimeField(auto_now_add=True)

class Answer(models.Model):
    question = models.OneToOneField(Question, on_delete=models.CASCADE, related_name='answer')
    answer_text = models.TextField()
    confidence_score = models.FloatField()
    source_document = models.ForeignKey(Document, on_delete=models.CASCADE, null=True, blank=True)
    source_chunk = models.TextField(blank=True)
    processing_time = models.FloatField()  # in seconds
    created_at = models.DateTimeField(auto_now_add=True)

class UserFeedback(models.Model):
    FEEDBACK_CHOICES = [
        ('helpful', 'Helpful'),
        ('not_helpful', 'Not Helpful'),
        ('partially_helpful', 'Partially Helpful')
    ]
    
    answer = models.ForeignKey(Answer, on_delete=models.CASCADE, related_name='feedback')
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_CHOICES)
    feedback_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)