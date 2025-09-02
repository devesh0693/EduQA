# backend/apps/core/admin.py
from django.contrib import admin
from .models import Document, DocumentEmbedding, QASession, Question, Answer, UserFeedback

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['title', 'file_type', 'uploaded_by', 'created_at', 'is_active']
    list_filter = ['file_type', 'is_active', 'created_at']
    search_fields = ['title', 'content']
    date_hierarchy = 'created_at'
    readonly_fields = ['created_at', 'updated_at']

@admin.register(DocumentEmbedding)
class DocumentEmbeddingAdmin(admin.ModelAdmin):
    list_display = ['document', 'chunk_index', 'created_at']
    list_filter = ['created_at']
    search_fields = ['document__title', 'chunk_text']
    readonly_fields = ['embedding_vector', 'created_at']

@admin.register(QASession)
class QASessionAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'user', 'created_at', 'is_active']
    list_filter = ['is_active', 'created_at']
    search_fields = ['session_id', 'user__username']
    date_hierarchy = 'created_at'

class AnswerInline(admin.StackedInline):
    model = Answer
    extra = 0
    readonly_fields = ['confidence_score', 'processing_time', 'created_at']

@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ['question_text_short', 'session', 'asked_at']
    list_filter = ['asked_at']
    search_fields = ['question_text', 'session__session_id']
    date_hierarchy = 'asked_at'
    inlines = [AnswerInline]
    
    def question_text_short(self, obj):
        return obj.question_text[:50] + '...' if len(obj.question_text) > 50 else obj.question_text
    question_text_short.short_description = 'Question'

@admin.register(Answer)
class AnswerAdmin(admin.ModelAdmin):
    list_display = ['answer_text_short', 'confidence_score', 'source_document', 'created_at']
    list_filter = ['confidence_score', 'created_at', 'source_document']
    search_fields = ['answer_text', 'question__question_text']
    date_hierarchy = 'created_at'
    readonly_fields = ['processing_time', 'created_at']
    
    def answer_text_short(self, obj):
        return obj.answer_text[:50] + '...' if len(obj.answer_text) > 50 else obj.answer_text
    answer_text_short.short_description = 'Answer'

@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ['feedback_type', 'answer', 'user', 'created_at']
    list_filter = ['feedback_type', 'created_at']
    search_fields = ['feedback_text', 'answer__answer_text']
    date_hierarchy = 'created_at'