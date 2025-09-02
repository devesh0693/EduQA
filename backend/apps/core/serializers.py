# backend/apps/core/serializers.py
from rest_framework import serializers
from .models import Document, QASession, Question, Answer, UserFeedback, DocumentEmbedding

class DocumentSerializer(serializers.ModelSerializer):
    uploaded_by = serializers.StringRelatedField(read_only=True)
    
    class Meta:
        model = Document
        fields = ['id', 'title', 'file_type', 'uploaded_by', 'created_at', 'is_active']

class AnswerSerializer(serializers.ModelSerializer):
    source_document = DocumentSerializer(read_only=True)
    
    class Meta:
        model = Answer
        fields = [
            'id', 'answer_text', 'confidence_score', 
            'source_document', 'processing_time', 'created_at'
        ]

class QuestionSerializer(serializers.ModelSerializer):
    answer = AnswerSerializer(read_only=True)
    
    class Meta:
        model = Question
        fields = ['id', 'question_text', 'asked_at', 'answer']

class QASessionSerializer(serializers.ModelSerializer):
    questions = QuestionSerializer(many=True, read_only=True)
    question_count = serializers.SerializerMethodField()
    
    class Meta:
        model = QASession
        fields = ['id', 'session_id', 'created_at', 'is_active', 'questions', 'question_count']
    
    def get_question_count(self, obj):
        return obj.questions.count()

class UserFeedbackSerializer(serializers.ModelSerializer):
    answer = AnswerSerializer(read_only=True)
    
    class Meta:
        model = UserFeedback
        fields = ['id', 'feedback_type', 'feedback_text', 'created_at', 'answer']

class DocumentSearchSerializer(serializers.Serializer):
    query = serializers.CharField(required=True)
    limit = serializers.IntegerField(default=5, min_value=1, max_value=20)
    
    def validate(self, attrs):
        return attrs