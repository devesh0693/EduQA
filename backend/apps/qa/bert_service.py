# backend/apps/qa/bert_service.py
import time
import numpy as np
import logging
import hashlib
import torch
from pathlib import Path
import sys
import os

# Add backend to Python path
backend_dir = str(Path(__file__).parent.parent.parent)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Try to import the ML model
EducationalQASystem = None
try:
    from ml_models.question_answering import EducationalQASystem
except ImportError as e:
    logging.warning(f"Could not import EducationalQASystem: {e}")
    EducationalQASystem = None

logger = logging.getLogger(__name__)

class BERTService:
    def __init__(self):
        self.confidence_threshold = 0.3
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        
        if EducationalQASystem is not None:
            self._load_model()
        else:
            logger.warning("EducationalQASystem not available, using fallback mode")
            
        logger.info(f"BERT Service initialized on {self.device}")
    
    def _load_model(self):
        """Load the trained QA model"""
        try:
            # Try multiple possible model paths
            possible_paths = [
                Path("backend/ml_models/trained_model"),
                Path("ml_models/trained_model"),
                Path(__file__).parent.parent.parent / "ml_models" / "trained_model",
                Path(__file__).parent.parent.parent.parent / "test_output"  # Add test_output path
            ]
            
            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    break
            
            if model_path is None:
                logger.warning("Trained model not found in any of the expected locations:")
                for path in possible_paths:
                    logger.warning(f"- {path.absolute()}")
                self.model_loaded = False
                return
                
            logger.info(f"Loading QA model from {model_path.absolute()}...")
            self.qa_system = EducationalQASystem(str(model_path))
            self.model_loaded = True
            logger.info("QA model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading QA model: {str(e)}", exc_info=True)
            self.model_loaded = False
    
    def answer_question(self, question, context, max_length=512):
        """
        Extract answer from context using our trained QA model with enhanced validation
        
        Args:
            question (str): The question to answer
            context (str): The context to search for answer
            max_length (int): Maximum token length for processing
            
        Returns:
            dict: Answer with confidence score and position
        """
        try:
            if not question or not context:
                logger.warning("Empty question or context provided")
                return self._fallback_answer(question, context)
                
            if not self.model_loaded:
                logger.warning("Using fallback answer extraction - model not loaded")
                return self._fallback_answer(question, context)
            
            # Ensure context is long enough but not too long
            if len(context.split()) < 5:
                logger.warning("Context too short, using fallback")
                return self._fallback_answer(question, context)
                
            # Get answer using our QA system
            logger.info(f"Processing question: {question[:100]}...")
            result = self.qa_system.answer_question(question, context)
            
            # Format the response
            return {
                'answer': result.get('answer', ''),
                'confidence': float(result.get('confidence', 0.0)),
                'start': result.get('start', 0),
                'end': result.get('end', 0),
                'context': context,
                'is_fallback': False
            }
            
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            return self._fallback_answer(question, context)
    
    def _fallback_answer(self, question, context):
        """Generate a more helpful fallback answer when model is not available"""
        if not question:
            return {
                'answer': "I didn't receive a question. Could you please ask me something?",
                'confidence': 0.1,
                'start': 0,
                'end': 0,
                'context': context or '',
                'is_fallback': True
            }
            
        # Simple question analysis for better fallback responses
        question_lower = question.lower()
        
        if not context or len(str(context).split()) < 5:
            # No or very short context
            if any(word in question_lower for word in ['what', 'who', 'when', 'where']):
                topic = question_lower.split(' ', 1)[1] if ' ' in question_lower else 'that'
                answer = f"I don't have enough information about {topic} in my knowledge base. Could you provide more context?"
            elif 'how' in question_lower:
                answer = "I'm not entirely sure about the exact process. Could you provide more details about what you're trying to accomplish?"
            elif 'why' in question_lower:
                answer = "I don't have enough information to explain why. Could you provide more context about what you're asking?"
            else:
                answer = "I'm not sure how to answer that. Could you rephrase your question or provide more context?"
        else:
            # We have some context but couldn't find a good answer
            answer = "I couldn't find a specific answer to your question in the provided context. Here's some relevant information: "
            
            # Try to extract a relevant snippet from the context
            sentences = [s.strip() for s in str(context).split('.') if s.strip()]
            if len(sentences) > 0:
                # Take the first sentence that contains a question word
                for sentence in sentences:
                    if any(word in sentence.lower() for word in ['because', 'since', 'as', 'due to']):
                        answer += sentence + ". "
                        break
                else:
                    # If no explanation found, just take the first sentence
                    answer += sentences[0] + ". "
            
            answer += "Would you like me to search for more information on this topic?"
        
        return {
            'answer': answer,
            'confidence': 0.3,
            'start': 0,
            'end': 0,
            'context': context or '',
            'is_fallback': True
        }
    
    def get_embeddings(self, text):
        """
        Generate simple hash-based embeddings
        
        Args:
            text (str): Input text
            
        Returns:
            np.array: Text embeddings
        """
        try:
            # Create a simple 768-dimensional vector based on text hash
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Convert hash to numbers
            numbers = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
            
            # Extend to 768 dimensions
            embedding = []
            for i in range(768):
                embedding.append(numbers[i % len(numbers)] / 255.0)
            
            return np.array(embedding)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return np.zeros(768)

# Global instance
bert_service = BERTService()