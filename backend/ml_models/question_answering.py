"""
Question Answering module for Educational QA System
Handles BERT-based answer extraction and inference
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    pipeline
)
from .config import MODEL_CONFIG, MODEL_PATHS, PERFORMANCE_TARGETS
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EducationalQASystem:
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize the QA system with BERT model
        
        Args:
            model_path: Path to fine-tuned model or model name
            use_gpu: Whether to use GPU if available
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        if model_path is None:
            model_path = MODEL_CONFIG['bert_model_name']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Create pipeline for easier inference
        self.qa_pipeline = pipeline(
            'question-answering',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device.type == 'cuda' else -1
        )
        
        # Configuration
        self.max_seq_length = MODEL_CONFIG['max_seq_length']
        self.doc_stride = MODEL_CONFIG['doc_stride']
        self.max_answer_length = MODEL_CONFIG['max_answer_length']
        self.n_best_size = MODEL_CONFIG['n_best_size']
        self.confidence_threshold = PERFORMANCE_TARGETS['confidence_threshold']
        
        logger.info("QA System initialized successfully")
    
    def preprocess_context(self, context: str) -> str:
        """Preprocess context text for better QA performance"""
        if not context:
            return ""
        
        # Basic text cleaning
        context = context.strip()
        # Remove excessive whitespace
        context = ' '.join(context.split())
        
        return context
    
    def answer_question(
        self, 
        question: str, 
        context: str,
        return_confidence: bool = True,
        return_all_scores: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a question given a context using BERT
        
        Args:
            question: The question to answer
            context: The context containing the answer
            return_confidence: Whether to return confidence score
            return_all_scores: Whether to return all candidate answers
            
        Returns:
            Dictionary containing answer, confidence, and metadata
        """
        start_time = time.time()
        
        # Preprocess inputs
        question = question.strip()
        context = self.preprocess_context(context)
        
        if not question or not context:
            return {
                'answer': '',
                'confidence': 0.0,
                'start': 0,
                'end': 0,
                'response_time': time.time() - start_time,
                'error': 'Empty question or context'
            }
        
        try:
            # Use pipeline for simple inference
            if not return_all_scores:
                result = self.qa_pipeline(
                    question=question,
                    context=context,
                    max_answer_len=self.max_answer_length,
                    handle_impossible_answer=True
                )
                
                response_time = time.time() - start_time
                
                return {
                    'answer': result['answer'],
                    'confidence': result['score'],
                    'start': result['start'],
                    'end': result['end'],
                    'response_time': response_time,
                    'meets_performance_target': response_time < PERFORMANCE_TARGETS['response_time_threshold'],
                    'high_confidence': result['score'] >= self.confidence_threshold
                }
            
            # Detailed inference with multiple candidates
            return self._detailed_inference(question, context, start_time)
            
        except Exception as e:
            logger.error(f"Error in question answering: {str(e)}")
            return {
                'answer': '',
                'confidence': 0.0,
                'start': 0,
                'end': 0,
                'response_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _detailed_inference(self, question: str, context: str, start_time: float) -> Dict[str, Any]:
        """Perform detailed inference with multiple answer candidates"""
        
        # Tokenize input
        inputs = self.tokenizer(
            question,
            context,
            max_length=self.max_seq_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]
        
        # Get input ids for token decoding
        input_ids = inputs['input_ids'][0]
        
        # Find the best answer spans
        answer_candidates = self._extract_answer_candidates(
            start_logits, end_logits, input_ids
        )
        
        response_time = time.time() - start_time
        
        if not answer_candidates:
            return {
                'answer': '',
                'confidence': 0.0,
                'start': 0,
                'end': 0,
                'response_time': response_time,
                'candidates': [],
                'no_answer_probability': 1.0
            }
        
        best_answer = answer_candidates[0]
        
        return {
            'answer': best_answer['text'],
            'confidence': best_answer['confidence'],
            'start': best_answer['start'],
            'end': best_answer['end'],
            'response_time': response_time,
            'candidates': answer_candidates[:5],  # Top 5 candidates
            'meets_performance_target': response_time < PERFORMANCE_TARGETS['response_time_threshold'],
            'high_confidence': best_answer['confidence'] >= self.confidence_threshold,
            'no_answer_probability': self._calculate_no_answer_probability(start_logits, end_logits)
        }
    
    def _extract_answer_candidates(
        self, 
        start_logits: torch.Tensor, 
        end_logits: torch.Tensor, 
        input_ids: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Extract and rank answer candidates from model logits"""
        
        # Apply softmax to get probabilities
        start_probs = F.softmax(start_logits, dim=0)
        end_probs = F.softmax(end_logits, dim=0)
        
        candidates = []
        
        # Get top start and end positions
        top_start_indices = torch.topk(start_probs, self.n_best_size).indices
        top_end_indices = torch.topk(end_probs, self.n_best_size).indices
        
        for start_idx in top_start_indices:
            for end_idx in top_end_indices:
                # Skip invalid spans
                if start_idx >= end_idx or end_idx - start_idx + 1 > self.max_answer_length:
                    continue
                
                # Calculate confidence score
                confidence = float(start_probs[start_idx] * end_probs[end_idx])
                
                # Decode answer text
                answer_tokens = input_ids[start_idx:end_idx + 1]
                answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                if answer_text.strip():  # Skip empty answers
                    candidates.append({
                        'text': answer_text.strip(),
                        'confidence': confidence,
                        'start': int(start_idx),
                        'end': int(end_idx),
                        'start_prob': float(start_probs[start_idx]),
                        'end_prob': float(end_probs[end_idx])
                    })
        
        # Sort by confidence score
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return candidates
    
    def _calculate_no_answer_probability(
        self, 
        start_logits: torch.Tensor, 
        end_logits: torch.Tensor
    ) -> float:
        """Calculate probability that there's no answer in the context"""
        # For BERT, CLS token (position 0) represents no answer
        cls_start_prob = F.softmax(start_logits, dim=0)[0]
        cls_end_prob = F.softmax(end_logits, dim=0)[0]
        
        no_answer_prob = float(cls_start_prob * cls_end_prob)
        return no_answer_prob
    
    def batch_answer_questions(
        self, 
        questions_contexts: List[Tuple[str, str]]
    ) -> List[Dict[str, Any]]:
        """Answer multiple questions in batch for efficiency"""
        logger.info(f"Processing batch of {len(questions_contexts)} questions")
        
        start_time = time.time()
        results = []
        
        for question, context in questions_contexts:
            result = self.answer_question(question, context, return_confidence=True)
            results.append(result)
        
        batch_time = time.time() - start_time
        logger.info(f"Batch processing completed in {batch_time:.2f} seconds")
        
        return results
    
    def evaluate_answer_quality(self, predicted_answer: str, true_answer: str) -> Dict[str, float]:
        """
        Evaluate the quality of a predicted answer against the true answer
        
        Returns:
            Dictionary with quality metrics
        """
        if not predicted_answer and not true_answer:
            return {'exact_match': 1.0, 'f1_score': 1.0}
        
        if not predicted_answer or not true_answer:
            return {'exact_match': 0.0, 'f1_score': 0.0}
        
        # Normalize answers for comparison
        pred_normalized = self._normalize_answer(predicted_answer)
        true_normalized = self._normalize_answer(true_answer)
        
        # Exact match
        exact_match = float(pred_normalized == true_normalized)
        
        # F1 score
        f1_score = self._compute_f1(pred_normalized, true_normalized)
        
        return {
            'exact_match': exact_match,
            'f1_score': f1_score
        }
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for evaluation"""
        import re
        import string
        
        # Convert to lowercase
        answer = answer.lower()
        
        # Remove punctuation
        answer = ''.join(char for char in answer if char not in string.punctuation)
        
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        return answer
    
    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute F1 score between prediction and ground truth"""
        pred_tokens = prediction.split()
        true_tokens = ground_truth.split()
        
        if len(pred_tokens) == 0 or len(true_tokens) == 0:
            return 0.0
        
        common_tokens = set(pred_tokens) & set(true_tokens)
        
        if len(common_tokens) == 0:
            return 0.0
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(true_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def save_model(self, output_dir: Path):
        """Save the fine-tuned model and tokenizer"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: Path):
        """Load a saved model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Recreate pipeline
        self.qa_pipeline = pipeline(
            'question-answering',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device.type == 'cuda' else -1
        )
        
        logger.info(f"Model loaded from {model_dir}")

if __name__ == "__main__":
    # Test the QA system
    qa_system = EducationalQASystem()
    
    # Test question
    context = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from data. BERT (Bidirectional Encoder Representations from Transformers) 
    is a transformer-based model developed by Google for natural language processing tasks. 
    It uses attention mechanisms to understand context bidirectionally.
    """
    
    question = "What is BERT?"
    
    result = qa_system.answer_question(question, context, return_all_scores=True)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Response Time: {result['response_time']:.3f} seconds")
    print(f"High Confidence: {result['high_confidence']}")