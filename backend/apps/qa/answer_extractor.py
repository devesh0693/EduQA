# backend/apps/qa/answer_extractor.py
import logging
from .bert_service import bert_service

logger = logging.getLogger(__name__)

class AnswerExtractor:
    def __init__(self):
        self.bert_service = bert_service
        logger.info("AnswerExtractor initialized")
    
    def extract_answer(self, question_text, context=None, document_processor=None):
        """Extract answer from context using BERT with enhanced context handling"""
        relevant_docs = []
        try:
            if not question_text or len(question_text.strip()) < 3:
                return self._get_error_response("Question is too short or empty")
                
            question_text = self._preprocess_question(question_text)
            
            # Get relevant context if not provided or too short
            if not context or len(context.split()) < 20:
                if document_processor:
                    # Get relevant chunks from document processor
                    relevant_docs = document_processor.get_relevant_chunks(question_text, top_k=3)
                    if relevant_docs:
                        # Combine top chunks with their context
                        context = " ".join([
                            f"[{doc.get('source', 'source')}] {doc.get('content', '')}" 
                            for doc in relevant_docs
                        ])
                        logger.info(f"Retrieved {len(relevant_docs)} context chunks")
                
                # If still no context, use question to generate some
                if not context or len(context.split()) < 10:
                    context = self._get_enhanced_context(question_text, context)
                    relevant_docs = [{
                        'title': 'Enhanced Context',
                        'content': context,
                        'source': 'system',
                        'score': 0.5
                    }]
            
            # Get answer from BERT
            result = self.bert_service.answer_question(question_text, context)
            
            # If no answer or low confidence, try fallback
            if not result.get('answer') or result.get('confidence', 0) < 0.4:
                fallback_answer = self._generate_fallback_answer(question_text, context)
                result.update({
                    'answer': fallback_answer,
                    'confidence': max(0.3, result.get('confidence', 0.3)),
                    'is_fallback': True,
                    'context_used': context[:500] + ('...' if len(context) > 500 else ''),
                    'relevant_docs': relevant_docs
                })
            else:
                result.update({
                    'context_used': context[:500] + ('...' if len(context) > 500 else ''),
                    'relevant_docs': relevant_docs,
                    'is_fallback': False
                })
                
            return result
            
        except Exception as e:
            logger.error(f"Error in extract_answer: {str(e)}")
            return self._get_error_response("An error occurred while processing your question.")
            
    def _get_error_response(self, message):
        """Generate a consistent error response"""
        return {
            'answer': f"I'm sorry, {message} Please try rephrasing your question or providing more context.",
            'confidence': 0.1,
            'is_fallback': True,
            'context_used': ''
        }
    
    def _preprocess_question(self, question_text):
        """Preprocess the question to improve answer quality"""
        if not question_text:
            return ""
        
        # Remove question marks and extra whitespace
        question_text = question_text.strip().rstrip('?').strip()
        
        # Handle simple questions by adding context
        question_lower = question_text.lower()
        
        # Common question starters that might need more context
        simple_starters = [
            'what is', 'who is', 'what are', 'when was', 'where is',
            'how to', 'how does', 'why is', 'can you', 'could you'
        ]
        
        # If it's a simple question, add more context to help the model
        if any(question_lower.startswith(starter) for starter in simple_starters):
            question_text = f"{question_text}? Please provide a detailed explanation."
        
        return question_text
    
    def _get_enhanced_context(self, question, current_context):
        """Enhance the context with additional relevant information"""
        if not question or len(question.strip()) < 3:
            return current_context
            
        # TODO: In a production system, you would query a knowledge base here
        # For now, we'll just return the current context with some enhancements
        
        # If context is too short, try to find more relevant information
        if not current_context or len(current_context.split()) < 10:
            # Add some generic educational context based on question keywords
            enhanced = f"{current_context} " if current_context else ""
            
            # Add context based on question type
            question_lower = question.lower()
            if 'what is' in question_lower:
                enhanced += "This is an educational explanation about the topic. "
            elif 'who is' in question_lower:
                enhanced += "This is information about a person or entity. "
            elif 'how to' in question_lower:
                enhanced += "Here are the steps to follow: "
            
            return enhanced.strip()
            
        return current_context
    
    def _generate_fallback_answer(self, question, context):
        """Generate a fallback answer using enhanced keyword matching"""
        if not question:
            return "I don't have enough information to answer that question."
            
        question_lower = question.lower()
        context_lower = context.lower() if context else ""
        
        # Try to extract named entities or key terms from the question
        question_terms = [word for word in question_lower.split() 
                         if len(word) > 3 and word not in ['what', 'when', 'where', 'who', 'how']]
        
        # If we have some context, try to find relevant sentences
        if context:
            sentences = [s.strip() for s in context.split('.') if s.strip()]
            relevant_sentences = []
            
            # Score sentences based on keyword matching
            for sentence in sentences:
                sentence_lower = sentence.lower()
                score = 0
                
                # Score based on question terms
                for term in question_terms:
                    if term in sentence_lower:
                        score += 1
                
                # Bonus for sentences that contain question words
                if any(word in sentence_lower for word in ['because', 'due to', 'since', 'as']):
                    score += 1
                
                if score > 0:
                    relevant_sentences.append((sentence, score))
            
            # Sort by score and take top 2 sentences
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            
            if relevant_sentences:
                best_sentences = [s[0] for s in relevant_sentences[:2]]
                return ". ".join(best_sentences) + "."
        
        # Fallback generic responses based on question type
        if any(word in question_lower for word in ['what', 'who', 'when', 'where']):
            return f"I don't have specific information about {question_lower.split(' ', 1)[1] if ' ' in question_lower else 'that'}. Could you provide more context or rephrase your question?"
        elif 'how' in question_lower:
            return "I'm not entirely sure about the exact process. Could you provide more details about what you're trying to accomplish?"
        else:
            return "I don't have enough information to answer that question. Could you provide more context or rephrase it?"
    
    def get_session_history(self, session_id):
        """Get session history (placeholder for future implementation)"""
        # This would typically query the database for session history
        # For now, return empty list
        return []

# Global instance
answer_extractor = AnswerExtractor()