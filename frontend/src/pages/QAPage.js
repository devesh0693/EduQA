import React, { useState, useRef, useEffect } from 'react';
import { gsap } from 'gsap';
import SearchBar from '../components/SearchBar';
import AnswerCard from '../components/AnswerCard';
import AnimatedLoader from '../components/AnimatedLoader';
import { askQuestion, getDocument } from '../services/api';
import './QAPage.css';

const QAPage = () => {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedSource, setExpandedSource] = useState(null);
  const [loadingSources, setLoadingSources] = useState({});

  // Load source content when a source is clicked
  const loadSourceContent = async (documentId) => {
    if (!documentId) return;
    
    setLoadingSources(prev => ({ ...prev, [documentId]: true }));
    setExpandedSource(prev => (prev === documentId ? null : documentId));
    
    try {
      const sourceData = await getDocument(documentId);
      setAnswer(prev => {
        if (!prev) return null;
        return {
          ...prev,
          sources: Array.isArray(prev.sources) 
            ? prev.sources
                .filter(Boolean)
                .map(source => 
                  source?.document_id === documentId 
                    ? { 
                        ...source, 
                        content: sourceData?.content || source.content || 'No content available' 
                      }
                    : source
                )
            : []
        };
      });
    } catch (error) {
      console.error('Error loading source content:', error);
      // Update the source with error message
      setAnswer(prev => ({
        ...prev,
        sources: Array.isArray(prev?.sources) 
          ? prev.sources.map(source => 
              source?.document_id === documentId
                ? { 
                    ...source, 
                    content: 'Failed to load content. Please try again later.' 
                  }
                : source
            )
          : []
      }));
    } finally {
      setLoadingSources(prev => ({
        ...prev, 
        [documentId]: false 
      }));
    }
  };

  const pageRef = useRef();
  const searchBarRef = useRef();
  const answerRef = useRef();

  // GSAP animations
  useEffect(() => {
    if (pageRef.current) {
      gsap.fromTo(pageRef.current, 
        { opacity: 0, y: 30 },
        { opacity: 1, y: 0, duration: 0.8, ease: "power2.out" }
      );
    }
  }, []);

  const generateFallbackResponse = (searchQuery) => {
    const demoQuestion = searchQuery || 'What is machine learning?';
    return {
      answer: `I'm showing you a response for "${demoQuestion}".`,
      confidence: 0.85,
      sources: [{
        document_id: 'fallback-1',
        title: 'Educational Content',
        content: 'This is a fallback response generated when the primary search encounters issues. The system is designed to always provide helpful information even when facing technical challenges.',
        score: 0.9
      }],
      isFallback: true,
      processingTime: 0.5,
      contextUsed: "General knowledge response",
      answerId: 'fallback-' + Date.now()
    };
  };

  const handleSearch = async (searchQuery) => {
    if (!searchQuery?.trim()) {
      setError('Please enter a question');
      return;
    }

    const trimmedQuery = searchQuery.trim();
    setQuestion(trimmedQuery);
    setLoading(true);
    setError(null);
    
    // Initialize with a loading state
    const loadingAnswer = {
      answer: "Searching for an answer...",
      sources: [],
      confidence: 0,
      isFallback: false,
      processingTime: 0,
      contextUsed: "Searching...",
      answerId: `temp-${Date.now()}`
    };
    setAnswer(loadingAnswer);

    // Show loading animation
    gsap.to(searchBarRef.current, {
      scale: 0.98,
      duration: 0.2,
      ease: "power2.out"
    });

    try {
      console.log(`[QAPage] Asking question: "${trimmedQuery}"`);
      const response = await askQuestion(trimmedQuery);
      
      if (!response) {
        throw new Error('No response received from server');
      }

      // Log the response for debugging
      console.log('[QAPage] Received response:', {
        hasAnswer: !!response.answer,
        sourceCount: response.sources?.length || 0,
        isFallback: response.is_fallback || false
      });

      // Format the answer with proper fallbacks and clean up whitespace
      const formatAnswerText = (text) => {
        if (!text) return "I don't have an answer for that question right now.";
        // Clean up whitespace and normalize newlines
        return text
          .replace(/\n{3,}/g, '\n\n')  // Replace 3+ newlines with double newlines
          .replace(/^\s+/, '')         // Remove leading whitespace
          .replace(/\s+$/, '')         // Remove trailing whitespace
          .replace(/[\u2018\u2019]/g, "'")  // Replace smart quotes
          .replace(/[\u201C\u201D]/g, '"');
      };

      const newAnswer = {
        answer: formatAnswerText(response.answer),
        confidence: Math.min(1, Math.max(0, parseFloat(response.confidence || 0.8))), // Ensure between 0 and 1
        sources: Array.isArray(response.sources) 
          ? response.sources
              .filter(Boolean)
              .map((source, idx) => ({
                ...source,
                document_id: source.document_id || `source-${Date.now()}-${idx}`,
                title: source.title || 'Document',
                content: formatAnswerText(source.content || ''),
                score: Math.min(1, Math.max(0, parseFloat(source.score || 0.0))),
                relevance: source.relevance || `${Math.round((source.score || 0) * 100)}%`
              }))
              .sort((a, b) => (b.score || 0) - (a.score || 0)) // Sort by score descending
          : [],
        isFallback: response.is_fallback || false,
        processingTime: Math.max(0, parseFloat(response.processing_time || 0.5)),
        contextUsed: response.context_used || "General knowledge",
        answerId: response.answer_id || `ans-${Date.now()}`,
        timestamp: new Date().toISOString()
      };
      
      console.log('[QAPage] Setting answer:', {
        answerLength: newAnswer.answer?.length,
        sourceCount: newAnswer.sources?.length,
        confidence: newAnswer.confidence,
        isFallback: newAnswer.isFallback
      });
      
      setAnswer(newAnswer);
      
      // Preload first source content if available
      if (newAnswer.sources.length > 0 && newAnswer.sources[0].document_id) {
        try {
          console.log(`[QAPage] Preloading source: ${newAnswer.sources[0].document_id}`);
          await loadSourceContent(newAnswer.sources[0].document_id);
        } catch (e) {
          console.warn('[QAPage] Failed to preload source content:', e);
          // Update the source with error message
          setAnswer(prev => ({
            ...prev,
            sources: prev.sources.map((source, idx) => 
              idx === 0 ? { 
                ...source, 
                content: 'Failed to load content. ' + (e.message || 'Please try again later.') 
              } : source
            )
          }));
        }
      }
    } catch (error) {
      console.error('[QAPage] Error in handleSearch:', {
        name: error.name,
        message: error.message,
        status: error.status,
        details: error.details
      });
      
      // Generate user-friendly error message
      let userMessage = 'Failed to get an answer. ';
      let detailedMessage = error.message || 'Please try again later.';
      
      if (error.status === 503) {
        userMessage = 'The server is currently overloaded. ';
        detailedMessage = 'Please try again in a few moments.';
      } else if (error.status === 404) {
        userMessage = 'The requested resource was not found. ';
      } else if (error.status === 400) {
        userMessage = 'Invalid request. ';
      }
      
      setError(userMessage + detailedMessage);
      
      // Set a fallback answer
      setAnswer({
        answer: userMessage + detailedMessage,
        sources: [],
        confidence: 0,
        isFallback: true,
        processingTime: 0,
        contextUsed: "Error",
        answerId: `error-${Date.now()}`
      });
    } finally {
      // Animate answer appearance
      if (answerRef.current) {
        gsap.fromTo(answerRef.current,
          { opacity: 0, y: 30 },
          { opacity: 1, y: 0, duration: 0.6, ease: "power2.out", delay: 0.2 }
        );
      }

      // Reset loading state and animations
      setLoading(false);
      gsap.to(searchBarRef.current, {
        scale: 1,
        duration: 0.2,
        ease: "power2.out"
      });
    }
  };

  return (
    <div className="qa-page" ref={pageRef} style={{
      maxWidth: '900px',
      margin: '0 auto',
      padding: '2rem 1rem',
      minHeight: '80vh'
    }}>
      <div className="search-container" ref={searchBarRef} style={{
        marginBottom: '2rem',
        maxWidth: '800px',
        margin: '0 auto 2rem',
        width: '100%',
        boxSizing: 'border-box',
        padding: '0 1rem'
      }}>
        <SearchBar 
          onSearch={handleSearch} 
          placeholder="Ask me anything about education..."
          disabled={loading}
        />
      </div>

      {loading && (
        <div className="loading-container" style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2rem',
          textAlign: 'center',
          color: '#4a5568'
        }}>
          <AnimatedLoader />
          <p style={{ marginTop: '1rem', fontSize: '1.1rem' }}>Searching for answers...</p>
        </div>
      )}

      {error && (
        <div className="error-message" style={{
          background: '#fff5f5',
          borderLeft: '4px solid #e53e3e',
          padding: '1rem',
          margin: '1rem auto',
          borderRadius: '4px',
          color: '#c53030',
          maxWidth: '800px',
          boxSizing: 'border-box'
        }}>
          <p style={{ margin: 0 }}>{error}</p>
        </div>
      )}

      {answer && (
        <div className="answer-container" ref={answerRef} style={{
          background: 'white',
          borderRadius: '12px',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.05)',
          padding: '2rem',
          margin: '0 auto',
          maxWidth: '800px',
          boxSizing: 'border-box'
        }}>
          <div className="question-section">
            <h3>📝 Question</h3>
            <div className="card">
              <div className="card-content">
                <p style={{ fontSize: '1.1rem', margin: 0 }}>{question}</p>
              </div>
            </div>
          </div>
          
          <div className="answer-section">
            <h3>💡 Answer</h3>
            <AnswerCard 
              answer={answer?.answer || "No answer available"}
              sources={Array.isArray(answer?.sources) ? answer.sources.filter(Boolean) : []}
              confidence={answer?.confidence || 0}
              isFallback={answer?.isFallback || false}
              processingTime={answer?.processingTime || 0}
              contextUsed={answer?.contextUsed || ""}
              answerId={answer?.answerId || `gen-${Date.now()}`}
              onSourceClick={loadSourceContent}
              expandedSource={expandedSource}
              loadingSources={loadingSources}
            />
          </div>
        </div>
      )}

      {/* Helpful tips when no question has been asked yet */}
      {!loading && !answer && !error && (
        <div className="card fade-in">
          <h3>🎓 Welcome to EduQA System</h3>
          <div className="card-content">
            <p>
              Ask any educational question and get intelligent answers powered by our AI system. 
              The system uses advanced natural language processing to understand your questions 
              and provide relevant, accurate responses.
            </p>
            <br />
            <p>
              <strong>Example questions:</strong>
            </p>
            <ul style={{ marginTop: '1rem', paddingLeft: '2rem' }}>
              <li>What is machine learning?</li>
              <li>How do neural networks work?</li>
              <li>Explain blockchain technology</li>
              <li>What is the difference between AI and ML?</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default QAPage;