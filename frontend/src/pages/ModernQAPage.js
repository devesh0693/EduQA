import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { gsap } from 'gsap';
import ThreeBackground from '../components/ThreeBackground';
import AnimatedLoader from '../components/AnimatedLoader';
import { askQuestion, getDocument } from '../services/api';
import cacheService from '../services/cache';
import '../ModernApp.css';

const ModernQAPage = () => {
  const [searchParams] = useSearchParams();
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedSource, setExpandedSource] = useState(null);
  const [loadingSources, setLoadingSources] = useState({});

  const urlQuestion = searchParams.get('q') || '';

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
  const answerRef = useRef();

  const handleSearch = useCallback(async (searchQuery) => {
    if (!searchQuery?.trim()) {
      setError('Please enter a question');
      return;
    }

    const trimmedQuery = searchQuery.trim();
    setQuestion(trimmedQuery);
    
    // Check cache first
    const cachedAnswer = cacheService.getCachedAnswer(trimmedQuery);
    if (cachedAnswer) {
      setAnswer(cachedAnswer);
      setLoading(false);
      setError(null);
      
      // Animate answer appearance
      if (answerRef.current) {
        gsap.fromTo(answerRef.current,
          { opacity: 0, y: 30 },
          { opacity: 1, y: 0, duration: 0.6, ease: "power2.out", delay: 0.2 }
        );
      }
      return;
    }
    
    setLoading(true);
    setError(null);

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
      
      // Cache the answer
      cacheService.setCachedAnswer(trimmedQuery, newAnswer);
      
      // Animate answer appearance
      if (answerRef.current) {
        gsap.fromTo(answerRef.current,
          { opacity: 0, y: 30 },
          { opacity: 1, y: 0, duration: 0.6, ease: "power2.out", delay: 0.2 }
        );
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
      setLoading(false);
    }
  }, [answerRef]);

  // Initialize question from URL on mount only
  useEffect(() => {
    if (urlQuestion && !question) {
      setQuestion(urlQuestion);
      handleSearch(urlQuestion);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run on mount

  // Handle search execution
  const executeSearch = () => {
    if (question.trim()) {
      handleSearch(question.trim());
    }
  };

  // GSAP animations
  useEffect(() => {
    if (pageRef.current) {
      gsap.fromTo(pageRef.current, 
        { opacity: 0, y: 30 },
        { opacity: 1, y: 0, duration: 0.8, ease: "power2.out" }
      );
    }
  }, []);


  return (
    <div className="App">
      {/* Three.js Background */}
      <ThreeBackground />
      
      <header className="header">
        <h1>EduQA</h1>
        <div className="nav-links">
          <a href="/">Home</a>
          <a href="/search">Search</a>
          <a href="/qa">Ask Questions</a>
        </div>
      </header>

      <main className="main-content">
        <div className="page" ref={pageRef}>
          <div className="search-hero">
            <h1>Ask Questions</h1>
            <p>Get instant answers from our educational knowledge base powered by advanced AI. 
               Ask anything related to education, science, technology, and more.</p>
            
            <div className="search-container">
              <div className="search-form">
                <input 
                  type="text"
                  className="search-input"
                  placeholder="Ask me anything about education..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && executeSearch()}
                />
                <button 
                  className="search-btn"
                  onClick={executeSearch}
                  disabled={loading}
                >
                  {loading ? 'Thinking...' : 'Ask'}
                </button>
              </div>
            </div>
          </div>

          {/* Loading State */}
          {loading && (
            <div className="loading-container">
              <AnimatedLoader message={`Finding answer for "${question}"`} />
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="modern-card" style={{ borderColor: 'var(--red)', background: 'rgba(239, 68, 68, 0.05)' }}>
              <h3 style={{ color: 'var(--red)' }}>⚠️ Error</h3>
              <p>{error}</p>
            </div>
          )}

          {/* Answer Display */}
          {answer && !loading && (
            <div ref={answerRef}>
              {/* Question Display */}
              <div className="modern-card" style={{ marginBottom: 'var(--space-6)' }}>
                <h3 style={{ color: 'var(--purple)', marginBottom: 'var(--space-3)' }}>📝 Your Question</h3>
                <p style={{ fontSize: '1.1rem', lineHeight: '1.6' }}>{question}</p>
              </div>

              {/* Answer Card */}
              <div className="modern-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 'var(--space-6)' }}>
                  <h3 style={{ color: 'var(--green)', margin: 0 }}>💡 Answer</h3>
                  
                  {/* Confidence Badge */}
                  <div className={`relevance-badge ${answer.confidence >= 0.8 ? 'relevance-high' : answer.confidence >= 0.6 ? 'relevance-medium' : 'relevance-low'}`}>
                    {Math.round(answer.confidence * 100)}% Confidence
                  </div>
                </div>

                {/* Answer Text */}
                <div style={{ 
                  fontSize: '1.15rem', 
                  lineHeight: '1.8', 
                  marginBottom: 'var(--space-6)',
                  color: 'var(--white)',
                  whiteSpace: 'pre-line',
                  textShadow: '0 1px 2px rgba(0, 0, 0, 0.5)'
                }}>
                  {answer.answer}
                </div>

                {/* Answer Meta */}
                <div style={{
                  display: 'flex',
                  gap: 'var(--space-6)',
                  flexWrap: 'wrap',
                  padding: 'var(--space-4)',
                  background: 'var(--gray-950)',
                  borderRadius: 'var(--radius-lg)',
                  marginBottom: answer.sources?.length > 0 ? 'var(--space-6)' : '0'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                    <span style={{ color: 'var(--gray-500)', fontSize: '0.9rem' }}>⏱️ Processing Time:</span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--purple-light)' }}>
                      {answer.processingTime?.toFixed(2) || '0.00'}s
                    </span>
                  </div>
                  
                  <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                    <span style={{ color: 'var(--gray-500)', fontSize: '0.9rem' }}>🔄 Context:</span>
                    <span style={{ fontSize: '0.85rem', color: 'var(--gray-300)' }}>
                      {answer.contextUsed || 'General knowledge'}
                    </span>
                  </div>

                  {answer.isFallback && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                      <span style={{ color: 'var(--orange)', fontSize: '0.9rem' }}>⚠️ Fallback Mode</span>
                    </div>
                  )}
                </div>

                {/* Sources Section */}
                {answer.sources && answer.sources.length > 0 && (
                  <div>
                    <h4 style={{ 
                      marginBottom: 'var(--space-4)', 
                      color: 'var(--white)',
                      fontSize: '1.1rem',
                      fontWeight: '600'
                    }}>
                      📚 Sources ({answer.sources.length})
                    </h4>
                    
                    {answer.sources.map((source, idx) => (
                      <div key={source.document_id || idx} className="modern-card" style={{ marginBottom: 'var(--space-4)' }}>
                        <div 
                          style={{ 
                            cursor: 'pointer',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'flex-start',
                            gap: 'var(--space-4)'
                          }}
                          onClick={() => loadSourceContent(source.document_id)}
                        >
                          <div style={{ flex: 1 }}>
                            <h5 style={{ 
                              marginBottom: 'var(--space-2)', 
                              color: 'var(--white)',
                              fontSize: '1rem',
                              fontWeight: '600'
                            }}>
                              {source.title || `Source ${idx + 1}`}
                            </h5>
                            
                            {source.content && expandedSource !== source.document_id && (
                              <p style={{ 
                                color: 'var(--gray-400)', 
                                fontSize: '0.9rem',
                                lineHeight: '1.5',
                                margin: 0
                              }}>
                                {source.content.substring(0, 200)}
                                {source.content.length > 200 ? '...' : ''}
                              </p>
                            )}
                          </div>
                          
                          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
                            <div className={`relevance-badge ${source.score >= 0.7 ? 'relevance-high' : source.score >= 0.4 ? 'relevance-medium' : 'relevance-low'}`}>
                              {Math.round(source.score * 100)}%
                            </div>
                            
                            <div className="expand-indicator">
                              {loadingSources[source.document_id] ? '⏳' : 
                               expandedSource === source.document_id ? '▲' : '▼'}
                            </div>
                          </div>
                        </div>

                        {/* Expanded Source Content */}
                        {expandedSource === source.document_id && source.content && (
                          <div style={{
                            marginTop: 'var(--space-4)',
                            padding: 'var(--space-4)',
                            background: 'var(--gray-950)',
                            borderRadius: 'var(--radius-lg)',
                            borderTop: '1px solid var(--gray-800)'
                          }}>
                            <h6 style={{ 
                              marginBottom: 'var(--space-3)', 
                              color: 'var(--white)',
                              fontSize: '0.9rem',
                              fontWeight: '600'
                            }}>
                              📄 Full Content
                            </h6>
                            <div style={{ 
                              color: 'var(--white)', 
                              fontSize: '1rem',
                              lineHeight: '1.8',
                              whiteSpace: 'pre-line',
                              textShadow: '0 1px 2px rgba(0, 0, 0, 0.5)'
                            }}>
                              {source.content}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Welcome State */}
          {!loading && !answer && !error && (
            <div className="modern-card">
              <h3 style={{ color: 'var(--purple)', marginBottom: 'var(--space-4)' }}>🎓 Welcome to EduQA System</h3>
              <p style={{ marginBottom: 'var(--space-4)' }}>
                Ask any educational question and get intelligent answers powered by our AI system. 
                The system uses advanced natural language processing to understand your questions 
                and provide relevant, accurate responses.
              </p>
              
              <div style={{
                background: 'var(--gray-950)',
                padding: 'var(--space-6)',
                borderRadius: 'var(--radius-lg)',
                marginTop: 'var(--space-6)'
              }}>
                <h4 style={{ 
                  color: 'var(--white)', 
                  marginBottom: 'var(--space-4)',
                  fontSize: '1rem',
                  fontWeight: '600'
                }}>
                  💡 Example Questions:
                </h4>
                <ul style={{ 
                  listStyle: 'none', 
                  color: 'var(--gray-300)',
                  fontSize: '0.95rem',
                  lineHeight: '1.6'
                }}>
                  <li style={{ 
                    padding: 'var(--space-2) 0', 
                    position: 'relative',
                    paddingLeft: 'var(--space-6)'
                  }}>
                    <span style={{ 
                      position: 'absolute', 
                      left: '0', 
                      color: 'var(--purple)' 
                    }}>→</span>
                    What is machine learning?
                  </li>
                  <li style={{ 
                    padding: 'var(--space-2) 0', 
                    position: 'relative',
                    paddingLeft: 'var(--space-6)'
                  }}>
                    <span style={{ 
                      position: 'absolute', 
                      left: '0', 
                      color: 'var(--purple)' 
                    }}>→</span>
                    How do neural networks work?
                  </li>
                  <li style={{ 
                    padding: 'var(--space-2) 0', 
                    position: 'relative',
                    paddingLeft: 'var(--space-6)'
                  }}>
                    <span style={{ 
                      position: 'absolute', 
                      left: '0', 
                      color: 'var(--purple)' 
                    }}>→</span>
                    Explain blockchain technology
                  </li>
                  <li style={{ 
                    padding: 'var(--space-2) 0', 
                    position: 'relative',
                    paddingLeft: 'var(--space-6)'
                  }}>
                    <span style={{ 
                      position: 'absolute', 
                      left: '0', 
                      color: 'var(--purple)' 
                    }}>→</span>
                    What is the difference between AI and ML?
                  </li>
                </ul>
              </div>
            </div>
          )}
        </div>
      </main>
      
      {/* Backend Status Indicator */}
      <div className="status-indicator connected">
        <div className="status-dot"></div>
        <span>Backend Connected</span>
      </div>
    </div>
  );
};

export default ModernQAPage;
