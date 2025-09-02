import React, { useState } from 'react';
import { submitFeedback } from '../services/api';

const AnswerCard = ({ 
  answer, 
  confidence, 
  sources = [], 
  isFallback = false, 
  processingTime = 0, 
  contextUsed = "", 
  answerId = null,
  onSourceClick,
  expandedSource,
  loadingSources = {}
}) => {
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [feedbackType, setFeedbackType] = useState(null);
  const [feedbackText, setFeedbackText] = useState('');
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);

  const handleFeedback = async (type) => {
    if (!answerId) {
      // If no answerId, just track locally
      setFeedbackType(type);
      setFeedbackSubmitted(true);
      return;
    }

    try {
      await submitFeedback(answerId, type, feedbackText);
      setFeedbackType(type);
      setFeedbackSubmitted(true);
      setShowFeedbackForm(false);
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      // Still track locally
      setFeedbackType(type);
      setFeedbackSubmitted(true);
    }
  };

  const getConfidenceColor = (conf) => {
    if (conf >= 0.8) return 'high';
    if (conf >= 0.6) return 'medium';
    return 'low';
  };

  const getConfidenceText = (conf) => {
    if (conf >= 0.8) return 'High Confidence';
    if (conf >= 0.6) return 'Medium Confidence';
    return 'Low Confidence';
  };

  const handleSourceClick = (documentId) => {
    if (onSourceClick && documentId) {
      onSourceClick(documentId);
    }
  };

  return (
    <div style={{
      background: 'white',
      borderRadius: '12px',
      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
      overflow: 'hidden',
      marginBottom: '2rem',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    }}>
      <div style={{
        padding: '1.5rem',
        borderBottom: '1px solid #edf2f7',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h3 style={{
          margin: 0,
          fontSize: '1.25rem',
          color: '#2d3748',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          <span style={{ color: '#4a5568' }}>🤔</span> Answer
        </h3>
        {confidence !== null && confidence !== undefined && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            background: getConfidenceColor(confidence) === 'high' ? '#ebf8ff' : 
                        getConfidenceColor(confidence) === 'medium' ? '#fffaf0' : '#fff5f5',
            color: getConfidenceColor(confidence) === 'high' ? '#2b6cb0' : 
                  getConfidenceColor(confidence) === 'medium' ? '#b7791f' : '#c53030',
            padding: '0.25rem 0.75rem',
            borderRadius: '9999px',
            fontSize: '0.875rem',
            fontWeight: '500'
          }}>
            <span>{getConfidenceText(confidence)}</span>
            <span style={{
              background: 'rgba(255, 255, 255, 0.7)',
              padding: '0.125rem 0.5rem',
              borderRadius: '9999px',
              fontWeight: 'bold'
            }}>
              {Math.round(confidence * 100)}%
            </span>
          </div>
        )}
      </div>
      
      <div style={{ padding: '1.5rem' }}>
        <div style={{
          fontSize: '1.1rem',
          lineHeight: '1.7',
          color: '#2d3748',
          marginBottom: '1.5rem',
          whiteSpace: 'pre-line'
        }}>
          {answer}
        </div>
        
        {isFallback && (
          <div style={{
            background: '#fffaf0',
            borderLeft: '4px solid #ecc94b',
            padding: '1rem',
            borderRadius: '0 4px 4px 0',
            margin: '1rem 0',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '0.75rem',
            color: '#975a16'
          }}>
            <span style={{ fontSize: '1.25rem' }}>⚠️</span>
            <div>
              <p style={{ margin: '0 0 0.5rem 0', fontWeight: '500' }}>Generated Answer</p>
              <p style={{ margin: 0, fontSize: '0.9375rem', opacity: 0.9 }}>
                This answer was generated based on available information but may not be a perfect match to your question.
              </p>
            </div>
          </div>
        )}

        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1rem',
          paddingTop: '1rem',
          borderTop: '1px solid #edf2f7',
          marginTop: '1.5rem',
          fontSize: '0.875rem',
          color: '#718096'
        }}>
          {processingTime > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span>⏱️</span>
              <span>{processingTime.toFixed(2)}s</span>
            </div>
          )}
        </div>
      </div>

      {sources && sources.length > 0 && (
        <div style={{
          borderTop: '1px solid #edf2f7',
          padding: '1.5rem',
          background: '#f8fafc'
        }}>
          <h4 style={{
            margin: '0 0 1rem 0',
            color: '#2d3748',
            fontSize: '1.1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <span>📚</span> Information Sources
          </h4>
          <ul style={{
            listStyle: 'none',
            padding: 0,
            margin: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem'
          }}>
            {sources.filter(Boolean).map((source, index) => {
              const sourceId = source?.document_id || `source-${index}`;
              const isLoading = loadingSources[sourceId] || false;
              const isExpanded = expandedSource === sourceId;
              const safeSource = source || {};
              
              return (
                <li key={sourceId} style={{
                  background: 'white',
                  borderRadius: '8px',
                  overflow: 'hidden',
                  boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
                  transition: 'all 0.2s ease',
                  border: '1px solid #e2e8f0',
                  ...(isExpanded && {
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
                  })
                }}>
                  <div 
                    style={{
                      cursor: 'pointer',
                      padding: '1rem',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      backgroundColor: isExpanded ? '#f7fafc' : 'white',
                      transition: 'background-color 0.2s ease'
                    }}
                    onClick={() => handleSourceClick(sourceId)}
                  >
                    <div style={{
                      flex: 1,
                      fontWeight: 500,
                      color: '#2d3748',
                      fontSize: '0.95rem',
                      lineHeight: '1.4',
                      paddingRight: '1rem'
                    }}>
                      {safeSource.title || `Source ${index + 1}`}
                      {isLoading && (
                        <span style={{
                          marginLeft: '0.75rem',
                          color: '#718096',
                          fontSize: '0.85rem',
                          fontStyle: 'italic'
                        }}>
                          Loading...
                        </span>
                      )}
                    </div>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '1rem'
                    }}>
                      {safeSource.score !== undefined && safeSource.score !== null && (
                        <span style={{
                          background: '#ebf8ff',
                          color: '#2b6cb0',
                          padding: '0.25rem 0.5rem',
                          borderRadius: '4px',
                          fontSize: '0.8rem',
                          fontWeight: 600
                        }}>
                          {Math.round(safeSource.score * 100)}%
                        </span>
                      )}
                      <span style={{
                        color: '#718096',
                        fontSize: '0.8rem',
                        minWidth: '1rem',
                        textAlign: 'center'
                      }}>
                        {isExpanded ? '▼' : '►'}
                      </span>
                    </div>
                  </div>
                  
                  {(isExpanded || !safeSource.document_id) && (
                    <div style={{
                      padding: '1rem',
                      borderTop: '1px solid #edf2f7',
                      background: 'white',
                      fontSize: '0.9rem',
                      lineHeight: '1.6',
                      color: '#4a5568'
                    }}>
                      {isLoading ? (
                        <div style={{
                          color: '#718096',
                          fontStyle: 'italic',
                          textAlign: 'center',
                          padding: '1rem 0'
                        }}>
                          Loading source content...
                        </div>
                      ) : safeSource.content ? (
                        <div className="source-document-content">
                          {safeSource.content}
                        </div>
                      ) : (
                        <div style={{
                          color: '#718096',
                          fontStyle: 'italic',
                          padding: '0.5rem 0',
                          fontSize: '0.9rem'
                        }}>
                          No content available for this source.
                        </div>
                      )}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
      
      {contextUsed && (
        <details style={{
          margin: '1.5rem 1.5rem 0',
          borderRadius: '6px',
          border: '1px solid #e2e8f0',
          overflow: 'hidden'
        }}>
          <summary style={{
            padding: '0.75rem 1rem',
            background: '#f7fafc',
            cursor: 'pointer',
            fontWeight: '500',
            color: '#4a5568',
            outline: 'none'
          }}>
            Context Used
          </summary>
          <div style={{
            padding: '1rem',
            background: 'white',
            borderTop: '1px solid #e2e8f0',
            fontSize: '0.9rem',
            lineHeight: '1.6',
            color: '#4a5568'
          }}>
            {contextUsed}
          </div>
        </details>
      )}

      {/* Feedback Section */}
      <div style={{
        padding: '1rem 1.5rem',
        borderTop: '1px solid #edf2f7',
        background: '#f8fafc'
      }}>
        {!feedbackSubmitted ? (
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '0.5rem',
            alignItems: 'center'
          }}>
            <span style={{
              fontSize: '0.875rem',
              color: '#4a5568',
              marginRight: '0.5rem'
            }}>Was this helpful?</span>
            
            <button
              onClick={() => handleFeedback('helpful')}
              style={{
                background: '#e6fffa',
                color: '#2c7a7b',
                border: '1px solid #b2f5ea',
                borderRadius: '4px',
                padding: '0.35rem 0.75rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontSize: '0.875rem'
              }}
            >
              👍 Helpful
            </button>
            
            <button
              onClick={() => handleFeedback('partially_helpful')}
              style={{
                background: '#fffaf0',
                color: '#975a16',
                border: '1px solid #feebc8',
                borderRadius: '4px',
                padding: '0.35rem 0.75rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontSize: '0.875rem'
              }}
            >
              🤔 Partially Helpful
            </button>
            
            <button
              onClick={() => handleFeedback('not_helpful')}
              style={{
                background: '#fff5f5',
                color: '#c53030',
                border: '1px solid #fed7d7',
                borderRadius: '4px',
                padding: '0.35rem 0.75rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontSize: '0.875rem'
              }}
            >
              👎 Not Helpful
            </button>
            
            <button
              onClick={() => setShowFeedbackForm(true)}
              style={{
                background: 'white',
                color: '#4a5568',
                border: '1px solid #e2e8f0',
                borderRadius: '4px',
                padding: '0.35rem 0.75rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontSize: '0.875rem'
              }}
            >
              💬 Give Detailed Feedback
            </button>
          </div>
        ) : (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            color: '#38a169',
            fontSize: '0.875rem',
            fontWeight: '500'
          }}>
            <span>✓</span>
            <span>Thank you for your feedback! {feedbackType && `(${feedbackType.replace('_', ' ')})`}</span>
          </div>
        )}

        {showFeedbackForm && (
          <div style={{
            marginTop: '1rem',
            paddingTop: '1rem',
            borderTop: '1px solid #e2e8f0'
          }}>
            <textarea
              value={feedbackText}
              onChange={(e) => setFeedbackText(e.target.value)}
              placeholder="Please provide detailed feedback about this answer..."
              rows="3"
              style={{
                width: '100%',
                padding: '0.75rem',
                borderRadius: '6px',
                border: '1px solid #e2e8f0',
                fontSize: '0.875rem',
                lineHeight: '1.5',
                marginBottom: '0.75rem',
                resize: 'vertical',
                fontFamily: 'inherit'
              }}
            />
            <div style={{
              display: 'flex',
              justifyContent: 'flex-end',
              gap: '0.75rem'
            }}>
              <button
                onClick={() => setShowFeedbackForm(false)}
                style={{
                  background: 'white',
                  color: '#4a5568',
                  border: '1px solid #cbd5e0',
                  borderRadius: '4px',
                  padding: '0.5rem 1rem',
                  cursor: 'pointer',
                  fontSize: '0.875rem'
                }}
              >
                Cancel
              </button>
              <button
                onClick={() => handleFeedback('not_helpful')}
                style={{
                  background: '#4299e1',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '0.5rem 1rem',
                  cursor: 'pointer',
                  fontSize: '0.875rem',
                  fontWeight: '500'
                }}
                disabled={!feedbackText.trim()}
              >
                Submit Feedback
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnswerCard;