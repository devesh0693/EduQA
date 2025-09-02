import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import SearchBar from '../components/SearchBar';
import { searchDocuments } from '../services/api';
import { gsap } from 'gsap';

// Clean text by removing garbage characters and normalizing whitespace
const cleanText = (text) => {
  if (!text) return '';
  return text
    .replace(/[^\x00-\x7F]/g, '')  // Remove non-ASCII
    .replace(/\s+/g, ' ')           // Normalize whitespace
    .trim();
};

// Format text with bold highlighting
const formatHighlightedText = (text) => {
  if (!text) return text;
  return text.replace(/\*\*(.*?)\*\*/g, '<mark style="background: #fff3cd; padding: 2px 4px; border-radius: 3px; font-weight: 600;">$1</mark>');
};

// Get color scheme based on relevance score (adjusted for unnormalized embeddings)
const getRelevanceColor = (score) => {
  if (score >= 0.25) return { bg: '#d4edda', border: '#28a745', text: '#155724' };
  if (score >= 0.20) return { bg: '#cce7ff', border: '#007bff', text: '#004085' };
  if (score >= 0.15) return { bg: '#fff3cd', border: '#ffc107', text: '#856404' };
  return { bg: '#f8d7da', border: '#dc3545', text: '#721c24' };
};

const SearchPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [hasSearched, setHasSearched] = useState(false);
  const [expandedDoc, setExpandedDoc] = useState(null);

  const query = searchParams.get('q') || '';

  useEffect(() => {
    if (query) {
      handleSearch(query);
    }
  }, [query]);

  const handleSearch = async (searchQuery) => {
    const trimmedQuery = searchQuery.trim();
    if (!trimmedQuery) {
      setError('Please enter a search term');
      return;
    }

    setLoading(true);
    setError('');
    setHasSearched(true);
    setExpandedDoc(null);

    try {
      const data = await searchDocuments(trimmedQuery);
      console.log('Raw search results:', data); // Debug log
      
      // Process and clean the results with better error handling
      const processedResults = (data.results || []).map((result, index) => {
        // Ensure we have valid data
        if (!result) {
          console.warn(`Skipping invalid result at index ${index}:`, result);
          return null;
        }
        
        // Ensure we have valid chunks
        let chunks = [];
        if (Array.isArray(result.chunks)) {
          chunks = result.chunks
            .filter(chunk => chunk && (chunk.text || chunk.preview)) // Filter out invalid chunks
            .map(chunk => ({
              ...chunk,
              text: cleanText(chunk.text || chunk.preview || ''),
              preview: cleanText(chunk.preview || chunk.text || ''),
              // Ensure similarity score is a valid number between 0 and 1
              similarity: Math.min(1, Math.max(0, parseFloat(chunk.similarity) || 0))
            }));
        }
        
        // Use the score directly from the improved backend
        const finalScore = Math.min(1, Math.max(0, parseFloat(result.score) || 0));
        
        return {
          id: result.id || `result-${index}`,
          title: cleanText(result.title || 'Untitled Document'),
          content: cleanText(result.content || ''),
          score: finalScore,
          chunks: chunks,
          relevance_category: result.relevance_category || 'Unknown'
        };
      }).filter(Boolean); // Remove any null results
      
      setResults(processedResults);
      setSearchParams({ q: trimmedQuery });
      
      // Animate results only if there are results to animate
      if (processedResults.length > 0) {
        setTimeout(() => {
          const resultCards = document.querySelectorAll('.result-card');
          if (resultCards.length > 0) {
            gsap.from(resultCards, {
              opacity: 0,
              y: 20,
              stagger: 0.1,
              duration: 0.3
            });
          }
        }, 100);
      }
    } catch (err) {
      console.error('Search error:', err);
      setError('Failed to fetch search results. Please try again.');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const toggleExpandDoc = (docId) => {
    setExpandedDoc(expandedDoc === docId ? null : docId);
  };

  return (
    <div style={{ 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '2rem 1rem'
    }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <h1 style={{ 
            color: 'white', 
            fontSize: '2.5rem', 
            fontWeight: '700',
            marginBottom: '0.5rem',
            textShadow: '0 2px 4px rgba(0,0,0,0.3)'
          }}>
            🔍 Search Educational Content
          </h1>
          <p style={{ 
            color: 'rgba(255,255,255,0.9)', 
            fontSize: '1.1rem',
            maxWidth: '600px',
            margin: '0 auto'
          }}>
            Discover insights from our comprehensive educational database
          </p>
        </div>
        
        {/* Search Bar */}
        <div style={{ marginBottom: '2rem' }}>
          <SearchBar 
            onSearch={handleSearch}
            placeholder="Search for topics, concepts, or specific information..."
            initialValue={query}
            disabled={loading}
          />
        </div>

        {/* Loading State */}
        {loading && (
          <div style={{ 
            textAlign: 'center', 
            padding: '3rem',
            background: 'rgba(255,255,255,0.95)',
            borderRadius: '12px',
            boxShadow: '0 4px 20px rgba(0,0,0,0.1)'
          }}>
            <div style={{
              width: '50px',
              height: '50px',
              border: '4px solid #e3f2fd',
              borderTop: '4px solid #2196f3',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              margin: '0 auto 1rem'
            }} />
            <p style={{ color: '#666', fontSize: '1.1rem', margin: 0 }}>
              Searching for "{query}"...
            </p>
          </div>
        )}
        
        {/* Error State */}
        {error && (
          <div style={{ 
            background: 'linear-gradient(135deg, #ff6b6b, #ee5a6f)',
            color: 'white',
            padding: '1.5rem', 
            borderRadius: '12px', 
            margin: '1rem 0',
            boxShadow: '0 4px 20px rgba(238, 90, 111, 0.3)',
            border: 'none'
          }}>
            <strong>⚠️ Search Error:</strong> {error}
          </div>
        )}

        {/* Results Section */}
        {hasSearched && !loading && (
          <div style={{
            background: 'rgba(255,255,255,0.95)',
            borderRadius: '12px',
            padding: '2rem',
            boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
            backdropFilter: 'blur(10px)'
          }}>
            {/* Results Header */}
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center', 
              marginBottom: '2rem',
              paddingBottom: '1rem',
              borderBottom: '2px solid #e9ecef'
            }}>
              <div>
                <h2 style={{ 
                  margin: '0 0 0.5rem 0', 
                  color: '#2c3e50',
                  fontSize: '1.8rem',
                  fontWeight: '600'
                }}>
                  {results.length > 0 
                    ? `${results.length} Result${results.length !== 1 ? 's' : ''} Found`
                    : 'No Results Found'
                  }
                </h2>
                <p style={{ 
                  margin: 0, 
                  color: '#6c757d',
                  fontSize: '1rem'
                }}>
                  Search query: <strong>"{query}"</strong>
                </p>
              </div>
              {results.length > 0 && (
                <div style={{
                  background: '#e3f2fd',
                  padding: '0.75rem 1rem',
                  borderRadius: '8px',
                  color: '#1565c0',
                  fontSize: '0.9rem',
                  fontWeight: '500'
                }}>
                  📊 {results.length} documents matched
                </div>
              )}
            </div>
            
            {/* Results List */}
            <div className="results-list">
              {results.length === 0 ? (
                <div style={{ 
                  textAlign: 'center',
                  padding: '3rem 2rem',
                  color: '#6c757d'
                }}>
                  <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>🔍</div>
                  <h3 style={{ color: '#495057', marginBottom: '1rem' }}>No matches found</h3>
                  <p style={{ marginBottom: '1.5rem' }}>We couldn't find any documents matching your search.</p>
                  <div style={{
                    background: '#f8f9fa',
                    padding: '1rem',
                    borderRadius: '8px',
                    fontSize: '0.9rem'
                  }}>
                    <strong>Try:</strong>
                    <ul style={{ textAlign: 'left', margin: '0.5rem 0', paddingLeft: '1.5rem' }}>
                      <li>Using different keywords</li>
                      <li>Checking for typos</li>
                      <li>Using more general terms</li>
                    </ul>
                  </div>
                </div>
              ) : (
                results.map((result, index) => {
                  const colors = getRelevanceColor(result.score);
                  return (
                    <div 
                      key={result.id} 
                      className="result-card"
                      style={{
                        background: 'white',
                        borderRadius: '12px',
                        boxShadow: expandedDoc === result.id 
                          ? '0 8px 32px rgba(0,0,0,0.12)' 
                          : '0 2px 12px rgba(0,0,0,0.08)',
                        marginBottom: '1.5rem',
                        overflow: 'hidden',
                        transition: 'all 0.3s ease',
                        border: `2px solid ${expandedDoc === result.id ? colors.border : 'transparent'}`,
                        transform: expandedDoc === result.id ? 'translateY(-2px)' : 'translateY(0)'
                      }}
                    >
                      {/* Card Header */}
                      <div 
                        style={{
                          padding: '1.5rem',
                          cursor: 'pointer',
                          background: expandedDoc === result.id 
                            ? `linear-gradient(135deg, ${colors.bg}, rgba(255,255,255,0.9))` 
                            : 'white',
                          borderBottom: expandedDoc === result.id ? `1px solid ${colors.border}` : '1px solid #e9ecef'
                        }}
                        onClick={() => toggleExpandDoc(result.id)}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem' }}>
                          <div style={{ flex: 1 }}>
                            <h3 style={{ 
                              margin: '0 0 0.75rem 0',
                              color: '#2c3e50',
                              fontSize: '1.25rem',
                              fontWeight: '600',
                              lineHeight: '1.4'
                            }}>
                              {result.title}
                            </h3>
                            {result.content && (
                              <p style={{
                                margin: 0,
                                color: '#6c757d',
                                fontSize: '0.95rem',
                                lineHeight: '1.5'
                              }}>
                                {result.content}
                              </p>
                            )}
                          </div>
                          
                          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexShrink: 0 }}>
                            {/* Relevance Badge */}
                            <div style={{
                              background: colors.bg,
                              color: colors.text,
                              padding: '0.5rem 1rem',
                              borderRadius: '20px',
                              fontSize: '0.85rem',
                              fontWeight: '600',
                              border: `1px solid ${colors.border}`,
                              textAlign: 'center',
                              minWidth: '140px'
                            }}>
                              <div>{Math.round((result.score || 0) * 100)}%</div>
                              <div style={{ fontSize: '0.75rem', marginTop: '2px' }}>
                                {result.relevance_category || (result.score > 0.7 ? 'Highly Relevant' : 
                                 result.score > 0.4 ? 'Moderately Relevant' : 'Low Relevance')}
                              </div>
                            </div>
                            
                            {/* Expand Icon */}
                            <div style={{
                              width: '32px',
                              height: '32px',
                              background: expandedDoc === result.id ? colors.border : '#e9ecef',
                              borderRadius: '50%',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              color: expandedDoc === result.id ? 'white' : '#666',
                              fontSize: '0.9rem',
                              transition: 'all 0.3s ease'
                            }}>
                              {expandedDoc === result.id ? '▲' : '▼'}
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      {/* Expanded Content */}
                      {expandedDoc === result.id && (
                        <div style={{ padding: '2rem', background: '#fafbfc' }}>
                          {result.chunks?.length > 0 ? (
                            <div>
                              <h4 style={{
                                margin: '0 0 1.5rem 0',
                                color: '#495057',
                                fontSize: '1.1rem',
                                fontWeight: '600'
                              }}>
                                📄 Relevant Content Sections
                              </h4>
                              {result.chunks.map((chunk, idx) => {
                                const chunkColors = getRelevanceColor(chunk.similarity);
                                return (
                                  <div key={idx} style={{ 
                                    background: 'white',
                                    padding: '1.5rem',
                                    borderRadius: '8px',
                                    marginBottom: idx < result.chunks.length - 1 ? '1rem' : 0,
                                    border: `1px solid ${chunkColors.border}`,
                                    borderLeft: `4px solid ${chunkColors.border}`
                                  }}>
                                    <div 
                                      style={{ 
                                        color: '#333', 
                                        margin: '0 0 1rem 0', 
                                        lineHeight: '1.7',
                                        fontSize: '1rem'
                                      }}
                                      dangerouslySetInnerHTML={{
                                        __html: formatHighlightedText(chunk.text || chunk.preview)
                                      }}
                                    />
                                    
                                    {/* Similarity Score */}
                                    <div style={{
                                      display: 'flex',
                                      alignItems: 'center',
                                      gap: '0.75rem',
                                      marginTop: '1rem',
                                      padding: '0.75rem',
                                      background: chunkColors.bg,
                                      borderRadius: '6px'
                                    }}>
                                      <div style={{
                                        flex: 1,
                                        height: '8px',
                                        background: '#e9ecef',
                                        borderRadius: '4px',
                                        overflow: 'hidden'
                                      }}>
                                        <div style={{
                                          width: `${Math.round(chunk.similarity * 100)}%`,
                                          height: '100%',
                                          background: `linear-gradient(90deg, ${chunkColors.border}, ${chunkColors.text})`,
                                          transition: 'width 0.6s ease',
                                          borderRadius: '4px'
                                        }} />
                                      </div>
                                      <span style={{
                                        color: chunkColors.text,
                                        fontSize: '0.9rem',
                                        fontWeight: '600',
                                        minWidth: '60px'
                                      }}>
                                        {Math.round(chunk.similarity * 100)}% match
                                      </span>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          ) : (
                            <div style={{
                              textAlign: 'center',
                              padding: '2rem',
                              color: '#6c757d',
                              background: 'white',
                              borderRadius: '8px',
                              border: '1px dashed #dee2e6'
                            }}>
                              <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📄</div>
                              <p style={{ margin: 0, fontStyle: 'italic' }}>
                                No detailed content preview available
                              </p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}
      </div>
      
      {/* CSS Animation */}
      <style jsx="true">{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        .result-card:hover {
          transform: translateY(-4px) !important;
          box-shadow: 0 8px 32px rgba(0,0,0,0.15) !important;
        }
      `}</style>
    </div>
  );
};

export default SearchPage;