import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import ThreeBackground from '../components/ThreeBackground';
import AnimatedLoader from '../components/AnimatedLoader';
import { searchDocuments } from '../services/api';
import cacheService from '../services/cache';
import { gsap } from 'gsap';
import '../ModernApp.css';

// Clean text by removing garbage characters and normalizing whitespace
const cleanText = (text) => {
  if (!text) return '';
  return text
    .replace(/[\u0000-\u001F]/g, '')  // Remove control characters
    .replace(/\s+/g, ' ')           // Normalize whitespace
    .trim();
};

// Format text with bold highlighting
const formatHighlightedText = (text) => {
  if (!text) return text;
  return text.replace(/\*\*(.*?)\*\*/g, '<mark>$1</mark>');
};

const ModernSearchPage = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [hasSearched, setHasSearched] = useState(false);
  const [expandedDoc, setExpandedDoc] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Refs for animations
  const heroRef = useRef();

  const query = searchParams.get('q') || '';

  const handleSearch = useCallback(async (searchQuery) => {
    const trimmedQuery = searchQuery.trim();
    if (!trimmedQuery) {
      setError('Please enter a search term');
      return;
    }

    // Check cache first
    const cachedResults = cacheService.getCachedSearch(trimmedQuery);
    if (cachedResults) {
      setResults(cachedResults);
      setHasSearched(true);
      setExpandedDoc(null);
      setLoading(false);
      setError('');
      
      // Still animate results
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
      
      // Cache the results
      cacheService.setCachedSearch(trimmedQuery, processedResults);
      
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
  }, []);

  // Initialize search query from URL on mount
  useEffect(() => {
    const urlQuery = searchParams.get('q') || '';
    if (urlQuery) {
      setSearchQuery(urlQuery);
      handleSearch(urlQuery);
    }
  }, []); // Only run on mount

  // Handle search execution
  const executeSearch = () => {
    if (searchQuery.trim()) {
      setSearchParams({ q: searchQuery.trim() });
      handleSearch(searchQuery.trim());
    }
  };

  // Page entrance animations
  useEffect(() => {
    if (heroRef.current) {
      gsap.fromTo(heroRef.current, 
        { opacity: 0, y: 50 },
        { opacity: 1, y: 0, duration: 1, ease: "power2.out" }
      );
    }
  }, []);


  const toggleExpandDoc = (docId) => {
    const isExpanding = expandedDoc !== docId;
    setExpandedDoc(expandedDoc === docId ? null : docId);
    
    // Animate the expand/collapse
    if (isExpanding) {
      setTimeout(() => {
        const expandedContent = document.querySelector(`[data-result-id="${docId}"] .result-expanded`);
        if (expandedContent) {
          gsap.fromTo(expandedContent, 
            { opacity: 0, height: 0, y: -20 },
            { opacity: 1, height: 'auto', y: 0, duration: 0.5, ease: "power2.out" }
          );
          
          const chunks = expandedContent.querySelectorAll('.content-chunk');
          if (chunks.length > 0) {
            gsap.fromTo(chunks, 
              { opacity: 0, x: -20 },
              { opacity: 1, x: 0, duration: 0.3, stagger: 0.1, delay: 0.2, ease: "power2.out" }
            );
          }
        }
      }, 50);
    }
  };

  // Helper function to determine relevance class
  const getRelevanceClass = (score) => {
    if (score >= 0.25) return 'relevance-high';
    if (score >= 0.15) return 'relevance-medium';
    return 'relevance-low';
  };

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
        <div className="search-hero" ref={heroRef}>
          <h1>Search Educational Content</h1>
          <p>Discover insights from our comprehensive educational database. 
             Find documents, articles, and resources to enhance your learning.</p>
          
          <div className="search-container">
            <div className="search-form">
              <input 
                type="text"
                className="search-input"
                placeholder="Search for topics, concepts, or specific information..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && executeSearch()}
              />
              <button 
                className="search-btn"
                onClick={executeSearch}
                disabled={loading}
              >
                {loading ? 'Searching...' : 'Search'}
              </button>
            </div>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="loading-container">
            <AnimatedLoader message={`Searching for "${searchQuery}"`} />
          </div>
        )}
        
        {/* Error State */}
        {error && (
          <div className="modern-card" style={{ borderColor: 'var(--red)', background: 'rgba(239, 68, 68, 0.05)' }}>
            <h3 style={{ color: 'var(--red)' }}>⚠️ Search Error</h3>
            <p>{error}</p>
          </div>
        )}

        {/* Results Section */}
        {hasSearched && !loading && results.length > 0 && (
          <div className="results-section">
            <div className="results-header">
              <h2 className="results-title">Search Results</h2>
              <div className="results-count">{results.length} documents found</div>
            </div>
            
            {/* Results List */}
            <div>
              {results.map((result) => (
                <div 
                  key={result.id} 
                  className="result-card"
                  data-result-id={result.id}
                >
                  {/* Card Header */}
                  <div 
                    className="result-header"
                    onClick={() => toggleExpandDoc(result.id)}
                  >
                    <h3 className="result-title">{result.title}</h3>
                    
                    {result.content && (
                      <p className="result-content">{result.content}</p>
                    )}
                    
                    <div className="result-meta">
                      <div className={`relevance-badge ${getRelevanceClass(result.score)}`}>
                        {Math.round(result.score * 100)}% Match
                      </div>
                      
                      <div className="expand-indicator">
                        {expandedDoc === result.id ? '▲' : '▼'}
                      </div>
                    </div>
                  </div>
                  
                  {/* Expanded Content */}
                  {expandedDoc === result.id && (
                    <div className="result-expanded">
                      {result.chunks?.length > 0 ? (
                        <div>
                          <h4 className="content-sections-title">📄 Relevant Content Sections</h4>
                          
                          {result.chunks.map((chunk, idx) => (
                            <div key={idx} className="content-chunk">
                              <div 
                                className="chunk-text"
                                dangerouslySetInnerHTML={{
                                  __html: formatHighlightedText(chunk.text || chunk.preview)
                                }}
                              />
                              
                              {/* Similarity Score */}
                              <div className="similarity-bar">
                                <div className="similarity-progress">
                                  <div 
                                    className="similarity-fill"
                                    style={{ width: `${Math.round(chunk.similarity * 100)}%` }}
                                  />
                                </div>
                                <div className="similarity-score">
                                  {Math.round(chunk.similarity * 100)}%
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="empty-state" style={{ padding: 'var(--space-6)' }}>
                          <p>No detailed content sections available for this document.</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Empty Results State */}
        {hasSearched && !loading && results.length === 0 && !error && (
          <div className="empty-state">
            <div className="empty-icon">🔍</div>
            <h3 className="empty-title">No results found</h3>
            <p className="empty-description">We couldn't find any documents matching your search query.</p>
            
            <div className="empty-suggestions">
              <h4>Try searching with:</h4>
              <ul>
                <li>Different keywords or phrases</li>
                <li>More general terms</li>
                <li>Check for any spelling mistakes</li>
                <li>Educational topics like "machine learning" or "neural networks"</li>
              </ul>
            </div>
          </div>
        )}
      </main>
      
      {/* Backend Status Indicator */}
      <div className="status-indicator connected">
        <div className="status-dot"></div>
        <span>Backend Connected</span>
      </div>
    </div>
  );
};

export default ModernSearchPage;
