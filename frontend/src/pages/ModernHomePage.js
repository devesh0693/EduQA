import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { gsap } from 'gsap';
import ThreeBackground from '../components/ThreeBackground';
import '../ModernApp.css';

const ModernHomePage = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const navigate = useNavigate();
  const heroRef = useRef();
  const featuresRef = useRef();

  useEffect(() => {
    // Entrance animations
    if (heroRef.current) {
      gsap.fromTo(heroRef.current, 
        { opacity: 0, y: 50 },
        { opacity: 1, y: 0, duration: 1, ease: "power2.out" }
      );
    }
    
    if (featuresRef.current) {
      gsap.fromTo(featuresRef.current.children, 
        { opacity: 0, y: 30 },
        { opacity: 1, y: 0, duration: 0.8, stagger: 0.2, delay: 0.5, ease: "power2.out" }
      );
    }
  }, []);

  const handleSearch = (query) => {
    if (query?.trim()) {
      navigate(`/search?q=${encodeURIComponent(query.trim())}`);
    }
  };

  const handleQuickAction = (action) => {
    switch (action) {
      case 'ask':
        navigate('/qa');
        break;
      case 'search':
        navigate('/search');
        break;
      case 'ml':
        navigate('/search?q=machine%20learning');
        break;
      case 'ai':
        navigate('/search?q=artificial%20intelligence');
        break;
      default:
        break;
    }
  };

  const exampleQuestions = [
    "What is machine learning?",
    "How do neural networks work?", 
    "Explain blockchain technology",
    "What is deep learning?",
    "How does natural language processing work?"
  ];

  const features = [
    {
      icon: "🔍",
      title: "Smart Search",
      description: "Advanced semantic search across thousands of educational documents using state-of-the-art AI embeddings.",
      action: () => handleQuickAction('search')
    },
    {
      icon: "💡",
      title: "AI Q&A",
      description: "Ask complex questions and get detailed, contextual answers powered by our educational knowledge base.",
      action: () => handleQuickAction('ask')
    },
    {
      icon: "📚",
      title: "Rich Content",
      description: "Access comprehensive educational materials covering machine learning, AI, technology, and more.",
      action: () => handleQuickAction('ml')
    }
  ];

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
        {/* Hero Section */}
        <div className="search-hero" ref={heroRef}>
          <h1>Educational Intelligence</h1>
          <p>Unlock knowledge with AI-powered search and question answering. 
             Explore thousands of educational resources with intelligent semantic understanding.</p>
          
          <div className="search-container">
            <div className="search-form">
              <input 
                type="text"
                className="search-input"
                placeholder="Search educational content or ask a question..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch(searchQuery)}
              />
              <button 
                className="search-btn"
                onClick={() => handleSearch(searchQuery)}
              >
                Search
              </button>
            </div>
          </div>

          {/* Quick Action Buttons */}
          <div style={{
            display: 'flex',
            gap: 'var(--space-4)',
            justifyContent: 'center',
            marginTop: 'var(--space-8)',
            flexWrap: 'wrap'
          }}>
            <button 
              className="btn btn-secondary"
              onClick={() => handleQuickAction('ask')}
            >
              💬 Ask a Question
            </button>
            <button 
              className="btn btn-secondary"
              onClick={() => handleQuickAction('ml')}
            >
              🤖 Explore ML
            </button>
            <button 
              className="btn btn-secondary"
              onClick={() => handleQuickAction('ai')}
            >
              🧠 Learn AI
            </button>
          </div>
        </div>

        {/* Features Section */}
        <div ref={featuresRef} style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: 'var(--space-8)',
          margin: 'var(--space-16) auto',
          maxWidth: '1200px'
        }}>
          {features.map((feature, index) => (
            <div 
              key={index}
              className="modern-card"
              style={{ cursor: 'pointer' }}
              onClick={feature.action}
            >
              <div style={{ 
                fontSize: '3rem', 
                marginBottom: 'var(--space-4)',
                textAlign: 'center' 
              }}>
                {feature.icon}
              </div>
              
              <h3 style={{ 
                marginBottom: 'var(--space-3)',
                textAlign: 'center',
                color: 'var(--white)'
              }}>
                {feature.title}
              </h3>
              
              <p style={{ 
                textAlign: 'center',
                color: 'var(--gray-400)',
                lineHeight: '1.6'
              }}>
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        {/* Example Questions */}
        <div className="modern-card" style={{ maxWidth: '800px', margin: '0 auto' }}>
          <h3 style={{ 
            color: 'var(--purple)', 
            marginBottom: 'var(--space-6)',
            textAlign: 'center'
          }}>
            🎯 Try These Questions
          </h3>
          
          <div style={{
            display: 'grid',
            gap: 'var(--space-3)'
          }}>
            {exampleQuestions.map((question, index) => (
              <div
                key={index}
                style={{
                  background: 'var(--gray-950)',
                  padding: 'var(--space-4)',
                  borderRadius: 'var(--radius-lg)',
                  cursor: 'pointer',
                  transition: 'var(--transition-normal)',
                  border: '1px solid var(--gray-800)'
                }}
                className="example-question"
                onClick={() => navigate(`/qa?q=${encodeURIComponent(question)}`)}
                onMouseEnter={(e) => {
                  e.target.style.background = 'var(--gray-800)';
                  e.target.style.borderColor = 'var(--gray-700)';
                  e.target.style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.background = 'var(--gray-950)';
                  e.target.style.borderColor = 'var(--gray-800)';
                  e.target.style.transform = 'translateY(0)';
                }}
              >
                <span style={{ color: 'var(--purple)', marginRight: 'var(--space-2)' }}>Q:</span>
                <span style={{ color: 'var(--gray-200)' }}>{question}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Stats Section */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: 'var(--space-6)',
          margin: 'var(--space-16) auto',
          maxWidth: '800px'
        }}>
          <div className="modern-card" style={{ textAlign: 'center' }}>
            <div style={{ 
              fontSize: '2.5rem', 
              fontWeight: '700',
              color: 'var(--green)',
              marginBottom: 'var(--space-2)',
              fontFamily: 'var(--font-mono)'
            }}>
              1000+
            </div>
            <p style={{ color: 'var(--gray-400)', fontSize: '0.9rem' }}>Educational Documents</p>
          </div>
          
          <div className="modern-card" style={{ textAlign: 'center' }}>
            <div style={{ 
              fontSize: '2.5rem', 
              fontWeight: '700',
              color: 'var(--blue)',
              marginBottom: 'var(--space-2)',
              fontFamily: 'var(--font-mono)'
            }}>
              AI
            </div>
            <p style={{ color: 'var(--gray-400)', fontSize: '0.9rem' }}>Powered Intelligence</p>
          </div>
          
          <div className="modern-card" style={{ textAlign: 'center' }}>
            <div style={{ 
              fontSize: '2.5rem', 
              fontWeight: '700',
              color: 'var(--purple)',
              marginBottom: 'var(--space-2)',
              fontFamily: 'var(--font-mono)'
            }}>
              24/7
            </div>
            <p style={{ color: 'var(--gray-400)', fontSize: '0.9rem' }}>Always Available</p>
          </div>
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

export default ModernHomePage;
