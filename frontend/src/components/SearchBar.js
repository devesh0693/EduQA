import React, { useState, useRef, useEffect } from 'react';
import { gsap } from 'gsap';

const SearchBar = ({ onSearch, placeholder = "Enter your search query...", disabled = false }) => {
  const [query, setQuery] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const inputRef = useRef(null);
  const buttonRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    // Initial animation
    if (containerRef.current) {
      gsap.fromTo(
        containerRef.current,
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.8, ease: "power2.out" }
      );
    }
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !disabled) {
      // Animate button press
      if (buttonRef.current) {
        gsap.to(buttonRef.current, {
          scale: 0.95,
          duration: 0.1,
          yoyo: true,
          repeat: 1,
          ease: "power2.out"
        });
      }
      
      onSearch(query.trim());
    }
  };

  const handleFocus = () => {
    setIsFocused(true);
    if (containerRef.current) {
      gsap.to(containerRef.current, {
        boxShadow: "0 0 30px rgba(102, 126, 234, 0.3)",
        scale: 1.02,
        duration: 0.3,
        ease: "power2.out"
      });
    }
  };

  const handleBlur = () => {
    setIsFocused(false);
    if (containerRef.current) {
      gsap.to(containerRef.current, {
        boxShadow: "0 8px 32px rgba(0,0,0,0.1)",
        scale: 1,
        duration: 0.3,
        ease: "power2.out"
      });
    }
  };

  const handleInputChange = (e) => {
    setQuery(e.target.value);
    
    // Animate input field
    if (inputRef.current) {
      gsap.to(inputRef.current, {
        borderColor: e.target.value ? "#667eea" : "transparent",
        duration: 0.2,
        ease: "power2.out"
      });
    }
  };

  return (
    <div className="search-container">
      <form 
        onSubmit={handleSubmit} 
        className="search-bar"
        ref={containerRef}
        style={{
          display: 'flex',
          gap: '1rem',
          maxWidth: '700px',
          margin: '0 auto',
          background: 'rgba(255, 255, 255, 0.95)',
          padding: '1rem',
          borderRadius: '16px',
          boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          transition: 'all 0.3s ease'
        }}
      >
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={handleInputChange}
          onFocus={handleFocus}
          onBlur={handleBlur}
          placeholder={placeholder}
          className={`search-input ${disabled ? 'disabled' : ''}`}
          disabled={disabled}
          style={{
            flex: 1,
            padding: '1rem 1.5rem',
            border: '2px solid #e2e8f0',
            borderRadius: '12px',
            fontSize: '1.1rem',
            outline: 'none',
            transition: 'all 0.3s ease',
            backgroundColor: 'white',
            color: '#1a202c',
            boxShadow: isFocused ? '0 0 0 3px rgba(102, 126, 234, 0.2)' : '0 2px 4px rgba(0, 0, 0, 0.05)',
            borderColor: isFocused ? '#667eea' : '#e2e8f0',
            width: '100%',
            boxSizing: 'border-box',
            minWidth: '300px',
            opacity: disabled ? 0.7 : 1,
            cursor: disabled ? 'not-allowed' : 'text'
          }} 
        />
        <button 
          ref={buttonRef}
          type="submit" 
          className={`search-btn ${disabled ? 'disabled' : ''}`}
          disabled={disabled}
          style={{
            background: 'linear-gradient(135deg, #667eea, #764ba2)',
            color: 'white',
            border: 'none',
            padding: '1rem 2rem',
            borderRadius: '12px',
            cursor: 'pointer',
            fontSize: '1.1rem',
            fontWeight: '600',
            transition: 'all 0.3s ease',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            minWidth: '120px',
            justifyContent: 'center',
            transform: 'scale(1)',
            boxShadow: '0 4px 15px rgba(102, 126, 234, 0.3)'
          }}
        >
          {disabled ? (
            <>
              <span className="loading-dots">Searching</span>
              <div 
                className="loading-spinner"
                style={{
                  width: '16px',
                  height: '16px',
                  border: '2px solid rgba(255, 255, 255, 0.3)',
                  borderTop: '2px solid white',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }}
              />
            </>
          ) : (
            <>
              <span>Search</span>
              <svg 
                width="16" 
                height="16" 
                viewBox="0 0 24 24" 
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <circle cx="11" cy="11" r="8"/>
                <path d="m21 21-4.35-4.35"/>
              </svg>
            </>
          )}
        </button>
      </form>
    </div>
  );
};

export default SearchBar;