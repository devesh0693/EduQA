import React, { useRef, useEffect } from 'react';
import { gsap } from 'gsap';

const AnimatedLoader = ({ message = "Finding answer..." }) => {
  const containerRef = useRef(null);
  const dotsRef = useRef([]);

  useEffect(() => {
    if (!containerRef.current) return;

    // Initial animation
    gsap.fromTo(
      containerRef.current,
      { 
        opacity: 0, 
        y: 20,
        scale: 0.9
      },
      { 
        opacity: 1, 
        y: 0,
        scale: 1,
        duration: 0.6,
        ease: "back.out(1.7)"
      }
    );

    // Animate dots
    const dots = dotsRef.current;
    if (dots.length > 0) {
      gsap.to(dots, {
        y: -10,
        duration: 0.6,
        stagger: 0.2,
        repeat: -1,
        yoyo: true,
        ease: "power2.inOut"
      });
    }

    // Pulse animation for the container
    const progressBar = containerRef.current.querySelector('.progress-bar');
    if (progressBar) {
      gsap.to(progressBar, {
        width: '100%',
        duration: 8,
        ease: 'power1.inOut',
        repeat: -1,
        onRepeat: () => {
          gsap.set(progressBar, { width: '0%' });
        }
      });
    }

    const brainIcon = containerRef.current.querySelector('.brain-icon svg');
    if (brainIcon) {
      gsap.to(brainIcon, {
        rotate: 5,
        scale: 1.05,
        duration: 2,
        ease: 'power1.inOut',
        repeat: -1,
        yoyo: true,
      });
    }

  }, []);

  return (
    <div 
      ref={containerRef}
      className="animated-loader"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '3rem',
        background: 'rgba(0, 0, 0, 0.8)',
        borderRadius: '20px',
        boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        minWidth: '300px'
      }}
    >
      {/* Animated brain icon */}
      <div className="brain-icon" style={{ marginBottom: '1rem' }}>
        <svg 
          width="60" 
          height="60" 
          viewBox="0 0 24 24" 
          fill="none"
          style={{ filter: 'drop-shadow(0 4px 8px rgba(102, 126, 234, 0.3))' }}
        >
          <path 
            d="M12 2a5.5 5.5 0 00-5.5 5.5c0 2.22 1.32 4.13 3.22 4.95a.75.75 0 01.31 1.2l-1.2 1.2a.75.75 0 01-1.06 0l-2-2a.75.75 0 010-1.06l2.3-2.3A5.48 5.48 0 016.5 7.5a5.5 5.5 0 0111 0c0 .92-.23 1.79-.64 2.56l.01.01c.02.03.04.06.06.09l.05.08a.75.75 0 01-1.3.78l-.05-.08-.06-.09a3.99 3.99 0 00-5.02-5.02L10 6.5a2.5 2.5 0 10-3.54 3.54l-1.2 1.2a.75.75 0 01-1.06 0l-2-2a.75.75 0 010-1.06l2.3-2.3A5.48 5.48 0 016.5 7.5a5.5 5.5 0 0111 0c0 2.22-1.32 4.13-3.22 4.95a.75.75 0 01-.31 1.2l1.2 1.2a.75.75 0 011.06 0l2-2a.75.75 0 010-1.06l-2.3-2.3A5.48 5.48 0 0117.5 7.5a5.5 5.5 0 00-5.5-5.5z"
            fill="url(#gradient)"
          />
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#667eea" />
              <stop offset="100%" stopColor="#764ba2" />
            </linearGradient>
          </defs>
        </svg>
      </div>

      {/* Loading text */}
      <div 
        className="loading-text"
        style={{
          fontSize: '1.2rem',
          fontWeight: '600',
          color: '#ffffff',
          marginBottom: '1rem',
          textAlign: 'center'
        }}
      >
        {message}
      </div>

      {/* Animated dots */}
      <div 
        className="loading-dots"
        style={{
          display: 'flex',
          gap: '0.5rem',
          alignItems: 'center'
        }}
      >
        {[0, 1, 2].map((index) => (
          <div
            key={index}
            ref={(el) => (dotsRef.current[index] = el)}
            style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #667eea, #764ba2)',
              boxShadow: '0 2px 8px rgba(102, 126, 234, 0.4)'
            }}
          />
        ))}
      </div>

      {/* Progress bar */}
      <div 
        className="progress-container"
        style={{
          width: '100%',
          height: '4px',
          background: 'rgba(102, 126, 234, 0.1)',
          borderRadius: '2px',
          marginTop: '1.5rem',
          overflow: 'hidden'
        }}
      >
        <div 
          className="progress-bar"
          style={{
            height: '100%',
            background: 'linear-gradient(90deg, #667eea, #764ba2)',
            borderRadius: '2px',
            width: '0%'
          }}
        />
      </div>
    </div>
  );
};

export default AnimatedLoader;
