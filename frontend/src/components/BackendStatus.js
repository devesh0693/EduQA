import React, { useState, useEffect } from 'react';

const BackendStatus = () => {
  const [status, setStatus] = useState('checking');
  const [lastChecked, setLastChecked] = useState(null);

  const checkBackendStatus = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/api/qa/health/');
      if (response.ok) {
        setStatus('connected');
      } else {
        setStatus('error');
      }
    } catch (error) {
      setStatus('error');
    }
    setLastChecked(new Date());
  };

  useEffect(() => {
    checkBackendStatus();
    
    // Check status every 30 seconds
    const interval = setInterval(checkBackendStatus, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = () => {
    switch (status) {
      case 'connected':
        return '#00d4aa';
      case 'error':
        return '#ef5350';
      default:
        return '#ffa726';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'connected':
        return 'Backend Connected';
      case 'error':
        return 'Backend Error';
      default:
        return 'Checking...';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'connected':
        return '●';
      case 'error':
        return '●';
      default:
        return '●';
    }
  };

  return (
    <div 
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        background: 'rgba(0, 0, 0, 0.8)',
        backdropFilter: 'blur(10px)',
        border: `1px solid ${getStatusColor()}`,
        borderRadius: '12px',
        padding: '8px 12px',
        fontSize: '0.8rem',
        fontWeight: '500',
        color: '#ffffff',
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        zIndex: 1000,
        boxShadow: `0 4px 15px rgba(0, 0, 0, 0.3)`,
        transition: 'all 0.3s ease',
        cursor: 'pointer'
      }}
      onClick={checkBackendStatus}
      title={`Last checked: ${lastChecked ? lastChecked.toLocaleTimeString() : 'Never'}`}
    >
      <span 
        style={{
          color: getStatusColor(),
          fontSize: '1rem',
          animation: status === 'checking' ? 'pulse 1.5s infinite' : 'none'
        }}
      >
        {getStatusIcon()}
      </span>
      <span>{getStatusText()}</span>
    </div>
  );
};

export default BackendStatus;
