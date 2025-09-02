import React from 'react';

const QuestionCard = ({ question, onClick, timestamp }) => {
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return null;
    
    const date = new Date(timestamp);
    const now = new Date();
    const diffInMinutes = Math.floor((now - date) / (1000 * 60));
    
    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes} minutes ago`;
    if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)} hours ago`;
    return date.toLocaleDateString();
  };

  return (
    <div 
      className={`card question-card ${onClick ? 'clickable' : ''}`} 
      onClick={onClick}
    >
      <div className="question-header">
        <h3>Question</h3>
        {timestamp && (
          <span className="timestamp">
            {formatTimestamp(timestamp)}
          </span>
        )}
      </div>
      <p className="question-text">{question}</p>
    </div>
  );
};

export default QuestionCard;