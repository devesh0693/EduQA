import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import SearchBar from '../components/SearchBar';
import QuestionCard from '../components/QuestionCard';
import { getQuestions } from '../services/api';

const HomePage = () => {
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchQuestions = async () => {
      try {
        const data = await getQuestions();
        setQuestions(data.questions || []);
      } catch (error) {
        console.error('Error fetching questions:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchQuestions();
  }, []);

  const handleSearch = (query) => {
    navigate(`/search?q=${encodeURIComponent(query)}`);
  };

  const handleQuestionClick = (question) => {
    navigate(`/qa?q=${encodeURIComponent(question)}`);
  };

  return (
    <div className="page">
      <h1>Educational Q&A System</h1>
      
      <SearchBar 
        onSearch={handleSearch}
        placeholder="Search for educational content..."
      />

      <section>
        <h2 style={{ textAlign: 'center', marginBottom: '2rem', color: '#2c3e50' }}>
          Popular Questions
        </h2>
        
        {loading ? (
          <div className="loading">Loading questions...</div>
        ) : (
          <div>
            {questions.map((question, index) => (
              <QuestionCard
                key={index}
                question={question}
                onClick={() => handleQuestionClick(question)}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
};

export default HomePage;