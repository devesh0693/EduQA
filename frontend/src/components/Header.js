import React from 'react';
import { Link } from 'react-router-dom';

const Header = () => {
  return (
    <header className="header">
      <Link to="/" className="logo">
        <h1>EduQA System</h1>
      </Link>
      <nav className="nav-links">
        <Link to="/">Home</Link>
        <Link to="/search">Search</Link>
        <Link to="/qa">Q&A</Link>
      </nav>
    </header>
  );
};

export default Header;