// src/App.js

import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import Speech from './components/Speech'; 
import HomePage from './components/HomePage';
import Results from './components/Results';

function App() {
  return (
    <Router>
      <div>
        <Header />
        <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/speech" element={<Speech />} />
        <Route path="/results" element={<Results />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
