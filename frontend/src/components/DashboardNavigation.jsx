import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Camera, TrendingUp, Sparkles, User } from 'lucide-react';

const DashboardNavigation = () => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <nav className="dashboard-nav">
      <button 
        className={`nav-btn ${location.pathname === '/skin-analysis' ? 'active' : ''}`}
        onClick={() => navigate('/skin-analysis')}
      >
        <span className="nav-icon"><Camera size={20} /></span>
        Skin Analysis
      </button>
      
      <button 
        className={`nav-btn ${location.pathname === '/progress' ? 'active' : ''}`}
        onClick={() => navigate('/progress')}
      >
        <span className="nav-icon"><TrendingUp size={20} /></span>
        Progress Tracker
      </button>
      
      <button 
        className={`nav-btn ${location.pathname === '/routine' ? 'active' : ''}`}
        onClick={() => navigate('/routine')}
      >
        <span className="nav-icon"><Sparkles size={20} /></span>
        My Routine
      </button>
      
      <button 
        className={`nav-btn ${location.pathname === '/profile' ? 'active' : ''}`}
        onClick={() => navigate('/profile')}
      >
        <span className="nav-icon"><User size={20} /></span>
        Profile
      </button>
    </nav>
  );
};

export default DashboardNavigation;
