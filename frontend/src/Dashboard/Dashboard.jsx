import React, { useState, useEffect } from 'react';
import { useUser } from '@clerk/clerk-react';
import { Camera, TrendingUp, Sparkles, User } from 'lucide-react';
import SkinAnalysis from './components/SkinAnalysis/SkinAnalysis';
import ProgressTracker from './components/ProgressTracker/ProgressTracker';
import SkinRoutine from './components/SkinRoutine/SkinRoutine';
import ProfileSection from './components/ProfileSection/ProfileSection';
import Header from './components/Header/Header';
import './Dashboard.css';

const Dashboard = () => {
  const { user, isLoaded } = useUser();
  const [activeTab, setActiveTab] = useState('analysis');
  const [userProfile, setUserProfile] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);

  useEffect(() => {
    if (isLoaded && user) {
      // Load user profile from localStorage or Clerk metadata
      const localProfile = localStorage.getItem('userProfile');
      if (localProfile) {
        setUserProfile(JSON.parse(localProfile));
      } else if (user.unsafeMetadata) {
        setUserProfile(user.unsafeMetadata);
      }

      // Fetch analysis history from backend DB
      (async () => {
        try {
          const res = await fetch(`http://localhost:8000/api/skin/history/${user.id}`);
          if (res.ok) {
            const data = await res.json();
            setAnalysisHistory(data);
          } else {
            // Fallback to localStorage if server history unavailable
            const history = localStorage.getItem(`skinAnalysis_${user.id}`);
            if (history) setAnalysisHistory(JSON.parse(history));
          }
        } catch (e) {
          const history = localStorage.getItem(`skinAnalysis_${user.id}`);
          if (history) setAnalysisHistory(JSON.parse(history));
        }
      })();
    }
  }, [user, isLoaded]);

  const updateUserProfile = (profileData) => {
    setUserProfile(profileData);
    localStorage.setItem('userProfile', JSON.stringify(profileData));
    
    // Also save to Clerk metadata if needed
    if (user) {
      user.update({
        unsafeMetadata: {
          ...user.unsafeMetadata,
          ...profileData
        }
      });
    }
  };

  const saveAnalysisResult = async (analysisData) => {
    const newAnalysis = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      ...analysisData
    };
    
    // Optimistic update UI
    const updatedHistory = [newAnalysis, ...analysisHistory];
    setAnalysisHistory(updatedHistory);
    localStorage.setItem(`skinAnalysis_${user.id}`, JSON.stringify(updatedHistory));

    // Refresh from backend to pull canonical saved record ordering
    try {
      const res = await fetch(`http://localhost:8000/api/skin/history/${user.id}`);
      if (res.ok) {
        const data = await res.json();
        setAnalysisHistory(data);
        localStorage.setItem(`skinAnalysis_${user.id}`, JSON.stringify(data));
      }
    } catch {}
  };

  if (!isLoaded) {
    return <div className="loading">Loading...</div>;
  }

  return (
    <div className="dashboard">
      <Header user={user} userProfile={userProfile} />
      
      <div className="dashboard-content">
        <nav className="dashboard-nav">
          <button 
            className={`nav-btn ${activeTab === 'analysis' ? 'active' : ''}`}
            onClick={() => setActiveTab('analysis')}
          >
            <span className="nav-icon"><Camera size={20} /></span>
            Skin Analysis
          </button>
          
          <button 
            className={`nav-btn ${activeTab === 'progress' ? 'active' : ''}`}
            onClick={() => setActiveTab('progress')}
          >
            <span className="nav-icon"><TrendingUp size={20} /></span>
            Progress Tracker
          </button>
          
          <button 
            className={`nav-btn ${activeTab === 'routine' ? 'active' : ''}`}
            onClick={() => setActiveTab('routine')}
          >
            <span className="nav-icon"><Sparkles size={20} /></span>
            My Routine
          </button>
          
          <button 
            className={`nav-btn ${activeTab === 'profile' ? 'active' : ''}`}
            onClick={() => setActiveTab('profile')}
          >
            <span className="nav-icon"><User size={20} /></span>
            Profile
          </button>
        </nav>

        <main className="dashboard-main">
          {activeTab === 'analysis' && (
            <SkinAnalysis 
              onAnalysisComplete={saveAnalysisResult}
              clerkUserId={user?.id}
            />
          )}
          
          {activeTab === 'progress' && (
            <ProgressTracker 
              analysisHistory={analysisHistory}
              clerkUserId={user?.id}
            />
          )}
          
          {activeTab === 'routine' && (
            <SkinRoutine 
              latestAnalysis={analysisHistory[0]}
            />
          )}
          
          {activeTab === 'profile' && (
            <ProfileSection 
              analysisHistory={analysisHistory}
              userProfile={userProfile}
              onProfileUpdate={updateUserProfile}
            />
          )}
        </main>
      </div>
    </div>
  );
};

export default Dashboard;
