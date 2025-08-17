import React, { useState, useEffect } from 'react';
import { useUser } from '@clerk/clerk-react';
import SkinAnalysis from '../Dashboard/components/SkinAnalysis/SkinAnalysis';
import Header from '../Dashboard/components/Header/Header';
import DashboardNavigation from '../components/DashboardNavigation';
import '../Dashboard/Dashboard.css';

const SkinAnalysisPage = () => {
  const { user, isLoaded } = useUser();
  const [userProfile, setUserProfile] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);

  useEffect(() => {
    if (isLoaded && user) {
      const localProfile = localStorage.getItem('userProfile');
      if (localProfile) {
        setUserProfile(JSON.parse(localProfile));
      } else if (user.unsafeMetadata) {
        setUserProfile(user.unsafeMetadata);
      }

      const history = localStorage.getItem(`skinAnalysis_${user.id}`);
      if (history) {
        setAnalysisHistory(JSON.parse(history));
      }
    }
  }, [user, isLoaded]);

  const saveAnalysisResult = (analysisData) => {
    const newAnalysis = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      ...analysisData
    };
    
    const updatedHistory = [newAnalysis, ...analysisHistory];
    setAnalysisHistory(updatedHistory);
    
    localStorage.setItem(`skinAnalysis_${user.id}`, JSON.stringify(updatedHistory));
  };

  if (!isLoaded) {
    return <div className="loading">Loading...</div>;
  }

  return (
    <div className="dashboard">
      <Header user={user} userProfile={userProfile} />
      <div className="dashboard-content">
        <DashboardNavigation />
        <main className="dashboard-main">
          <SkinAnalysis onAnalysisComplete={saveAnalysisResult} />
        </main>
      </div>
    </div>
  );
};

export default SkinAnalysisPage;
