import React from 'react';
import { useUser } from '@clerk/clerk-react';
import { User, BarChart3, TrendingUp, TrendingDown, Calendar, Target, Search, FileText } from 'lucide-react';
import './ProfileSection.css';

const ProfileSection = ({ analysisHistory, userProfile /* onProfileUpdate (unused) */ }) => {
  const { user } = useUser();

  const getAnalysisStats = () => {
    if (!analysisHistory || analysisHistory.length === 0) {
      return {
        total: 0,
        thisMonth: 0,
        averageHealth: 0,
        improvementTrend: 0,
        lastAnalysis: null
      };
    }

    const now = new Date();
    const thisMonth = new Date(now.getFullYear(), now.getMonth(), 1);
    const thisMonthAnalyses = analysisHistory.filter(analysis => 
      new Date(analysis.timestamp) >= thisMonth
    );

    const healthScores = analysisHistory.map(analysis => analysis.analysis?.skinHealth || 0);
    const averageHealth = healthScores.reduce((sum, score) => sum + score, 0) / healthScores.length;

    let improvementTrend = 0;
    if (analysisHistory.length >= 2) {
      const recent = analysisHistory.slice(0, 3);
      const older = analysisHistory.slice(-3);
      const recentAvg = recent.reduce((sum, analysis) => sum + (analysis.analysis?.skinHealth || 0), 0) / recent.length;
      const olderAvg = older.reduce((sum, analysis) => sum + (analysis.analysis?.skinHealth || 0), 0) / older.length;
      improvementTrend = recentAvg - olderAvg;
    }

    return {
      total: analysisHistory.length,
      thisMonth: thisMonthAnalyses.length,
      averageHealth: Math.round(averageHealth),
      improvementTrend: Math.round(improvementTrend * 10) / 10,
      lastAnalysis: analysisHistory[0]
    };
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', { 
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const stats = getAnalysisStats();

  return (
    <div className="profile-section">
      <div className="profile-header">
        <h2><User size={24} className="inline-icon" /> Your Profile</h2>
        <p>Manage your personal information and track your skincare journey</p>
      </div>

      <div className="profile-content">
        <div className="profile-card">
          <div className="profile-info">
            <div className="profile-avatar">
              <img 
                src={user?.imageUrl || '/default-avatar.png'} 
                alt="Profile" 
                onError={(e) => {
                  e.target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(userProfile?.fullName || 'User')}&background=ffb79e&color=fff&size=120`;
                }}
              />
            </div>
            
            <div className="profile-details">
              <h3>{userProfile?.fullName || 'Complete Your Profile'}</h3>
              <div className="profile-meta">
                {userProfile?.age && <span>Age: {userProfile.age}</span>}
                {userProfile?.gender && <span>Gender: {userProfile.gender}</span>}
              </div>
              <div className="profile-email">
                {user?.primaryEmailAddress?.emailAddress}
              </div>
            </div>
          </div>
        </div>

        <div className="profile-stats">
          <h3><BarChart3 size={20} className="inline-icon" /> Your Skincare Journey</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-icon"><TrendingUp size={20} /></div>
              <div className="stat-content">
                <div className="stat-number">{stats.total}</div>
                <div className="stat-label">Total Analyses</div>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon"><Calendar size={20} /></div>
              <div className="stat-content">
                <div className="stat-number">{stats.thisMonth}</div>
                <div className="stat-label">This Month</div>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon"><Target size={20} /></div>
              <div className="stat-content">
                <div className="stat-number">{stats.averageHealth}%</div>
                <div className="stat-label">Avg. Skin Health</div>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon">{stats.improvementTrend >= 0 ? <TrendingUp size={20} /> : <TrendingDown size={20} />}</div>
              <div className="stat-content">
                <div className={`stat-number ${stats.improvementTrend >= 0 ? 'positive' : 'negative'}`}>
                  {stats.improvementTrend > 0 ? '+' : ''}{stats.improvementTrend}%
                </div>
                <div className="stat-label">Progress Trend</div>
              </div>
            </div>
          </div>

          {stats.lastAnalysis && (
            <div className="last-analysis">
              <h4><Search size={16} className="inline-icon" /> Latest Analysis</h4>
              <div className="analysis-summary">
                <div className="analysis-date">
                  {formatDate(stats.lastAnalysis.timestamp)}
                </div>
                <div className="analysis-details">
                  <div className="analysis-health">
                    Overall Health: <span className="health-score">{stats.lastAnalysis.analysis.skinHealth}%</span>
                  </div>
                  <div className="analysis-type">
                    Skin Type: <span className="skin-type">{stats.lastAnalysis.analysis.skinType}</span>
                  </div>
                  <div className="top-concerns">
                    <strong>Top Concerns:</strong>
                    <div className="concerns-tags">
                      {stats.lastAnalysis.analysis.concerns.slice(0, 3).map((concern, index) => (
                        <span key={index} className="concern-tag">{concern}</span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {(!analysisHistory || analysisHistory.length === 0) && (
          <div className="no-data-card">
            <div className="no-data-icon"><BarChart3 size={32} /></div>
            <h3>Start Your Skincare Journey</h3>
            <p>Take your first skin analysis to begin tracking your progress and get personalized recommendations!</p>
            <div className="journey-features">
              <div className="feature"><TrendingUp size={16} className="inline-icon" /> Track progress over time</div>
              <div className="feature"><Target size={16} className="inline-icon" /> Get personalized recommendations</div>
              <div className="feature"><FileText size={16} className="inline-icon" /> Keep detailed analysis history</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ProfileSection;
