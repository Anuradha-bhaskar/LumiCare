import React, { useState } from 'react';
import { useUser } from '@clerk/clerk-react';
import { User, Edit3, Save, X, BarChart3, TrendingUp, TrendingDown, Calendar, Target, Search, FileText } from 'lucide-react';
import './ProfileSection.css';

const ProfileSection = ({ analysisHistory, userProfile, onProfileUpdate }) => {
  const { user } = useUser();
  const [isEditing, setIsEditing] = useState(false);
  const [editForm, setEditForm] = useState({
    fullName: userProfile?.fullName || '',
    age: userProfile?.age || '',
    gender: userProfile?.gender || '',
    skinGoals: userProfile?.skinGoals || [],
    skinConcerns: userProfile?.skinConcerns || [],
    allergies: userProfile?.allergies || [],
    currentProducts: userProfile?.currentProducts || []
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setEditForm(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleArrayInputChange = (field, value) => {
    const items = value.split(',').map(item => item.trim()).filter(item => item);
    setEditForm(prev => ({
      ...prev,
      [field]: items
    }));
  };

  const handleSave = () => {
    onProfileUpdate(editForm);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setEditForm({
      fullName: userProfile?.fullName || '',
      age: userProfile?.age || '',
      gender: userProfile?.gender || '',
      skinGoals: userProfile?.skinGoals || [],
      skinConcerns: userProfile?.skinConcerns || [],
      allergies: userProfile?.allergies || [],
      currentProducts: userProfile?.currentProducts || []
    });
    setIsEditing(false);
  };

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
              {!isEditing ? (
                <>
                  <h3>{userProfile?.fullName || 'Complete Your Profile'}</h3>
                  <div className="profile-meta">
                    {userProfile?.age && <span>Age: {userProfile.age}</span>}
                    {userProfile?.gender && <span>Gender: {userProfile.gender}</span>}
                  </div>
                  <div className="profile-email">
                    {user?.primaryEmailAddress?.emailAddress}
                  </div>
                  <button 
                    className="edit-profile-btn"
                    onClick={() => setIsEditing(true)}
                  >
                    <Edit3 size={16} className="inline-icon" /> Edit Profile
                  </button>
                </>
              ) : (
                <div className="edit-form">
                  <div className="form-group">
                    <label>Full Name</label>
                    <input
                      type="text"
                      name="fullName"
                      value={editForm.fullName}
                      onChange={handleInputChange}
                      placeholder="Enter your full name"
                    />
                  </div>
                  
                  <div className="form-row">
                    <div className="form-group">
                      <label>Age</label>
                      <input
                        type="number"
                        name="age"
                        value={editForm.age}
                        onChange={handleInputChange}
                        placeholder="Age"
                        min="13"
                        max="100"
                      />
                    </div>
                    
                    <div className="form-group">
                      <label>Gender</label>
                      <select
                        name="gender"
                        value={editForm.gender}
                        onChange={handleInputChange}
                      >
                        <option value="">Select Gender</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Non-binary">Non-binary</option>
                        <option value="Prefer not to say">Prefer not to say</option>
                      </select>
                    </div>
                  </div>

                  <div className="form-group">
                    <label>Skin Goals</label>
                    <input
                      type="text"
                      value={editForm.skinGoals.join(', ')}
                      onChange={(e) => handleArrayInputChange('skinGoals', e.target.value)}
                      placeholder="e.g., Clear skin, Anti-aging, Hydration (separate with commas)"
                    />
                  </div>

                  <div className="form-group">
                    <label>Skin Concerns</label>
                    <input
                      type="text"
                      value={editForm.skinConcerns.join(', ')}
                      onChange={(e) => handleArrayInputChange('skinConcerns', e.target.value)}
                      placeholder="e.g., Acne, Dark spots, Wrinkles (separate with commas)"
                    />
                  </div>

                  <div className="form-group">
                    <label>Known Allergies</label>
                    <input
                      type="text"
                      value={editForm.allergies.join(', ')}
                      onChange={(e) => handleArrayInputChange('allergies', e.target.value)}
                      placeholder="e.g., Fragrances, Sulfates, Parabens (separate with commas)"
                    />
                  </div>

                  <div className="form-group">
                    <label>Current Products</label>
                    <input
                      type="text"
                      value={editForm.currentProducts.join(', ')}
                      onChange={(e) => handleArrayInputChange('currentProducts', e.target.value)}
                      placeholder="e.g., CeraVe Cleanser, Neutrogena Moisturizer (separate with commas)"
                    />
                  </div>

                  <div className="form-actions">
                    <button className="save-btn" onClick={handleSave}>
                      💾 Save Changes
                    </button>
                    <button className="cancel-btn" onClick={handleCancel}>
                      ❌ Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="profile-stats">
          <h3>📊 Your Skincare Journey</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-icon">📈</div>
              <div className="stat-content">
                <div className="stat-number">{stats.total}</div>
                <div className="stat-label">Total Analyses</div>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon">📅</div>
              <div className="stat-content">
                <div className="stat-number">{stats.thisMonth}</div>
                <div className="stat-label">This Month</div>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon">🎯</div>
              <div className="stat-content">
                <div className="stat-number">{stats.averageHealth}%</div>
                <div className="stat-label">Avg. Skin Health</div>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon">{stats.improvementTrend >= 0 ? '📈' : '📉'}</div>
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
              <h4>🔍 Latest Analysis</h4>
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

        {userProfile && (
          <div className="profile-preferences">
            <h3>🎯 Your Skincare Profile</h3>
            
            {userProfile.skinGoals && userProfile.skinGoals.length > 0 && (
              <div className="preference-section">
                <h4>Skin Goals</h4>
                <div className="tags-list">
                  {userProfile.skinGoals.map((goal, index) => (
                    <span key={index} className="tag goal-tag">{goal}</span>
                  ))}
                </div>
              </div>
            )}

            {userProfile.skinConcerns && userProfile.skinConcerns.length > 0 && (
              <div className="preference-section">
                <h4>Current Concerns</h4>
                <div className="tags-list">
                  {userProfile.skinConcerns.map((concern, index) => (
                    <span key={index} className="tag concern-tag">{concern}</span>
                  ))}
                </div>
              </div>
            )}

            {userProfile.allergies && userProfile.allergies.length > 0 && (
              <div className="preference-section">
                <h4>Allergies & Sensitivities</h4>
                <div className="tags-list">
                  {userProfile.allergies.map((allergy, index) => (
                    <span key={index} className="tag allergy-tag">{allergy}</span>
                  ))}
                </div>
              </div>
            )}

            {userProfile.currentProducts && userProfile.currentProducts.length > 0 && (
              <div className="preference-section">
                <h4>Current Products</h4>
                <div className="tags-list">
                  {userProfile.currentProducts.map((product, index) => (
                    <span key={index} className="tag product-tag">{product}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {(!analysisHistory || analysisHistory.length === 0) && (
          <div className="no-data-card">
            <div className="no-data-icon">📊</div>
            <h3>Start Your Skincare Journey</h3>
            <p>Take your first skin analysis to begin tracking your progress and get personalized recommendations!</p>
            <div className="journey-features">
              <div className="feature">📈 Track progress over time</div>
              <div className="feature">🎯 Get personalized recommendations</div>
              <div className="feature">📝 Keep detailed analysis history</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ProfileSection;
