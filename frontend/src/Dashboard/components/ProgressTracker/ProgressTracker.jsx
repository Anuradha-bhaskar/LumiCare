import React, { useState, useMemo } from 'react';
import { TrendingUp, TrendingDown, ArrowRight, BarChart3, Calendar, Target } from 'lucide-react';
import './ProgressTracker.css';

const ProgressTracker = ({ analysisHistory }) => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('all'); // all, week, month, 3months

  const filteredHistory = useMemo(() => {
    if (!analysisHistory || analysisHistory.length === 0) return [];
    
    const now = new Date();
    let cutoffDate;
    
    switch (selectedTimeframe) {
      case 'week':
        cutoffDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        break;
      case 'month':
        cutoffDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        break;
      case '3months':
        cutoffDate = new Date(now.getTime() - 90 * 24 * 60 * 60 * 1000);
        break;
      default:
        return analysisHistory;
    }
    
    return analysisHistory.filter(analysis => 
      new Date(analysis.timestamp) >= cutoffDate
    );
  }, [analysisHistory, selectedTimeframe]);

  const progressData = useMemo(() => {
    if (filteredHistory.length === 0) return null;
    
    const latest = filteredHistory[0];
    const oldest = filteredHistory[filteredHistory.length - 1];
    
    if (!latest.analysis || !oldest.analysis) return null;
    
    const improvements = {
      hydration: latest.analysis.metrics.hydration - oldest.analysis.metrics.hydration,
      clarity: latest.analysis.metrics.clarity - oldest.analysis.metrics.clarity,
      texture: latest.analysis.metrics.texture - oldest.analysis.metrics.texture,
      poreSize: latest.analysis.metrics.poreSize - oldest.analysis.metrics.poreSize,
      overall: latest.analysis.skinHealth - oldest.analysis.skinHealth
    };
    
    return {
      latest: latest.analysis,
      oldest: oldest.analysis,
      improvements,
      totalAnalyses: filteredHistory.length,
      timeSpan: Math.ceil((new Date(latest.timestamp) - new Date(oldest.timestamp)) / (1000 * 60 * 60 * 24))
    };
  }, [filteredHistory]);

  const getImprovementColor = (value) => {
    if (value > 5) return 'var(--success)';
    if (value < -5) return 'var(--error)';
    return 'var(--warning)';
  };

  const getImprovementIcon = (value) => {
    if (value > 5) return <TrendingUp size={16} />;
    if (value < -5) return <TrendingDown size={16} />;
    return <ArrowRight size={16} />;
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      year: 'numeric'
    });
  };

  if (!analysisHistory || analysisHistory.length === 0) {
    return (
      <div className="progress-tracker">
        <div className="empty-state">
          <div className="empty-icon"><BarChart3 size={48} /></div>
          <h2>No Progress Data Yet</h2>
          <p>Complete your first skin analysis to start tracking your progress!</p>
          <div className="empty-benefits">
            <div className="benefit"><TrendingUp size={20} className="benefit-icon" /> Track skin improvements over time</div>
            <div className="benefit"><Calendar size={20} className="benefit-icon" /> Compare results across different periods</div>
            <div className="benefit"><Target size={20} className="benefit-icon" /> See which treatments work best</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="progress-tracker">
      <div className="progress-header">
        <h2><TrendingUp size={24} className="inline-icon" /> Your Skin Progress Journey</h2>
        <p>Track your skin health improvements over time</p>
      </div>

      <div className="timeframe-selector">
        <button 
          className={`timeframe-btn ${selectedTimeframe === 'all' ? 'active' : ''}`}
          onClick={() => setSelectedTimeframe('all')}
        >
          All Time
        </button>
        <button 
          className={`timeframe-btn ${selectedTimeframe === '3months' ? 'active' : ''}`}
          onClick={() => setSelectedTimeframe('3months')}
        >
          3 Months
        </button>
        <button 
          className={`timeframe-btn ${selectedTimeframe === 'month' ? 'active' : ''}`}
          onClick={() => setSelectedTimeframe('month')}
        >
          1 Month
        </button>
        <button 
          className={`timeframe-btn ${selectedTimeframe === 'week' ? 'active' : ''}`}
          onClick={() => setSelectedTimeframe('week')}
        >
          1 Week
        </button>
      </div>

      {progressData && (
        <>
          <div className="progress-overview">
            <div className="overview-card">
              <h3>Progress Summary</h3>
              <div className="summary-stats">
                <div className="stat">
                  <span className="stat-number">{progressData.totalAnalyses}</span>
                  <span className="stat-label">Total Analyses</span>
                </div>
                <div className="stat">
                  <span className="stat-number">{progressData.timeSpan}</span>
                  <span className="stat-label">Days Tracked</span>
                </div>
                <div className="stat">
                  <span 
                    className="stat-number"
                    style={{ color: getImprovementColor(progressData.improvements.overall) }}
                  >
                    {progressData.improvements.overall > 0 ? '+' : ''}
                    {progressData.improvements.overall.toFixed(1)}%
                  </span>
                  <span className="stat-label">Overall Change</span>
                </div>
              </div>
            </div>
          </div>

          <div className="metrics-comparison">
            <h3>Metric Improvements</h3>
            <div className="metrics-grid">
              <div className="metric-comparison">
                <div className="metric-header">
                  <span className="metric-name">Hydration</span>
                  <span className="metric-change" style={{ color: getImprovementColor(progressData.improvements.hydration) }}>
                    {getImprovementIcon(progressData.improvements.hydration)}
                    {progressData.improvements.hydration > 0 ? '+' : ''}
                    {progressData.improvements.hydration.toFixed(1)}%
                  </span>
                </div>
                <div className="metric-bars">
                  <div className="metric-bar-container">
                    <span className="bar-label">Before</span>
                    <div className="metric-bar old">
                      <div 
                        className="metric-fill old" 
                        style={{ width: `${progressData.oldest.metrics.hydration}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.oldest.metrics.hydration}%</span>
                  </div>
                  <div className="metric-bar-container">
                    <span className="bar-label">Now</span>
                    <div className="metric-bar new">
                      <div 
                        className="metric-fill new" 
                        style={{ width: `${progressData.latest.metrics.hydration}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.latest.metrics.hydration}%</span>
                  </div>
                </div>
              </div>

              <div className="metric-comparison">
                <div className="metric-header">
                  <span className="metric-name">Clarity</span>
                  <span className="metric-change" style={{ color: getImprovementColor(progressData.improvements.clarity) }}>
                    {getImprovementIcon(progressData.improvements.clarity)}
                    {progressData.improvements.clarity > 0 ? '+' : ''}
                    {progressData.improvements.clarity.toFixed(1)}%
                  </span>
                </div>
                <div className="metric-bars">
                  <div className="metric-bar-container">
                    <span className="bar-label">Before</span>
                    <div className="metric-bar old">
                      <div 
                        className="metric-fill old" 
                        style={{ width: `${progressData.oldest.metrics.clarity}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.oldest.metrics.clarity}%</span>
                  </div>
                  <div className="metric-bar-container">
                    <span className="bar-label">Now</span>
                    <div className="metric-bar new">
                      <div 
                        className="metric-fill new" 
                        style={{ width: `${progressData.latest.metrics.clarity}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.latest.metrics.clarity}%</span>
                  </div>
                </div>
              </div>

              <div className="metric-comparison">
                <div className="metric-header">
                  <span className="metric-name">Texture</span>
                  <span className="metric-change" style={{ color: getImprovementColor(progressData.improvements.texture) }}>
                    {getImprovementIcon(progressData.improvements.texture)}
                    {progressData.improvements.texture > 0 ? '+' : ''}
                    {progressData.improvements.texture.toFixed(1)}%
                  </span>
                </div>
                <div className="metric-bars">
                  <div className="metric-bar-container">
                    <span className="bar-label">Before</span>
                    <div className="metric-bar old">
                      <div 
                        className="metric-fill old" 
                        style={{ width: `${progressData.oldest.metrics.texture}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.oldest.metrics.texture}%</span>
                  </div>
                  <div className="metric-bar-container">
                    <span className="bar-label">Now</span>
                    <div className="metric-bar new">
                      <div 
                        className="metric-fill new" 
                        style={{ width: `${progressData.latest.metrics.texture}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.latest.metrics.texture}%</span>
                  </div>
                </div>
              </div>

              <div className="metric-comparison">
                <div className="metric-header">
                  <span className="metric-name">Pore Size</span>
                  <span className="metric-change" style={{ color: getImprovementColor(progressData.improvements.poreSize) }}>
                    {getImprovementIcon(progressData.improvements.poreSize)}
                    {progressData.improvements.poreSize > 0 ? '+' : ''}
                    {progressData.improvements.poreSize.toFixed(1)}%
                  </span>
                </div>
                <div className="metric-bars">
                  <div className="metric-bar-container">
                    <span className="bar-label">Before</span>
                    <div className="metric-bar old">
                      <div 
                        className="metric-fill old" 
                        style={{ width: `${progressData.oldest.metrics.poreSize}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.oldest.metrics.poreSize}%</span>
                  </div>
                  <div className="metric-bar-container">
                    <span className="bar-label">Now</span>
                    <div className="metric-bar new">
                      <div 
                        className="metric-fill new" 
                        style={{ width: `${progressData.latest.metrics.poreSize}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.latest.metrics.poreSize}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="analysis-timeline">
            <h3>Analysis History</h3>
            <div className="timeline">
              {filteredHistory.map((analysis, index) => (
                <div key={analysis.id} className="timeline-item">
                  <div className="timeline-marker">
                    <div className="marker-dot"></div>
                    {index < filteredHistory.length - 1 && <div className="marker-line"></div>}
                  </div>
                  <div className="timeline-content">
                    <div className="timeline-date">{formatDate(analysis.timestamp)}</div>
                    <div className="timeline-photo">
                      <img src={analysis.image} alt={`Analysis ${index + 1}`} />
                    </div>
                    <div className="timeline-data">
                      <div className="timeline-health">
                        Overall Health: <span className="health-score">{analysis.analysis.skinHealth}%</span>
                      </div>
                      <div className="timeline-type">
                        Skin Type: <span className="skin-type">{analysis.analysis.skinType}</span>
                      </div>
                      <div className="timeline-concerns">
                        <strong>Main Concerns:</strong>
                        <ul>
                          {analysis.analysis.concerns.slice(0, 2).map((concern, i) => (
                            <li key={i}>{concern}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {filteredHistory.length < 2 && (
        <div className="insufficient-data">
          <div className="info-card">
            <h3><BarChart3 size={20} className="inline-icon" /> Need More Data for Progress Tracking</h3>
            <p>Complete at least 2 analyses to see your progress trends and improvements over time.</p>
            <div className="next-analysis-tip">
              <strong>Tip:</strong> For best results, take analyses consistently (weekly or bi-weekly) 
              to track meaningful changes in your skin health.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProgressTracker;
