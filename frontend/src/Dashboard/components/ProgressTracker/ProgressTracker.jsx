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

  const toPercent = (value) => {
    if (typeof value !== 'number') return null;
    if (value <= 1) return Math.round(value * 100);
    return Math.round(Math.max(0, Math.min(100, value)));
  };

  const getMetricPercent = (analysis, key) => {
    const score = analysis?.analysis?.metrics?.[key]?.score;
    return toPercent(score);
  };

  const progressData = useMemo(() => {
    if (filteredHistory.length < 2) return null;
    
    const latest = filteredHistory[0];
    const oldest = filteredHistory[filteredHistory.length - 1];
    
    const latestSkinHealth = typeof latest?.analysis?.skinHealth === 'number' ? latest.analysis.skinHealth : null;
    const oldestSkinHealth = typeof oldest?.analysis?.skinHealth === 'number' ? oldest.analysis.skinHealth : null;

    const latestMetrics = {
      hydration: getMetricPercent(latest, 'hydration'),
      pigmentation: getMetricPercent(latest, 'pigmentation'),
      wrinkles: getMetricPercent(latest, 'wrinkles'),
      pores: getMetricPercent(latest, 'pores'),
    };
    const oldestMetrics = {
      hydration: getMetricPercent(oldest, 'hydration'),
      pigmentation: getMetricPercent(oldest, 'pigmentation'),
      wrinkles: getMetricPercent(oldest, 'wrinkles'),
      pores: getMetricPercent(oldest, 'pores'),
    };

    const improvements = {
      hydration: latestMetrics.hydration != null && oldestMetrics.hydration != null ? (latestMetrics.hydration - oldestMetrics.hydration) : null,
      pigmentation: latestMetrics.pigmentation != null && oldestMetrics.pigmentation != null ? (latestMetrics.pigmentation - oldestMetrics.pigmentation) : null,
      wrinkles: latestMetrics.wrinkles != null && oldestMetrics.wrinkles != null ? (latestMetrics.wrinkles - oldestMetrics.wrinkles) : null,
      pores: latestMetrics.pores != null && oldestMetrics.pores != null ? (latestMetrics.pores - oldestMetrics.pores) : null,
      overall: latestSkinHealth != null && oldestSkinHealth != null ? (latestSkinHealth - oldestSkinHealth) : null,
    };
    
    return {
      latest,
      oldest,
      latestMetrics,
      oldestMetrics,
      improvements,
      totalAnalyses: filteredHistory.length,
      timeSpan: Math.ceil((new Date(latest.timestamp) - new Date(oldest.timestamp)) / (1000 * 60 * 60 * 24))
    };
  }, [filteredHistory]);

  const getImprovementColor = (value) => {
    if (value == null) return 'var(--text-secondary)';
    if (value > 5) return 'var(--success)';
    if (value < -5) return 'var(--error)';
    return 'var(--warning)';
  };

  const getImprovementIcon = (value) => {
    if (value == null) return <ArrowRight size={16} />;
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
                    {progressData.improvements.overall != null && progressData.improvements.overall > 0 ? '+' : ''}
                    {progressData.improvements.overall != null ? progressData.improvements.overall.toFixed(1) : '—'}%
                  </span>
                  <span className="stat-label">Overall Change</span>
                </div>
              </div>
            </div>
          </div>

          <div className="metrics-comparison">
            <h3>Metric Improvements</h3>
            <div className="metrics-grid">
              {/* Hydration */}
              <div className="metric-comparison">
                <div className="metric-header">
                  <span className="metric-name">Hydration</span>
                  <span className="metric-change" style={{ color: getImprovementColor(progressData.improvements.hydration) }}>
                    {getImprovementIcon(progressData.improvements.hydration)}
                    {progressData.improvements.hydration != null && progressData.improvements.hydration > 0 ? '+' : ''}
                    {progressData.improvements.hydration != null ? progressData.improvements.hydration.toFixed(1) : '—'}%
                  </span>
                </div>
                <div className="metric-bars">
                  <div className="metric-bar-container">
                    <span className="bar-label">Before</span>
                    <div className="metric-bar old">
                      <div 
                        className="metric-fill old" 
                        style={{ width: `${progressData.oldestMetrics.hydration ?? 0}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.oldestMetrics.hydration != null ? `${progressData.oldestMetrics.hydration}%` : '—'}</span>
                  </div>
                  <div className="metric-bar-container">
                    <span className="bar-label">Now</span>
                    <div className="metric-bar new">
                      <div 
                        className="metric-fill new" 
                        style={{ width: `${progressData.latestMetrics.hydration ?? 0}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.latestMetrics.hydration != null ? `${progressData.latestMetrics.hydration}%` : '—'}</span>
                  </div>
                </div>
              </div>

              {/* Pigmentation */}
              <div className="metric-comparison">
                <div className="metric-header">
                  <span className="metric-name">Pigmentation</span>
                  <span className="metric-change" style={{ color: getImprovementColor(progressData.improvements.pigmentation) }}>
                    {getImprovementIcon(progressData.improvements.pigmentation)}
                    {progressData.improvements.pigmentation != null && progressData.improvements.pigmentation > 0 ? '+' : ''}
                    {progressData.improvements.pigmentation != null ? progressData.improvements.pigmentation.toFixed(1) : '—'}%
                  </span>
                </div>
                <div className="metric-bars">
                  <div className="metric-bar-container">
                    <span className="bar-label">Before</span>
                    <div className="metric-bar old">
                      <div 
                        className="metric-fill old" 
                        style={{ width: `${progressData.oldestMetrics.pigmentation ?? 0}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.oldestMetrics.pigmentation != null ? `${progressData.oldestMetrics.pigmentation}%` : '—'}</span>
                  </div>
                  <div className="metric-bar-container">
                    <span className="bar-label">Now</span>
                    <div className="metric-bar new">
                      <div 
                        className="metric-fill new" 
                        style={{ width: `${progressData.latestMetrics.pigmentation ?? 0}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.latestMetrics.pigmentation != null ? `${progressData.latestMetrics.pigmentation}%` : '—'}</span>
                  </div>
                </div>
              </div>

              {/* Wrinkles */}
              <div className="metric-comparison">
                <div className="metric-header">
                  <span className="metric-name">Wrinkles</span>
                  <span className="metric-change" style={{ color: getImprovementColor(progressData.improvements.wrinkles) }}>
                    {getImprovementIcon(progressData.improvements.wrinkles)}
                    {progressData.improvements.wrinkles != null && progressData.improvements.wrinkles > 0 ? '+' : ''}
                    {progressData.improvements.wrinkles != null ? progressData.improvements.wrinkles.toFixed(1) : '—'}%
                  </span>
                </div>
                <div className="metric-bars">
                  <div className="metric-bar-container">
                    <span className="bar-label">Before</span>
                    <div className="metric-bar old">
                      <div 
                        className="metric-fill old" 
                        style={{ width: `${progressData.oldestMetrics.wrinkles ?? 0}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.oldestMetrics.wrinkles != null ? `${progressData.oldestMetrics.wrinkles}%` : '—'}</span>
                  </div>
                  <div className="metric-bar-container">
                    <span className="bar-label">Now</span>
                    <div className="metric-bar new">
                      <div 
                        className="metric-fill new" 
                        style={{ width: `${progressData.latestMetrics.wrinkles ?? 0}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.latestMetrics.wrinkles != null ? `${progressData.latestMetrics.wrinkles}%` : '—'}</span>
                  </div>
                </div>
              </div>

              {/* Pores */}
              <div className="metric-comparison">
                <div className="metric-header">
                  <span className="metric-name">Pores</span>
                  <span className="metric-change" style={{ color: getImprovementColor(progressData.improvements.pores) }}>
                    {getImprovementIcon(progressData.improvements.pores)}
                    {progressData.improvements.pores != null && progressData.improvements.pores > 0 ? '+' : ''}
                    {progressData.improvements.pores != null ? progressData.improvements.pores.toFixed(1) : '—'}%
                  </span>
                </div>
                <div className="metric-bars">
                  <div className="metric-bar-container">
                    <span className="bar-label">Before</span>
                    <div className="metric-bar old">
                      <div 
                        className="metric-fill old" 
                        style={{ width: `${progressData.oldestMetrics.pores ?? 0}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.oldestMetrics.pores != null ? `${progressData.oldestMetrics.pores}%` : '—'}</span>
                  </div>
                  <div className="metric-bar-container">
                    <span className="bar-label">Now</span>
                    <div className="metric-bar new">
                      <div 
                        className="metric-fill new" 
                        style={{ width: `${progressData.latestMetrics.pores ?? 0}%` }}
                      ></div>
                    </div>
                    <span className="bar-value">{progressData.latestMetrics.pores != null ? `${progressData.latestMetrics.pores}%` : '—'}</span>
                  </div>
                </div>
              </div>

            </div>
          </div>

          <div className="analysis-timeline">
            <h3>Analysis History</h3>
            <div className="timeline">
              {filteredHistory.map((analysis, index) => (
                <div key={analysis.id || index} className="timeline-item">
                  <div className="timeline-marker">
                    <div className="marker-dot"></div>
                    {index < filteredHistory.length - 1 && <div className="marker-line"></div>}
                  </div>
                  <div className="timeline-content">
                    <div className="timeline-date">{formatDate(analysis.timestamp)}</div>
                    {analysis.image && (
                      <div className="timeline-photo">
                        <img src={analysis.image} alt={`Analysis ${index + 1}`} />
                      </div>
                    )}
                    <div className="timeline-data">
                      {typeof analysis.analysis?.skinHealth === 'number' && (
                        <div className="timeline-health">
                          Overall Health: <span className="health-score">{analysis.analysis.skinHealth}%</span>
                        </div>
                      )}
                      {analysis.analysis?.skinType && (
                        <div className="timeline-type">
                          Skin Type: <span className="skin-type">{analysis.analysis.skinType}</span>
                        </div>
                      )}
                      {Array.isArray(analysis.analysis?.concerns) && analysis.analysis.concerns.length > 0 && (
                        <div className="timeline-concerns">
                          <strong>Main Concerns:</strong>
                          <ul>
                            {analysis.analysis.concerns.slice(0, 2).map((concern, i) => (
                              <li key={i}>{concern}</li>
                            ))}
                          </ul>
                        </div>
                      )}
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
