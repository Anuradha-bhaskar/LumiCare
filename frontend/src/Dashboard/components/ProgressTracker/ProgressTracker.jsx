import React, { useState, useMemo, useEffect } from 'react';
import { TrendingUp, TrendingDown, ArrowRight, BarChart3, Calendar, Target } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, AreaChart, Area, Brush } from 'recharts';
import './ProgressTracker.css';

const ProgressTracker = ({ analysisHistory, clerkUserId }) => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('all'); // all, week, month, 3months
  const [history, setHistory] = useState(analysisHistory || []);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('progress'); // progress | history

  useEffect(() => {
    setHistory(analysisHistory || []);
  }, [analysisHistory]);

  useEffect(() => {
    const fetchIfNeeded = async () => {
      if ((!history || history.length === 0) && clerkUserId) {
        try {
          setLoading(true);
          setError(null);
          const res = await fetch(`http://localhost:8000/api/skin/history/${clerkUserId}`);
          if (!res.ok) throw new Error('Failed to fetch history');
          const data = await res.json();
          setHistory(data || []);
        } catch (e) {
          setError('Unable to load progress history');
        } finally {
          setLoading(false);
        }
      }
    };
    fetchIfNeeded();
  }, [clerkUserId]);

  const filteredHistory = useMemo(() => {
    const list = history || [];
    if (list.length === 0) return [];
    
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
        return list;
    }
    
    return list.filter(analysis => 
      new Date(analysis.timestamp) >= cutoffDate
    );
  }, [history, selectedTimeframe]);

  // Helper conversions
  const toPercent = (value) => {
    if (typeof value !== 'number') return null;
    if (value <= 1) return Math.round(value * 100);
    return Math.round(Math.max(0, Math.min(100, value)));
  };
  const getMetricPercent = (analysis, key) => {
    const score = analysis?.analysis?.metrics?.[key]?.score;
    return toPercent(score);
  };

  // Raw set of metric keys from data
  const allMetricKeys = useMemo(() => {
    const keys = new Set();
    filteredHistory.forEach(a => {
      const m = a?.analysis?.metrics || {};
      Object.keys(m).forEach(k => keys.add(k));
    });
    return Array.from(keys);
  }, [filteredHistory]);

  // Only keep metrics that have numeric score at least once
  const numericMetricKeys = useMemo(() => {
    return allMetricKeys.filter((k) => {
      return filteredHistory.some((a) => typeof a?.analysis?.metrics?.[k]?.score === 'number');
    });
  }, [filteredHistory, allMetricKeys]);

  // Metrics that have at least one numeric point -> for trend chart display
  const trendMetricKeys = useMemo(() => {
    return numericMetricKeys.filter((k) => {
      // Include any metric that has at least one numeric value
      return filteredHistory.some(a => typeof a?.analysis?.metrics?.[k]?.score === 'number');
    });
  }, [filteredHistory, numericMetricKeys]);

  // Direction map for clarity (true = higher is better)
  const higherIsBetter = {
    hydration: true,
    skinHealth: true,
    // Most other metrics represent issues (higher => worse)
    pigmentation: false,
    wrinkles: false,
    pores: false,
    oiliness: false,
    acne: false,
    darkCircles: false,
    redness: false,
  };
  const isHigherBetter = (key) => higherIsBetter[key] === true;

  // Build chronological series - ensure proper sorting by timestamp
  const chronological = useMemo(() => {
    return filteredHistory.slice().sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  }, [filteredHistory]);

  const progressData = useMemo(() => {
    if (chronological.length < 1) return null;

    // Overall skin health earliest/latest - ensure proper percentage conversion
    const overallBefore = (() => {
      for (const a of chronological) {
        const health = a?.analysis?.skinHealth;
        if (typeof health === 'number') {
          // Convert to percentage if needed (0-1 range to 0-100)
          return health <= 1 ? Math.round(health * 100) : Math.round(health);
        }
      }
      return null;
    })();
    const overallNow = (() => {
      for (let i = chronological.length - 1; i >= 0; i--) {
        const a = chronological[i];
        const health = a?.analysis?.skinHealth;
        if (typeof health === 'number') {
          // Convert to percentage if needed (0-1 range to 0-100)
          return health <= 1 ? Math.round(health * 100) : Math.round(health);
        }
      }
      return null;
    })();

    // Per-metric earliest/latest non-null
    const beforeMetrics = {};
    const nowMetrics = {};
    trendMetricKeys.forEach((key) => {
      // earliest non-null
      let b = null;
      for (const a of chronological) {
        const v = getMetricPercent(a, key);
        if (v != null) { b = v; break; }
      }
      // latest non-null
      let n = null;
      for (let i = chronological.length - 1; i >= 0; i--) {
        const v = getMetricPercent(chronological[i], key);
        if (v != null) { n = v; break; }
      }
      beforeMetrics[key] = b;
      nowMetrics[key] = n;
    });

    const improvements = trendMetricKeys.reduce((acc, key) => {
      const b = beforeMetrics[key];
      const n = nowMetrics[key];
      acc[key] = b != null && n != null ? (n - b) : null;
      return acc;
    }, {});

    const overallChange = overallBefore != null && overallNow != null ? (overallNow - overallBefore) : null;

    return {
      beforeMetrics,
      nowMetrics,
      improvements: { ...improvements, overall: overallChange },
      totalAnalyses: filteredHistory.length,
      timeSpan: Math.ceil((new Date(filteredHistory[0].timestamp) - new Date(filteredHistory[filteredHistory.length - 1].timestamp)) / (1000 * 60 * 60 * 24))
    };
  }, [chronological, filteredHistory, trendMetricKeys]);

  // Chart series: chronological order and dynamic metric keys
  const chartSeries = useMemo(() => {
    console.log('Building chart series from chronological data:', chronological);
    console.log('Available trend metric keys:', trendMetricKeys);
    
    const allSeries = chronological.map((a) => {
      const row = {
        date: new Date(a.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        fullDate: a.timestamp,
        skinHealth: typeof a?.analysis?.skinHealth === 'number' ? a.analysis.skinHealth : null,
      };
      trendMetricKeys.forEach((k) => {
        row[k] = getMetricPercent(a, k);
      });
      console.log('Chart row for', row.date, ':', row);
      return row;
    });

    console.log('Final chart series:', allSeries);
    return allSeries;
  }, [chronological, trendMetricKeys]);

  const colorPalette = ['#4caf50', '#ff9800', '#9c27b0', '#03a9f4', '#f44336', '#795548', '#607d8b', '#8bc34a', '#ff5722', '#3f51b5'];
  const getColorForMetric = (key) => {
    const idx = trendMetricKeys.indexOf(key);
    return colorPalette[idx % colorPalette.length];
  };
  const prettyMetricName = (key) => key.replace(/([A-Z])/g, ' $1').replace(/^./, (s) => s.toUpperCase());

  const getImprovementColor = (key, delta) => {
    if (delta == null) return 'var(--text-secondary)';
    const better = isHigherBetter(key) ? delta > 0 : delta < 0;
    const worse = isHigherBetter(key) ? delta < 0 : delta > 0;
    if (better && Math.abs(delta) > 2) return 'var(--success)';
    if (worse && Math.abs(delta) > 2) return 'var(--error)';
    return 'var(--warning)';
  };

  const getImprovementIcon = (key, delta) => {
    if (delta == null) return <ArrowRight size={16} />;
    const better = isHigherBetter(key) ? delta > 0 : delta < 0;
    const worse = isHigherBetter(key) ? delta < 0 : delta > 0;
    if (better) return <TrendingUp size={16} />;
    if (worse) return <TrendingDown size={16} />;
    return <ArrowRight size={16} />;
  };

  const improvementLabel = (key, delta) => {
    if (delta == null) return 'No data';
    const abs = Math.abs(delta).toFixed(1);
    if (abs === '0.0') return 'No change';
    const better = isHigherBetter(key) ? delta > 0 : delta < 0;
    return better ? `Improved by ${abs}%` : `Worsened by ${abs}%`;
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      year: 'numeric'
    });
  };

  const percentTick = (v) => `${v}%`;
  const chartTooltip = (value, name) => {
    if (typeof value === 'number') return [`${value}%`, prettyMetricName(name)];
    return ['—', prettyMetricName(name)];
  };

  if (loading) {
    return (
      <div className="progress-tracker">
        <div className="empty-state">
          <div className="empty-icon"><BarChart3 size={48} /></div>
          <h2>Loading your progress...</h2>
          <p>Please wait while we fetch your analysis history.</p>
        </div>
      </div>
    );
  }

  if ((!history || history.length === 0) && !loading) {
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
        {error && <p style={{ color: 'var(--error)', marginTop: '0.5rem' }}>{error}</p>}
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

      <div className="section-toggle">
        <button 
          className={`toggle-btn ${activeTab === 'progress' ? 'active' : ''}`}
          onClick={() => setActiveTab('progress')}
        >
          Progress
        </button>
        <button 
          className={`toggle-btn ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          Analysis History
        </button>
      </div>

      {activeTab === 'progress' && (
        <>
          {/* Overview & Overall Trend */}
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
                        style={{ color: getImprovementColor('skinHealth', progressData.improvements.overall) }}
                      >
                        {progressData.improvements.overall != null && progressData.improvements.overall > 0 ? '+' : ''}
                        {progressData.improvements.overall != null ? progressData.improvements.overall.toFixed(1) : '—'}%
                      </span>
                      <span className="stat-label">Overall Change</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Overall Skin Health Trend */}
              <div className="metrics-comparison" style={{ marginBottom: '2rem' }}>
                <h3>Overall Skin Health Trend</h3>
                <div className="chart-container chart-health">
                  <ResponsiveContainer>
                    <AreaChart data={chartSeries} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="colorHealth" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#ff6b35" stopOpacity={0.4} />
                          <stop offset="100%" stopColor="#ff6b35" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                      <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} tickFormatter={percentTick} />
                      <Tooltip formatter={(v) => (typeof v === 'number' ? [`${v}%`, 'Skin Health'] : ['—', 'Skin Health'])} />
                    <Area type="monotone" dataKey="skinHealth" stroke="#ff6b35" fill="url(#colorHealth)" name="Skin Health" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}

          {/* Metric Trends (All metrics with >= 2 points) */}
          {trendMetricKeys.length > 0 && (
            <div className="metrics-comparison">
              <h3>Metric Trends</h3>
              <div className="chart-container chart-metrics">
                <ResponsiveContainer>
                  <LineChart data={chartSeries} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f3f3" />
                    <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} tickFormatter={percentTick} />
                    <Tooltip formatter={chartTooltip} />
                    <Legend />
                    {trendMetricKeys.map((key) => (
                      <Line key={key} type="monotone" dataKey={key} stroke={getColorForMetric(key)} name={prettyMetricName(key)} strokeWidth={2} dot={{ r: 2 }} activeDot={{ r: 4 }} />
                    ))}
                    <Brush dataKey="date" height={20} travellerWidth={8} stroke="#ccc" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Before vs Now Comparisons for all metrics with >= 2 points */}
          {progressData && trendMetricKeys.length > 0 && (
            <div className="metrics-comparison">
              <h3>Metric Improvements</h3>
              <div className="metrics-grid">
                {trendMetricKeys.map((key) => {
                  const before = progressData.beforeMetrics[key];
                  const now = progressData.nowMetrics[key];
                  const delta = progressData.improvements[key];
                  return (
                    <div className="metric-comparison" key={key}>
                      <div className="metric-header">
                        <span className="metric-name">{prettyMetricName(key)}</span>
                        <span className="metric-change" style={{ color: getImprovementColor(key, delta) }}>
                          {getImprovementIcon(key, delta)} {improvementLabel(key, delta)}
                        </span>
                      </div>
                      <div className="metric-bars">
                        <div className="metric-bar-container">
                          <span className="bar-label">Before</span>
                          <div className="metric-bar old">
                            <div 
                              className="metric-fill old" 
                              style={{ width: `${before ?? 0}%`, background: `linear-gradient(90deg, ${getColorForMetric(key)}33, ${getColorForMetric(key)}66)` }}
                            ></div>
                          </div>
                          <span className="bar-value">{before != null ? `${before}%` : '—'}</span>
                        </div>
                        <div className="metric-bar-container">
                          <span className="bar-label">Now</span>
                          <div className="metric-bar new">
                            <div 
                              className="metric-fill new" 
                              style={{ width: `${now ?? 0}%`, background: `linear-gradient(90deg, ${getColorForMetric(key)}, #ff6b35)` }}
                            ></div>
                          </div>
                          <span className="bar-value">{now != null ? `${now}%` : '—'}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
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
        </>
      )}

      {activeTab === 'history' && (
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
                    {/* Compact metrics snapshot - only metrics with numeric score on this entry */}
                    {analysis.analysis?.metrics && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.5rem' }}>
                        {Object.entries(analysis.analysis.metrics)
                          .filter(([k, v]) => typeof v?.score === 'number')
                          .slice(0, 8)
                          .map(([key, v]) => {
                            const val = toPercent(v.score);
                            return (
                              <span key={key} style={{ background: 'var(--peach-light)', padding: '0.25rem 0.5rem', borderRadius: '6px', fontSize: '0.8rem', color: 'var(--text-primary)' }}>
                                {prettyMetricName(key)}: {val != null ? `${val}%` : '—'}
                              </span>
                            );
                          })}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ProgressTracker;
