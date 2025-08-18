import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { Camera, Search, BarChart3, Lightbulb, MapPin, Target } from 'lucide-react';
import { useUser } from '@clerk/clerk-react';
import { useNavigate } from 'react-router-dom';
import './SkinAnalysis.css';

const SkinAnalysis = ({ onAnalysisComplete, clerkUserId }) => {
  const webcamRef = useRef(null);
  const { user } = useUser();
  const navigate = useNavigate();
  const [step, setStep] = useState('ready'); // ready, camera, analyzing, results
  const [cameraChecks, setCameraChecks] = useState({
    lighting: false,
    position: false,
    stillness: false,
    positionDetails: null,
    lightingDetails: null // Add this to store lighting details
  });

  const [analysisResult, setAnalysisResult] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [countdown, setCountdown] = useState(0);
  const [highlightedMetric, setHighlightedMetric] = useState(null);
  const [isGeneratingRoutine, setIsGeneratingRoutine] = useState(false);

  // Real AI analysis function using MediaPipe, OpenCV, and Gemini
  const analyzeImage = async (imageData) => {
    try {
      console.log('Starting real skin analysis...');
      const effectiveUserId = clerkUserId || user?.id || null;
      console.log('Sending clerk_user_id:', effectiveUserId);
      
      // Call the backend skin analysis API
      const response = await fetch('http://localhost:8000/api/skin/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image_data: imageData, clerk_user_id: effectiveUserId }),
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const analysisResult = await response.json();
      console.log('Skin analysis completed:', analysisResult);
      
      return {
        skinType: analysisResult?.comprehensive?.skinType,
        concerns: analysisResult?.comprehensive?.concerns,
        skinHealth: analysisResult?.comprehensive?.skinHealth,
        recommendations: analysisResult?.comprehensive?.recommendations,
        metrics: analysisResult?.metrics || {},
        faceRegion: analysisResult?.face_region,
        priorityActions: analysisResult?.comprehensive?.priorityActions
      };
    } catch (error) {
      console.error('Skin analysis failed:', error);
      throw error;
    }
  };

  // Real camera quality checks using MediaPipe and OpenCV
  useEffect(() => {
    if (step === 'camera' && webcamRef.current) {
      const interval = setInterval(async () => {
        try {
          // Capture current frame
          const imageSrc = webcamRef.current.getScreenshot();
          if (!imageSrc) return;

          // Send to backend for analysis
          const response = await fetch('http://localhost:8000/api/camera/analyze', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image_data: imageSrc }),
          });

          if (response.ok) {
            const analysis = await response.json();
            setCameraChecks({
              lighting: analysis.lighting.is_good,
              position: analysis.position.is_good,
              stillness: analysis.stillness.is_good,
              lightingDetails: analysis.lighting, // Store lighting details
              positionDetails: analysis.position
            });
          } else {
            console.error('Camera analysis failed:', response.statusText);
            // Fallback to basic checks if API fails
            setCameraChecks({
              lighting: true,
              position: true,
              stillness: true,
              lightingDetails: { quality: 'good' },
              positionDetails: null
            });
          }
        } catch (error) {
          console.error('Error during camera analysis:', error);
          // Fallback to basic checks if API fails
          setCameraChecks({
            lighting: true,
            position: true,
            stillness: true,
            lightingDetails: { quality: 'good' },
            positionDetails: null
          });
        }
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [step]);

  const startAnalysis = async () => {
    try {
      // Reset the camera analysis state on backend
      await fetch('http://localhost:8000/api/camera/reset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    } catch (error) {
      console.error('Failed to reset camera analysis:', error);
    }
    setStep('camera');
  };

  const capture = useCallback(async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setCapturedImage(imageSrc);
    setStep('analyzing');

    try {
      const result = await analyzeImage(imageSrc);
      setAnalysisResult(result);
      setStep('results');
      onAnalysisComplete({
        image: imageSrc,
        analysis: result
      });
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed. Please try again.');
      setStep('camera');
    }
  }, [webcamRef, onAnalysisComplete]);

  // Auto-capture when all checks pass
  useEffect(() => {
    if (step === 'camera' && cameraChecks.lighting && cameraChecks.position && cameraChecks.stillness) {
      if (countdown === 0) {
        setCountdown(3);
      }
    } else {
      setCountdown(0);
    }
  }, [step, cameraChecks, countdown]);

  useEffect(() => {
    if (countdown > 0) {
      const timer = setTimeout(() => {
        if (countdown === 1) {
          capture();
        } else {
          setCountdown(countdown - 1);
        }
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [countdown, capture]);

  const restartAnalysis = async () => {
    try {
      // Reset the camera analysis state on backend
      await fetch('http://localhost:8000/api/camera/reset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
    } catch (error) {
      console.error('Failed to reset camera analysis:', error);
    }
    setStep('ready');
    setCapturedImage(null);
    setAnalysisResult(null);
    setCountdown(0);
    setHighlightedMetric(null);
  };

  const highlightMetric = (metricType) => {
    setHighlightedMetric(metricType);
    // Here you could add logic to highlight specific areas on the image
    console.log(`Highlighting ${metricType} on image`);
  };

  // Helpers for results view
  const calculatePercent = (score) => {
    if (score == null) return 0;
    // If score already looks like 0-1, convert to percentage
    if (score <= 1) return Math.round(score * 100);
    // If score in 0-100 range, clamp
    return Math.round(Math.min(100, Math.max(0, score)));
  };

  const getSeverityBadgeClass = (severity) => {
    if (!severity) return 'severity-badge warn';
    const s = String(severity).toLowerCase();
    if (['none', 'clear', 'fine', 'even', 'normal', 'adequately hydrated', 'well hydrated'].some(k => s.includes(k))) {
      return 'severity-badge good';
    }
    if (['mild'].some(k => s.includes(k))) {
      return 'severity-badge warn';
    }
    if (['moderate-severe', 'pronounced', 'very oily', 'severely dehydrated', 'severe'].some(k => s.includes(k))) {
      return 'severity-badge bad';
    }
    if (['moderate', 'oily', 'dry', 'enlarged', 'very enlarged', 'dehydrated', 'sensitive'].some(k => s.includes(k))) {
      return 'severity-badge warn';
    }
    return 'severity-badge warn';
  };

  const StatItem = ({ label, value, isPercent = false, digits = 2 }) => (
    value === null || value === undefined || (typeof value === 'number' && Number.isNaN(value)) ? null : (
      <li className="stat-item">
        <span className="stat-label">{label}</span>
        <span className="stat-value">{isPercent ? `${Math.round(value * 100)}%` : (typeof value === 'number' ? value.toFixed(digits) : value)}</span>
      </li>
    )
  );

  const generateRoutineAndDietThenNavigate = async () => {
    setIsGeneratingRoutine(true);
    try {
      const effectiveUserId = clerkUserId || user?.id || null;
      if (effectiveUserId) {
        localStorage.setItem('last_clerk_user_id', effectiveUserId);
        // Fire and forget generate calls; backend uses latest saved analysis
        await Promise.allSettled([
          fetch('http://localhost:8000/api/skin/routine', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ clerk_user_id: effectiveUserId })
          }),
          fetch('http://localhost:8000/api/skin/diet', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ clerk_user_id: effectiveUserId })
          })
        ]);
      }
    } catch (e) {
      console.error('Failed to pre-generate routine/diet:', e);
    } finally {
      setIsGeneratingRoutine(false);
      navigate('/routine');
    }
  };

  return (
    <div className="skin-analysis">
      {step === 'ready' && (
        <div className="analysis-start">
          <div className="start-card">
            <div className="start-header">
              <h2><Camera size={24} className="inline-icon" /> Skin Analysis</h2>
              <p>Get personalized insights about your skin health</p>
            </div>
            
            <div className="analysis-benefits">
              <div className="benefit-item">
                <span className="benefit-icon"><Search size={20} /></span>
                <span>Detailed skin assessment</span>
              </div>
              <div className="benefit-item">
                <span className="benefit-icon"><BarChart3 size={20} /></span>
                <span>Progress tracking over time</span>
              </div>
              <div className="benefit-item">
                <span className="benefit-icon"><Lightbulb size={20} /></span>
                <span>Personalized recommendations</span>
              </div>
            </div>
            
            <div className="tips">
              <h3>Tips for best results:</h3>
              <ul>
                <li>Ensure good, natural lighting</li>
                <li>Position your face close to the camera (60-85% of frame)</li>
                <li>Look directly at the camera</li>
                <li>Remove makeup if possible</li>
                <li>Stay still during capture</li>
              </ul>
            </div>
            
            <button className="start-btn" onClick={startAnalysis}>
              Start Analysis
            </button>
          </div>
        </div>
      )}

      {step === 'camera' && (
        <div className="camera-view">
          <div className="camera-container">
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              className="webcam"
              videoConstraints={{
                width: 640,
                height: 480,
                facingMode: "user"
              }}
            />
            
            {countdown > 0 && (
              <div className="countdown-overlay">
                <div className="countdown-circle">
                  <span className="countdown-number">{countdown}</span>
                </div>
              </div>
            )}
            
            <div className="camera-checks">
              <div className={`check-item ${cameraChecks.lighting ? 'good' : 'bad'}`}>
                <span className="check-icon"><Lightbulb size={16} /></span>
                <span>Lighting: {cameraChecks.lighting ? 'Good' : 'Adjust lighting'}</span>
              </div>
              <div className={`check-item ${cameraChecks.position ? 'good' : 'bad'}`}>
                <span className="check-icon"><MapPin size={16} /></span>
                <span>Position: {cameraChecks.positionDetails?.distance_feedback || cameraChecks.positionDetails?.message || (cameraChecks.position ? 'Perfect' : 'Center your face')}</span>
              </div>
              <div className={`check-item ${cameraChecks.stillness ? 'good' : 'bad'}`}>
                <span className="check-icon"><Target size={16} /></span>
                <span>Stillness: {cameraChecks.stillness ? 'Good' : 'Stay still'}</span>
              </div>
            </div>
          </div>
          
          <div className="camera-controls">
            <button className="cancel-btn" onClick={restartAnalysis}>
              Cancel
            </button>
            <button 
              className="capture-btn" 
              onClick={capture}
              disabled={countdown > 0}
            >
              {countdown > 0 ? `Capturing in ${countdown}...` : 'Capture Now'}
            </button>
          </div>
        </div>
      )}

      {step === 'analyzing' && (
        <div className="analyzing-view">
          <div className="analyzing-card">
            <div className="analyzing-animation">
              <div className="spinner"></div>
            </div>
            <h2>Analyzing Your Skin...</h2>
            <p>Our AI is examining your photo to provide detailed insights</p>
            <div className="progress-steps">
              <div className="progress-step active">Detecting face</div>
              <div className="progress-step active">Analyzing skin texture</div>
              <div className="progress-step">Generating recommendations</div>
            </div>
          </div>
        </div>
      )}

      {step === 'results' && analysisResult && (
        <div className="results-view">
          <div className="results-header">
            <h2><BarChart3 size={24} className="inline-icon" /> Your Skin Analysis Results</h2>
            <div className="results-date">
              {new Date().toLocaleDateString('en-US', { 
                weekday: 'long', 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
              })}
            </div>
          </div>

          <div className="results-content">
            <div className="results-main">
              <div className="captured-photo-grid">
                <div className="captured-photo">
                  <img src={capturedImage} alt="Analyzed Face" />
                </div>
              </div>
              
              <div className="skin-metrics">
                <h3>Skin Analysis Metrics</h3>
                <div className="metrics-grid">
                  {/* Acne */}
                  {analysisResult.metrics?.acne && (
                  <div className={`metric-item interactive ${highlightedMetric === 'acne' ? 'active' : ''}`} onClick={() => highlightMetric('acne')}>
                    <div className="metric-header">
                      <span className="metric-label">Acne</span>
                      {analysisResult.metrics.acne?.severity && (
                        <span className={getSeverityBadgeClass(analysisResult.metrics.acne?.severity)}>{analysisResult.metrics.acne?.severity}</span>
                      )}
                    </div>
                    {typeof analysisResult.metrics.acne?.score === 'number' && (
                      <div className="metric-bar">
                        <div
                          className="metric-fill"
                          style={{ width: `${calculatePercent(analysisResult.metrics.acne?.score)}%` }}
                        ></div>
                      </div>
                    )}
                    <ul className="stat-list">
                      <StatItem label="Spots" value={analysisResult.metrics.acne?.count} digits={0} />
                      <StatItem label="Avg Spot Size" value={analysisResult.metrics.acne?.avg_size} />
                    </ul>
                  </div>
                  )}
                  
                  {/* Oiliness */}
                  {analysisResult.metrics?.oiliness && (
                  <div className={`metric-item interactive ${highlightedMetric === 'oiliness' ? 'active' : ''}`} onClick={() => highlightMetric('oiliness')}>
                    <div className="metric-header">
                      <span className="metric-label">Oiliness</span>
                      {analysisResult.metrics.oiliness?.severity && (
                        <span className={getSeverityBadgeClass(analysisResult.metrics.oiliness?.severity)}>{analysisResult.metrics.oiliness?.severity}</span>
                      )}
                    </div>
                    {typeof analysisResult.metrics.oiliness?.score === 'number' && (
                      <div className="metric-bar">
                        <div className="metric-fill" style={{ width: `${calculatePercent(analysisResult.metrics.oiliness?.score)}%` }}></div>
                      </div>
                    )}
                    <ul className="stat-list">
                      <StatItem label="T-zone Brightness" value={analysisResult.metrics.oiliness?.t_zone_brightness} />
                      <StatItem label="Texture Variance" value={analysisResult.metrics.oiliness?.texture_variance} />
                    </ul>
                  </div>
                  )}

                  {/* Pigmentation */}
                  {analysisResult.metrics?.pigmentation && (
                  <div className={`metric-item interactive ${highlightedMetric === 'pigmentation' ? 'active' : ''}`} onClick={() => highlightMetric('pigmentation')}>
                    <div className="metric-header">
                      <span className="metric-label">Pigmentation</span>
                      {analysisResult.metrics.pigmentation?.severity && (
                        <span className={getSeverityBadgeClass(analysisResult.metrics.pigmentation?.severity)}>{analysisResult.metrics.pigmentation?.severity}</span>
                      )}
                    </div>
                    {typeof analysisResult.metrics.pigmentation?.score === 'number' && (
                      <div className="metric-bar">
                        <div className="metric-fill" style={{ width: `${calculatePercent(analysisResult.metrics.pigmentation?.score)}%` }}></div>
                      </div>
                    )}
                    <ul className="stat-list">
                      <StatItem label="Color Variation" value={analysisResult.metrics.pigmentation?.color_variation} />
                      <StatItem label="Dark Spots Area" value={analysisResult.metrics.pigmentation?.dark_spots_percentage} isPercent />
                    </ul>
                  </div>
                  )}

                  {/* Wrinkles */}
                  {analysisResult.metrics?.wrinkles && (
                  <div className={`metric-item interactive ${highlightedMetric === 'wrinkles' ? 'active' : ''}`} onClick={() => highlightMetric('wrinkles')}>
                    <div className="metric-header">
                      <span className="metric-label">Wrinkles</span>
                      {analysisResult.metrics.wrinkles?.severity && (
                        <span className={getSeverityBadgeClass(analysisResult.metrics.wrinkles?.severity)}>{analysisResult.metrics.wrinkles?.severity}</span>
                      )}
                    </div>
                    {typeof analysisResult.metrics.wrinkles?.score === 'number' && (
                      <div className="metric-bar">
                        <div className="metric-fill" style={{ width: `${calculatePercent(analysisResult.metrics.wrinkles?.score)}%` }}></div>
                      </div>
                    )}
                    <ul className="stat-list">
                      <StatItem label="Density" value={analysisResult.metrics.wrinkles?.density} isPercent />
                      <StatItem label="Line Count" value={analysisResult.metrics.wrinkles?.line_count} digits={0} />
                    </ul>
                  </div>
                  )}

                  {/* Pores */}
                  {analysisResult.metrics?.pores && (
                  <div className={`metric-item interactive ${highlightedMetric === 'pores' ? 'active' : ''}`} onClick={() => highlightMetric('pores')}>
                    <div className="metric-header">
                      <span className="metric-label">Pores</span>
                      {analysisResult.metrics.pores?.severity && (
                        <span className={getSeverityBadgeClass(analysisResult.metrics.pores?.severity)}>{analysisResult.metrics.pores?.severity}</span>
                      )}
                    </div>
                    {typeof analysisResult.metrics.pores?.score === 'number' && (
                      <div className="metric-bar">
                        <div className="metric-fill" style={{ width: `${calculatePercent(analysisResult.metrics.pores?.score)}%` }}></div>
                      </div>
                    )}
                    <ul className="stat-list">
                      <StatItem label="Count" value={analysisResult.metrics.pores?.count} digits={0} />
                      <StatItem label="Avg Size" value={analysisResult.metrics.pores?.avg_size} />
                      <StatItem label="Density (per 100x100)" value={analysisResult.metrics.pores?.density} />
                    </ul>
                  </div>
                  )}

                  {/* Hydration */}
                  {analysisResult.metrics?.hydration && (
                  <div className={`metric-item interactive ${highlightedMetric === 'hydration' ? 'active' : ''}`} onClick={() => highlightMetric('hydration')}>
                    <div className="metric-header">
                      <span className="metric-label">Hydration</span>
                      {analysisResult.metrics.hydration?.severity && (
                        <span className={getSeverityBadgeClass(analysisResult.metrics.hydration?.severity)}>{analysisResult.metrics.hydration?.severity}</span>
                      )}
                    </div>
                    {typeof analysisResult.metrics.hydration?.score === 'number' && (
                      <div className="metric-bar">
                        <div className="metric-fill" style={{ width: `${calculatePercent(analysisResult.metrics.hydration?.score)}%` }}></div>
                      </div>
                    )}
                    <ul className="stat-list">
                      <StatItem label="Smoothness" value={analysisResult.metrics.hydration?.smoothness} isPercent />
                      <StatItem label="Texture Uniformity" value={analysisResult.metrics.hydration?.texture_uniformity} isPercent />
                    </ul>
                  </div>
                  )}

                  {/* Dark Circles */}
                  {analysisResult.metrics?.darkCircles && (
                  <div className={`metric-item interactive ${highlightedMetric === 'darkCircles' ? 'active' : ''}`} onClick={() => highlightMetric('darkCircles')}>
                    <div className="metric-header">
                      <span className="metric-label">Dark Circles</span>
                      {analysisResult.metrics.darkCircles?.severity && (
                        <span className={getSeverityBadgeClass(analysisResult.metrics.darkCircles?.severity)}>{analysisResult.metrics.darkCircles?.severity}</span>
                      )}
                    </div>
                    {typeof analysisResult.metrics.darkCircles?.score === 'number' && (
                      <div className="metric-bar">
                        <div className="metric-fill" style={{ width: `${calculatePercent(analysisResult.metrics.darkCircles?.score)}%` }}></div>
                      </div>
                    )}
                    <ul className="stat-list">
                      <StatItem label="Darkness" value={analysisResult.metrics.darkCircles?.darkness_score} isPercent />
                      <StatItem label="Color Score" value={analysisResult.metrics.darkCircles?.color_score} isPercent />
                      <StatItem label="Left Eye" value={analysisResult.metrics.darkCircles?.left_eye} isPercent />
                      <StatItem label="Right Eye" value={analysisResult.metrics.darkCircles?.right_eye} isPercent />
                    </ul>
                  </div>
                  )}

                  {/* Redness */}
                  {analysisResult.metrics?.redness && (
                  <div className={`metric-item interactive ${highlightedMetric === 'redness' ? 'active' : ''}`} onClick={() => highlightMetric('redness')}>
                    <div className="metric-header">
                      <span className="metric-label">Redness</span>
                      {analysisResult.metrics.redness?.severity && (
                        <span className={getSeverityBadgeClass(analysisResult.metrics.redness?.severity)}>{analysisResult.metrics.redness?.severity}</span>
                      )}
                    </div>
                    {typeof analysisResult.metrics.redness?.score === 'number' && (
                      <div className="metric-bar">
                        <div className="metric-fill" style={{ width: `${calculatePercent(analysisResult.metrics.redness?.score)}%` }}></div>
                      </div>
                    )}
                    <ul className="stat-list">
                      <StatItem label="Area" value={analysisResult.metrics.redness?.percentage} isPercent />
                      <StatItem label="Red Intensity" value={analysisResult.metrics.redness?.red_intensity} />
                      <StatItem label="Red Dominance" value={analysisResult.metrics.redness?.red_dominance} />
                    </ul>
                  </div>
                  )}
                </div>
              </div>
            </div>

            <div className="results-sidebar">
              <div className="skin-type-card">
                <h3>Skin Type</h3>
                {analysisResult.skinType && (
                  <div className="skin-type">{analysisResult.skinType}</div>
                )}
                {typeof analysisResult.skinHealth === 'number' && (
                  <div className="skin-health">
                    <span>Overall Health: </span>
                    <span className="health-score">{analysisResult.skinHealth}%</span>
                  </div>
                )}
              </div>

              <div className="concerns-card">
                <h3>Primary Concerns</h3>
                {Array.isArray(analysisResult.concerns) && analysisResult.concerns.length > 0 && (
                  <ul className="concerns-list">
                    {analysisResult.concerns.map((concern, index) => (
                      <li key={index}>{concern}</li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="priority-actions-card">
                <h3>Priority Actions</h3>
                {Array.isArray(analysisResult.priorityActions) && analysisResult.priorityActions.length > 0 && (
                  <ul className="priority-list">
                    {analysisResult.priorityActions.map((action, index) => (
                      <li key={index}>{action}</li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="recommendations-card">
                <h3>Recommendations</h3>
                {Array.isArray(analysisResult.recommendations) && analysisResult.recommendations.length > 0 && (
                  <ul className="recommendations-list">
                    {analysisResult.recommendations.map((rec, index) => (
                      <li key={index}>{rec}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>

          <div className="results-actions">
            <button className="secondary-btn" onClick={restartAnalysis}>
              New Analysis
            </button>
            <button 
              className="primary-btn"
              onClick={generateRoutineAndDietThenNavigate}
              disabled={isGeneratingRoutine}
            >
              {isGeneratingRoutine ? 'Loading...' : 'Get My Routine'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SkinAnalysis;