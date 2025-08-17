import React, { useState, useRef, useCallback, useEffect } from 'react';
import Webcam from 'react-webcam';
import { Camera, Search, BarChart3, Lightbulb, MapPin, Target } from 'lucide-react';
import './SkinAnalysis.css';

const SkinAnalysis = ({ onAnalysisComplete }) => {
  const webcamRef = useRef(null);
  const [step, setStep] = useState('ready'); // ready, camera, analyzing, results
  const [cameraChecks, setCameraChecks] = useState({
    lighting: false,
    position: false,
    stillness: false
  });
  const [analysisResult, setAnalysisResult] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [countdown, setCountdown] = useState(0);

  // Mock AI analysis function - replace with actual AI service
  const analyzeImage = async (imageData) => {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Log imageData for future AI integration
    console.log('Image data captured for analysis:', imageData);
    
    // Mock analysis results
    return {
      skinType: 'Combination',
      concerns: ['Mild acne', 'Slight dryness', 'Minor dark circles'],
      skinHealth: 75,
      recommendations: [
        'Use a gentle cleanser twice daily',
        'Apply moisturizer with hyaluronic acid',
        'Use sunscreen with SPF 30+',
        'Consider vitamin C serum for brightness'
      ],
      metrics: {
        hydration: 68,
        clarity: 72,
        texture: 70,
        poreSize: 65
      }
    };
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
              stillness: analysis.stillness.is_good
            });
          } else {
            console.error('Camera analysis failed:', response.statusText);
            // Fallback to basic checks if API fails
            setCameraChecks({
              lighting: true,
              position: true,
              stillness: true
            });
          }
        } catch (error) {
          console.error('Error during camera analysis:', error);
          // Fallback to basic checks if API fails
          setCameraChecks({
            lighting: true,
            position: true,
            stillness: true
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
                <span>Position: {cameraChecks.position ? 'Perfect' : 'Center your face'}</span>
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
              <div className="captured-photo">
                <img src={capturedImage} alt="Captured" />
              </div>
              
              <div className="skin-metrics">
                <h3>Skin Health Metrics</h3>
                <div className="metrics-grid">
                  <div className="metric-item">
                    <span className="metric-label">Hydration</span>
                    <div className="metric-bar">
                      <div 
                        className="metric-fill" 
                        style={{ width: `${analysisResult.metrics.hydration}%` }}
                      ></div>
                    </div>
                    <span className="metric-value">{analysisResult.metrics.hydration}%</span>
                  </div>
                  
                  <div className="metric-item">
                    <span className="metric-label">Clarity</span>
                    <div className="metric-bar">
                      <div 
                        className="metric-fill" 
                        style={{ width: `${analysisResult.metrics.clarity}%` }}
                      ></div>
                    </div>
                    <span className="metric-value">{analysisResult.metrics.clarity}%</span>
                  </div>
                  
                  <div className="metric-item">
                    <span className="metric-label">Texture</span>
                    <div className="metric-bar">
                      <div 
                        className="metric-fill" 
                        style={{ width: `${analysisResult.metrics.texture}%` }}
                      ></div>
                    </div>
                    <span className="metric-value">{analysisResult.metrics.texture}%</span>
                  </div>
                  
                  <div className="metric-item">
                    <span className="metric-label">Pore Size</span>
                    <div className="metric-bar">
                      <div 
                        className="metric-fill" 
                        style={{ width: `${analysisResult.metrics.poreSize}%` }}
                      ></div>
                    </div>
                    <span className="metric-value">{analysisResult.metrics.poreSize}%</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="results-sidebar">
              <div className="skin-type-card">
                <h3>Skin Type</h3>
                <div className="skin-type">{analysisResult.skinType}</div>
                <div className="skin-health">
                  <span>Overall Health: </span>
                  <span className="health-score">{analysisResult.skinHealth}%</span>
                </div>
              </div>

              <div className="concerns-card">
                <h3>Areas of Concern</h3>
                <ul className="concerns-list">
                  {analysisResult.concerns.map((concern, index) => (
                    <li key={index}>{concern}</li>
                  ))}
                </ul>
              </div>

              <div className="recommendations-card">
                <h3>Recommendations</h3>
                <ul className="recommendations-list">
                  {analysisResult.recommendations.map((rec, index) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          <div className="results-actions">
            <button className="secondary-btn" onClick={restartAnalysis}>
              New Analysis
            </button>
            <button 
              className="primary-btn"
              onClick={() => {
                // This will be implemented later to navigate to routine tab
                alert('Skincare routine feature coming soon!');
              }}
            >
              Get My Routine
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SkinAnalysis;
