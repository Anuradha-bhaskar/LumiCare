import React from "react";
import { useNavigate } from "react-router-dom";
import { useUser } from "@clerk/clerk-react";
import { Sparkles, Camera, Droplets, Target } from "lucide-react";
import "./LandingPage.css";

const LandingPage = () => {
  const navigate = useNavigate();
  const { isSignedIn } = useUser();

  return (
    <div className="landing-page">
      {/* ---------- HEADER ---------- */}
      <header className="landing-header">
        <h1 className="logo">LumiCare</h1>
        <div className="header-actions">
          {!isSignedIn ? (
            <>
              <button className="btn-outline" onClick={() => navigate("/sign-in")}>
                Sign In
              </button>
              <button className="btn-primary" onClick={() => navigate("/sign-up")}>
                Get Started
              </button>
            </>
          ) : (
            <button className="btn-primary" onClick={() => navigate("/skin-analysis")}>
              Go to Dashboard
            </button>
          )}
        </div>
      </header>

      {/* ---------- HERO SECTION ---------- */}
      <main className="landing-main">
        <section className="hero-section">
          <div className="hero-text">
            <h1>Your Personal AI Skincare Companion</h1>
            <p>
              LumiCare helps you <strong>analyze your skin</strong>, track progress with{" "}
              <strong>photo journals</strong>, and get a personalized{" "}
              <strong>skincare & diet routine</strong> powered by AI.
            </p>
            {!isSignedIn ? (
              <button className="btn-primary hero-btn" onClick={() => navigate("/sign-up")}>
                Start Your Analysis
              </button>
            ) : (
              <button className="btn-primary hero-btn" onClick={() => navigate("/skin-analysis")}>
                Continue Your Journey
              </button>
            )}
          </div>

          <div className="hero-image">
            <img
              src="https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?auto=format&fit=crop&w=600&q=80"
              alt="Skincare model"
            />
          </div>
        </section>
      </main>

      {/* ---------- FEATURES ---------- */}
      <section className="features-section">
        <div className="features-container">
          <h2>Why Choose LumiCare?</h2>
          <p className="features-subtitle">Experience the future of personalized skincare with AI-powered insights</p>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">
                <Sparkles size={32} />
              </div>
              <h3>AI Skin Analysis</h3>
              <p>Advanced webcam-powered skin scanning with intelligent lighting and posture detection for clinical-grade accuracy.</p>
              <div className="feature-highlight">Real-time results</div>
            </div>
            <div className="feature-card">
              <div className="feature-icon">
                <Camera size={32} />
              </div>
              <h3>Progress Tracking</h3>
              <p>Visual journey documentation with smart photo comparisons and detailed progress analytics over time.</p>
              <div className="feature-highlight">Photo timeline</div>
            </div>
            <div className="feature-card">
              <div className="feature-icon">
                <Droplets size={32} />
              </div>
              <h3>Personalized Routines</h3>
              <p>Custom-tailored skincare and dietary recommendations based on your unique skin profile and lifestyle.</p>
              <div className="feature-highlight">Custom plans</div>
            </div>
            <div className="feature-card">
              <div className="feature-icon">
                <Target size={32} />
              </div>
              <h3>Smart Recommendations</h3>
              <p>AI-curated product suggestions and ingredient analysis to optimize your skincare routine effectiveness.</p>
              <div className="feature-highlight">Expert insights</div>
            </div>
          </div>
        </div>
      </section>

      {/* ---------- HOW IT WORKS ---------- */}
      <section className="how-it-works-section">
        <div className="container">
          <h2>How LumiCare Works</h2>
          <div className="steps-container">
            <div className="step">
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>Capture Your Skin</h3>
                <p>Use our advanced AI camera to take detailed photos of your skin in optimal lighting conditions</p>
              </div>
            </div>
            <div className="step-connector"></div>
            <div className="step">
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>AI Analysis</h3>
                <p>Our machine learning algorithms analyze your skin type, concerns, and current condition</p>
              </div>
            </div>
            <div className="step-connector"></div>
            <div className="step">
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>Get Your Plan</h3>
                <p>Receive personalized skincare and dietary recommendations tailored to your unique needs</p>
              </div>
            </div>
            <div className="step-connector"></div>
            <div className="step">
              <div className="step-number">4</div>
              <div className="step-content">
                <h3>Track Progress</h3>
                <p>Monitor your skin's improvement with regular check-ins and progress visualizations</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ---------- CTA SECTION ---------- */}
      <section className="cta-section">
        <div className="cta-container">
          <div className="cta-content">
            <h2>Ready to Transform Your Skin?</h2>
            <p>Join thousands of users who have discovered their perfect skincare routine with LumiCare's AI-powered analysis.</p>
            
            {!isSignedIn ? (
              <button className="btn-primary cta-btn" onClick={() => navigate("/sign-up")}>
                Start Your Free Analysis
              </button>
            ) : (
              <button className="btn-primary cta-btn" onClick={() => navigate("/skin-analysis")}>
                Continue Your Journey
              </button>
            )}
          </div>
          <div className="cta-image">
            <img
              src="https://images.unsplash.com/photo-1556228578-0d85b1a4d571?auto=format&fit=crop&w=600&q=80"
              alt="Skincare products and routine"
            />
          </div>
        </div>
      </section>

      {/* ---------- FOOTER ---------- */}
      <footer className="landing-footer">
        <p>© {new Date().getFullYear()} LumiCare · Your AI Skincare Partner</p>
      </footer>
    </div>
  );
};

export default LandingPage;