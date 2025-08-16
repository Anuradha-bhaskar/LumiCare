import React, { useState } from 'react';
import { useSignIn } from '@clerk/clerk-react';
import { useNavigate, Link } from 'react-router-dom';
import './SignIn.css';

const CustomSignIn = () => {
  const { signIn, isLoaded, setActive } = useSignIn();
  const navigate = useNavigate();
  
  const [formData, setFormData] = useState({
    emailAddress: '',
    password: ''
  });
  
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.emailAddress) {
      newErrors.emailAddress = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.emailAddress)) {
      newErrors.emailAddress = 'Please enter a valid email';
    }
    
    if (!formData.password) {
      newErrors.password = 'Password is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSignIn = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    if (!isLoaded) return;
    
    setIsSubmitting(true);
    
    try {
      const result = await signIn.create({
        identifier: formData.emailAddress,
        password: formData.password,
      });

      if (result.status === 'complete') {
        await setActive({ session: result.createdSessionId });
        navigate('/');
      }
    } catch (err) {
      console.error('Sign in error:', err);
      let errorMessage = 'Sign in failed. Please check your credentials.';
      
      if (err.errors) {
        const clerkError = err.errors[0];
        if (clerkError.code === 'form_identifier_not_found') {
          errorMessage = 'No account found with this email address.';
        } else if (clerkError.code === 'form_password_incorrect') {
          errorMessage = 'Incorrect password. Please try again.';
        } else if (clerkError.message) {
          errorMessage = clerkError.message;
        }
      }
      
      setErrors({ submit: errorMessage });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card signin-card">
        <div className="auth-header">
          <h2>Welcome Back</h2>
          <p>Sign in to your LumiCare account</p>
        </div>
        
        <form onSubmit={handleSignIn} className="auth-form">
          <div className="form-group">
            <label htmlFor="emailAddress">Email Address</label>
            <input
              type="email"
              id="emailAddress"
              name="emailAddress"
              value={formData.emailAddress}
              onChange={handleInputChange}
              className={errors.emailAddress ? 'error' : ''}
              placeholder="Enter your email address"
            />
            {errors.emailAddress && (
              <span className="error-message">{errors.emailAddress}</span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleInputChange}
              className={errors.password ? 'error' : ''}
              placeholder="Enter your password"
            />
            {errors.password && (
              <span className="error-message">{errors.password}</span>
            )}
          </div>

          {errors.submit && (
            <div className="error-message submit-error">{errors.submit}</div>
          )}

          <button 
            type="submit" 
            className="auth-btn"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Signing In...' : 'Sign In'}
          </button>
        </form>

        <div className="auth-footer">
          <p>Don't have an account? <Link to="/sign-up">Sign Up</Link></p>
        </div>
      </div>
    </div>
  );
};

export default CustomSignIn;
