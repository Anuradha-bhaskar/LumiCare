import React, { useState } from 'react';
import { useUser } from '@clerk/clerk-react';
import { useNavigate } from 'react-router-dom';
import './UserInfo.css';

const UserInfoForm = () => {
  const { user } = useUser();
  const navigate = useNavigate();
  
  const [formData, setFormData] = useState({
    fullName: '',
    age: '',
    gender: '',
    sensitiveSkin: ''
  });
  
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.fullName.trim()) {
      newErrors.fullName = 'Full name is required';
    }
    
    if (!formData.age) {
      newErrors.age = 'Age is required';
    } else if (formData.age < 13 || formData.age > 120) {
      newErrors.age = 'Please enter a valid age between 13 and 120';
    }
    
    if (!formData.gender) {
      newErrors.gender = 'Please select your gender';
    }

    if (!formData.sensitiveSkin) {
      newErrors.sensitiveSkin = 'Please specify if you have sensitive skin';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      const userProfileData = {
        fullName: formData.fullName,
        age: parseInt(formData.age),
        gender: formData.gender,
        sensitiveSkin: formData.sensitiveSkin === 'yes',
        profileCompleted: true,
        userId: user.id
      };
      
      localStorage.setItem('userProfile', JSON.stringify(userProfileData));
      await user.update({
        unsafeMetadata: {
          ...user.unsafeMetadata,
          ...userProfileData
        }
      });
      await user.reload();
      
      try {
        const syncPayload = {
          clerk_user_id: user.id,
          email: user.emailAddresses[0]?.emailAddress || user.email || 'test@example.com'
        };
        await fetch('http://localhost:8000/users/sync', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(syncPayload)
        });
        const profilePayload = {
          full_name: formData.fullName,
          age: parseInt(formData.age),
          gender: formData.gender,
          sensitive_skin: formData.sensitiveSkin === 'yes'
        };
        await fetch(`http://localhost:8000/users/profile/${user.id}`, {
          method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(profilePayload)
        });
      } catch {}
      
      navigate('/');
      
    } catch (error) {
      let errorMessage = 'Failed to save information. Please try again.';
      if (error.message && error.message.includes('network')) {
        errorMessage = 'Network error. Please check your connection and try again.';
      } else if (error.message && error.message.includes('permission')) {
        errorMessage = 'Permission error. Please try signing out and back in.';
      }
      setErrors({ submit: errorMessage });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="user-info-container">
      <div className="user-info-card">
        <div className="user-info-header">
          <h2>Complete Your Profile</h2>
          <p>Help us personalize your LumiCare experience</p>
        </div>
        
        <form onSubmit={handleSubmit} className="user-info-form">
          <div className="form-group">
            <label htmlFor="fullName">Full Name</label>
            <input
              type="text"
              id="fullName"
              name="fullName"
              value={formData.fullName}
              onChange={handleInputChange}
              className={errors.fullName ? 'error' : ''}
              placeholder="Enter your full name"
            />
            {errors.fullName && <span className="error-message">{errors.fullName}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="age">Age</label>
            <input
              type="number"
              id="age"
              name="age"
              value={formData.age}
              onChange={handleInputChange}
              className={errors.age ? 'error' : ''}
              placeholder="Enter your age"
              min="13"
              max="120"
            />
            {errors.age && <span className="error-message">{errors.age}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="gender">Gender</label>
            <select
              id="gender"
              name="gender"
              value={formData.gender}
              onChange={handleInputChange}
              className={errors.gender ? 'error' : ''}
            >
              <option value="">Select your gender</option>
              <option value="female">Female</option>
              <option value="male">Male</option>
            </select>
            {errors.gender && <span className="error-message">{errors.gender}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="sensitiveSkin">Sensitive Skin</label>
            <select
              id="sensitiveSkin"
              name="sensitiveSkin"
              value={formData.sensitiveSkin}
              onChange={handleInputChange}
              className={errors.sensitiveSkin ? 'error' : ''}
            >
              <option value="">Select an option</option>
              <option value="yes">Yes</option>
              <option value="no">No</option>
            </select>
            {errors.sensitiveSkin && <span className="error-message">{errors.sensitiveSkin}</span>}
          </div>

          {errors.submit && (
            <div className="error-message submit-error">{errors.submit}</div>
          )}

          <button 
            type="submit" 
            className="submit-btn"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Saving...' : 'Complete Profile'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default UserInfoForm;
