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
    gender: ''
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
      // Store user profile data for faster access
      const userProfileData = {
        fullName: formData.fullName,
        age: parseInt(formData.age),
        gender: formData.gender,
        profileCompleted: true,
        userId: user.id
      };
      
      localStorage.setItem('userProfile', JSON.stringify(userProfileData));
      console.log('Stored user profile in localStorage:', userProfileData);
      
      // Update user metadata in Clerk
      await user.update({
        unsafeMetadata: {
          ...user.unsafeMetadata,
          ...userProfileData
        }
      });
      console.log('Updated Clerk metadata successfully');
      
      // Force a reload of user data
      await user.reload();
      
      // Try to save to database (optional - won't fail if backend is down)
      try {
        console.log('Attempting to sync user with database...');
        
        // First, ensure user exists in database by calling sync endpoint
        const syncPayload = {
          clerk_user_id: user.id,
          email: user.emailAddresses[0]?.emailAddress || user.email || 'test@example.com'
        };
        console.log('Sync payload:', syncPayload);
        
        const syncResponse = await fetch('http://localhost:8000/users/sync', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(syncPayload)
        });
        
        console.log('Sync response status:', syncResponse.status);
        
        if (syncResponse.ok) {
          const syncData = await syncResponse.json();
          console.log('Sync successful:', syncData);
          
          // Then, save profile to database
          const profilePayload = {
            full_name: formData.fullName,
            age: parseInt(formData.age),
            gender: formData.gender
          };
          console.log('Profile update payload:', profilePayload);
          
          const response = await fetch(`http://localhost:8000/users/profile/${user.id}`, {
            method: 'PUT',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(profilePayload)
          });
          
          console.log('Profile update response status:', response.status);
          
          if (response.ok) {
            const profileData = await response.json();
            console.log('Successfully saved to database:', profileData);
          } else {
            const errorData = await response.text();
            console.log('Profile update failed:', errorData);
          }
        } else {
          const errorData = await syncResponse.text();
          console.log('Sync failed:', errorData);
        }
      } catch (dbError) {
        console.warn('Database save failed, but user profile saved locally:', dbError);
      }
      
      // Navigate after successful update
      navigate('/');
      
    } catch (error) {
      console.error('Error updating user profile:', error);
      
      // More specific error messages
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
              <option value="non-binary">Non-binary</option>
              <option value="prefer-not-to-say">Prefer not to say</option>
            </select>
            {errors.gender && <span className="error-message">{errors.gender}</span>}
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
