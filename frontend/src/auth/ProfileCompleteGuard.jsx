import React, { useEffect, useState } from 'react';
import { useUser } from '@clerk/clerk-react';
import { useLocation } from 'react-router-dom';
import UserInfoForm from './components/UserInfo';

const ProfileCompleteGuard = ({ children }) => {
  const { user, isLoaded } = useUser();
  const location = useLocation();
  const [shouldShowUserInfo, setShouldShowUserInfo] = useState(false);

  useEffect(() => {
    if (isLoaded && user) {
      // Check if user has completed their profile
      const profileCompleted = user.unsafeMetadata?.profileCompleted || user.publicMetadata?.profileCompleted;
      
      // Also check localStorage as backup
      const storedProfile = localStorage.getItem('userProfile');
      let localProfileCompleted = false;
      
      if (storedProfile) {
        try {
          const parsedProfile = JSON.parse(storedProfile);
          localProfileCompleted = parsedProfile.userId === user.id && parsedProfile.profileCompleted && !parsedProfile.skipped;
        } catch {
          // Ignore JSON parsing errors
        }
      }
      
      const isOnAuthPages = location.pathname.includes('/sign-in') || location.pathname.includes('/sign-up');
      
      // Show user info form only if:
      // 1. User is signed in
      // 2. Profile is not completed (neither in Clerk nor localStorage)
      // 3. User is not on auth pages
      if (!profileCompleted && !localProfileCompleted && !isOnAuthPages) {
        setShouldShowUserInfo(true);
      } else {
        setShouldShowUserInfo(false);
      }
    }
  }, [user, isLoaded, location.pathname]);

  // Show loading while Clerk is loading
  if (!isLoaded) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #fff5f2 0%, #ffd4c4 100%)'
      }}>
        <div style={{
          fontSize: '1.2rem',
          color: '#4a4a4a',
          fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
        }}>
          Loading...
        </div>
      </div>
    );
  }

  // Show user info form if profile is incomplete
  if (shouldShowUserInfo) {
    return <UserInfoForm />;
  }

  // Show normal content
  return children;
};

export default ProfileCompleteGuard;
