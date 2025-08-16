import React from 'react';
import { SignedIn, SignedOut } from '@clerk/clerk-react';
import { useLocation } from 'react-router-dom';
import CustomSignIn from './components/SignIn';
import CustomSignUp from './components/SignUp';

const AuthenticationPage = () => {
  const location = useLocation();
  const isSignUp = location.pathname.includes('/sign-up');

  return (
    <div>
      <SignedOut>
        {isSignUp ? <CustomSignUp /> : <CustomSignIn />}
      </SignedOut>
      <SignedIn>
        <div className='auth-container'>
          <div className='auth-card'>
            <div className='auth-header'>
              <h2>Already Signed In</h2>
              <p>You are already signed in to LumiCare</p>
            </div>
          </div>
        </div>
      </SignedIn>
    </div>
  );
};

export default AuthenticationPage;
