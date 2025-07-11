import React from 'react'
import { SignedIn, SignedOut, SignIn, SignUp } from '@clerk/clerk-react';


const AuthenticationPage = () => {
  return (
    <div className='auth-container'>
      <header>
      <SignedOut>
        <SignIn routing='path' path='/sign-in' />
        <SignUp routing='path' path='/sign-up' />
      </SignedOut>
      <SignedIn>
        <div className='redirect-message'>
            <p>You are already Signed in</p>
        </div>
      </SignedIn>
    </header>
    </div>
  )
}

export default AuthenticationPage
