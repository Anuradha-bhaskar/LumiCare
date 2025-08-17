import React from 'react'
import { SignedIn, SignedOut } from '@clerk/clerk-react'
import { Outlet, Navigate } from 'react-router-dom'

const Layout = () => {
  return (
    <div className="app-layout">
      <main className="app-main">
        <SignedOut>
          <Navigate to="/sign-in" replace />
        </SignedOut>
        <SignedIn>
          <Outlet />
        </SignedIn>
      </main>
    </div>
  )
}

export default Layout
