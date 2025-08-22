import React from 'react';
import { SignOutButton } from '@clerk/clerk-react';
import { LogOut } from 'lucide-react';
import { Link } from "react-router-dom";
import './Header.css';

const Header = ({ user, userProfile }) => {
  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 18) return 'Good afternoon';
    return 'Good evening';
  };

  const getUserName = () => {
    if (userProfile?.fullName) return userProfile.fullName;
    if (user?.firstName) return user.firstName;
    return 'User';
  };

  return (
    <header className="dashboard-header">
      <div className="header-content">
        <div className="header-left">
          <Link to="/" className="logo-link">
            <h1 className="app-title">LumiCare</h1>
          </Link>
          <div className="greeting">
            <span className="greeting-text">
              {getGreeting()}, {getUserName()}! 
            </span>
          </div>
        </div>
        
        <div className="header-right">
          <div className="user-info">
            <div className="user-avatar">
              {user?.imageUrl ? (
                <img src={user.imageUrl} alt="Profile" className="avatar-img" />
              ) : (
                <div className="avatar-placeholder">
                  {getUserName().charAt(0).toUpperCase()}
                </div>
              )}
            </div>
            <div className="user-details">
              <span className="user-name">{getUserName()}</span>
              <span className="user-email">{user?.emailAddresses?.[0]?.emailAddress}</span>
            </div>
          </div>
          
          <SignOutButton>
            <button className="sign-out-btn">
              <span className="sign-out-icon"><LogOut size={16} /></span>
              Sign Out
            </button>
          </SignOutButton>
        </div>
      </div>
    </header>
  );
};

export default Header;
