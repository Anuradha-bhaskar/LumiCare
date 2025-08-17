import  ClerkProviderWithRoutes  from './auth/ClerkProviderWithRoutes.jsx'
import './App.css'
import { Routes, Route } from 'react-router-dom'
import AuthenticationPage  from './auth/AuthenticationPage.jsx'
import Layout from './layout/Layout.jsx'
import ProfileCompleteGuard from './auth/ProfileCompleteGuard.jsx'
import LandingPage from './pages/LandingPage.jsx'
import SkinAnalysisPage from './pages/SkinAnalysisPage.jsx'
import ProgressPage from './pages/ProgressPage.jsx'
import RoutinePage from './pages/RoutinePage.jsx'
import ProfilePage from './pages/ProfilePage.jsx'


function App() {
  return (
    
      <ClerkProviderWithRoutes>
        <ProfileCompleteGuard>
          <Routes>
            <Route path="/sign-in/*" element={<AuthenticationPage />} />
            <Route path="/sign-up/*" element={<AuthenticationPage />} />
            <Route path="/" element={<LandingPage />} />
            <Route path="/skin-analysis" element={<Layout />}>
              <Route index element={<SkinAnalysisPage />} />
            </Route>
            <Route path="/progress" element={<Layout />}>
              <Route index element={<ProgressPage />} />
            </Route>
            <Route path="/routine" element={<Layout />}>
              <Route index element={<RoutinePage />} />
            </Route>
            <Route path="/profile" element={<Layout />}>
              <Route index element={<ProfilePage />} />
            </Route>
          </Routes>
        </ProfileCompleteGuard>
      </ClerkProviderWithRoutes>
    
  )
}

export default App
