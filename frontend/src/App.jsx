import  ClerkProviderWithRoutes  from './auth/ClerkProviderWithRoutes.jsx'
import './App.css'
import { Routes, Route } from 'react-router-dom'
import AuthenticationPage  from './auth/AuthenticationPage.jsx'
import Layout from './layout/Layout.jsx'
import ProfileCompleteGuard from './auth/ProfileCompleteGuard.jsx'


function App() {
  return (
    
      <ClerkProviderWithRoutes>
        <ProfileCompleteGuard>
          <Routes>
            <Route path="/sign-in/*" element={<AuthenticationPage />} />
            <Route path="/sign-up/*" element={<AuthenticationPage />} />
            <Route path="/" element={<Layout />}>
              <Route index element={<div>Welcome to LumiCare!</div>} />
              {/* Add other routes here */}
            </Route>
          </Routes>
        </ProfileCompleteGuard>
      </ClerkProviderWithRoutes>
    
  )
}

export default App
