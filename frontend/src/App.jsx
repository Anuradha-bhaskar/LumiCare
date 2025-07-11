import  ClerkProviderWithRoutes  from './auth/ClerkProviderWithRoutes.jsx'
import './App.css'
import { Routes, Route } from 'react-router-dom'
import AuthenticationPage  from './auth/AuthenticationPage.jsx'
import Layout from './layout/Layout.jsx'


function App() {
  return (
    
      <ClerkProviderWithRoutes>
        <Routes>
          <Route path="/sign-in/*" element={<AuthenticationPage />} />
          <Route path="/sign-up/*" element={<AuthenticationPage />} />
          <Route path="/" element={<Layout />}>
            <Route index element={<div>Welcome to LumiCare!</div>} />
            {/* Add other routes here */}
          </Route>
        </Routes>
      </ClerkProviderWithRoutes>
    
  )
}

export default App
