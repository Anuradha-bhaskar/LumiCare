# LumiCare - AI-Powered Skin Health Analysis Platform

## 🎯 Project Overview

LumiCare is a comprehensive skin health analysis platform that leverages computer vision and artificial intelligence to provide personalized skincare insights. The platform analyzes skin conditions in real-time using advanced image processing algorithms, offering users detailed assessments of their skin health and personalized recommendations.

### Key Features
- **Real-time Skin Analysis**: Advanced computer vision algorithms for detecting skin conditions
- **AI-Powered Diagnostics**: Machine learning models for skin texture, acne, and pigmentation analysis
- **Personalized Recommendations**: Customized skincare routines based on analysis results
- **Progress Tracking**: Monitor skin health improvements over time


## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Database**: MongoDB with PyMongo
- **Authentication**: Clerk Backend API
- **Computer Vision**: OpenCV, MediaPipe

### Frontend
- **Framework**: React.js 19


## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Node.js 18+
- UV (Python package manager)
- MongoDB instance

### Installation

#### Backend Setup
```bash
cd backend

# Install UV if not already installed
pip install uv

# Install dependencies and create virtual environment
uv sync

# Run the application
uv run main.py
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 📁 Project Structure

```
LumiCare/
├── backend/                 # FastAPI backend
│   ├── src/
│   │   ├── routes/         # API endpoints
│   │   ├── services/       # Business logic
│   │   ├── models/         # Data models
│   │   └── database/       # Database configuration
│   └── main.py            # Application entry point
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components
│   │   ├── auth/           # Authentication components
│   │   └── Dashboard/      # Dashboard components
│   └── package.json
└── README.md
```


