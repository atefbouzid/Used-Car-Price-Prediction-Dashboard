# Frontend-Backend Integration Guide

## Overview
This document explains how the frontend is integrated with the backend API for the Used Car Price Prediction Dashboard.

## API Integration

### Configuration
- **API Base URL**: `http://localhost:8000` (configurable via environment variables)
- **CORS**: Enabled for `http://localhost:3000` (frontend development server)

### API Endpoints Used

1. **Health Check** - `GET /health`
   - Checks if backend is running and models are loaded
   - Returns model availability status

2. **Available Models** - `GET /available_models`
   - Fetches list of trained models
   - Returns model information and metadata

3. **Model Info** - `GET /model_info/{model_name}`
   - Gets detailed information about a specific model
   - Returns model parameters and status

4. **Prediction** - `POST /predict`
   - Makes car price predictions using selected models
   - Supports single model and ensemble predictions

### Frontend Components Updated

#### 1. ModelSelector Component
- **Fetches available models** from backend on component mount
- **Dynamic model loading** with loading states and error handling
- **Ensemble option** automatically enabled when multiple models selected
- **Real-time model availability** display

#### 2. PredictionResults Component
- **Handles backend response format** with individual and ensemble predictions
- **Error display** for failed predictions
- **Price range analysis** for multiple model predictions
- **Loading states** during prediction requests

#### 3. Main Page (page.tsx)
- **Real API integration** replacing mock data
- **Form validation** before making predictions
- **Error handling** with user-friendly messages
- **State management** for predictions and loading states

### Data Flow

```
User Input → Form Validation → API Request → Backend Processing → Response → UI Update
```

1. **User fills car form** and selects models
2. **Frontend validates** required fields
3. **API request sent** to backend with car data and model selection
4. **Backend processes** data through preprocessing and models
5. **Response returned** with individual and/or ensemble predictions
6. **UI updates** with results, errors, or loading states

### Error Handling

- **Connection errors**: Display retry option
- **Validation errors**: Show specific field requirements
- **API errors**: Display backend error messages
- **Loading states**: Show progress indicators

### Environment Configuration

Create `.env.local` in the frontend directory:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Running the Application

1. **Start Backend**:
   ```bash
   cd backend
   python run_server.py
   ```

2. **Start Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access Application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Testing the Integration

Use the provided test files in the backend:
- `test_api.py` - Python test script
- `test_curl.sh` - Bash curl tests
- Individual JSON files for different scenarios

### Troubleshooting

1. **CORS Issues**: Ensure backend CORS is configured for frontend URL
2. **Connection Refused**: Check if backend server is running
3. **Model Not Found**: Ensure models are trained and available
4. **Validation Errors**: Check required fields in car form

### API Response Format

#### Prediction Response
```json
{
  "individual_predictions": {
    "xgboost": 25000.50,
    "catboost": 24800.75,
    "lightgbm": 25200.25
  },
  "selected_models": ["xgboost", "catboost", "lightgbm"],
  "ensemble_prediction": 25000.50
}
```

#### Available Models Response
```json
{
  "available_models": ["xgboost", "catboost", "lightgbm", "random_forest"],
  "models_count": 4,
  "models_info": {
    "xgboost": {
      "model_name": "XGBRegressor",
      "model_type": "XGBRegressor",
      "is_fitted": true,
      "has_predict": true
    }
  }
}
```
