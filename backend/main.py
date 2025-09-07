from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from services.predict import prediction_service
from services.preprocessing import data_preprocessor
from pathlib import Path

app = FastAPI(title="Used Car Price Prediction API", version="1.0.0")

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models and preprocessor on startup"""
    print("🚀 Starting Used Car Price Prediction API...")
    
    # Load models
    try:
        models = prediction_service.load_models()
        print(f"✅ Loaded {len(models)} models: {list(models.keys())}")
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
    
    # Try to load preprocessor
    try:
        preprocessor_path = Path("preprocessor.pkl")
        if preprocessor_path.exists():
            data_preprocessor.load_preprocessor("preprocessor.pkl")
            print("✅ Loaded preprocessor")
        else:
            print("⚠️ Preprocessor not found - will need to be fitted during first prediction")
    except Exception as e:
        print(f"⚠️ Error loading preprocessor: {e}")
    
    print("🎉 API startup complete!")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Car(BaseModel):
    brand: str
    model: str
    model_year: int
    milage: int
    fuel_type: str
    engine: str
    transmission: str
    ext_col: str
    int_col: str
    accident: str
    clean_title: str


class PredictionRequest(BaseModel):
    car_data: Car
    selected_models: List[str]
    use_ensemble: Optional[bool] = False


@app.get("/")
async def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Used Car Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "models": "/available_models",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        available_models = prediction_service.get_available_models()
        if len(available_models) == 0:
            return {
                "status": "error", 
                "message": "No models found",
                "models_loaded": False
            }
        
        return {
            "status": "success", 
            "message": "API is healthy",
            "models_loaded": True,
            "available_models": available_models,
            "models_count": len(available_models)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "models_loaded": False
        }

@app.get("/available_models")
async def get_available_models_endpoint():
    """Get list of available models"""
    try:
        available_models = prediction_service.get_available_models()
        models_info = prediction_service.get_all_models_info()
        
        return {
            "available_models": available_models,
            "models_count": len(available_models),
            "models_info": models_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make car price predictions using selected models"""
    try:
        # Validate model availability
        model_availability = prediction_service.validate_model_availability(request.selected_models)
        unavailable_models = [model for model, available in model_availability.items() if not available]
        
        if unavailable_models:
            raise HTTPException(
                status_code=400, 
                detail=f"Models not found: {unavailable_models}. Available models: {prediction_service.get_available_models()}"
            )
        
        # Check if preprocessor is fitted, if not try to load it
        if not data_preprocessor.is_fitted:
            preprocessor_path = Path("preprocessor.pkl")
            if preprocessor_path.exists():
                data_preprocessor.load_preprocessor("preprocessor.pkl")
            else:
                raise HTTPException(
                    status_code=500, 
                    detail="Preprocessor not fitted and no saved preprocessor found. Please train models first."
                )
        
        # Preprocess the car data
        preprocessed_data = data_preprocessor.preprocess_single_sample(request.car_data.dict())
        
        # Make predictions
        result = prediction_service.predict_with_ensemble(
            model_names=request.selected_models,
            features=preprocessed_data,
            use_ensemble=request.use_ensemble,
            ensemble_method='mean'
        )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model_info/{model_name}")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    try:
        model_info = prediction_service.get_model_info(model_name)
        if 'error' in model_info:
            raise HTTPException(status_code=404, detail=model_info['error'])
        
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")
