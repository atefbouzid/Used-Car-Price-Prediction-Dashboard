"""
Prediction Service
Handles model predictions for the API
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

class PredictionService:
    """Service for making predictions using trained models"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize PredictionService
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.is_loaded = False
    
    def load_models(self) -> Dict[str, Any]:
        """
        Load all trained models from disk
        
        Returns:
            Dictionary with loaded models
        """
        print("📂 Loading trained models...")
        
        model_files = {
            'catboost': 'CatBoostRegressor.pkl',
            'xgboost': 'XGBRegressor.pkl',
            'lightgbm': 'LGBMRegressor.pkl',
            'random_forest': 'RandomForestRegressor.pkl'
        }
        
        loaded_models = {}
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    loaded_models[model_name] = model
                    print(f"✅ Loaded {model_name}")
                except Exception as e:
                    print(f"❌ Error loading {model_name}: {e}")
            else:
                print(f"⚠️ Model file not found: {model_path}")
        
        self.models = loaded_models
        self.is_loaded = True
        
        if not loaded_models:
            print("⚠️ No models loaded successfully")
        
        return loaded_models
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models
        
        Returns:
            List of available model names
        """
        if not self.is_loaded:
            self.load_models()
        
        return list(self.models.keys())
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a specific model by name
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance or None if not found
        """
        if not self.is_loaded:
            self.load_models()
        
        return self.models.get(model_name)
    
    def predict_single_model(self, model_name: str, features: np.ndarray) -> float:
        """
        Make prediction using a single model
        
        Args:
            model_name: Name of the model to use
            features: Preprocessed feature array
            
        Returns:
            Prediction value
        """
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        try:
            prediction = model.predict(features)
            
            # Convert to float
            if hasattr(prediction, 'item'):
                return float(prediction.item())
            else:
                return float(prediction[0])
        
        except Exception as e:
            raise ValueError(f"Prediction failed for {model_name}: {str(e)}")
    
    def predict_multiple_models(self, model_names: List[str], features: np.ndarray) -> Dict[str, float]:
        """
        Make predictions using multiple models
        
        Args:
            model_names: List of model names to use
            features: Preprocessed feature array
            
        Returns:
            Dictionary with predictions from each model
        """
        predictions = {}
        
        for model_name in model_names:
            try:
                prediction = self.predict_single_model(model_name, features)
                predictions[model_name] = prediction
            except Exception as e:
                print(f"⚠️ Error predicting with {model_name}: {e}")
                continue
        
        return predictions
    
    def ensemble_predict(self, predictions: List[float], method: str = 'mean') -> float:
        """
        Create ensemble prediction from individual model predictions
        
        Args:
            predictions: List of predictions from different models
            method: Ensemble method ('mean', 'median', 'weighted')
            
        Returns:
            Ensemble prediction
        """
        if not predictions:
            raise ValueError("No predictions provided for ensemble")
        
        predictions_array = np.array(predictions)
        
        if method == 'mean':
            return float(np.mean(predictions_array))
        elif method == 'median':
            return float(np.median(predictions_array))
        elif method == 'weighted':
            # Simple weighted average (can be improved with validation scores)
            weights = np.ones(len(predictions_array)) / len(predictions_array)
            return float(np.average(predictions_array, weights=weights))
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def predict_with_ensemble(self, model_names: List[str], features: np.ndarray, 
                             use_ensemble: bool = True, ensemble_method: str = 'mean') -> Dict[str, Any]:
        """
        Make predictions with optional ensemble
        
        Args:
            model_names: List of model names to use
            features: Preprocessed feature array
            use_ensemble: Whether to include ensemble prediction
            ensemble_method: Method for ensemble prediction
            
        Returns:
            Dictionary with individual and ensemble predictions
        """
        # Get individual predictions
        individual_predictions = self.predict_multiple_models(model_names, features)
        
        result = {
            'individual_predictions': individual_predictions,
            'selected_models': model_names
        }
        
        # Add ensemble prediction if requested and we have multiple models
        if use_ensemble and len(individual_predictions) > 1:
            predictions_list = list(individual_predictions.values())
            ensemble_prediction = self.ensemble_predict(predictions_list, ensemble_method)
            result['ensemble_prediction'] = ensemble_prediction
        
        return result
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        model = self.get_model(model_name)
        if model is None:
            return {'error': f"Model '{model_name}' not found"}
        
        # Check if model is fitted
        is_fitted = False
        if hasattr(model, 'feature_importances_'):
            is_fitted = True
        elif hasattr(model, 'get_booster'):
            is_fitted = True
        elif hasattr(model, 'estimators_'):
            is_fitted = True
        elif hasattr(model, 'coef_'):
            is_fitted = True
        
        info = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'is_fitted': is_fitted,
            'has_predict': hasattr(model, 'predict')
        }
        
        # Add model-specific information
        if hasattr(model, 'n_estimators'):
            info['n_estimators'] = model.n_estimators
        if hasattr(model, 'max_depth'):
            info['max_depth'] = model.max_depth
        if hasattr(model, 'learning_rate'):
            info['learning_rate'] = model.learning_rate
        
        return info
    
    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available models
        
        Returns:
            Dictionary with information about all models
        """
        if not self.is_loaded:
            self.load_models()
        
        models_info = {}
        for model_name in self.models.keys():
            models_info[model_name] = self.get_model_info(model_name)
        
        return models_info
    
    def validate_model_availability(self, model_names: List[str]) -> Dict[str, bool]:
        """
        Validate which models are available
        
        Args:
            model_names: List of model names to check
            
        Returns:
            Dictionary with availability status for each model
        """
        if not self.is_loaded:
            self.load_models()
        
        availability = {}
        for model_name in model_names:
            availability[model_name] = model_name in self.models
        
        return availability
    
    def reload_models(self) -> Dict[str, Any]:
        """
        Reload all models from disk
        
        Returns:
            Dictionary with reloaded models
        """
        print("🔄 Reloading models...")
        self.is_loaded = False
        return self.load_models()

# Global instance
prediction_service = PredictionService()
