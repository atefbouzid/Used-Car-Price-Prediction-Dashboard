"""
Training Service
Handles model training and saving
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Service for training machine learning models"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize ModelTrainer
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.trained_models = {}
        self.training_results = {}
    
    def create_default_models(self) -> Dict[str, Any]:
        """
        Create default model instances
        
        Returns:
            Dictionary with default model instances
        """
        models = {
            'catboost': cb.CatBoostRegressor(
                loss_function='RMSE',
                verbose=0,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                objective='regression',
                metric='rmse',
                random_state=42,
                verbose=-1
            ),
            'random_forest': RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        }
        
        return models
    
    def create_optimized_models(self, best_params: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create model instances with optimized parameters
        
        Args:
            best_params: Dictionary with best parameters for each model
            
        Returns:
            Dictionary with optimized model instances
        """
        models = {}
        
        # CatBoost
        if 'catboost' in best_params:
            cat_params = best_params['catboost'].copy()
            cat_params.update({
                'loss_function': 'RMSE',
                'verbose': 0,
                'random_state': 42
            })
            models['catboost'] = cb.CatBoostRegressor(**cat_params)
        
        # XGBoost
        if 'xgboost' in best_params:
            xgb_params = best_params['xgboost'].copy()
            xgb_params.update({
                'objective': 'reg:squarederror',
                'random_state': 42
            })
            models['xgboost'] = xgb.XGBRegressor(**xgb_params)
        
        # LightGBM
        if 'lightgbm' in best_params:
            lgb_params = best_params['lightgbm'].copy()
            lgb_params.update({
                'objective': 'regression',
                'metric': 'rmse',
                'random_state': 42,
                'verbose': -1
            })
            models['lightgbm'] = lgb.LGBMRegressor(**lgb_params)
        
        # Random Forest
        if 'random_forest' in best_params:
            rf_params = best_params['random_forest'].copy()
            rf_params.update({
                'random_state': 42,
                'n_jobs': -1
            })
            models['random_forest'] = RandomForestRegressor(**rf_params)
        
        return models
    
    def train_single_model(self, model_name: str, model: Any, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Train a single model
        
        Args:
            model_name: Name of the model
            model: Model instance to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Dictionary with training results
        """
        print(f"🏋️ Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        results = {
            'model_name': model_name,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'model': model
        }
        
        # Validation results if provided
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            results.update({
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2
            })
            
            print(f"✅ {model_name} - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
        else:
            print(f"✅ {model_name} - Train RMSE: {train_rmse:.4f}")
        
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_val: pd.DataFrame = None, y_val: pd.Series = None,
                        use_optimized_params: bool = True) -> Dict[str, Any]:
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            use_optimized_params: Whether to use optimized parameters
            
        Returns:
            Dictionary with training results for all models
        """
        print("🚀 Starting model training...")
        
        # Create models
        if use_optimized_params:
            try:
                from .hyperparameter_tuning import hyperparameter_tuner
                best_params = hyperparameter_tuner.load_optimization_results()
                models = self.create_optimized_models(best_params)
                print("📊 Using optimized hyperparameters")
            except:
                models = self.create_default_models()
                print("⚠️ Using default hyperparameters (optimization results not found)")
        else:
            models = self.create_default_models()
            print("📊 Using default hyperparameters")
        
        # Train each model
        for model_name, model in models.items():
            results = self.train_single_model(
                model_name, model, X_train, y_train, X_val, y_val
            )
            self.training_results[model_name] = results
            self.trained_models[model_name] = model
        
        print("✅ All models trained successfully!")
        return self.training_results
    
    def save_models(self, models: Dict[str, Any] = None) -> None:
        """
        Save trained models to disk
        
        Args:
            models: Dictionary of models to save (uses self.trained_models if None)
        """
        if models is None:
            models = self.trained_models
        
        if not models:
            raise ValueError("No models to save. Train models first.")
        
        print("💾 Saving trained models...")
        
        for model_name, model in models.items():
            model_filename = self.models_dir / f"{model.__class__.__name__}.pkl"
            joblib.dump(model, model_filename)
            print(f"✅ Saved {model_name} to {model_filename}")
        
        print("✅ All models saved successfully!")
    
    def load_models(self) -> Dict[str, Any]:
        """
        Load trained models from disk
        
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
                model = joblib.load(model_path)
                loaded_models[model_name] = model
                print(f"✅ Loaded {model_name}")
            else:
                print(f"⚠️ Model file not found: {model_path}")
        
        self.trained_models = loaded_models
        return loaded_models
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a specific trained model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Trained model instance
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.trained_models.keys())}")
        
        return self.trained_models[model_name]
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available trained models
        
        Returns:
            List of model names
        """
        return list(self.trained_models.keys())
    
    def get_training_summary(self) -> pd.DataFrame:
        """
        Get training results summary
        
        Returns:
            DataFrame with training results
        """
        if not self.training_results:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, results in self.training_results.items():
            summary_data.append({
                'Model': model_name,
                'Train_RMSE': results.get('train_rmse', np.nan),
                'Train_MAE': results.get('train_mae', np.nan),
                'Train_R2': results.get('train_r2', np.nan),
                'Val_RMSE': results.get('val_rmse', np.nan),
                'Val_MAE': results.get('val_mae', np.nan),
                'Val_R2': results.get('val_r2', np.nan)
            })
        
        return pd.DataFrame(summary_data)
    
    def make_predictions(self, X: pd.DataFrame, model_names: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Make predictions using trained models
        
        Args:
            X: Features for prediction
            model_names: List of model names to use (uses all if None)
            
        Returns:
            Dictionary with predictions from each model
        """
        if not self.trained_models:
            raise ValueError("No trained models available. Train models first.")
        
        if model_names is None:
            model_names = list(self.trained_models.keys())
        
        predictions = {}
        for model_name in model_names:
            if model_name in self.trained_models:
                pred = self.trained_models[model_name].predict(X)
                predictions[model_name] = pred
            else:
                print(f"⚠️ Model '{model_name}' not found")
        
        return predictions
    
    def ensemble_predict(self, X: pd.DataFrame, model_names: List[str] = None, 
                        method: str = 'mean') -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Features for prediction
            model_names: List of model names to use (uses all if None)
            method: Ensemble method ('mean', 'median', 'weighted')
            
        Returns:
            Ensemble predictions
        """
        predictions = self.make_predictions(X, model_names)
        
        if not predictions:
            raise ValueError("No predictions available for ensemble")
        
        pred_array = np.array(list(predictions.values()))
        
        if method == 'mean':
            return np.mean(pred_array, axis=0)
        elif method == 'median':
            return np.median(pred_array, axis=0)
        elif method == 'weighted':
            # Simple weighted average (can be improved with validation scores)
            weights = np.ones(len(pred_array)) / len(pred_array)
            return np.average(pred_array, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

# Global instance
model_trainer = ModelTrainer()
