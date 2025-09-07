"""
Hyperparameter Tuning Service
Uses Optuna for automated hyperparameter optimization
"""

import optuna
from optuna.samplers import TPESampler
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, Tuple
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    """Service for hyperparameter optimization using Optuna"""
    
    def __init__(self, n_trials: int = 50, cv_folds: int = 5):
        """
        Initialize HyperparameterTuner
        
        Args:
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_params = {}
        self.study_results = {}
    
    def objective_catboost(self, trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """
        CatBoost hyperparameter optimization objective
        
        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training target
            
        Returns:
            Cross-validation RMSE score
        """
        params = {
            'iterations': trial.suggest_int('cat_iterations', 100, 1000),
            'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('cat_depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1, 10),
            'loss_function': 'RMSE',
            'verbose': 0,
            'random_state': 42
        }
        
        model = cb.CatBoostRegressor(**params)
        
        # Cross-validation
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=self.cv_folds, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        return np.sqrt(-scores.mean())
    
    def objective_xgboost(self, trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """
        XGBoost hyperparameter optimization objective
        
        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training target
            
        Returns:
            Cross-validation RMSE score
        """
        params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**params)
        
        # Cross-validation
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=self.cv_folds, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        return np.sqrt(-scores.mean())
    
    def objective_lightgbm(self, trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """
        LightGBM hyperparameter optimization objective
        
        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training target
            
        Returns:
            Cross-validation RMSE score
        """
        params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
            'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        
        # Cross-validation
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=self.cv_folds, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        return np.sqrt(-scores.mean())
    
    def objective_random_forest(self, trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        """
        Random Forest hyperparameter optimization objective
        
        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training target
            
        Returns:
            Cross-validation RMSE score
        """
        params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
            'max_depth': trial.suggest_int('rf_max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        
        # Cross-validation
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=self.cv_folds, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        return np.sqrt(-scores.mean())
    
    def optimize_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific model
        
        Args:
            model_name: Name of the model to optimize
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary with optimization results
        """
        print(f"🔍 Optimizing {model_name} hyperparameters...")
        
        # Select objective function
        objective_functions = {
            'catboost': self.objective_catboost,
            'xgboost': self.objective_xgboost,
            'lightgbm': self.objective_lightgbm,
            'random_forest': self.objective_random_forest
        }
        
        if model_name not in objective_functions:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(
            lambda trial: objective_functions[model_name](trial, X_train, y_train),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Store results
        self.best_params[model_name] = study.best_params
        self.study_results[model_name] = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'best_trial': study.best_trial.number
        }
        
        print(f"✅ {model_name} optimization completed!")
        print(f"📊 Best RMSE: {study.best_value:.4f}")
        print(f"🎯 Best parameters: {study.best_params}")
        
        return self.study_results[model_name]
    
    def optimize_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Optimize hyperparameters for all models
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary with all optimization results
        """
        print("🚀 Starting hyperparameter optimization for all models...")
        
        models = ['catboost', 'xgboost', 'lightgbm', 'random_forest']
        
        for model_name in models:
            self.optimize_model(model_name, X_train, y_train)
        
        print("✅ All models optimized!")
        
        # Save results
        self.save_optimization_results()
        
        return self.study_results
    
    def save_optimization_results(self, filepath: str = "model_params/best_hyperparameters.json"):
        """
        Save optimization results to file
        
        Args:
            filepath: Path to save the results
        """
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(exist_ok=True)
        
        # Prepare data for saving
        save_data = {}
        for model_name, results in self.study_results.items():
            save_data[model_name] = results['best_params']
        
        # Save to JSON
        import json
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=4)
        
        print(f"✅ Optimization results saved to {filepath}")
    
    def load_optimization_results(self, filepath: str = "model_params/best_hyperparameters.json") -> Dict[str, Any]:
        """
        Load optimization results from file
        
        Args:
            filepath: Path to load the results from
            
        Returns:
            Dictionary with optimization results
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        import json
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Hyperparameters file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            self.best_params = json.load(f)
        
        print(f"✅ Optimization results loaded from {filepath}")
        print(f"📊 Available models: {list(self.best_params.keys())}")
        return self.best_params
    
    def get_best_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get best parameters for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with best parameters
        """
        if model_name not in self.best_params:
            raise ValueError(f"No optimization results found for {model_name}")
        
        return self.best_params[model_name]
    
    def create_optimized_models(self) -> Dict[str, Any]:
        """
        Create model instances with optimized parameters
        
        Returns:
            Dictionary with optimized model instances
        """
        if not self.best_params:
            raise ValueError("No optimization results found. Run optimization first.")
        
        models = {}
        
        # CatBoost
        if 'catboost' in self.best_params:
            cat_params = self.best_params['catboost'].copy()
            cat_params.update({
                'loss_function': 'RMSE',
                'verbose': 0,
                'random_state': 42
            })
            models['catboost'] = cb.CatBoostRegressor(**cat_params)
        
        # XGBoost
        if 'xgboost' in self.best_params:
            xgb_params = self.best_params['xgboost'].copy()
            xgb_params.update({
                'objective': 'reg:squarederror',
                'random_state': 42
            })
            models['xgboost'] = xgb.XGBRegressor(**xgb_params)
        
        # LightGBM
        if 'lightgbm' in self.best_params:
            lgb_params = self.best_params['lightgbm'].copy()
            lgb_params.update({
                'objective': 'regression',
                'metric': 'rmse',
                'random_state': 42,
                'verbose': -1
            })
            models['lightgbm'] = lgb.LGBMRegressor(**lgb_params)
        
        # Random Forest
        if 'random_forest' in self.best_params:
            rf_params = self.best_params['random_forest'].copy()
            rf_params.update({
                'random_state': 42,
                'n_jobs': -1
            })
            models['random_forest'] = RandomForestRegressor(**rf_params)
        
        return models

# Global instance
hyperparameter_tuner = HyperparameterTuner()
