"""
Model Evaluation Service
Provides comprehensive model evaluation and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error, median_absolute_error
)
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelEvaluator:
    """Service for comprehensive model evaluation"""
    
    def __init__(self, output_dir: str = "imgs"):
        """
        Initialize ModelEvaluator
        
        Args:
            output_dir: Directory to save evaluation plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'MedAE': median_absolute_error(y_true, y_pred),
            'NRMSE': np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(y_true) * 100
        }
        
        return metrics
    
    def evaluate_single_model(self, model_name: str, model: Any, X_test: pd.DataFrame, 
                             y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate a single model
        
        Args:
            model_name: Name of the model
            model: Trained model instance
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Calculate residuals
        residuals = y_test - y_pred
        
        results = {
            'model_name': model_name,
            'predictions': y_pred,
            'residuals': residuals,
            'metrics': metrics
        }
        
        return results
    
    def evaluate_all_models(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                           y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate all models
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation results for all models
        """
        print("📊 Evaluating all models...")
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            print(f"🔍 Evaluating {model_name}...")
            results = self.evaluate_single_model(model_name, model, X_test, y_test)
            evaluation_results[model_name] = results
        
        print("✅ All models evaluated!")
        return evaluation_results
    
    def plot_predictions_vs_actual(self, evaluation_results: Dict[str, Any], 
                                  y_test: pd.Series, max_models: int = 4):
        """
        Plot predictions vs actual values
        
        Args:
            evaluation_results: Results from model evaluation
            y_test: True test values
            max_models: Maximum number of models to plot
        """
        n_models = min(len(evaluation_results), max_models)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, (model_name, results) in enumerate(list(evaluation_results.items())[:max_models]):
            plt.subplot(n_rows, n_cols, i + 1)
            
            y_pred = results['predictions']
            r2 = results['metrics']['R2']
            
            # Scatter plot
            plt.scatter(y_test, y_pred, alpha=0.5, s=1)
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            plt.xlabel('Actual Price')
            plt.ylabel('Predicted Price')
            plt.title(f'{model_name} (R² = {r2:.4f})')
            
            # Add grid
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "predictions_vs_actual.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved predictions vs actual plot")
    
    def plot_residuals(self, evaluation_results: Dict[str, Any], y_test: pd.Series, max_models: int = 4):
        """
        Plot residuals analysis
        
        Args:
            evaluation_results: Results from model evaluation
            y_test: True test values
            max_models: Maximum number of models to plot
        """
        n_models = min(len(evaluation_results), max_models)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, (model_name, results) in enumerate(list(evaluation_results.items())[:max_models]):
            plt.subplot(n_rows, n_cols, i + 1)
            
            residuals = results['residuals']
            rmse = results['metrics']['RMSE']
            
            # Residuals vs predicted
            plt.scatter(results['predictions'], residuals, alpha=0.5, s=1)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Price')
            plt.ylabel('Residuals')
            plt.title(f'{model_name} Residuals (RMSE = {rmse:.2f})')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "residuals_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved residuals analysis plot")
    
    def plot_metrics_comparison(self, evaluation_results: Dict[str, Any]):
        """
        Plot metrics comparison across models
        
        Args:
            evaluation_results: Results from model evaluation
        """
        # Prepare data for plotting
        metrics_data = []
        for model_name, results in evaluation_results.items():
            metrics = results['metrics']
            metrics_data.append({
                'Model': model_name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'MAPE': metrics['MAPE'],
                'NRMSE': metrics['NRMSE']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        metrics_to_plot = ['RMSE', 'MAE', 'R2', 'MAPE', 'NRMSE']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            bars = ax.bar(df_metrics['Model'], df_metrics[metric])
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved metrics comparison plot")
    
    def plot_feature_importance(self, models: Dict[str, Any], feature_names: List[str], 
                               max_features: int = 15):
        """
        Plot feature importance for tree-based models
        
        Args:
            models: Dictionary of trained models
            feature_names: List of feature names
            max_features: Maximum number of features to show
        """
        tree_models = ['catboost', 'xgboost', 'lightgbm', 'random_forest']
        
        n_models = len([m for m in tree_models if m in models])
        if n_models == 0:
            print("⚠️ No tree-based models found for feature importance")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, model_name in enumerate(tree_models):
            if model_name not in models:
                axes[i].remove()
                continue
            
            model = models[model_name]
            ax = axes[i]
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
            else:
                print(f"⚠️ {model_name} doesn't support feature importance")
                axes[i].remove()
                continue
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True).tail(max_features)
            
            # Plot
            bars = ax.barh(importance_df['feature'], importance_df['importance'])
            ax.set_title(f'{model_name.title()} Feature Importance')
            ax.set_xlabel('Importance')
        
        # Remove empty subplots
        for i in range(n_models, 4):
            if i < len(axes):
                axes[i].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved feature importance plot")
    
    def plot_error_distribution(self, evaluation_results: Dict[str, Any]):
        """
        Plot error distribution for all models
        
        Args:
            evaluation_results: Results from model evaluation
        """
        n_models = len(evaluation_results)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            plt.subplot(n_rows, n_cols, i + 1)
            
            residuals = results['residuals']
            
            # Histogram of residuals
            plt.hist(residuals, bins=50, alpha=0.7, density=True, edgecolor='black')
            plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
            plt.xlabel('Residuals')
            plt.ylabel('Density')
            plt.title(f'{model_name} Error Distribution')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "error_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved error distribution plot")
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate comprehensive evaluation report
        
        Args:
            evaluation_results: Results from model evaluation
            
        Returns:
            DataFrame with evaluation summary
        """
        report_data = []
        
        for model_name, results in evaluation_results.items():
            metrics = results['metrics']
            report_data.append({
                'Model': model_name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R²': metrics['R2'],
                'MAPE (%)': metrics['MAPE'],
                'NRMSE (%)': metrics['NRMSE'],
                'MedAE': metrics['MedAE']
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('RMSE')
        
        return report_df
    
    def run_full_evaluation(self, models: Dict[str, Any], X_test: pd.DataFrame, 
                           y_test: pd.Series, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Run complete model evaluation
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            feature_names: List of feature names (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        print("🔍 Starting comprehensive model evaluation...")
        
        # Evaluate all models
        evaluation_results = self.evaluate_all_models(models, X_test, y_test)
        
        # Generate plots
        self.plot_predictions_vs_actual(evaluation_results, y_test)
        self.plot_residuals(evaluation_results, y_test)
        self.plot_metrics_comparison(evaluation_results)
        self.plot_error_distribution(evaluation_results)
        
        if feature_names:
            self.plot_feature_importance(models, feature_names)
        
        # Generate report
        report = self.generate_evaluation_report(evaluation_results)
        
        print("✅ Model evaluation completed!")
        print(f"📊 Evaluation plots saved to: {self.output_dir}")
        print("\n📋 Model Performance Summary:")
        print(report.to_string(index=False))
        
        return {
            'evaluation_results': evaluation_results,
            'report': report
        }

# Global instance
model_evaluator = ModelEvaluator()
