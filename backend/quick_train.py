#!/usr/bin/env python3
"""
Quick Training Script
Fast model training using existing hyperparameters and preprocessor
"""

import sys
import os
import argparse
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.data_loader import data_loader
from services.preprocessing import data_preprocessor
from services.train import model_trainer
from services.evaluate import model_evaluator
from sklearn.model_selection import train_test_split

def main():
    """Quick training pipeline"""
    parser = argparse.ArgumentParser(description='Quick Model Training')
    parser.add_argument('--skip-eval', action='store_true', 
                       help='Skip model evaluation to save time')
    
    args = parser.parse_args()
    
    print("⚡ Quick Model Training Pipeline")
    print("=" * 50)
    
    try:
        # Step 1: Load Data
        print("\n📂 Step 1: Loading Data...")
        train_data, test_data, sample_submission = data_loader.load_all_data()
        print(f"✅ Loaded training data: {train_data.shape}")
        
        # Step 2: Data Preprocessing
        print("\n🔄 Step 2: Data Preprocessing...")
        train_processed, test_processed = data_preprocessor.preprocess_data(train_data, test_data)
        print("✅ Preprocessing completed")
        
        # Step 3: Train-Validation Split
        print("\n✂️ Step 3: Train-Validation Split...")
        X = train_processed.drop('price', axis=1)
        y = train_processed['price']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"✅ Training set: {X_train.shape}")
        print(f"✅ Validation set: {X_val.shape}")
        
        # Step 4: Model Training (using existing hyperparameters)
        print("\n🏋️ Step 4: Model Training...")
        print("📊 Using existing hyperparameters from model_params/")
        training_results = model_trainer.train_all_models(
            X_train, y_train, X_val, y_val, use_optimized_params=True
        )
        
        # Save trained models
        model_trainer.save_models()
        print("✅ Model training completed and models saved")
        
        # Step 5: Model Evaluation (optional)
        if not args.skip_eval:
            print("\n📊 Step 5: Model Evaluation...")
            evaluation_results = model_evaluator.run_full_evaluation(
                model_trainer.trained_models, X_val, y_val, 
                feature_names=data_preprocessor.get_feature_importance_names()
            )
            print("✅ Model evaluation completed")
            
            # Show results
            evaluation_summary = evaluation_results['report']
            print("\n📈 Evaluation Results:")
            print(evaluation_summary.to_string(index=False))
        else:
            print("\n⏭️ Step 5: Skipping evaluation (--skip-eval flag used)")
        
        print("\n" + "=" * 50)
        print("🎉 Quick Training Completed!")
        print("=" * 50)
        print("🚀 You can now start the API server with: python run_server.py")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
