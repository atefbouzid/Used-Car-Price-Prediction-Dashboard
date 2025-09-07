#!/usr/bin/env python3
"""
Complete Training Pipeline
Runs the entire ML pipeline from data loading to model training
"""

import sys
import os
import argparse
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.data_loader import data_loader
from services.eda import eda_analyzer
from services.preprocessing import data_preprocessor
from services.hyperparameter_tuning import hyperparameter_tuner
from services.train import model_trainer
from services.evaluate import model_evaluator
from sklearn.model_selection import train_test_split

def main():
    """Main training pipeline"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ML Training Pipeline')
    parser.add_argument('--tune', action='store_true', 
                       help='Run hyperparameter tuning (default: use existing parameters)')
    parser.add_argument('--no-tune', action='store_true', 
                       help='Skip hyperparameter tuning and use existing parameters')
    parser.add_argument('--interactive', action='store_true', 
                       help='Ask user for hyperparameter tuning preference')
    
    args = parser.parse_args()
    
    print("🚀 Starting Complete ML Training Pipeline")
    print("=" * 60)
    
    # Determine hyperparameter tuning preference
    if args.interactive:
        # Ask user if they want to run hyperparameter tuning
        print("\n🎯 Hyperparameter Tuning Option:")
        print("   - 'yes' or 'y': Run hyperparameter optimization (takes time)")
        print("   - 'no' or 'n': Use existing optimized parameters from model_params/")
        
        while True:
            user_input = input("\nDo you want to run hyperparameter tuning? (yes/no): ").lower().strip()
            if user_input in ['yes', 'y', 'no', 'n']:
                run_hyperparameter_tuning = user_input in ['yes', 'y']
                break
            else:
                print("Please enter 'yes' or 'no'")
    else:
        # Use command line arguments
        if args.tune and args.no_tune:
            print("❌ Error: Cannot specify both --tune and --no-tune")
            sys.exit(1)
        elif args.tune:
            run_hyperparameter_tuning = True
        elif args.no_tune:
            run_hyperparameter_tuning = False
        else:
            # Default: use existing parameters
            run_hyperparameter_tuning = False
    
    try:
        # Step 1: Load Data
        print("\n📂 Step 1: Loading Data...")
        train_data, test_data, sample_submission = data_loader.load_all_data()
        print(f"✅ Loaded training data: {train_data.shape}")
        print(f"✅ Loaded test data: {test_data.shape}")
        
        # Step 2: Exploratory Data Analysis
        print("\n🔍 Step 2: Exploratory Data Analysis...")
        eda_summary = eda_analyzer.run_full_eda(train_data, target_col='price')
        print("✅ EDA completed - visualizations saved to imgs/")
        
        # Step 3: Data Preprocessing
        print("\n🔄 Step 3: Data Preprocessing...")
        train_processed, test_processed = data_preprocessor.preprocess_data(train_data, test_data)
        
        # Save preprocessor state
        data_preprocessor.save_preprocessor("preprocessor.pkl")
        print("✅ Preprocessing completed and saved")
        
        # Step 4: Train-Validation Split
        print("\n✂️ Step 4: Train-Validation Split...")
        X = train_processed.drop('price', axis=1)
        y = train_processed['price']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"✅ Training set: {X_train.shape}")
        print(f"✅ Validation set: {X_val.shape}")
        
        # Step 5: Hyperparameter Optimization (Optional)
        if run_hyperparameter_tuning:
            print("\n🎯 Step 5: Hyperparameter Optimization...")
            print("This may take a while...")
            optimization_results = hyperparameter_tuner.optimize_all_models(X_train, y_train)
            print("✅ Hyperparameter optimization completed")
        else:
            print("\n🎯 Step 5: Using Existing Hyperparameters...")
            try:
                # Try to load existing parameters
                existing_params = hyperparameter_tuner.load_optimization_results()
                print("✅ Loaded existing hyperparameters from model_params/")
                print("📊 Available optimized models:", list(existing_params.keys()))
            except FileNotFoundError:
                print("⚠️ No existing hyperparameters found. Using default parameters.")
                existing_params = None
        
        # Step 6: Model Training
        print("\n🏋️ Step 6: Model Training...")
        training_results = model_trainer.train_all_models(
            X_train, y_train, X_val, y_val, use_optimized_params=not run_hyperparameter_tuning
        )
        
        # Save trained models
        model_trainer.save_models()
        print("✅ Model training completed and models saved")
        
        # Step 7: Model Evaluation
        print("\n📊 Step 7: Model Evaluation...")
        evaluation_results = model_evaluator.run_full_evaluation(
            model_trainer.trained_models, X_val, y_val, 
            feature_names=data_preprocessor.get_feature_importance_names()
        )
        print("✅ Model evaluation completed - plots saved to imgs/")
        
        # Step 8: Generate Summary Report
        print("\n📋 Step 8: Training Summary...")
        training_summary = model_trainer.get_training_summary()
        evaluation_summary = evaluation_results['report']
        
        print("\n🏆 Training Results:")
        print(training_summary.to_string(index=False))
        
        print("\n📈 Evaluation Results:")
        print(evaluation_summary.to_string(index=False))
        
        # Save summaries
        training_summary.to_csv("training_summary.csv", index=False)
        evaluation_summary.to_csv("evaluation_summary.csv", index=False)
        print("✅ Summaries saved to CSV files")
        
        print("\n" + "=" * 60)
        print("🎉 Training Pipeline Completed Successfully!")
        print("=" * 60)
        print("📁 Files created:")
        print("   - models/ (trained models)")
        print("   - imgs/ (EDA and evaluation plots)")
        print("   - model_params/ (hyperparameters)")
        print("   - preprocessor.pkl (preprocessing state)")
        print("   - training_summary.csv")
        print("   - evaluation_summary.csv")
        print("\n🚀 You can now start the API server with: python run_server.py")
        
        print("\n" + "=" * 60)
        print("📖 Usage Examples:")
        print("=" * 60)
        print("python train_pipeline.py                    # Use existing parameters (default)")
        print("python train_pipeline.py --no-tune          # Use existing parameters")
        print("python train_pipeline.py --tune             # Run hyperparameter tuning")
        print("python train_pipeline.py --interactive      # Ask for user input")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
