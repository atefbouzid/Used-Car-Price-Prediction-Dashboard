# Used Car Price Prediction - Backend API

A comprehensive machine learning backend for used car price prediction with multiple models and ensemble capabilities.

## 🏗️ Architecture

The backend is organized into modular services:

- **`data_loader.py`** - Data loading and directory management
- **`eda.py`** - Exploratory Data Analysis with visualizations
- **`preprocessing.py`** - Data preprocessing and feature engineering
- **`hyperparameter_tuning.py`** - Optuna-based hyperparameter optimization
- **`train.py`** - Model training and management
- **`evaluate.py`** - Model evaluation and performance analysis
- **`predict.py`** - Prediction service for API endpoints

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Training Options

#### Option A: Complete Pipeline (Recommended for first run)
```bash
# Interactive mode - asks for hyperparameter tuning preference
python train_pipeline.py --interactive

# Use existing hyperparameters (fast)
python train_pipeline.py --no-tune

# Run hyperparameter tuning (slow but optimal)
python train_pipeline.py --tune
```

#### Option B: Quick Training (Fast iteration)
```bash
# Train models with existing hyperparameters
python quick_train.py

# Skip evaluation for even faster training
python quick_train.py --skip-eval
```

### 3. Start API Server

```bash
python run_server.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📊 Training Pipeline Options

### Full Pipeline (`train_pipeline.py`)

**Features:**
- Complete EDA with visualizations
- Data preprocessing and feature engineering
- Optional hyperparameter tuning
- Model training and evaluation
- Comprehensive reporting

**Usage:**
```bash
# Default: Use existing hyperparameters
python train_pipeline.py

# Interactive mode
python train_pipeline.py --interactive

# Force hyperparameter tuning
python train_pipeline.py --tune

# Skip hyperparameter tuning
python train_pipeline.py --no-tune
```

### Quick Training (`quick_train.py`)

**Features:**
- Fast model training
- Uses existing hyperparameters
- Optional evaluation
- Perfect for rapid iterations

**Usage:**
```bash
# Full quick training
python quick_train.py

# Skip evaluation for maximum speed
python quick_train.py --skip-eval
```

## 🎯 Hyperparameter Management

The system intelligently manages hyperparameters:

1. **Existing Parameters**: If `model_params/best_hyperparameters.json` exists, it will be used by default
2. **New Optimization**: Use `--tune` flag to run fresh hyperparameter optimization
3. **Fallback**: If no existing parameters are found, default parameters are used

### File Structure:
```
model_params/
└── best_hyperparameters.json  # Optimized parameters from previous runs
```

## 🔧 API Endpoints

### Core Endpoints

- **`GET /`** - API information and available endpoints
- **`GET /health`** - Health check with model status
- **`GET /available_models`** - List all available trained models
- **`POST /predict`** - Make car price predictions
- **`GET /model_info/{model_name}`** - Get detailed model information

### Prediction Request Format

```json
{
  "car_data": {
    "brand": "Toyota",
    "model": "Camry",
    "model_year": 2020,
    "milage": 50000,
    "fuel_type": "Gasoline",
    "engine": "2.5L 4-Cylinder",
    "transmission": "Automatic",
    "ext_col": "White",
    "int_col": "Black",
    "accident": "None reported",
    "clean_title": "Yes"
  },
  "selected_models": ["xgboost", "catboost", "lightgbm"],
  "use_ensemble": true
}
```

### Response Format

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

## 📈 Model Performance

The system includes comprehensive evaluation metrics:

- **RMSE** - Root Mean Square Error
- **MAE** - Mean Absolute Error
- **R²** - Coefficient of Determination
- **MAPE** - Mean Absolute Percentage Error
- **NRMSE** - Normalized RMSE
- **MedAE** - Median Absolute Error

## 📁 Output Files

After training, the following files are generated:

```
backend/
├── models/                          # Trained model files
│   ├── CatBoostRegressor.pkl
│   ├── XGBRegressor.pkl
│   ├── LGBMRegressor.pkl
│   └── RandomForestRegressor.pkl
├── imgs/                           # EDA and evaluation plots
│   ├── missing_values_analysis.png
│   ├── target_distribution.png
│   ├── correlation_matrix.png
│   ├── predictions_vs_actual.png
│   └── feature_importance.png
├── model_params/                   # Hyperparameters
│   └── best_hyperparameters.json
├── preprocessor.pkl                # Preprocessing state
├── training_summary.csv            # Training results
└── evaluation_summary.csv          # Evaluation results
```

## 🔄 Workflow Recommendations

### First Time Setup
1. Run complete pipeline with hyperparameter tuning:
   ```bash
   python train_pipeline.py --tune
   ```

### Regular Development
1. Use quick training for fast iterations:
   ```bash
   python quick_train.py
   ```

### Production Deployment
1. Use existing hyperparameters:
   ```bash
   python train_pipeline.py --no-tune
   ```

## 🛠️ Troubleshooting

### Common Issues

1. **No models found**: Run training pipeline first
2. **Hyperparameter file not found**: Use `--tune` flag or run full pipeline
3. **Memory issues**: Use `--skip-eval` flag in quick training
4. **Import errors**: Ensure all dependencies are installed

### Performance Tips

- Use `quick_train.py` for rapid model iterations
- Skip evaluation with `--skip-eval` for faster training
- Use existing hyperparameters to avoid expensive optimization
- Monitor memory usage during hyperparameter tuning

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs (when server is running)
- **Model Performance**: Check `evaluation_summary.csv` after training
- **Feature Importance**: View plots in `imgs/feature_importance.png`
- **Data Insights**: Explore EDA plots in `imgs/` directory
