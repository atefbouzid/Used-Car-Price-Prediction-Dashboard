export type ModelType = 'catboost' | 'lightgbm' | 'xgboost' | 'random_forest'

export interface CarFeatures {
  brand: string
  model: string
  model_year: number
  milage: number
  fuel_type: string
  engine: string
  transmission: string
  ext_col: string
  int_col: string
  accident: string
  clean_title: string
}

export interface PredictionResult {
  model: ModelType
  prediction: number
}

export interface ModelInfo {
  id: ModelType
  name: string
  description: string
  color: string
}

// Backend API Types
export interface ApiResponse<T> {
  data?: T
  error?: string
  message?: string
}

export interface PredictionRequest {
  car_data: CarFeatures
  selected_models: ModelType[]
  use_ensemble: boolean
}

export interface PredictionResponse {
  individual_predictions: Record<ModelType, number>
  selected_models: ModelType[]
  ensemble_prediction?: number
}

export interface ModelInfoResponse {
  model_name: string
  model_type: string
  is_fitted: boolean
  has_predict: boolean
  n_estimators?: number
  max_depth?: number
  learning_rate?: number
}

export interface AvailableModelsResponse {
  available_models: ModelType[]
  models_count: number
  models_info: Record<ModelType, ModelInfoResponse>
}

export interface HealthResponse {
  status: 'success' | 'error'
  message: string
  models_loaded: boolean
  available_models?: ModelType[]
  models_count?: number
}
