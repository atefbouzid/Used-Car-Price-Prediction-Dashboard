export type ModelType = 'catboost' | 'lgbm' | 'xgboost' | 'randomforest'

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
  confidence: number
}

export interface ModelInfo {
  id: ModelType
  name: string
  description: string
  color: string
}
