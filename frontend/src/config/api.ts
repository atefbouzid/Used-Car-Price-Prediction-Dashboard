// API Configuration
export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  TIMEOUT: 30000, // 30 seconds
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // 1 second
}

// API Endpoints
export const API_ENDPOINTS = {
  HEALTH: '/health',
  AVAILABLE_MODELS: '/available_models',
  MODEL_INFO: '/model_info',
  PREDICT: '/predict',
} as const
