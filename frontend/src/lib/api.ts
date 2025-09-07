import { 
  PredictionRequest, 
  PredictionResponse, 
  AvailableModelsResponse, 
  HealthResponse,
  ModelInfoResponse,
  ModelType
} from '@/types'
import { API_CONFIG, API_ENDPOINTS } from '@/config/api'

class ApiService {
  private baseUrl: string

  constructor(baseUrl: string = API_CONFIG.BASE_URL) {
    this.baseUrl = baseUrl
  }

  private async request<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    }

    const response = await fetch(url, { ...defaultOptions, ...options })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(
        errorData.detail || 
        errorData.message || 
        `HTTP error! status: ${response.status}`
      )
    }

    return response.json()
  }

  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>(API_ENDPOINTS.HEALTH)
  }

  async getAvailableModels(): Promise<AvailableModelsResponse> {
    return this.request<AvailableModelsResponse>(API_ENDPOINTS.AVAILABLE_MODELS)
  }

  async getModelInfo(modelName: ModelType): Promise<ModelInfoResponse> {
    return this.request<ModelInfoResponse>(`${API_ENDPOINTS.MODEL_INFO}/${modelName}`)
  }

  async predictPrice(request: PredictionRequest): Promise<PredictionResponse> {
    return this.request<PredictionResponse>(API_ENDPOINTS.PREDICT, {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  async checkConnection(): Promise<boolean> {
    try {
      await this.getHealth()
      return true
    } catch (error) {
      console.error('API connection failed:', error)
      return false
    }
  }
}

// Export singleton instance
export const apiService = new ApiService()

// Export class for testing
export { ApiService }
