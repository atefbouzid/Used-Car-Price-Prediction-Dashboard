'use client'

import { useState, useEffect } from 'react'
import { ModelType, ModelInfo, AvailableModelsResponse } from '@/types'
import { Check, Brain, Zap, Target, TreePine, TrendingUp, AlertCircle, Loader2 } from 'lucide-react'
import { apiService } from '@/lib/api'

const modelInfo: ModelInfo[] = [
  {
    id: 'catboost',
    name: 'CatBoost',
    description: 'Gradient boosting with categorical feature support',
    color: 'bg-yellow-500'
  },
  {
    id: 'lightgbm',
    name: 'LightGBM',
    description: 'Fast gradient boosting framework',
    color: 'bg-green-500'
  },
  {
    id: 'xgboost',
    name: 'XGBoost',
    description: 'Extreme gradient boosting algorithm',
    color: 'bg-blue-500'
  },
  {
    id: 'random_forest',
    name: 'Random Forest',
    description: 'Ensemble of decision trees',
    color: 'bg-purple-500'
  }
]

const modelIcons = {
  catboost: Target,
  lightgbm: Zap,
  xgboost: Brain,
  random_forest: TreePine
}

interface ModelSelectorProps {
  selectedModels: ModelType[]
  onModelChange: (models: ModelType[]) => void
  useEnsemble: boolean
  onEnsembleChange: (useEnsemble: boolean) => void
}

export default function ModelSelector({ 
  selectedModels, 
  onModelChange, 
  useEnsemble, 
  onEnsembleChange 
}: ModelSelectorProps) {
  const [availableModels, setAvailableModels] = useState<ModelType[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Fetch available models from backend
  useEffect(() => {
    const fetchAvailableModels = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const response = await apiService.getAvailableModels()
        setAvailableModels(response.available_models)
        
        // If no models are currently selected, select the first available model
        if (selectedModels.length === 0 && response.available_models.length > 0) {
          onModelChange([response.available_models[0]])
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch available models')
        console.error('Error fetching available models:', err)
      } finally {
        setIsLoading(false)
      }
    }

    fetchAvailableModels()
  }, [])

  const handleModelToggle = (modelId: ModelType) => {
    const newSelection = selectedModels.includes(modelId)
      ? selectedModels.filter(id => id !== modelId)
      : [...selectedModels, modelId]
    
    onModelChange(newSelection)
    
    // Enable ensemble if multiple models selected
    if (newSelection.length > 1) {
      onEnsembleChange(true)
    } else {
      onEnsembleChange(false)
    }
  }

  const handleEnsembleToggle = () => {
    onEnsembleChange(!useEnsemble)
  }

  if (isLoading) {
    return (
      <div className="bg-dark-800 rounded-xl p-6 border border-dark-700">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center space-x-2">
          <Brain className="w-5 h-5 text-primary-500" />
          <span>Model Selection</span>
        </h2>
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 text-primary-500 animate-spin mr-2" />
          <span className="text-dark-300">Loading available models...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-dark-800 rounded-xl p-6 border border-dark-700">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center space-x-2">
          <Brain className="w-5 h-5 text-primary-500" />
          <span>Model Selection</span>
        </h2>
        <div className="flex items-center justify-center py-8">
          <AlertCircle className="w-6 h-6 text-red-500 mr-2" />
          <div className="text-center">
            <p className="text-red-400 font-medium">Failed to load models</p>
            <p className="text-dark-400 text-sm mt-1">{error}</p>
            <button 
              onClick={() => window.location.reload()} 
              className="mt-2 px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-dark-800 rounded-xl p-6 border border-dark-700">
      <h2 className="text-xl font-semibold text-white mb-4 flex items-center space-x-2">
        <Brain className="w-5 h-5 text-primary-500" />
        <span>Model Selection</span>
        <span className="text-sm text-dark-400">({availableModels.length} available)</span>
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {modelInfo
          .filter(model => availableModels.includes(model.id))
          .map((model) => {
            const Icon = modelIcons[model.id]
            const isSelected = selectedModels.includes(model.id)
            
            return (
              <div
                key={model.id}
                onClick={() => handleModelToggle(model.id)}
                className={`
                  p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 hover:shadow-lg
                  ${isSelected 
                    ? 'border-primary-500 bg-primary-500/10' 
                    : 'border-dark-600 bg-dark-900 hover:border-dark-500'
                  }
                `}
              >
                <div className="flex items-start space-x-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${model.color}`}>
                    <Icon className="w-5 h-5 text-white" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <h3 className="font-semibold text-white">{model.name}</h3>
                      {isSelected && (
                        <div className="w-5 h-5 bg-primary-500 rounded-full flex items-center justify-center">
                          <Check className="w-3 h-3 text-white" />
                        </div>
                      )}
                    </div>
                    <p className="text-dark-300 text-sm mt-1">{model.description}</p>
                  </div>
                </div>
              </div>
            )
          })}
      </div>

      {/* Ensemble Option */}
      {selectedModels.length > 1 && (
        <div className="border-t border-dark-700 pt-4">
          <div
            onClick={handleEnsembleToggle}
            className={`
              p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 hover:shadow-lg
              ${useEnsemble 
                ? 'border-orange-500 bg-orange-500/10' 
                : 'border-dark-600 bg-dark-900 hover:border-dark-500'
              }
            `}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-orange-500 rounded-lg flex items-center justify-center">
                  <TrendingUp className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold text-white">Ensemble Prediction</h3>
                  <p className="text-dark-300 text-sm">Average predictions from selected models</p>
                </div>
              </div>
              {useEnsemble && (
                <div className="w-5 h-5 bg-orange-500 rounded-full flex items-center justify-center">
                  <Check className="w-3 h-3 text-white" />
                </div>
              )}
            </div>
          </div>
          <p className="text-dark-400 text-xs mt-2 px-2">
            Ensemble averaging often provides more robust predictions by combining multiple models
          </p>
        </div>
      )}
    </div>
  )
}
