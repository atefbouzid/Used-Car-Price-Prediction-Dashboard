'use client'

import { ModelType, PredictionResult } from '@/types'
import { DollarSign, TrendingUp, Target, Brain, Zap, TreePine, Loader2, BarChart3 } from 'lucide-react'

interface PredictionResultsProps {
  predictions: PredictionResult[]
  selectedModels: ModelType[]
  isLoading: boolean
}

const modelInfo = {
  catboost: { name: 'CatBoost', color: 'text-yellow-400', bgColor: 'bg-yellow-500/10', icon: Target },
  lgbm: { name: 'LightGBM', color: 'text-green-400', bgColor: 'bg-green-500/10', icon: Zap },
  xgboost: { name: 'XGBoost', color: 'text-blue-400', bgColor: 'bg-blue-500/10', icon: Brain },
  randomforest: { name: 'Random Forest', color: 'text-purple-400', bgColor: 'bg-purple-500/10', icon: TreePine }
}

export default function PredictionResults({ predictions, selectedModels, isLoading }: PredictionResultsProps) {
  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(price)
  }

  const calculateEnsemble = () => {
    if (predictions.length === 0) return 0
    return predictions.reduce((sum, pred) => sum + pred.prediction, 0) / predictions.length
  }

  const ensemblePrice = calculateEnsemble()
  const showEnsemble = predictions.length > 1

  return (
    <div className="bg-dark-800 rounded-xl p-6 border border-dark-700">
      <h2 className="text-xl font-semibold text-white mb-6 flex items-center space-x-2">
        <BarChart3 className="w-5 h-5 text-primary-500" />
        <span>Prediction Results</span>
      </h2>

      {isLoading && (
        <div className="flex flex-col items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-primary-500 animate-spin mb-4" />
          <p className="text-dark-300 text-center">
            Analyzing vehicle data with selected models...
          </p>
        </div>
      )}

      {!isLoading && predictions.length === 0 && (
        <div className="text-center py-12">
          <DollarSign className="w-12 h-12 text-dark-500 mx-auto mb-4" />
          <p className="text-dark-400 text-lg mb-2">No predictions yet</p>
          <p className="text-dark-500 text-sm">
            Fill in the vehicle information and select models to get started
          </p>
        </div>
      )}

      {!isLoading && predictions.length > 0 && (
        <div className="space-y-4">
          {/* Ensemble Result (if multiple models) */}
          {showEnsemble && (
            <div className="bg-gradient-to-r from-orange-500/10 to-orange-600/10 rounded-lg p-4 border border-orange-500/30">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-orange-500 rounded-lg flex items-center justify-center">
                    <TrendingUp className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">Ensemble Average</h3>
                    <p className="text-orange-300 text-xs">{predictions.length} models combined</p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-orange-400">
                    {formatPrice(ensemblePrice)}
                  </div>
                  <div className="text-orange-300 text-xs">Recommended</div>
                </div>
              </div>
              <div className="w-full bg-dark-700 rounded-full h-2">
                <div className="bg-orange-500 h-2 rounded-full transition-all duration-1000 ease-out w-full" />
              </div>
            </div>
          )}

          {/* Individual Model Results */}
          <div className="space-y-3">
            {predictions.map((result) => {
              const model = modelInfo[result.model]
              const Icon = model.icon
              const confidencePercentage = Math.round(result.confidence * 100)
              
              return (
                <div
                  key={result.model}
                  className={`rounded-lg p-4 border border-dark-600 ${model.bgColor}`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center bg-dark-700`}>
                        <Icon className={`w-4 h-4 ${model.color}`} />
                      </div>
                      <div>
                        <h3 className="font-semibold text-white">{model.name}</h3>
                        <p className="text-dark-300 text-xs">
                          Confidence: {confidencePercentage}%
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-xl font-bold ${model.color}`}>
                        {formatPrice(result.prediction)}
                      </div>
                    </div>
                  </div>
                  
                  {/* Confidence Bar */}
                  <div className="w-full bg-dark-700 rounded-full h-1.5">
                    <div 
                      className={`h-1.5 rounded-full transition-all duration-1000 ease-out ${
                        result.model === 'catboost' ? 'bg-yellow-500' :
                        result.model === 'lgbm' ? 'bg-green-500' :
                        result.model === 'xgboost' ? 'bg-blue-500' :
                        'bg-purple-500'
                      }`}
                      style={{ width: `${confidencePercentage}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>

          {/* Price Range Analysis */}
          {predictions.length > 1 && (
            <div className="mt-6 p-4 bg-dark-900 rounded-lg border border-dark-600">
              <h4 className="text-sm font-medium text-dark-300 mb-3">Price Range Analysis</h4>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <p className="text-dark-400 text-xs">Minimum</p>
                  <p className="text-red-400 font-semibold">
                    {formatPrice(Math.min(...predictions.map(p => p.prediction)))}
                  </p>
                </div>
                <div>
                  <p className="text-dark-400 text-xs">Average</p>
                  <p className="text-orange-400 font-semibold">
                    {formatPrice(ensemblePrice)}
                  </p>
                </div>
                <div>
                  <p className="text-dark-400 text-xs">Maximum</p>
                  <p className="text-green-400 font-semibold">
                    {formatPrice(Math.max(...predictions.map(p => p.prediction)))}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Disclaimer */}
          <div className="mt-4 p-3 bg-dark-900 rounded-lg border border-dark-600">
            <p className="text-dark-400 text-xs">
              <strong>Disclaimer:</strong> These predictions are estimates based on machine learning models. 
              Actual market prices may vary due to condition, location, demand, and other factors.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
