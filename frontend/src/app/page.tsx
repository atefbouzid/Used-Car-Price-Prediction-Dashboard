'use client'

import { useState } from 'react'
import Header from '@/components/Header'
import ModelSelector from '@/components/ModelSelector'
import CarForm from '@/components/CarForm'
import PredictionResults from '@/components/PredictionResults'
import { ModelType, CarFeatures, PredictionResponse, PredictionRequest } from '@/types'
import { apiService } from '@/lib/api'

export default function Home() {
  const [selectedModels, setSelectedModels] = useState<ModelType[]>([])
  const [useEnsemble, setUseEnsemble] = useState(false)
  const [carFeatures, setCarFeatures] = useState<CarFeatures>({
    brand: '',
    model: '',
    model_year: new Date().getFullYear(),
    milage: 0,
    fuel_type: '',
    engine: '',
    transmission: '',
    ext_col: '',
    int_col: '',
    accident: 'None reported',
    clean_title: 'Yes'
  })
  const [predictionResponse, setPredictionResponse] = useState<PredictionResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleModelChange = (models: ModelType[]) => {
    setSelectedModels(models)
  }

  const handleEnsembleChange = (ensemble: boolean) => {
    setUseEnsemble(ensemble)
  }

  const handleFeatureChange = (features: CarFeatures) => {
    setCarFeatures(features)
  }

  const handlePredict = async () => {
    if (selectedModels.length === 0) {
      setError('Please select at least one model')
      return
    }

    // Validate required fields
    const requiredFields = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col']
    const missingFields = requiredFields.filter(field => !carFeatures[field as keyof CarFeatures])
    
    if (missingFields.length > 0) {
      setError(`Please fill in all required fields: ${missingFields.join(', ')}`)
      return
    }

    setIsLoading(true)
    setError(null)
    setPredictionResponse(null)

    try {
      const request: PredictionRequest = {
        car_data: carFeatures,
        selected_models: selectedModels,
        use_ensemble: useEnsemble
      }

      const response = await apiService.predictPrice(request)
      setPredictionResponse(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
      console.error('Prediction error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen">
      <Header />
      
      <main className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Panel - Configuration */}
          <div className="lg:col-span-2 space-y-6">
            <ModelSelector 
              selectedModels={selectedModels}
              onModelChange={handleModelChange}
              useEnsemble={useEnsemble}
              onEnsembleChange={handleEnsembleChange}
            />
            
            <CarForm 
              features={carFeatures}
              onFeatureChange={handleFeatureChange}
              onPredict={handlePredict}
              isLoading={isLoading}
            />
          </div>
          
          {/* Right Panel - Results */}
          <div className="lg:col-span-1">
            <PredictionResults 
              predictionResponse={predictionResponse}
              selectedModels={selectedModels}
              isLoading={isLoading}
              error={error}
            />
          </div>
        </div>
      </main>
    </div>
  )
}
