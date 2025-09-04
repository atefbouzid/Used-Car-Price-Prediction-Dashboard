'use client'

import { useState } from 'react'
import Header from '@/components/Header'
import ModelSelector from '@/components/ModelSelector'
import CarForm from '@/components/CarForm'
import PredictionResults from '@/components/PredictionResults'
import { ModelType, CarFeatures, PredictionResult } from '@/types'

export default function Home() {
  const [selectedModels, setSelectedModels] = useState<ModelType[]>(['catboost'])
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
  const [predictions, setPredictions] = useState<PredictionResult[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const handleModelChange = (models: ModelType[]) => {
    setSelectedModels(models)
  }

  const handleFeatureChange = (features: CarFeatures) => {
    setCarFeatures(features)
  }

  const handlePredict = async () => {
    setIsLoading(true)
    
    // Simulate API call - replace with actual backend call later
    setTimeout(() => {
      const mockPredictions: PredictionResult[] = selectedModels.map(model => ({
        model,
        prediction: Math.floor(Math.random() * 50000) + 15000,
        confidence: Math.random() * 0.3 + 0.7
      }))
      
      setPredictions(mockPredictions)
      setIsLoading(false)
    }, 2000)
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
              predictions={predictions}
              selectedModels={selectedModels}
              isLoading={isLoading}
            />
          </div>
        </div>
      </main>
    </div>
  )
}
