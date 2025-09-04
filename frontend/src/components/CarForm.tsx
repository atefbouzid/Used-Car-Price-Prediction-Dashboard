'use client'

import { CarFeatures } from '@/types'
import { Car, Calendar, Gauge, Fuel, Settings, Palette, Shield, CheckCircle, Loader2 } from 'lucide-react'

interface CarFormProps {
  features: CarFeatures
  onFeatureChange: (features: CarFeatures) => void
  onPredict: () => void
  isLoading: boolean
}

// Sample data based on the dataset
const brands = [
  'MINI', 'Lincoln', 'Chevrolet', 'Genesis', 'Mercedes-Benz', 'Audi', 'Ford', 
  'BMW', 'Toyota', 'Honda', 'Nissan', 'Hyundai', 'Kia', 'Volkswagen', 'Subaru',
  'Mazda', 'Infiniti', 'Acura', 'Lexus', 'Cadillac', 'Buick', 'GMC', 'Ram',
  'Jeep', 'Dodge', 'Chrysler', 'Volvo', 'Jaguar', 'Land Rover', 'Porsche'
]

const fuelTypes = [
  'Gasoline', 'E85 Flex Fuel', 'Hybrid', 'Electric', 'Diesel', 'Plug-in Hybrid'
]

const transmissionTypes = [
  'A/T', 'M/T', '7-Speed A/T', '8-Speed A/T', '9-Speed A/T', '10-Speed Automatic',
  'Transmission w/Dual Shift Mode', 'CVT', '6-Speed Manual', '5-Speed Manual'
]

const exteriorColors = [
  'Black', 'White', 'Silver', 'Gray', 'Blue', 'Red', 'Yellow', 'Green', 'Brown', 'Orange', 'Purple', 'Gold'
]

const interiorColors = [
  'Black', 'Gray', 'Beige', 'Brown', 'Tan', 'White', 'Red', 'Blue', '–'
]

const accidentOptions = [
  'None reported', 'At least 1 accident or damage reported'
]

const cleanTitleOptions = [
  'Yes', 'No'
]

export default function CarForm({ features, onFeatureChange, onPredict, isLoading }: CarFormProps) {
  const handleInputChange = (field: keyof CarFeatures, value: string | number) => {
    onFeatureChange({
      ...features,
      [field]: value
    })
  }

  const isFormValid = () => {
    return features.brand && features.model && features.model_year && 
           features.milage >= 0 && features.fuel_type && features.engine &&
           features.transmission && features.ext_col && features.int_col
  }

  return (
    <div className="bg-dark-800 rounded-xl p-6 border border-dark-700">
      <h2 className="text-xl font-semibold text-white mb-6 flex items-center space-x-2">
        <Car className="w-5 h-5 text-primary-500" />
        <span>Vehicle Information</span>
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Brand */}
        <div>
          <label htmlFor="brand" className="block text-sm font-medium text-dark-300 mb-2">
            Brand *
          </label>
          <select
            id="brand"
            value={features.brand}
            onChange={(e) => handleInputChange('brand', e.target.value)}
            className="w-full px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">Select Brand</option>
            {brands.map(brand => (
              <option key={brand} value={brand}>{brand}</option>
            ))}
          </select>
        </div>

        {/* Model */}
        <div>
          <label htmlFor="model" className="block text-sm font-medium text-dark-300 mb-2">
            Model *
          </label>
          <input
            id="model"
            type="text"
            value={features.model}
            onChange={(e) => handleInputChange('model', e.target.value)}
            placeholder="e.g., Cooper S Base"
            className="w-full px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Model Year */}
        <div>
          <label htmlFor="model_year" className="block text-sm font-medium text-dark-300 mb-2 flex items-center space-x-1">
            <Calendar className="w-4 h-4" />
            <span>Model Year *</span>
          </label>
          <input
            id="model_year"
            type="number"
            value={features.model_year}
            onChange={(e) => handleInputChange('model_year', parseInt(e.target.value) || 0)}
            min="1990"
            max={new Date().getFullYear()}
            className="w-full px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Mileage */}
        <div>
          <label htmlFor="mileage" className="block text-sm font-medium text-dark-300 mb-2 flex items-center space-x-1">
            <Gauge className="w-4 h-4" />
            <span>Mileage *</span>
          </label>
          <input
            id="mileage"
            type="number"
            value={features.milage}
            onChange={(e) => handleInputChange('milage', parseInt(e.target.value) || 0)}
            min="0"
            placeholder="Miles driven"
            className="w-full px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Fuel Type */}
        <div>
          <label htmlFor="fuel_type" className="block text-sm font-medium text-dark-300 mb-2 flex items-center space-x-1">
            <Fuel className="w-4 h-4" />
            <span>Fuel Type *</span>
          </label>
          <select
            id="fuel_type"
            value={features.fuel_type}
            onChange={(e) => handleInputChange('fuel_type', e.target.value)}
            className="w-full px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">Select Fuel Type</option>
            {fuelTypes.map(fuel => (
              <option key={fuel} value={fuel}>{fuel}</option>
            ))}
          </select>
        </div>

        {/* Engine */}
        <div>
          <label htmlFor="engine" className="block text-sm font-medium text-dark-300 mb-2 flex items-center space-x-1">
            <Settings className="w-4 h-4" />
            <span>Engine *</span>
          </label>
          <input
            id="engine"
            type="text"
            value={features.engine}
            onChange={(e) => handleInputChange('engine', e.target.value)}
            placeholder="e.g., 172.0HP 1.6L 4 Cylinder Engine"
            className="w-full px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-white placeholder-dark-400 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Transmission */}
        <div>
          <label htmlFor="transmission" className="block text-sm font-medium text-dark-300 mb-2">
            Transmission *
          </label>
          <select
            id="transmission"
            value={features.transmission}
            onChange={(e) => handleInputChange('transmission', e.target.value)}
            className="w-full px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">Select Transmission</option>
            {transmissionTypes.map(trans => (
              <option key={trans} value={trans}>{trans}</option>
            ))}
          </select>
        </div>

        {/* Exterior Color */}
        <div>
          <label htmlFor="ext_col" className="block text-sm font-medium text-dark-300 mb-2 flex items-center space-x-1">
            <Palette className="w-4 h-4" />
            <span>Exterior Color *</span>
          </label>
          <select
            id="ext_col"
            value={features.ext_col}
            onChange={(e) => handleInputChange('ext_col', e.target.value)}
            className="w-full px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">Select Color</option>
            {exteriorColors.map(color => (
              <option key={color} value={color}>{color}</option>
            ))}
          </select>
        </div>

        {/* Interior Color */}
        <div>
          <label htmlFor="int_col" className="block text-sm font-medium text-dark-300 mb-2">
            Interior Color *
          </label>
          <select
            id="int_col"
            value={features.int_col}
            onChange={(e) => handleInputChange('int_col', e.target.value)}
            className="w-full px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="">Select Color</option>
            {interiorColors.map(color => (
              <option key={color} value={color}>{color}</option>
            ))}
          </select>
        </div>

        {/* Accident History */}
        <div>
          <label htmlFor="accident" className="block text-sm font-medium text-dark-300 mb-2 flex items-center space-x-1">
            <Shield className="w-4 h-4" />
            <span>Accident History</span>
          </label>
          <select
            id="accident"
            value={features.accident}
            onChange={(e) => handleInputChange('accident', e.target.value)}
            className="w-full px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            {accidentOptions.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>

        {/* Clean Title */}
        <div>
          <label htmlFor="clean_title" className="block text-sm font-medium text-dark-300 mb-2 flex items-center space-x-1">
            <CheckCircle className="w-4 h-4" />
            <span>Clean Title</span>
          </label>
          <select
            id="clean_title"
            value={features.clean_title}
            onChange={(e) => handleInputChange('clean_title', e.target.value)}
            className="w-full px-4 py-3 bg-dark-900 border border-dark-600 rounded-lg text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            {cleanTitleOptions.map(option => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Predict Button */}
      <div className="mt-8">
        <button
          onClick={onPredict}
          disabled={!isFormValid() || isLoading}
          className={`
            w-full py-4 px-6 rounded-lg font-medium text-white transition-all duration-200 flex items-center justify-center space-x-2
            ${isFormValid() && !isLoading
              ? 'bg-primary-600 hover:bg-primary-700 focus:ring-4 focus:ring-primary-500/50' 
              : 'bg-dark-600 cursor-not-allowed'
            }
          `}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Predicting...</span>
            </>
          ) : (
            <>
              <Settings className="w-5 h-5" />
              <span>Predict Price</span>
            </>
          )}
        </button>
      </div>
    </div>
  )
}
