import { Car, TrendingUp } from 'lucide-react'

export default function Header() {
  return (
    <header className="bg-dark-900/80 backdrop-blur-sm border-b border-dark-700">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center space-x-3">
          <div className="flex items-center justify-center w-12 h-12 bg-primary-600 rounded-lg">
            <Car className="w-7 h-7 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">
              Used Car Price Prediction
            </h1>
            <p className="text-dark-400 text-sm flex items-center space-x-1">
              <TrendingUp className="w-4 h-4" />
              <span>Professional ML-powered pricing dashboard</span>
            </p>
          </div>
        </div>
      </div>
    </header>
  )
}
