# Used Car Price Prediction Dashboard

A professional Next.js dashboard for predicting used car prices using multiple machine learning models.

## Features

- **Multi-Model Predictions**: Support for CatBoost, LightGBM, XGBoost, and Random Forest models
- **Ensemble Averaging**: Combine predictions from multiple models for more robust estimates
- **Professional Dark Theme**: Modern, responsive design optimized for professional use
- **Comprehensive Input Form**: All necessary vehicle features for accurate predictions
- **Real-time Results**: Instant prediction updates with confidence scores

## Tech Stack

- **Frontend**: Next.js 14, React 18, TypeScript
- **Styling**: Tailwind CSS with custom dark theme
- **Icons**: Lucide React
- **Deployment**: Ready for Vercel, Netlify, or any static host

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Run the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

## Usage

1. **Select Models**: Choose one or more ML models for prediction
2. **Enter Vehicle Information**: Fill in all required vehicle details
3. **Get Predictions**: Click "Predict Price" to see results
4. **Ensemble Results**: When multiple models are selected, view both individual and averaged predictions

## Model Information

- **CatBoost**: Gradient boosting with categorical feature support
- **LightGBM**: Fast gradient boosting framework  
- **XGBoost**: Extreme gradient boosting algorithm
- **Random Forest**: Ensemble of decision trees

## Backend Integration

This frontend is designed to integrate with a backend API. Update the prediction logic in `src/app/page.tsx` to connect to your ML model endpoints.

## Development

### Project Structure

```
src/
├── app/                 # Next.js app directory
├── components/          # React components
├── lib/                # Utility functions
└── types/              # TypeScript type definitions
```

### Building for Production

```bash
npm run build
npm start
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.
