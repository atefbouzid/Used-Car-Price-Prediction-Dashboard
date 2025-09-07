"""
Preprocessing Service
Handles all data preprocessing including feature engineering
"""

import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Service for data preprocessing and feature engineering"""
    
    def __init__(self):
        """Initialize DataPreprocessor"""
        self.label_encoders = {}
        self.feature_names = None
        self.is_fitted = False
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering to the dataframe
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Calculate car age
        current_year = datetime.datetime.now().year
        df['age'] = df['model_year'].apply(lambda x: current_year - x)
        
        # Risk indicators
        df['risk_dead_engine'] = df['milage'].map(lambda x: 1 if x > 300000 else 0)
        
        # Overworked car indicator
        def overworked(row):
            if row['milage'] > 50000 and row['age'] < 1:
                return 1
            elif row['milage'] > 100000 and row['age'] < 2:
                return 1
            elif row['milage'] > 300000 and row['age'] < 10:
                return 1
            return 0
        
        # Fresh engine indicator
        def fresh_engine(row):
            if row['milage'] < 10000:
                return 1
            if row['milage'] < 30000 and row['age'] >= 2:
                return 1
            return 0
        
        df['overworked'] = df.apply(overworked, axis=1)
        df['fresh_engine'] = df.apply(fresh_engine, axis=1)
        
        # Extract engine specifications
        df['Cylinder'] = df['engine'].str.extract(r'(\d+)\s+Cylinder', expand=False).fillna(-1).astype(int)
        df['engine_Litr'] = df['engine'].str.extract(r'(\d+\.\d+)\s+L', expand=False).fillna(-1).astype(float)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Handle fuel_type missing values based on engine description
        mask = df['fuel_type'].isnull()
        df.loc[mask, 'fuel_type'] = df.loc[mask, 'engine'].map(
            lambda x: 'Plug-In Hybrid' if 'Plug-In' in str(x) else 
                     'Hybrid' if 'Hybrid' in str(x) else 
                     'Gasoline' if 'Gasoline' in str(x) else 
                     'E85 Flex Fuel' if 'Flex Fuel' in str(x) else  
                     'Diesel' if 'Diesel' in str(x) or 'GDI' in str(x) else 'Other'
        )
        
        # Handle clean_title missing values
        mask = df['clean_title'].isnull()
        df.loc[mask, 'clean_title'] = 'No'
        
        # Handle accident missing values
        mask = df['accident'].isnull()
        df.loc[mask, 'accident'] = 'None reported'
        
        return df
    
    def encode_categorical_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical features using label encoding
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame (optional)
            
        Returns:
            Tuple of (encoded_train_df, encoded_test_df)
        """
        train_df = train_df.copy()
        if test_df is not None:
            test_df = test_df.copy()
        
        # Get categorical columns
        categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if present
        if 'price' in categorical_cols:
            categorical_cols.remove('price')
        
        # Fit label encoders on training data
        for col in tqdm(categorical_cols, desc='Encoding categorical features'):
            if col in train_df.columns:
                # Fit encoder on training data
                le = LabelEncoder()
                train_df[col] = le.fit_transform(train_df[col].astype(str))
                
                # Store encoder for later use
                self.label_encoders[col] = le
                
                # Transform test data if provided
                if test_df is not None and col in test_df.columns:
                    # Handle unseen categories in test data
                    test_categories = set(test_df[col].astype(str).unique())
                    train_categories = set(le.classes_)
                    
                    # Map unseen categories to -1
                    test_df[col] = test_df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in train_categories else -1
                    )
        
        return train_df, test_df
    
    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame (optional)
            
        Returns:
            Tuple of (processed_train_df, processed_test_df)
        """
        print("🔄 Starting data preprocessing...")
        
        # Handle missing values
        train_df = self.handle_missing_values(train_df)
        if test_df is not None:
            test_df = self.handle_missing_values(test_df)
        
        # Apply feature engineering
        train_df = self.feature_engineering(train_df)
        if test_df is not None:
            test_df = self.feature_engineering(test_df)
        
        # Drop ID column if present
        if 'id' in train_df.columns:
            train_df = train_df.drop('id', axis=1)
        if test_df is not None and 'id' in test_df.columns:
            test_df = test_df.drop('id', axis=1)
        
        # One-hot encode accident feature
        train_df = pd.get_dummies(train_df, columns=['accident'])
        if test_df is not None:
            test_df = pd.get_dummies(test_df, columns=['accident'])
        
        # Ensure both datasets have the same columns
        if test_df is not None:
            # Add missing columns to test_df
            for col in train_df.columns:
                if col not in test_df.columns and col != 'price':
                    test_df[col] = 0
            
            # Remove extra columns from test_df
            test_df = test_df[train_df.columns.drop('price')]
        
        # Encode categorical features
        train_df, test_df = self.encode_categorical_features(train_df, test_df)
        
        # Store feature names for later use
        self.feature_names = [col for col in train_df.columns if col != 'price']
        self.is_fitted = True
        
        print("✅ Data preprocessing completed!")
        print(f"📊 Training data shape: {train_df.shape}")
        if test_df is not None:
            print(f"📊 Test data shape: {test_df.shape}")
        
        return train_df, test_df
    
    def preprocess_single_sample(self, car_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess a single car sample for prediction
        
        Args:
            car_data: Dictionary containing car features
            
        Returns:
            Preprocessed feature array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before preprocessing single samples")
        
        # Convert to DataFrame
        df = pd.DataFrame([car_data])
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Apply feature engineering
        df = self.feature_engineering(df)
        
        # One-hot encode accident
        df = pd.get_dummies(df, columns=['accident'])
        
        # Ensure all expected columns are present
        for col in self.feature_names:
            if col not in df.columns:
                if col.startswith('accident_'):
                    df[col] = 0
                else:
                    df[col] = 0
        
        # Select only the expected features
        df = df[self.feature_names]
        
        # Apply label encoding
        for col in self.feature_names:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories
                if df[col].iloc[0] not in le.classes_:
                    df[col] = -1
                else:
                    df[col] = le.transform([df[col].iloc[0]])[0]
        
        return df.values.reshape(1, -1)
    
    def save_preprocessor(self, filepath: str):
        """
        Save preprocessor state
        
        Args:
            filepath: Path to save the preprocessor
        """
        import joblib
        
        preprocessor_state = {
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_state, filepath)
        print(f"✅ Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """
        Load preprocessor state
        
        Args:
            filepath: Path to load the preprocessor from
        """
        import joblib
        
        preprocessor_state = joblib.load(filepath)
        self.label_encoders = preprocessor_state['label_encoders']
        self.feature_names = preprocessor_state['feature_names']
        self.is_fitted = preprocessor_state['is_fitted']
        
        print(f"✅ Preprocessor loaded from {filepath}")
    
    def get_feature_importance_names(self) -> List[str]:
        """
        Get feature names for model interpretation
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        return self.feature_names.copy()

# Global instance
data_preprocessor = DataPreprocessor()
