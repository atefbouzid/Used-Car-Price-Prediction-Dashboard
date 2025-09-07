"""
Data Loading Service
Handles loading of datasets and creation of necessary directories
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

class DataLoader:
    """Service for loading and managing datasets"""
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize DataLoader
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories for the project"""
        directories = [
            "models",
            "imgs",
            "submission", 
            "model_params",
            "catboost_info"
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            dir_path.mkdir(exist_ok=True)
            print(f"✓ Created directory: {directory}")
    
    def load_train_data(self) -> pd.DataFrame:
        """
        Load training dataset
        
        Returns:
            Training DataFrame
        """
        train_path = self.data_dir / "train.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        train_data = pd.read_csv(train_path)
        print(f"✓ Loaded training data: {train_data.shape}")
        return train_data
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load test dataset
        
        Returns:
            Test DataFrame
        """
        test_path = self.data_dir / "test.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found at {test_path}")
        
        test_data = pd.read_csv(test_path)
        print(f"✓ Loaded test data: {test_data.shape}")
        return test_data
    
    def load_sample_submission(self) -> pd.DataFrame:
        """
        Load sample submission file
        
        Returns:
            Sample submission DataFrame
        """
        submission_path = self.data_dir / "sample_submission.csv"
        if not submission_path.exists():
            raise FileNotFoundError(f"Sample submission not found at {submission_path}")
        
        submission_data = pd.read_csv(submission_path)
        print(f"✓ Loaded sample submission: {submission_data.shape}")
        return submission_data
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all datasets
        
        Returns:
            Tuple of (train_data, test_data, sample_submission)
        """
        train_data = self.load_train_data()
        test_data = self.load_test_data()
        sample_submission = self.load_sample_submission()
        
        return train_data, test_data, sample_submission
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get basic information about a dataset
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "duplicates": df.duplicated().sum()
        }
        
        return info
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, subfolder: str = "processed"):
        """
        Save processed data to file
        
        Args:
            df: DataFrame to save
            filename: Name of the file
            subfolder: Subfolder to save in
        """
        save_dir = Path(subfolder)
        save_dir.mkdir(exist_ok=True)
        
        file_path = save_dir / filename
        df.to_csv(file_path, index=False)
        print(f"✓ Saved processed data: {file_path}")

# Global instance
data_loader = DataLoader()
