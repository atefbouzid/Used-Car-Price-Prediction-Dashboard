"""
Exploratory Data Analysis Service
Generates insights and visualizations for the dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EDAAnalyzer:
    """Service for exploratory data analysis"""
    
    def __init__(self, output_dir: str = "imgs"):
        """
        Initialize EDA Analyzer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def analyze_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze basic dataset information
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with basic information
        """
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.value_counts().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
        
        return info
    
    def plot_missing_values(self, df: pd.DataFrame, title: str = "Missing Values Analysis"):
        """
        Plot missing values heatmap
        
        Args:
            df: DataFrame to analyze
            title: Title for the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Calculate missing values
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        # Create DataFrame for plotting
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        })
        
        # Filter out columns with no missing values
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            # Plot missing values
            plt.subplot(2, 1, 1)
            missing_df['Missing Count'].plot(kind='bar')
            plt.title(f'{title} - Missing Count')
            plt.xticks(rotation=45)
            plt.ylabel('Count')
            
            plt.subplot(2, 1, 2)
            missing_df['Missing Percentage'].plot(kind='bar', color='orange')
            plt.title(f'{title} - Missing Percentage')
            plt.xticks(rotation=45)
            plt.ylabel('Percentage (%)')
        else:
            plt.text(0.5, 0.5, 'No Missing Values Found', 
                    ha='center', va='center', fontsize=16)
            plt.title(title)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "missing_values_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved missing values analysis")
    
    def plot_target_distribution(self, df: pd.DataFrame, target_col: str = 'price'):
        """
        Plot target variable distribution
        
        Args:
            df: DataFrame containing target variable
            target_col: Name of target column
        """
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Original distribution
        plt.subplot(2, 2, 1)
        plt.hist(df[target_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'{target_col} Distribution')
        plt.xlabel(target_col)
        plt.ylabel('Frequency')
        
        # Log transformation
        plt.subplot(2, 2, 2)
        log_price = np.log1p(df[target_col])
        plt.hist(log_price, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title(f'Log({target_col}) Distribution')
        plt.xlabel(f'Log({target_col})')
        plt.ylabel('Frequency')
        
        # Box plot
        plt.subplot(2, 2, 3)
        plt.boxplot(df[target_col])
        plt.title(f'{target_col} Box Plot')
        plt.ylabel(target_col)
        
        # Q-Q plot
        plt.subplot(2, 2, 4)
        from scipy import stats
        stats.probplot(df[target_col], dist="norm", plot=plt)
        plt.title(f'{target_col} Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "target_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved target distribution analysis")
    
    def plot_categorical_analysis(self, df: pd.DataFrame, categorical_cols: List[str]):
        """
        Analyze categorical variables
        
        Args:
            df: DataFrame to analyze
            categorical_cols: List of categorical column names
        """
        n_cols = min(3, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, col in enumerate(categorical_cols):
            if col not in df.columns:
                continue
                
            plt.subplot(n_rows, n_cols, i + 1)
            
            # Count plot
            value_counts = df[col].value_counts().head(10)  # Top 10 values
            value_counts.plot(kind='bar')
            plt.title(f'{col} Distribution (Top 10)')
            plt.xticks(rotation=45)
            plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "categorical_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved categorical analysis")
    
    def plot_numerical_analysis(self, df: pd.DataFrame, numerical_cols: List[str]):
        """
        Analyze numerical variables
        
        Args:
            df: DataFrame to analyze
            numerical_cols: List of numerical column names
        """
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, col in enumerate(numerical_cols):
            if col not in df.columns:
                continue
                
            plt.subplot(n_rows, n_cols, i + 1)
            plt.hist(df[col], bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'{col} Distribution')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "numerical_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved numerical analysis")
    
    def plot_correlation_matrix(self, df: pd.DataFrame, target_col: str = 'price'):
        """
        Plot correlation matrix
        
        Args:
            df: DataFrame to analyze
            target_col: Name of target column
        """
        # Select only numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numerical_cols].corr()
        
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plot correlation matrix
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved correlation matrix")
        
        # Plot target correlations
        if target_col in corr_matrix.columns:
            target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
            
            plt.figure(figsize=(10, 8))
            target_corr.plot(kind='barh')
            plt.title(f'Feature Correlations with {target_col}')
            plt.xlabel('Correlation')
            plt.tight_layout()
            plt.savefig(self.output_dir / "target_correlations.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved target correlations")
    
    def plot_feature_vs_target(self, df: pd.DataFrame, features: List[str], target_col: str = 'price'):
        """
        Plot features vs target variable
        
        Args:
            df: DataFrame to analyze
            features: List of feature column names
            target_col: Name of target column
        """
        n_cols = min(3, len(features))
        n_rows = (len(features) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, feature in enumerate(features):
            if feature not in df.columns or target_col not in df.columns:
                continue
                
            plt.subplot(n_rows, n_cols, i + 1)
            
            if df[feature].dtype == 'object':
                # Categorical feature
                df.boxplot(column=target_col, by=feature, ax=plt.gca())
                plt.title(f'{target_col} vs {feature}')
                plt.xticks(rotation=45)
            else:
                # Numerical feature
                plt.scatter(df[feature], df[target_col], alpha=0.5)
                plt.xlabel(feature)
                plt.ylabel(target_col)
                plt.title(f'{target_col} vs {feature}')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_vs_target.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved feature vs target analysis")
    
    def generate_summary_report(self, df: pd.DataFrame, target_col: str = 'price') -> Dict[str, Any]:
        """
        Generate comprehensive EDA summary report
        
        Args:
            df: DataFrame to analyze
            target_col: Name of target column
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "dataset_info": self.analyze_basic_info(df),
            "target_stats": {},
            "feature_stats": {}
        }
        
        # Target statistics
        if target_col in df.columns:
            summary["target_stats"] = {
                "mean": df[target_col].mean(),
                "median": df[target_col].median(),
                "std": df[target_col].std(),
                "min": df[target_col].min(),
                "max": df[target_col].max(),
                "skewness": df[target_col].skew(),
                "kurtosis": df[target_col].kurtosis()
            }
        
        # Feature statistics
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        summary["feature_stats"] = {
            "numerical_features": list(numerical_cols),
            "categorical_features": list(categorical_cols),
            "numerical_summary": df[numerical_cols].describe().to_dict(),
            "categorical_summary": {col: df[col].value_counts().head().to_dict() 
                                  for col in categorical_cols}
        }
        
        return summary
    
    def run_full_eda(self, df: pd.DataFrame, target_col: str = 'price'):
        """
        Run complete EDA analysis
        
        Args:
            df: DataFrame to analyze
            target_col: Name of target column
        """
        print("🔍 Starting Exploratory Data Analysis...")
        
        # Basic analysis
        self.plot_missing_values(df)
        
        # Target analysis
        if target_col in df.columns:
            self.plot_target_distribution(df, target_col)
        
        # Feature analysis
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if numerical_cols:
            self.plot_numerical_analysis(df, numerical_cols)
        
        if categorical_cols:
            self.plot_categorical_analysis(df, categorical_cols)
        
        # Correlation analysis
        self.plot_correlation_matrix(df, target_col)
        
        # Feature vs target
        if target_col in df.columns:
            # Select top features by correlation
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numerical_cols:
                numerical_cols.remove(target_col)
            
            if numerical_cols:
                # Calculate correlations with target
                correlations = df[numerical_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
                # Select top 6 features (excluding target itself)
                top_features = correlations.drop(target_col).head(6).index.tolist()
            else:
                top_features = []
            
            if top_features:
                self.plot_feature_vs_target(df, top_features, target_col)
        
        # Generate summary
        summary = self.generate_summary_report(df, target_col)
        
        print("✅ EDA analysis completed!")
        print(f"📊 Visualizations saved to: {self.output_dir}")
        return summary

# Global instance
eda_analyzer = EDAAnalyzer()
