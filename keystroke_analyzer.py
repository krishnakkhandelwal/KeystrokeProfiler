import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

class KeystrokeAnalyzer:
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)
        print("Columns in dataset:", self.data.columns.tolist())
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing values in dataset:")
            print(missing_values[missing_values > 0])
        
    def perform_exploratory_analysis(self):
        """Perform exploratory data analysis on the keystroke dataset"""
        print("Dataset Shape:", self.data.shape)
        print("\nDataset Summary:")
        print(self.data.describe())
        
        print("\nUser Distribution:")
        print(self.data['user_id'].value_counts())
        
        # Identify numeric and non-numeric columns
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = self.data.select_dtypes(exclude=['number']).columns.tolist()
        
        print("\nNumeric columns:", len(numeric_cols))
        print("Non-numeric columns:", non_numeric_cols)
        
        # Drop user_id and non-numeric columns for correlation analysis
        numeric_data = self.data[numeric_cols].drop(columns=['user_id'] if 'user_id' in numeric_cols else [])
        
        # Handle missing values for correlation matrix
        numeric_data_filled = numeric_data.fillna(numeric_data.mean())
        
        # Create correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_data_filled.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()
        
        # Visualize key features by user - only use features that exist in the dataset
        desired_features = ['mean_hold_time', 'mean_down_down_time', 'keystrokes_per_second']
        features_to_plot = [f for f in desired_features if f in numeric_cols]
        
        if not features_to_plot:
            print("Warning: None of the desired features exist in the dataset.")
            # Use the first 3 numeric columns instead
            features_to_plot = numeric_cols[:min(3, len(numeric_cols))]
            print(f"Using these features instead: {features_to_plot}")
        
        if features_to_plot:
            plt.figure(figsize=(15, 8))
            for i, feature in enumerate(features_to_plot):
                plt.subplot(1, len(features_to_plot), i+1)
                # Drop NaN values for boxplot
                plot_data = self.data.dropna(subset=[feature])
                if len(plot_data) > 0:
                    sns.boxplot(x='user_id', y=feature, data=plot_data)
                    plt.title(f'{feature} by User')
                    plt.xticks(rotation=90)
                else:
                    plt.text(0.5, 0.5, f"No data for {feature}", 
                             horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            plt.savefig('features_by_user.png')
            plt.close()
        
        return "Analysis completed and saved as images"
    
    def perform_data_preprocessing(self):
        """Preprocess data for modeling"""
        # Separate features and target
        # First, identify numeric columns
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        # Remove non-numeric columns and keep only numeric features
        X = self.data[numeric_cols].drop(columns=['user_id'] if 'user_id' in numeric_cols else [])
        y = self.data['user_id']
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Apply PCA for dimensionality reduction
        # Only apply PCA if we have enough samples and features
        if X_scaled.shape[0] > 1 and X_scaled.shape[1] > 1:
            n_components = min(2, X_scaled.shape[1])
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Visualize PCA results
            plt.figure(figsize=(10, 8))
            for user in y.unique():
                indices = y == user
                if X_pca.shape[1] > 1:
                    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=user, alpha=0.7)
                else:
                    plt.scatter(X_pca[indices, 0], np.zeros(sum(indices)), label=user, alpha=0.7)
            
            plt.title('PCA: User Clustering by Keystroke Patterns')
            plt.xlabel('PC1')
            plt.ylabel('PC2' if X_pca.shape[1] > 1 else 'N/A')
            plt.legend()
            plt.savefig('user_clustering_pca.png')
            plt.close()
        else:
            print("Not enough data for PCA visualization")
            pca = None
            X_pca = X_scaled
        
        # Return preprocessed data
        return X_scaled, y, pca, scaler

if __name__ == "__main__":
    analyzer = KeystrokeAnalyzer("keystroke_data/keystroke_dataset.csv")
    analyzer.perform_exploratory_analysis()
    X_scaled, y, pca, scaler = analyzer.perform_data_preprocessing() 