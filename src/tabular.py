# *Tabular data generator*

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from .utils import get_random_generator


class SimpleTabularGenerator:
    """
    Creates fake tabular datasets for classification or regression.
    Classification for predicting categories and regression for predicting numbers.
    """
    
    def __init__(self, seed=42):
        # # Initialize the generator
        self.seed = seed
        self.rng = get_random_generator(seed)
    
    def generate_classification(self, n_samples=1000, n_features=10, n_classes=2, difficulty='medium'):
        """
        Generate data for classification problems
        
        Args:
            n_samples: How many rows of data
            n_features: How many columns/features
            n_classes: How many categories (n_classes= 2 --> binary)
            difficulty: How hard to classify ('easy', 'medium', 'hard')
        
        Returns:
            DataFrame with features and a target column
        """
        
        # Map difficulty to separation parameter
        difficulty_map = {'easy': 2.0, 'medium': 1.0, 'hard': 0.5}
        separation = difficulty_map.get(difficulty, 1.0)
        
        # Use sklearn to generate fake data where features actually relate to the target
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,  # Half the features are useful
            n_redundant=n_features // 4,     # Some features are duplicates
            n_classes=n_classes,
            class_sep=separation,            # How separated the classes are
            random_state=self.seed
        )
        
        # Create column names and put it in a DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    def generate_regression(self, n_samples=1000, n_features=10, noise_level=10.0):
        """
        Generate data for regression problems
        
        Args:
            n_samples: How many rows of data
            n_features: How many columns/features
            noise_level: How noisy the target is (higher = more random)
        
        Returns:
            DataFrame with features and a target column
        """
        
        # Use sklearn to generate the data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2, 
            noise=noise_level,               # Add some randomness
            random_state=self.seed
        )
        
        # Create column names and put it in a DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    def make_imbalanced(self, df, minority_ratio=0.1):
        """
        Make the classes unbalanced
        
        Args:
            df: DataFrame
            minority_ratio: Fraction of minority class (0.1 = 10%)
        
        Returns:
            DataFrame with imbalanced classes
        """
        
        # Separate the two classes
        class_0 = df[df['target'] == 0]
        class_1 = df[df['target'] == 1]
        
        # Calculate how many samples is needed
        n_total = len(df)
        n_minority = int(n_total * minority_ratio)
        n_majority = n_total - n_minority
        
        # Sample the right amounts
        class_1_sampled = class_1.sample(n=min(n_minority, len(class_1)), 
                                         random_state=self.seed, 
                                         replace=True)
        class_0_sampled = class_0.sample(n=min(n_majority, len(class_0)), 
                                         random_state=self.seed, 
                                         replace=True)
        
        # Shuffle
        df_imbalanced = pd.concat([class_0_sampled, class_1_sampled])
        df_imbalanced = df_imbalanced.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return df_imbalanced
    
    def save_csv(self, df, filename='tabular_data.csv'):
        """
        Save the data to a CSV file.
        """
        df.to_csv(filename, index=False)
        # print(f"Saved to {filename}")
    
    def show_info(self, df):
        """
        Print information about the dataset.
        """
        print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\nColumn names: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        if 'target' in df.columns:
            print(f"\nTarget distribution:")
            print(df['target'].value_counts())


# If you want to run this file directly
# if __name__ == '__main__':
    # Create generator
    # gen = SimpleTabularGenerator(seed=42)
    
    # Generate classification data
    # print("CLASSIFICATION EXAMPLE")
    # data_clf = gen.generate_classification(n_samples=500, n_features=5, difficulty='medium')
    # gen.show_info(data_clf)
    # gen.save_csv(data_clf, 'example_classification.csv')
    # print("=" * 50)

    # print("REGRESSION EXAMPLE")
    # data_reg = gen.generate_regression(n_samples=500, n_features=5, noise_level=5.0)
    # gen.show_info(data_reg)
    # gen.save_csv(data_reg, 'example_regression.csv')
    # print("=" * 50)

    # print("IMBALANCED CLASSIFICATION EXAMPLE")
    # data_imb = gen.make_imbalanced(data_clf, minority_ratio=0.2)
    # gen.show_info(data_imb)