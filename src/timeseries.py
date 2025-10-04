# *Time series data generator*

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import get_random_generator, add_simple_noise


class SimpleTimeSeriesGenerator:
    """
    Creates fake time series data for testing your ML models.
    """
    
    def __init__(self, seed=42):
        # Initialize the generator
        self.seed = seed
        self.rng = get_random_generator(seed)
    
    def generate(self, n_points=1000, trend='up', seasonality=True, noise_level=0.1):
        """
        Generate a simple time series.
        
        Args:
            n_points: How many data points to generate
            trend: Direction of trend
            seasonality: Add repeating patterns
            noise_level: How noisy the data should be (0.0 = perfect, 0.5 = very noisy)
        
        Returns:
            pandas DataFrame with timestamp and value columns
        """
        
        # Step 1: Create time points (hours starting from 2020-01-01)
        timestamps = pd.date_range(start='2020-01-01', periods=n_points, freq='H')
        time_index = np.arange(n_points)
        
        # Step 2: Create the trend
        if trend == 'up':  # --> Slowly increases
            trend_values = time_index * 0.01  
        elif trend == 'down': # --> Slowly decreases
            trend_values = -time_index * 0.01  
        else:  # flat --> stays the same
            trend_values = np.zeros(n_points)  
        
        # Step 3: Add seasonality
        if seasonality:
            # Creates a wave pattern that repeats every 24 hours
            seasonal_values = 5 * np.sin(2 * np.pi * time_index / 24)
        else:
            seasonal_values = np.zeros(n_points)
        
        # Step 4: trend + seasonality
        values = trend_values + seasonal_values
        
        # Step 5: Add noise to make it realistic
        if noise_level > 0:
            values = add_simple_noise(values, noise_level, self.seed)
        
        # Step 6: Create a nice DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        
        return df
    
    def plot(self, df, title='Time Series'):
        """
        A simple plot of the time series.
        
        Args:
            df: DataFrame from generate() function
            title: Title for the plot
        """
        plt.figure(figsize=(12, 4))
        plt.plot(df['timestamp'], df['value'], linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def save_csv(self, df, filename='timeseries.csv'):
        """
        Save the data to a CSV file.
        """
        df.to_csv(filename, index=False)
        # print(f"Saved to {filename}")


# If you want to run this file directly
# if __name__ == '__main__':
    # Create generator
    # gen = SimpleTimeSeriesGenerator(seed=42)
    
    # Generate data
    # data = gen.generate(n_points=500, trend='up', seasonality=True, noise_level=0.2)
    
    # print(data.head())
    
    # gen.plot(data)
    # gen.save_csv(data, 'example_timeseries.csv')