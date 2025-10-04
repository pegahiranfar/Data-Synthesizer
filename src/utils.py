# *Utility functions for data generation*

import numpy as np


def get_random_generator(seed=None):
    """
    Create a random number generator to make results reproducible
    Args:
        seed: A number to make results reproducible. 
    
    Returns:
        A numpy random generator
    """
    if seed is None:
        seed = 42  # Default seed
    return np.random.default_rng(seed)


def add_simple_noise(data, noise_amount=0.1, seed=None):
    """
    Add random noise to make data more realistic (Real data is never perfect).
    
    Args:
        data: data (numpy array)
        noise_amount: How much noise (0.1 = 10% noise)
        seed: Random seed for reproducibility
    
    Returns:
        Data with noise added
    """
    rng = get_random_generator(seed)
    
    # Create random noise with same shape as data
    noise = rng.normal(0, noise_amount, size=data.shape)
    
    # Add noise to data
    noisy_data = data + noise
    
    return noisy_data