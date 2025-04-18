#!/usr/bin/env python
"""
Generate diagnostic plots for Poisson regression.
"""
import os
import numpy as np
from utils.diagnostic_plots.poisson_regression import generate_poisson_regression_plots

def generate_sample_data(n_samples=200):
    """Generate sample data for Poisson regression
    
    Returns:
        X: Features matrix
        y: Count target variable
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate predictors
    X = np.random.normal(0, 1, (n_samples, 3))
    
    # Generate log(lambda) = intercept + X1 + 0.5*X2 + 0.25*X3
    log_lambda = 0.5 + X[:, 0] + 0.5 * X[:, 1] + 0.25 * X[:, 2]
    lambda_vals = np.exp(log_lambda)
    
    # Generate count data from Poisson distribution
    y = np.random.poisson(lambda_vals)
    
    return X, y

def main():
    """Main function to generate and save Poisson regression plots"""
    print("Generating Poisson regression diagnostic plots...")
    
    # Generate sample data
    X, y = generate_sample_data()
    
    # Generate plots
    plots = generate_poisson_regression_plots(X, y)
    
    print(f"Generated {len(plots)} diagnostic plots for Poisson regression")
    print(f"Plots saved to: static/diagnostic_plots/poisson_regression/")

if __name__ == "__main__":
    main() 