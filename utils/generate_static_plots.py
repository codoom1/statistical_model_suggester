#!/usr/bin/env python
"""
Script to generate static diagnostic plots.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid display issues
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA

def ensure_directory(path):
    """Ensure the directory exists, create if not."""
    if not os.path.exists(path):
        os.makedirs(path)

def generate_linear_regression_plots():
    """Generate static diagnostic plots for linear regression."""
    # Create output directory
    output_dir = os.path.join('static', 'diagnostic_plots', 'linear_regression')
    ensure_directory(output_dir)
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.normal(0, 1, (100, 2))
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 1, 100)
    
    # Fit linear regression model
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Create plots
    plots_info = [
        {
            "title": "Residuals vs Fitted",
            "interpretation": "This plot checks the linearity assumption. Look for random scatter around the horizontal line. Patterns or curves indicate non-linearity that should be addressed."
        },
        {
            "title": "Normal Q-Q Plot",
            "interpretation": "This plot checks if residuals follow a normal distribution. Points should fall along the diagonal line. Deviations suggest non-normality."
        },
        {
            "title": "Scale-Location",
            "interpretation": "This plot checks the homoscedasticity assumption. Look for random scatter with constant spread. A funnel shape indicates heteroscedasticity."
        },
        {
            "title": "Residuals vs Leverage",
            "interpretation": "This plot helps identify influential observations. Points outside the dashed lines (Cook's distance) have high influence on the model and should be examined."
        }
    ]
    
    # Plot 1: Residuals vs Fitted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.savefig(os.path.join(output_dir, '1_residuals_vs_fitted.png'))
    plt.close()
    
    # Plot 2: Normal Q-Q Plot
    plt.figure(figsize=(10, 6))
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.linspace(0.01, 0.99, len(residuals))
    theoretical_quantiles = np.quantile(np.random.normal(0, 1, 10000), theoretical_quantiles)
    plt.scatter(theoretical_quantiles, sorted_residuals)
    plt.plot([-3, 3], [-3, 3], 'r-')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title('Normal Q-Q Plot')
    plt.savefig(os.path.join(output_dir, '2_normal_qq.png'))
    plt.close()
    
    # Plot 3: Scale-Location
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, np.sqrt(np.abs(residuals)))
    plt.xlabel('Fitted values')
    plt.ylabel('√|Residuals|')
    plt.title('Scale-Location')
    plt.savefig(os.path.join(output_dir, '3_scale_location.png'))
    plt.close()
    
    # Plot 4: Residuals vs Leverage
    plt.figure(figsize=(10, 6))
    leverage = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
    plt.scatter(leverage, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Leverage')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Leverage')
    plt.savefig(os.path.join(output_dir, '4_residuals_vs_leverage.png'))
    plt.close()
    
    # Save plot info as JSON
    for i, plot_info in enumerate(plots_info, 1):
        json_path = os.path.join(output_dir, f'{i}_{plot_info["title"].replace(" ", "_").lower()}.json')
        with open(json_path, 'w') as f:
            json.dump(plot_info, f, indent=4)
    
    print(f"Generated linear regression plots in {output_dir}")

def main():
    """Main function to generate all diagnostic plots."""
    print("Generating static diagnostic plots...")
    generate_linear_regression_plots()
    print("Done!")

if __name__ == "__main__":
    main() 