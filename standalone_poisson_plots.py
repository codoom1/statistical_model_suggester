#!/usr/bin/env python
"""
Standalone script to generate diagnostic plots for Poisson regression.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import json
from pathlib import Path

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

def save_plot_with_interpretation(plot_name, title, interpretation, plot_dir):
    """Save the current plot with interpretation
    
    Args:
        plot_name: Filename for the plot
        title: Title of the plot
        interpretation: Text explaining how to interpret the plot
        plot_dir: Directory to save the plot
    """
    # Remove any numerical prefixes for the file name
    clean_name = plot_name
    if "_" in plot_name and plot_name[0].isdigit():
        clean_name = plot_name.split("_", 1)[1]  # Remove prefix like "1_"
    
    # Create full path
    plot_path = os.path.join(plot_dir, f"{clean_name}.png")
    json_path = os.path.join(plot_dir, f"{clean_name}.json")
    
    # Save the plot as PNG
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    
    # Save the interpretation as JSON
    with open(json_path, 'w') as f:
        json.dump({
            "title": title,
            "interpretation": interpretation
        }, f)
    
    plt.close()
    print(f"Saved: {plot_path}")

def generate_poisson_plots():
    """Generate and save diagnostic plots for Poisson regression"""
    # Generate sample data
    X, y = generate_sample_data()
    
    # Create output directory
    plot_dir = "static/diagnostic_plots/poisson_regression"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Fit Poisson regression model
    X_with_const = sm.add_constant(X)
    try:
        model = sm.GLM(y, X_with_const, family=sm.families.Poisson()).fit()
        fitted_vals = model.fittedvalues
        resid_deviance = model.resid_deviance
        resid_pearson = model.resid_pearson
        
        # Plot 1: Observed vs Predicted Counts
        plt.figure(figsize=(10, 6))
        plt.scatter(fitted_vals, y)
        # Add a 45-degree reference line
        max_val = max(np.max(fitted_vals), np.max(y))
        plt.plot([0, max_val], [0, max_val], 'r--')
        plt.xlabel('Predicted counts')
        plt.ylabel('Observed counts')
        plt.title('Observed vs Predicted Counts')
        save_plot_with_interpretation(
            "observed_vs_predicted", 
            "Observed vs Predicted Counts",
            "Points should lie close to the diagonal line. Systematic deviations suggest model misspecification.",
            plot_dir
        )
        
        # Plot 2: Deviance Residuals vs Fitted
        plt.figure(figsize=(10, 6))
        plt.scatter(fitted_vals, resid_deviance)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Fitted values')
        plt.ylabel('Deviance residuals')
        plt.title('Deviance Residuals vs Fitted')
        save_plot_with_interpretation(
            "deviance_residuals", 
            "Deviance Residuals vs Fitted",
            "Check for patterns in residuals. Points should be randomly scattered around zero with constant variance.",
            plot_dir
        )
        
        # Plot 3: Q-Q Plot of Deviance Residuals
        plt.figure(figsize=(10, 6))
        qqplot(resid_deviance, line='45', fit=True)
        plt.title('Q-Q Plot of Deviance Residuals')
        save_plot_with_interpretation(
            "qq_plot", 
            "Q-Q Plot of Deviance Residuals",
            "Check for normality of deviance residuals. Deviations from the line suggest non-normality.",
            plot_dir
        )
        
        # Plot 4: Scale-Location Plot (Sqrt of deviance residuals vs fitted)
        plt.figure(figsize=(10, 6))
        plt.scatter(fitted_vals, np.sqrt(np.abs(resid_deviance)))
        plt.xlabel('Fitted values')
        plt.ylabel('√|Deviance residuals|')
        plt.title('Scale-Location Plot')
        save_plot_with_interpretation(
            "scale_location", 
            "Scale-Location Plot",
            "Check for homogeneity of variance. Points should show a constant spread across fitted values.",
            plot_dir
        )
        
        # Plot 5: Pearson Residuals Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(resid_pearson, bins=20, edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='-')
        plt.xlabel('Pearson residuals')
        plt.ylabel('Frequency')
        plt.title('Histogram of Pearson Residuals')
        save_plot_with_interpretation(
            "pearson_residuals_hist", 
            "Histogram of Pearson Residuals",
            "Check for skewness or outliers in residuals. Histogram should be approximately symmetric around zero.",
            plot_dir
        )
        
        # Plot 6: Check for overdispersion
        plt.figure(figsize=(10, 6))
        plt.scatter(fitted_vals, resid_pearson**2)
        plt.axhline(y=1, color='r', linestyle='-', label='Reference line (y=1)')
        plt.xlabel('Fitted values')
        plt.ylabel('Squared Pearson residuals')
        plt.title('Overdispersion Check')
        plt.legend()
        
        # Calculate dispersion parameter
        dispersion = sum(resid_pearson**2) / model.df_resid
        plt.figtext(0.5, 0.01, f'Dispersion parameter: {dispersion:.4f}', 
                    ha='center', fontsize=12, 
                    bbox={'facecolor':'orange', 'alpha':0.1, 'pad':5})
        
        save_plot_with_interpretation(
            "overdispersion_check", 
            "Overdispersion Check",
            f"Dispersion parameter = {dispersion:.4f}. Values substantially greater than 1 indicate overdispersion, suggesting negative binomial may be more appropriate than Poisson regression.",
            plot_dir
        )
        
        print(f"Generated 6 diagnostic plots for Poisson regression")
        print(f"Plots saved to: {plot_dir}")
        
    except Exception as e:
        print(f"Error generating Poisson regression plots: {e}")

if __name__ == "__main__":
    generate_poisson_plots() 