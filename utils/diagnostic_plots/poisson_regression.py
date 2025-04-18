"""Poisson regression diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import io
import base64
import os
from pathlib import Path

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def save_plot(plot_name):
    """Save the current plot to the appropriate directory"""
    # Create directory if it doesn't exist
    plot_dir = Path("static/diagnostic_plots/poisson_regression")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(plot_dir / f"{plot_name}.png", dpi=100, bbox_inches='tight')
    plt.close()

def generate_poisson_regression_plots(X, y):
    """Generate diagnostic plots for Poisson regression
    
    Args:
        X: Features (numpy array)
        y: Target variable (numpy array) - count data
        
    Returns:
        List of dictionaries with plot information
    """
    # Fit the model
    X_with_const = sm.add_constant(X)
    model = sm.GLM(y, X_with_const, family=sm.families.Poisson()).fit()
    fitted_vals = model.fittedvalues
    resid_deviance = model.resid_deviance
    resid_pearson = model.resid_pearson
    
    plots = []
    
    # Plot 1: Observed vs Predicted Counts
    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_vals, y)
    # Add a 45-degree reference line
    max_val = max(np.max(fitted_vals), np.max(y))
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel('Predicted counts')
    plt.ylabel('Observed counts')
    plt.title('Observed vs Predicted Counts')
    plots.append({
        "title": "Observed vs Predicted Counts",
        "img_data": get_base64_plot(),
        "interpretation": "Points should lie close to the diagonal line. Systematic deviations suggest model misspecification."
    })
    save_plot("observed_vs_predicted")
    
    # Plot 2: Deviance Residuals vs Fitted
    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_vals, resid_deviance)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Fitted values')
    plt.ylabel('Deviance residuals')
    plt.title('Deviance Residuals vs Fitted')
    plots.append({
        "title": "Deviance Residuals vs Fitted",
        "img_data": get_base64_plot(),
        "interpretation": "Check for patterns in residuals. Points should be randomly scattered around zero with constant variance."
    })
    save_plot("deviance_residuals")
    
    # Plot 3: Q-Q Plot of Deviance Residuals
    plt.figure(figsize=(10, 6))
    qqplot(resid_deviance, line='45', fit=True)
    plt.title('Q-Q Plot of Deviance Residuals')
    plots.append({
        "title": "Q-Q Plot of Deviance Residuals",
        "img_data": get_base64_plot(),
        "interpretation": "Check for normality of deviance residuals. Deviations from the line suggest non-normality."
    })
    save_plot("qq_plot")
    
    # Plot 4: Scale-Location Plot (Sqrt of deviance residuals vs fitted)
    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_vals, np.sqrt(np.abs(resid_deviance)))
    plt.xlabel('Fitted values')
    plt.ylabel('√|Deviance residuals|')
    plt.title('Scale-Location Plot')
    plots.append({
        "title": "Scale-Location Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Check for homogeneity of variance. Points should show a constant spread across fitted values."
    })
    save_plot("scale_location")
    
    # Plot 5: Pearson Residuals Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(resid_pearson, bins=20, edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='-')
    plt.xlabel('Pearson residuals')
    plt.ylabel('Frequency')
    plt.title('Histogram of Pearson Residuals')
    plots.append({
        "title": "Histogram of Pearson Residuals",
        "img_data": get_base64_plot(),
        "interpretation": "Check for skewness or outliers in residuals. Histogram should be approximately symmetric around zero."
    })
    save_plot("pearson_residuals_hist")
    
    # Plot 6: Check for overdispersion
    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_vals, resid_pearson**2)
    plt.axhline(y=1, color='r', linestyle='-', label='Reference line (y=1)')
    plt.xlabel('Fitted values')
    plt.ylabel('Squared Pearson residuals')
    plt.title('Squared Pearson Residuals vs Fitted (Overdispersion Check)')
    plt.legend()
    
    # Calculate dispersion parameter
    dispersion = sum(resid_pearson**2) / model.df_resid
    plt.figtext(0.5, 0.01, f'Dispersion parameter: {dispersion:.4f}', 
                ha='center', fontsize=12, 
                bbox={'facecolor':'orange', 'alpha':0.1, 'pad':5})
    
    plots.append({
        "title": "Overdispersion Check",
        "img_data": get_base64_plot(),
        "interpretation": f"Dispersion parameter = {dispersion:.4f}. Values substantially greater than 1 indicate overdispersion, suggesting negative binomial may be more appropriate than Poisson regression."
    })
    save_plot("overdispersion_check")
    
    return plots 