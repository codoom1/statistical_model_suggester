"""Linear regression diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import io
import base64

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_linear_regression_plots(X, y):
    """Generate diagnostic plots for linear regression
    
    Args:
        X: Features (numpy array)
        y: Target variable (numpy array)
        
    Returns:
        List of dictionaries with plot information
    """
    # Fit the model
    model = sm.OLS(y, sm.add_constant(X)).fit()
    fitted_vals = model.fittedvalues
    residuals = model.resid
    
    plots = []
    
    # Plot 1: Residuals vs Fitted
    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_vals, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plots.append({
        "title": "Residuals vs Fitted",
        "img_data": get_base64_plot(),
        "interpretation": "Check for linearity. Residuals should be randomly scattered around the horizontal line at zero with no patterns. Curved patterns indicate non-linearity in the relationship."
    })
    
    # Plot 2: Q-Q Plot
    plt.figure(figsize=(10, 6))
    qqplot(residuals, line='45', fit=True)
    plt.title('Normal Q-Q Plot')
    plots.append({
        "title": "Normal Q-Q Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Check for normality of residuals. Points should follow the diagonal line closely. Deviations at the ends indicate heavy or light tails in the distribution."
    })
    
    # Plot 3: Scale-Location Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(fitted_vals, np.sqrt(np.abs(residuals)))
    plt.xlabel('Fitted values')
    plt.ylabel('√|Residuals|')
    plt.title('Scale-Location Plot')
    plots.append({
        "title": "Scale-Location Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Check for homoscedasticity. Points should be randomly scattered with a constant spread. A funnel shape indicates heteroscedasticity (non-constant variance)."
    })
    
    # Plot 4: Leverage Plot
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    plt.figure(figsize=(10, 6))
    plt.scatter(leverage, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Leverage')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Leverage')
    plots.append({
        "title": "Residuals vs Leverage",
        "img_data": get_base64_plot(),
        "interpretation": "Identify influential observations. Points with high leverage and large residuals may disproportionately influence the model. Points outside the dashed lines (Cook's distance) require examination."
    })
    
    return plots 