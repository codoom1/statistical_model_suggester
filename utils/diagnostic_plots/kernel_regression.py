"""Kernel regression diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from statsmodels.nonparametric.smoothers_lowess import lowess

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_kernel_regression_plots(X, y, model, X_test=None, y_test=None, bandwidth=None):
    """Generate diagnostic plots for kernel regression models
    
    Args:
        X: Feature matrix (could be single or multi-dimensional)
        y: Target variable
        model: Fitted kernel regression model 
        X_test: Optional test data features
        y_test: Optional test data target
        bandwidth: Optional bandwidth value used in the model
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Convert X to numpy array if it's not already
    X_np = np.array(X)
    y_np = np.array(y)
    
    # Plot 1: Actual vs Fitted values
    plt.figure(figsize=(10, 6))
    
    # Get fitted values
    if hasattr(model, 'predict'):
        y_pred = model.predict(X)
    elif hasattr(model, 'fit'):
        y_pred = model.fit(X)
    else:
        # If we can't get predictions directly, try using the model as a callable
        try:
            y_pred = model(X)
        except:
            y_pred = None
            
    if y_pred is not None:
        plt.scatter(y, y_pred, alpha=0.6)
        plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Fitted Values')
        plt.title('Actual vs Fitted Values')
        
        plots.append({
            "title": "Actual vs Fitted Values",
            "img_data": get_base64_plot(),
            "interpretation": "Shows how well the model's fitted values match the actual values. Points should ideally lie close to the diagonal line, indicating good model fit."
        })
    
    # Plot 2: Residuals vs Fitted values
    if y_pred is not None:
        plt.figure(figsize=(10, 6))
        residuals = y - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add lowess trend line
        try:
            smooth = lowess(residuals, y_pred, frac=0.6)
            plt.plot(smooth[:, 0], smooth[:, 1], 'g-', lw=2)
        except:
            # Skip if lowess fails
            pass
            
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted Values')
        
        plots.append({
            "title": "Residuals vs Fitted Values",
            "img_data": get_base64_plot(),
            "interpretation": "Helps assess if there are non-linear patterns that weren't captured by the kernel. Ideally, residuals should be randomly distributed around zero with no clear pattern."
        })
        
        # Plot 3: Residual histogram
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        
        plots.append({
            "title": "Residual Distribution",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the distribution of residuals. While kernel regression doesn't assume normally distributed residuals, this plot helps identify if there are extreme outliers or skewness that might affect the model."
        })
    
    # Plot 4: Visualization of kernel fit (for 1D feature only)
    if X_np.ndim == 1 or (X_np.ndim == 2 and X_np.shape[1] == 1):
        # Ensure X is 1D for plotting
        x_1d = X_np.ravel() if X_np.ndim > 1 else X_np
        
        # Sort points for smooth line
        sort_idx = np.argsort(x_1d)
        x_sorted = x_1d[sort_idx]
        y_sorted = y_np[sort_idx]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(x_sorted, y_sorted, alpha=0.6, label='Data Points')
        
        if y_pred is not None:
            y_pred_sorted = y_pred[sort_idx]
            plt.plot(x_sorted, y_pred_sorted, 'r-', lw=2, label='Kernel Fit')
            
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Kernel Regression Fit')
        plt.legend()
        
        plots.append({
            "title": "Kernel Regression Fit",
            "img_data": get_base64_plot(),
            "interpretation": "Visualizes how the kernel regression line fits the data points. The flexibility of the line shows how the model adapts to local patterns in the data."
        })
    
    # Plot 5: 2D visualization if possible
    if X_np.ndim == 2 and X_np.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        
        # Create a scatter plot colored by the response variable
        scatter = plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Response Value')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('2D Feature Space Colored by Response')
        
        plots.append({
            "title": "Feature Space Visualization",
            "img_data": get_base64_plot(),
            "interpretation": "Shows how the response variable (color) varies across the 2D feature space. This helps visualize the patterns that the kernel regression is modeling."
        })
    
    # Plot 6: Bandwidth selection impact (if bandwidth is provided)
    if bandwidth is not None and X_np.ndim == 1:
        plt.figure(figsize=(12, 8))
        
        # Sort for consistent plotting
        sort_idx = np.argsort(X_np.ravel())
        x_sorted = X_np.ravel()[sort_idx]
        y_sorted = y_np[sort_idx]
        
        plt.scatter(x_sorted, y_sorted, alpha=0.6, label='Data')
        
        # Try different bandwidths centered around the provided one
        bandwidths = [bandwidth * 0.5, bandwidth, bandwidth * 2]
        colors = ['green', 'red', 'blue']
        labels = ['Small Bandwidth', 'Selected Bandwidth', 'Large Bandwidth']
        
        # This is a simplistic approach - in practice, you'd refit the model with different bandwidths
        # or use a function that can generate fits for multiple bandwidths
        for i, bw in enumerate(bandwidths):
            try:
                # Try to get a new prediction with this bandwidth
                # This is a placeholder and would need to be adapted to the specific kernel regression implementation
                if hasattr(model, 'set_params') and hasattr(model, 'predict'):
                    model_copy = model.set_params(bandwidth=bw)
                    y_pred_bw = model_copy.predict(X_np)
                    plt.plot(x_sorted, y_pred_bw[sort_idx], colors[i], lw=2, label=labels[i])
            except:
                # Skip if we can't get prediction for this bandwidth
                continue
            
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Effect of Bandwidth Selection')
        plt.legend()
        
        plots.append({
            "title": "Bandwidth Selection Impact",
            "img_data": get_base64_plot(),
            "interpretation": "Illustrates how different bandwidth choices affect the smoothness of the fit. A smaller bandwidth creates a more flexible curve that may overfit, while a larger bandwidth produces a smoother curve that may underfit."
        })
        
    # Plot 7: Leave-one-out cross-validation if test data is provided
    if X_test is not None and y_test is not None and hasattr(model, 'predict'):
        plt.figure(figsize=(10, 6))
        
        # Get predictions on test data
        y_test_pred = model.predict(X_test)
        
        plt.scatter(y_test, y_test_pred, alpha=0.6)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
        
        plt.xlabel('Actual Test Values')
        plt.ylabel('Predicted Test Values')
        plt.title('Prediction Performance on Test Data')
        
        # Calculate and display RMSE
        rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
        plt.annotate(f'RMSE: {rmse:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
        
        plots.append({
            "title": "Test Data Prediction",
            "img_data": get_base64_plot(),
            "interpretation": f"Evaluates the model's performance on unseen data. Points should be close to the diagonal line, indicating good predictive ability. The Root Mean Squared Error (RMSE) is {rmse:.4f}."
        })
    
    return plots 