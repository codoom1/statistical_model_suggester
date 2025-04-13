"""Logistic regression diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
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

def generate_logistic_regression_plots(X, y):
    """Generate diagnostic plots for logistic regression
    
    Args:
        X: Features (numpy array)
        y: Binary target variable (numpy array)
        
    Returns:
        List of dictionaries with plot information
    """
    # Fit the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    plots = []
    
    # Plot 1: ROC Curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plots.append({
        "title": "ROC Curve",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the model's ability to discriminate between classes at different threshold settings. The AUC (Area Under Curve) of {:.2f} indicates the overall model performance. AUC ranges from 0.5 (no discrimination) to 1.0 (perfect discrimination).".format(roc_auc)
    })
    
    # Plot 2: Predicted Probabilities Histogram
    plt.figure(figsize=(10, 6))
    for i in range(2):
        if np.sum(y == i) > 0:  # Check if class exists in the dataset
            plt.hist(y_pred_proba[y == i], alpha=0.5, bins=20, label=f'Class {i}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Histogram of Predicted Probabilities by Class')
    plots.append({
        "title": "Prediction Distribution",
        "img_data": get_base64_plot(),
        "interpretation": "Shows how well the model separates classes. Ideally, the distributions should have minimal overlap, indicating good discrimination. Higher separation means better classification performance."
    })
    
    # Plot 3: Calibration Plot
    plt.figure(figsize=(10, 6))
    # Create bins of predicted probabilities
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices[bin_indices == len(bins) - 1] = len(bins) - 2  # Handle edge case
    bin_sums = np.bincount(bin_indices, weights=y, minlength=len(bins)-1)
    bin_counts = np.bincount(bin_indices, minlength=len(bins)-1)
    nonzero_mask = bin_counts > 0
    observed_probs = np.zeros(len(bin_centers))
    observed_probs[nonzero_mask] = bin_sums[nonzero_mask] / bin_counts[nonzero_mask]
    
    plt.plot(bin_centers, observed_probs, 'o-', label='Calibration curve')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.legend()
    plt.title('Calibration Plot')
    plots.append({
        "title": "Calibration Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Assesses how well predicted probabilities match observed frequencies. A well-calibrated model follows the diagonal line. Curves above the line indicate underestimation, while curves below indicate overestimation of probabilities."
    })
    
    # Plot 4: Influence Plot
    # Convert to standardized Pearson residuals
    from scipy.stats import norm
    p = y_pred_proba
    residuals = (y - p) / np.sqrt(p * (1 - p))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(p, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Standardized Pearson Residuals')
    plt.title('Residual vs Predicted Probability')
    plots.append({
        "title": "Residual Analysis",
        "img_data": get_base64_plot(),
        "interpretation": "Helps identify poorly fit observations. Large residuals suggest observations that are not well explained by the model. Patterns may indicate missing predictors or non-linear relationships."
    })
    
    return plots 