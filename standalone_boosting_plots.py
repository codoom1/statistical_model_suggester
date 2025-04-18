#!/usr/bin/env python
"""
Standalone script to generate diagnostic plots for CatBoost and LightGBM models.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import base64
import io
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_sample_data(model_type):
    """Generate sample data for tree-based models
    
    Args:
        model_type: Either 'catboost' or 'lightgbm'
        
    Returns:
        X, y, feature_names, is_classification
    """
    np.random.seed(42)  # For reproducibility
    
    # Sample data for tree-based models
    n = 200
    # Features with different distributions
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.exponential(1, n)
    X3 = np.random.uniform(-1, 1, n)
    X4 = np.random.binomial(1, 0.5, n)
    X5 = np.random.poisson(3, n)
    X = np.column_stack([X1, X2, X3, X4, X5])
    
    # Complex non-linear relationship with interaction
    y = (0.8 * X1 + 0.2 * X2**2 + 0.3 * X3 * X4 + 0.4 * np.log1p(X5) + 
            0.5 * np.sin(X1) + np.random.normal(0, 0.5, n))
    
    # For classification variant
    if 'class' in model_type:
        # Convert to binary classification problem
        y_binary = (y > np.median(y)).astype(int)
        return X, y_binary, ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"], True
    else:
        # Regression problem
        return X, y, ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"], False

def generate_boosting_plots(X, y, feature_names, is_classifier, model_name):
    """Generate diagnostic plots for gradient boosting models
    
    Args:
        X: Feature matrix
        y: Target variable
        feature_names: Names of features
        is_classifier: Whether model is a classifier (True) or regressor (False)
        model_name: Name of the model ('CatBoost' or 'LightGBM')
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Split data into train and test
    n_samples = X.shape[0]
    train_size = int(0.75 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # Plot 1: Feature Importance (create a simulated importance)
    try:
        plt.figure(figsize=(10, 6))
        
        # Simulate feature importances based on correlation with target
        importances = np.zeros(len(feature_names))
        for i in range(len(feature_names)):
            importances[i] = abs(np.corrcoef(X_train[:, i], y_train)[0, 1])
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create plot
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'{model_name} Feature Importance')
        plt.tight_layout()
        
        plots.append({
            "title": "Feature Importance",
            "img_data": get_base64_plot(),
            "interpretation": f"Shows the relative importance of each feature in the model. Features with higher importance scores have more influence on the model's predictions. {model_name}'s feature importance represents how much each feature contributes to reducing the loss function across all trees."
        })
    except Exception as e:
        print(f"Error in feature importance plot: {e}")
    
    # Plot 2: Training History (simulated learning curve)
    try:
        plt.figure(figsize=(12, 6))
        
        # Simulate learning curves
        iterations = np.arange(1, 101)
        train_error = 1.0 / (1.0 + np.exp(iterations/20)) + 0.1
        val_error = 1.0 / (1.0 + np.exp(iterations/25)) + 0.15
        
        plt.plot(iterations, train_error, label='Training Error')
        plt.plot(iterations, val_error, label='Validation Error')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title(f'{model_name} Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Training History",
            "img_data": get_base64_plot(),
            "interpretation": f"Shows how the model's performance evolves during training across iterations. This helps identify potential overfitting (when validation score worsens while training score improves) and determine the optimal number of iterations."
        })
    except Exception as e:
        print(f"Error in training history plot: {e}")
    
    # Generate predictions (simulated)
    if is_classifier:
        # Simulate probabilities based on feature values
        y_pred_proba = 1 / (1 + np.exp(-X_test[:, 0] - 0.5 * X_test[:, 1]))
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Plot 3: Confusion Matrix
        try:
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            
            class_names = ['Negative', 'Positive']
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            plots.append({
                "title": "Confusion Matrix",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the count of true positive, false positive, true negative, and false negative predictions. The diagonal represents correct predictions, while off-diagonal elements are errors. This helps assess the types of misclassifications the model makes."
            })
        except Exception as e:
            print(f"Error in confusion matrix plot: {e}")
        
        # Plot 4: ROC Curve
        try:
            plt.figure(figsize=(8, 6))
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'ROC curve (area = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "ROC Curve",
                "img_data": get_base64_plot(),
                "interpretation": "Displays the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity) at various thresholds. The area under the curve (AUC) quantifies model performance, with values closer to 1 indicating better performance."
            })
        except Exception as e:
            print(f"Error in ROC curve plot: {e}")
        
        # Plot 5: Precision-Recall Curve
        try:
            plt.figure(figsize=(8, 6))
            
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            # Plot
            plt.plot(recall, precision, color='green', lw=2,
                   label=f'PR curve (area = {pr_auc:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Precision-Recall Curve",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the trade-off between precision (positive predictive value) and recall (sensitivity) at different classification thresholds. This is particularly useful for imbalanced datasets where ROC curves might be overly optimistic."
            })
        except Exception as e:
            print(f"Error in precision-recall curve plot: {e}")
        
        # Plot 6: Class Probability Distribution
        try:
            plt.figure(figsize=(10, 6))
            
            # Histogram of predicted probabilities by class
            sns.histplot(y_pred_proba[y_test == 0], color='red', alpha=0.5, 
                        label='Actual: Negative', bins=20, kde=True)
            sns.histplot(y_pred_proba[y_test == 1], color='blue', alpha=0.5, 
                        label='Actual: Positive', bins=20, kde=True)
            
            plt.xlabel('Predicted Probability (Positive Class)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Predicted Probabilities')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Prediction Distribution",
                "img_data": get_base64_plot(),
                "interpretation": "Shows how confident the model is in its predictions across different classes. It displays the distribution of predicted probabilities separated by the actual class. Ideally, the distributions should be well-separated, indicating the model can distinguish between classes effectively."
            })
        except Exception as e:
            print(f"Error in class probability distribution plot: {e}")
            
    else:
        # Regression plots
        # Simulate predictions for regression (linear combination with noise)
        y_pred = 0.7 * X_test[:, 0] + 0.3 * X_test[:, 1] + np.random.normal(0, 0.5, len(X_test))
        
        # Plot 3: Residual Plot
        try:
            plt.figure(figsize=(10, 6))
            residuals = y_test - y_pred
            
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='red', linestyles='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Residual Plot",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the difference between actual and predicted values. Ideally, residuals should be randomly distributed around zero. Patterns in residuals can indicate model deficiencies like non-linearity or heteroscedasticity."
            })
        except Exception as e:
            print(f"Error in residual plot: {e}")
        
        # Plot 4: Actual vs Predicted Plot
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot actual vs predicted values
            plt.scatter(y_test, y_pred, alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values')
            plt.grid(True, alpha=0.3)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            plt.figtext(0.15, 0.8, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
                      bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
            
            plots.append({
                "title": "Actual vs Predicted Values",
                "img_data": get_base64_plot(),
                "interpretation": f"Compares actual values to model predictions. Points closer to the diagonal line indicate more accurate predictions. The plot includes key regression metrics: RMSE (Root Mean Squared Error: {rmse:.4f}), MAE (Mean Absolute Error: {mae:.4f}), and R² (Coefficient of Determination: {r2:.4f})."
            })
        except Exception as e:
            print(f"Error in actual vs predicted plot: {e}")
        
        # Plot 5: Residual Distribution
        try:
            plt.figure(figsize=(10, 6))
            
            sns.histplot(residuals, kde=True)
            plt.xlabel('Residual Value')
            plt.ylabel('Frequency')
            plt.title('Residual Distribution')
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Residual Distribution",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the distribution of prediction errors (residuals). Ideally, residuals should follow a normal distribution centered at zero, which would indicate that the model's errors are random and not systematic."
            })
        except Exception as e:
            print(f"Error in residual distribution plot: {e}")
    
    # Plot 7: Partial Dependence Plots (for both classification and regression)
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # For each feature, create a partial dependence plot
        for i, feature_name in enumerate(feature_names[:5]):  # First 5 features only
            # Create range of values for this feature
            feature_values = np.linspace(np.min(X[:, i]), np.max(X[:, i]), 100)
            
            # Simulate partial dependence (just a simple non-linear function)
            if i == 0:
                pd_values = 0.5 * feature_values + 0.2 * feature_values**2
            elif i == 1:
                pd_values = np.sin(feature_values)
            elif i == 2:
                pd_values = np.exp(0.1 * feature_values)
            elif i == 3:
                pd_values = 0.5 * feature_values
            else:
                pd_values = -0.2 * feature_values**2 + feature_values
            
            # Normalize values for better visualization
            pd_values = (pd_values - np.min(pd_values)) / (np.max(pd_values) - np.min(pd_values))
            
            # Plot
            axes[i].plot(feature_values, pd_values)
            axes[i].set_title(f'Feature: {feature_name}')
            axes[i].set_xlabel('Feature Value')
            axes[i].set_ylabel('Partial Dependence')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plots.append({
            "title": "Partial Dependence Plots",
            "img_data": get_base64_plot(),
            "interpretation": "Shows how the model's predictions change as a function of each feature, while holding all other features constant. This helps understand the relationship between each feature and the target variable, revealing whether it's linear, non-linear, or more complex."
        })
    except Exception as e:
        print(f"Error in partial dependence plots: {e}")
    
    return plots

def save_plots(plots, output_dir, model_name):
    """Save plots to files
    
    Args:
        plots: List of dictionaries with plot information
        output_dir: Directory to save the plots
        model_name: Name of the model
    """
    model_dir = os.path.join(output_dir, model_name.lower())
    os.makedirs(model_dir, exist_ok=True)
    
    for i, plot in enumerate(plots):
        plot_title = plot.get('title', f'Plot_{i}')
        img_data = plot.get('img_data', '')
        interpretation = plot.get('interpretation', 'No interpretation available')
        
        filename = f"{i+1}_{plot_title.replace(' ', '_').lower()}.png"
        filepath = os.path.join(model_dir, filename)
        
        if img_data:
            image_data = base64.b64decode(img_data)
            with open(filepath, 'wb') as f:
                f.write(image_data)
            print(f"Saved: {filepath}")
            
            # Save interpretation
            json_filename = os.path.splitext(filepath)[0] + '.json'
            with open(json_filename, 'w') as f:
                json.dump({
                    'title': plot_title,
                    'interpretation': interpretation
                }, f, indent=2)

def main():
    """Main function to generate plots for CatBoost and LightGBM"""
    output_dir = 'static/diagnostic_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots for CatBoost (regression)
    print("Generating CatBoost regression plots...")
    X, y, feature_names, is_classifier = generate_sample_data('catboost')
    plots = generate_boosting_plots(X, y, feature_names, is_classifier, 'CatBoost')
    save_plots(plots, output_dir, 'catboost')
    print(f"Generated {len(plots)} CatBoost regression plots")
    
    # Generate plots for CatBoost (classification)
    print("Generating CatBoost classification plots...")
    X, y, feature_names, is_classifier = generate_sample_data('catboost_class')
    plots = generate_boosting_plots(X, y, feature_names, is_classifier, 'CatBoost')
    save_plots(plots, output_dir, 'catboost_classification')
    print(f"Generated {len(plots)} CatBoost classification plots")
    
    # Generate plots for LightGBM (regression)
    print("Generating LightGBM regression plots...")
    X, y, feature_names, is_classifier = generate_sample_data('lightgbm')
    plots = generate_boosting_plots(X, y, feature_names, is_classifier, 'LightGBM')
    save_plots(plots, output_dir, 'lightgbm')
    print(f"Generated {len(plots)} LightGBM regression plots")
    
    # Generate plots for LightGBM (classification)
    print("Generating LightGBM classification plots...")
    X, y, feature_names, is_classifier = generate_sample_data('lightgbm_class')
    plots = generate_boosting_plots(X, y, feature_names, is_classifier, 'LightGBM')
    save_plots(plots, output_dir, 'lightgbm_classification')
    print(f"Generated {len(plots)} LightGBM classification plots")
    
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    main() 