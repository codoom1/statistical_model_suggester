#!/usr/bin/env python
"""
Isolated plot generator for specific models without external dependencies.

This script implements diagnostic plots for selected models directly,
without depending on other modules.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

# Import for CatBoost and LightGBM
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    catboost_available = True
except ImportError:
    catboost_available = False
    
try:
    import lightgbm as lgb
    lightgbm_available = True
except ImportError:
    lightgbm_available = False

# Import for XGBoost
try:
    import xgboost as xgb
    xgboost_available = True
except ImportError:
    xgboost_available = False

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_sample_data(model_type, n_samples=200):
    """Generate synthetic data for plotting
    
    Args:
        model_type: Type of model
        n_samples: Number of samples
        
    Returns:
        Data appropriate for the model
    """
    np.random.seed(42)
    
    if model_type in ['ridge_regression', 'lasso_regression', 'elastic_net']:
        # Generate data with multicollinearity
        n_features = 10
        
        # Generate coefficients (some are zero for sparsity)
        true_coef = np.zeros(n_features)
        true_coef[:5] = np.array([1.0, -0.5, 0.25, -0.75, 1.5])
        
        # Generate correlated features
        X = np.random.randn(n_samples, n_features)
        
        # Add multicollinearity
        X[:, 5] = 0.8 * X[:, 0] + 0.2 * np.random.randn(n_samples)
        X[:, 6] = 0.8 * X[:, 1] + 0.2 * np.random.randn(n_samples)
        
        # Generate response with noise
        y = X @ true_coef + np.random.randn(n_samples) * 0.5
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
        
        return X_train, y_train, X_test, y_test, feature_names, true_coef
    
    elif model_type in ['random_forest', 'gradient_boosting', 'lightgbm', 'catboost', 'xgboost']:
        # Generate data for tree-based models
        n_features = 5
        
        # Generate features with different distributions
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.exponential(1, n_samples)
        X3 = np.random.uniform(-1, 1, n_samples)
        X4 = np.random.binomial(1, 0.5, n_samples)
        X5 = np.random.poisson(3, n_samples)
        X = np.column_stack([X1, X2, X3, X4, X5])
        
        # Generate target with non-linear relationships
        y = (0.8 * X1 + 0.2 * X2**2 + 0.3 * X3 * X4 + 0.4 * np.log1p(X5) + 
             0.5 * np.sin(X1) + np.random.normal(0, 0.5, n_samples))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
        
        return X_train, y_train, X_test, y_test, feature_names
    
    elif model_type in ['logistic_regression']:
        # Generate classification data with non-linear decision boundary
        n_features = 2  # Keep 2D for easy visualization
        
        # Generate data for two features
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)
        X = np.column_stack([X1, X2])
        
        # Generate binary target with non-linear decision boundary
        z = 1.5 + 2 * X1 - 3 * X2 + 0.5 * X1**2 + 0.5 * X1 * X2
        prob = 1 / (1 + np.exp(-z))  # Sigmoid function to get probabilities
        y = (prob > 0.5).astype(int)  # Binarize probabilities
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
        
        return X_train, y_train, X_test, y_test, feature_names
    
    # Default case
    X = np.random.randn(n_samples, 5)
    y = 2 + 0.5 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n_samples) * 0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    feature_names = [f"Feature {i+1}" for i in range(5)]
    
    return X_train, y_train, X_test, y_test, feature_names

def generate_ridge_regression_plots(X_train, y_train, X_test, y_test, feature_names, true_coef=None):
    """Generate plots for ridge regression
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: Names of features
        true_coef: True coefficients if known
        
    Returns:
        List of plot dictionaries
    """
    plots = []
    
    # Fit models with different regularization strengths
    alphas = np.logspace(-6, 6, 13)
    ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
    ridge_cv.fit(X_train, y_train)
    
    # Optimal model
    best_alpha = ridge_cv.alpha_
    best_model = ridge_cv
    
    # Get predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Plot 1: Feature Importance (Coefficients)
    plt.figure(figsize=(10, 6))
    coef = best_model.coef_
    sorted_idx = np.argsort(np.abs(coef))
    plt.barh(np.array(feature_names)[sorted_idx], coef[sorted_idx])
    plt.title(f'Ridge Regression Coefficients (α={best_alpha:.4f})')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    
    plots.append({
        "title": "Feature Importance",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the coefficients (weights) assigned to each feature by the Ridge regression model. The magnitude of a coefficient indicates the importance of the feature. Ridge regression shrinks coefficients toward zero to reduce overfitting, with more regularization leading to smaller coefficients."
    })
    
    # Plot 2: Training History (Cross-Validation)
    plt.figure(figsize=(10, 6))
    mse_path = np.mean(ridge_cv.cv_values_, axis=0)
    plt.semilogx(alphas, mse_path, marker='o', linestyle='-')
    plt.axvline(x=best_alpha, color='red', linestyle='--', label=f'Best α={best_alpha:.4f}')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Mean Squared Error')
    plt.title('Ridge Regression Cross-Validation MSE vs. Alpha')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plots.append({
        "title": "Training History",
        "img_data": get_base64_plot(),
        "interpretation": f"Shows the cross-validation mean squared error as a function of the regularization parameter α. The optimal α value of {best_alpha:.4f} minimizes the cross-validation error. Too small α may lead to overfitting while too large α may lead to underfitting."
    })
    
    # Plot 3: Residual Plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred_test
    plt.scatter(y_pred_test, residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Ridge Regression Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Residual Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the difference between actual and predicted values plotted against predicted values. Ideally, residuals should be randomly distributed around zero with no pattern. Patterns in residuals may indicate model inadequacy."
    })
    
    # Plot 4: Actual vs. Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.7)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Ridge Regression: Actual vs. Predicted (Test R²={test_r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Actual vs Predicted Values",
        "img_data": get_base64_plot(),
        "interpretation": f"Compares actual values against model predictions. Points should ideally fall along the diagonal red line. The test R² score of {test_r2:.3f} indicates the proportion of variance in the dependent variable that is predictable from the independent variables."
    })
    
    # Plot 5: Coefficient Path (if true coefficients are known)
    if true_coef is not None:
        plt.figure(figsize=(12, 6))
        
        # Train models with different alphas and store coefficients
        coef_paths = []
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            coef_paths.append(model.coef_)
        
        coef_paths = np.array(coef_paths)
        
        # Plot coefficient paths
        for i in range(len(feature_names)):
            plt.semilogx(alphas, coef_paths[:, i], label=feature_names[i])
        
        plt.axvline(x=best_alpha, color='black', linestyle='--', label=f'Best α={best_alpha:.4f}')
        plt.xlabel('Alpha (Regularization Strength)')
        plt.ylabel('Coefficient Value')
        plt.title('Ridge Regression Coefficient Paths')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plots.append({
            "title": "Regularization Path",
            "img_data": get_base64_plot(),
            "interpretation": "Shows how coefficient values change as the regularization parameter α increases. Ridge regression shrinks coefficients toward zero but rarely sets them exactly to zero, preserving most features in the model."
        })
    
    return plots

def generate_lasso_regression_plots(X_train, y_train, X_test, y_test, feature_names, true_coef=None):
    """Generate plots for lasso regression
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: Names of features
        true_coef: True coefficients if known
        
    Returns:
        List of plot dictionaries
    """
    plots = []
    
    # Fit models with different regularization strengths
    alphas = np.logspace(-6, 2, 100)
    lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X_train, y_train)
    
    # Optimal model
    best_alpha = lasso_cv.alpha_
    best_model = lasso_cv
    
    # Get predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Plot 1: Feature Importance (Coefficients)
    plt.figure(figsize=(10, 6))
    coef = best_model.coef_
    sorted_idx = np.argsort(np.abs(coef))
    plt.barh(np.array(feature_names)[sorted_idx], coef[sorted_idx])
    plt.title(f'Lasso Regression Coefficients (α={best_alpha:.4f})')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    
    plots.append({
        "title": "Feature Importance",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the coefficients (weights) assigned to each feature by the Lasso regression model. Lasso performs feature selection by shrinking some coefficients exactly to zero. Non-zero coefficients represent selected features, with larger magnitudes indicating more important features."
    })
    
    # Plot 2: Training History (Cross-Validation)
    plt.figure(figsize=(10, 6))
    plt.semilogx(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=1), marker='o', linestyle='-')
    plt.axvline(x=best_alpha, color='red', linestyle='--', label=f'Best α={best_alpha:.4f}')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Mean Squared Error')
    plt.title('Lasso Regression Cross-Validation MSE vs. Alpha')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plots.append({
        "title": "Training History",
        "img_data": get_base64_plot(),
        "interpretation": f"Shows the cross-validation mean squared error as a function of the regularization parameter α. The optimal α value of {best_alpha:.4f} minimizes the cross-validation error. This helps balance between underfitting (high α) and overfitting (low α)."
    })
    
    # Plot 3: Residual Plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred_test
    plt.scatter(y_pred_test, residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Lasso Regression Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Residual Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the difference between actual and predicted values plotted against predicted values. Ideally, residuals should be randomly distributed around zero with no pattern. Systematic patterns may indicate model inadequacy or violation of regression assumptions."
    })
    
    # Plot 4: Actual vs. Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.7)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Lasso Regression: Actual vs. Predicted (Test R²={test_r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Actual vs Predicted Values",
        "img_data": get_base64_plot(),
        "interpretation": f"Compares actual values against model predictions. Points should ideally fall along the diagonal red line. The test R² score of {test_r2:.3f} indicates the proportion of variance in the dependent variable that is predictable from the independent variables."
    })
    
    # Plot 5: Sparse Coefficient Comparison
    plt.figure(figsize=(10, 6))
    
    # Get the number of non-zero coefficients
    non_zero = np.sum(coef != 0)
    zero = np.sum(coef == 0)
    
    # Create bar plot
    bars = plt.bar(['Non-Zero', 'Zero'], [non_zero, zero])
    plt.title(f'Lasso Regression Sparsity: {non_zero} Non-Zero / {zero} Zero Coefficients')
    plt.ylabel('Count')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plots.append({
        "title": "Feature Sparsity",
        "img_data": get_base64_plot(),
        "interpretation": f"Shows the sparsity induced by Lasso regularization. Out of {len(feature_names)} features, Lasso has selected {non_zero} features by setting their coefficients to non-zero values, while {zero} features have been eliminated (coefficients = 0)."
    })
    
    # Plot 6: Coefficient path (if true coefficients are known)
    if true_coef is not None:
        from sklearn.linear_model import lasso_path
        
        plt.figure(figsize=(12, 6))
        _, coefs_lasso, _ = lasso_path(X_train, y_train, alphas=alphas, max_iter=10000)
        
        # Plot coefficient paths
        for i, name in enumerate(feature_names):
            plt.semilogx(alphas, coefs_lasso[i], label=name)
        
        plt.axvline(x=best_alpha, color='black', linestyle='--', label=f'Best α={best_alpha:.4f}')
        plt.xlabel('Alpha (Regularization Strength)')
        plt.ylabel('Coefficient Value')
        plt.title('Lasso Regression Coefficient Paths')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plots.append({
            "title": "Regularization Path",
            "img_data": get_base64_plot(),
            "interpretation": "Shows how coefficient values change as the regularization parameter α increases. Unlike Ridge regression, Lasso can shrink coefficients exactly to zero, performing feature selection. Coefficients that remain non-zero at higher α values are more important features."
        })
    
    return plots

def generate_random_forest_plots(X_train, y_train, X_test, y_test, feature_names):
    """Generate plots for random forest regression
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: Names of features
        
    Returns:
        List of plot dictionaries
    """
    plots = []
    
    # Fit model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Plot 1: Feature Importance
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    plots.append({
        "title": "Feature Importance",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the importance of each feature in the Random Forest model. Features with higher values have more influence on predictions. Random Forest feature importance is based on how much each feature contributes to decreasing impurity across all trees."
    })
    
    # Plot 2: Actual vs. Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.7)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Random Forest: Actual vs. Predicted (Test R²={test_r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Actual vs Predicted Values",
        "img_data": get_base64_plot(),
        "interpretation": f"Compares actual values against model predictions. Points should ideally fall along the diagonal red line. The test R² score of {test_r2:.3f} shows the proportion of variance explained by the model."
    })
    
    # Plot 3: Residual Plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred_test
    plt.scatter(y_pred_test, residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Random Forest Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Residual Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the difference between actual and predicted values. Residuals should ideally be randomly distributed around zero. Patterns may indicate non-linear relationships not captured by the model or heteroscedasticity."
    })
    
    # Plot 4: Training vs. Test Performance
    plt.figure(figsize=(10, 6))
    metrics = [train_mse, test_mse, train_r2, test_r2]
    labels = ['Training MSE', 'Test MSE', 'Training R²', 'Test R²']
    colors = ['blue', 'red', 'green', 'orange']
    
    plt.bar(labels, metrics, color=colors)
    plt.title('Random Forest: Training vs. Test Performance')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plots.append({
        "title": "Model Performance",
        "img_data": get_base64_plot(),
        "interpretation": f"Compares the model's performance on training and test sets. The difference between training MSE ({train_mse:.3f}) and test MSE ({test_mse:.3f}) indicates the degree of overfitting. Similarly, the difference between training R² ({train_r2:.3f}) and test R² ({test_r2:.3f}) reflects the model's generalization ability."
    })
    
    # Plot 5: Error Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Random Forest Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Error Distribution",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the distribution of prediction errors (residuals). Ideally, errors should be normally distributed around zero. Skewness or multiple peaks may indicate systematic bias or separate regimes in the data."
    })
    
    return plots

def generate_catboost_plots(X_train, y_train, X_test, y_test, feature_names, is_classifier=False):
    """Generate plots for CatBoost model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: Names of features
        is_classifier: Whether this is a classification task
        
    Returns:
        List of plot dictionaries
    """
    if not catboost_available:
        return [{
            "title": "CatBoost Not Available",
            "img_data": "",
            "interpretation": "CatBoost is not installed. Please install it with 'pip install catboost'."
        }]
    
    plots = []
    
    # Select model type based on task
    if is_classifier:
        model = CatBoostClassifier(iterations=100, learning_rate=0.1, random_seed=42, verbose=0)
    else:
        model = CatBoostRegressor(iterations=100, learning_rate=0.1, random_seed=42, verbose=0)
    
    # Train the model
    model.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)
    
    # Get predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    if is_classifier:
        # For classification
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # For probabilistic predictions
        y_pred_proba_test = model.predict_proba(X_test)[:, 1] if y_pred_test.ndim == 1 else model.predict_proba(X_test)
        
    else:
        # For regression
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
    
    # Plot 1: Feature Importance
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.barh(np.array(feature_names)[indices], importances[indices])
    plt.title('CatBoost Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    plots.append({
        "title": "Feature Importance",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the contribution of each feature to the model's predictions. Features with higher values have more influence on predictions. This helps identify the most predictive variables in your model."
    })
    
    # Plot 2: Learning Curve
    plt.figure(figsize=(10, 6))
    train_metrics = model.get_evals_result()['learn']
    test_metrics = model.get_evals_result()['validation']
    
    if is_classifier:
        # Use appropriate metric based on classifier output
        metric_name = list(train_metrics.keys())[0]  # Usually 'Logloss'
    else:
        metric_name = 'RMSE'
        
    plt.plot(train_metrics[metric_name], label='Train')
    plt.plot(test_metrics[metric_name], label='Test')
    plt.xlabel('Iterations')
    plt.ylabel(metric_name)
    plt.title(f'CatBoost Learning Curve ({metric_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Training History",
        "img_data": get_base64_plot(),
        "interpretation": f"Shows the model's learning progress over iterations. The gap between training and test curves indicates potential overfitting. Ideally, both curves should converge to a low error value."
    })
    
    # Plot 3: Residual Plot (for regression) or ROC Curve (for classification)
    if is_classifier:
        # ROC Curve
        plt.figure(figsize=(10, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('CatBoost ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "ROC Curve",
            "img_data": get_base64_plot(),
            "interpretation": f"The Receiver Operating Characteristic (ROC) curve shows the trade-off between true positive rate and false positive rate at different classification thresholds. The Area Under the Curve (AUC) of {roc_auc:.3f} quantifies the overall ability of the model to discriminate between classes."
        })
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred_test)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('CatBoost Confusion Matrix')
        plt.colorbar()
        
        # Add text annotations
        threshold = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > threshold else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plots.append({
            "title": "Confusion Matrix",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the count of true positives, false positives, true negatives, and false negatives. This helps understand where the model makes correct predictions and where it makes errors."
        })
        
    else:
        # Residual Plot for regression
        plt.figure(figsize=(10, 6))
        residuals = y_test - y_pred_test
        plt.scatter(y_pred_test, residuals, alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('CatBoost Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Residual Plot",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the difference between actual and predicted values plotted against predicted values. Ideally, residuals should be randomly distributed around zero with no pattern. Patterns may indicate model inadequacy."
        })
        
        # Actual vs Predicted Plot for regression
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.7)
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'CatBoost: Actual vs. Predicted (Test R²={test_r2:.3f})')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Actual vs Predicted Values",
            "img_data": get_base64_plot(),
            "interpretation": f"Compares actual values against model predictions. Points should ideally fall along the diagonal red line. The test R² score of {test_r2:.3f} indicates the proportion of variance explained by the model."
        })
    
    return plots

def generate_lightgbm_plots(X_train, y_train, X_test, y_test, feature_names, is_classifier=False):
    """Generate plots for LightGBM model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: Names of features
        is_classifier: Whether this is a classification task
        
    Returns:
        List of plot dictionaries
    """
    if not lightgbm_available:
        return [{
            "title": "LightGBM Not Available",
            "img_data": "",
            "interpretation": "LightGBM is not installed. Please install it with 'pip install lightgbm'."
        }]
    
    plots = []
    
    # Create dataset
    if is_classifier:
        # For classification task
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'seed': 42
        }
        eval_name = 'binary_logloss'
    else:
        # For regression task
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'seed': 42
        }
        eval_name = 'rmse'
    
    # Train model
    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        evals_result=evals_result,
        num_boost_round=100,
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Get predictions
    if is_classifier:
        y_pred_train = np.round(model.predict(X_train))
        y_pred_test = np.round(model.predict(X_test))
        y_pred_proba_test = model.predict(X_test)
    else:
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    if is_classifier:
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
    else:
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
    
    # Plot 1: Feature Importance
    plt.figure(figsize=(10, 6))
    importances = model.feature_importance()
    features = np.array(feature_names)
    indices = np.argsort(importances)
    
    plt.barh(features[indices], importances[indices])
    plt.title('LightGBM Feature Importance (Gain)')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    plots.append({
        "title": "Feature Importance",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the contribution of each feature to the model's predictions based on the total gain of splits which use the feature. Features with higher values are more important for making predictions."
    })
    
    # Plot 2: Learning Curve
    plt.figure(figsize=(10, 6))
    
    plt.plot(evals_result['train'][eval_name], label='Train')
    plt.plot(evals_result['test'][eval_name], label='Test')
    plt.xlabel('Iterations')
    plt.ylabel(eval_name.upper())
    plt.title(f'LightGBM Learning Curve ({eval_name.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Training History",
        "img_data": get_base64_plot(),
        "interpretation": f"Shows the model's learning progress over iterations. The gap between training and test curves indicates potential overfitting. Ideally, both curves should converge to a low error value."
    })
    
    # Specific plots for classification or regression
    if is_classifier:
        # ROC Curve for classification
        plt.figure(figsize=(10, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('LightGBM ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "ROC Curve",
            "img_data": get_base64_plot(),
            "interpretation": f"The Receiver Operating Characteristic (ROC) curve shows the trade-off between true positive rate and false positive rate at different classification thresholds. The Area Under the Curve (AUC) of {roc_auc:.3f} quantifies the overall ability of the model to discriminate between classes."
        })
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred_test)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('LightGBM Confusion Matrix')
        plt.colorbar()
        
        # Add text annotations
        threshold = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > threshold else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plots.append({
            "title": "Confusion Matrix",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the count of true positives, false positives, true negatives, and false negatives. This helps understand where the model makes correct predictions and where it makes errors."
        })
        
        # Classification Metrics
        plt.figure(figsize=(10, 6))
        metrics = [
            accuracy_score(y_test, y_pred_test),
            precision_score(y_test, y_pred_test),
            recall_score(y_test, y_pred_test),
            f1_score(y_test, y_pred_test)
        ]
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        plt.bar(metric_names, metrics)
        plt.title('LightGBM Classification Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(metrics):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
            
        plt.tight_layout()
        
        plots.append({
            "title": "Classification Metrics",
            "img_data": get_base64_plot(),
            "interpretation": f"Shows key classification metrics: Accuracy measures overall correctness, Precision measures the proportion of true positives among predicted positives, Recall measures the proportion of actual positives correctly identified, and F1 score is the harmonic mean of precision and recall."
        })
        
    else:
        # Residual Plot for regression
        plt.figure(figsize=(10, 6))
        residuals = y_test - y_pred_test
        plt.scatter(y_pred_test, residuals, alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('LightGBM Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Residual Plot",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the difference between actual and predicted values plotted against predicted values. Ideally, residuals should be randomly distributed around zero with no pattern. Patterns in residuals may indicate model inadequacy."
        })
        
        # Actual vs Predicted Plot for regression
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.7)
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'LightGBM: Actual vs. Predicted (Test R²={test_r2:.3f})')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Actual vs Predicted Values",
            "img_data": get_base64_plot(),
            "interpretation": f"Compares actual values against model predictions. Points should ideally fall along the diagonal red line. The test R² score of {test_r2:.3f} indicates the proportion of variance explained by the model."
        })
    
    return plots

def generate_gradient_boosting_plots(X_train, y_train, X_test, y_test, feature_names, is_classifier=False):
    """Generate plots for Gradient Boosting model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: Names of features
        is_classifier: Whether this is a classification task
        
    Returns:
        List of plot dictionaries
    """
    plots = []
    
    # Choose model type based on task
    if is_classifier:
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    if is_classifier:
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # For probabilistic predictions
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    else:
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
    
    # Plot 1: Feature Importance
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.barh(np.array(feature_names)[indices], importances[indices])
    plt.title('Gradient Boosting Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    plots.append({
        "title": "Feature Importance",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the contribution of each feature to the model's predictions. Features with higher values have more influence on predictions."
    })
    
    # Plot 2: Learning Curve
    plt.figure(figsize=(10, 6))
    
    test_scores = np.zeros((model.n_estimators,), dtype=float)
    train_scores = np.zeros((model.n_estimators,), dtype=float)
    
    if is_classifier:
        # For classification
        for i, pred in enumerate(model.staged_predict(X_test)):
            test_scores[i] = accuracy_score(y_test, pred)
            
        for i, pred in enumerate(model.staged_predict(X_train)):
            train_scores[i] = accuracy_score(y_train, pred)
            
        ylabel = 'Accuracy'
    else:
        # For regression
        for i, pred in enumerate(model.staged_predict(X_test)):
            test_scores[i] = r2_score(y_test, pred)
            
        for i, pred in enumerate(model.staged_predict(X_train)):
            train_scores[i] = r2_score(y_train, pred)
            
        ylabel = 'R² Score'
    
    plt.plot(np.arange(model.n_estimators) + 1, train_scores, label='Train')
    plt.plot(np.arange(model.n_estimators) + 1, test_scores, label='Test')
    plt.xlabel('Number of Trees')
    plt.ylabel(ylabel)
    plt.title(f'Gradient Boosting Learning Curve ({ylabel})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Training History",
        "img_data": get_base64_plot(),
        "interpretation": f"Shows the model's learning progress as trees are added. The gap between training and test curves indicates potential overfitting."
    })
    
    # Different plots based on task type
    if is_classifier:
        # ROC Curve for classification
        plt.figure(figsize=(10, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Gradient Boosting ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "ROC Curve",
            "img_data": get_base64_plot(),
            "interpretation": f"The ROC curve shows the trade-off between true positive rate and false positive rate. The AUC of {roc_auc:.3f} indicates the model's ability to distinguish between classes."
        })
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred_test)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Gradient Boosting Confusion Matrix')
        plt.colorbar()
        
        # Add text annotations
        threshold = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > threshold else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plots.append({
            "title": "Confusion Matrix",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the count of true positives, false positives, true negatives, and false negatives. This helps understand where the model makes correct predictions and where it makes errors."
        })
        
    else:
        # Residual Plot for regression
        plt.figure(figsize=(10, 6))
        residuals = y_test - y_pred_test
        plt.scatter(y_pred_test, residuals, alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Gradient Boosting Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Residual Plot",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the difference between actual and predicted values. Ideally, residuals should be randomly distributed around zero with no pattern."
        })
        
        # Actual vs Predicted Plot for regression
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.7)
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Gradient Boosting: Actual vs. Predicted (Test R²={test_r2:.3f})')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Actual vs Predicted Values",
            "img_data": get_base64_plot(),
            "interpretation": f"Compares actual values against model predictions. Points should ideally fall along the diagonal red line. The test R² score of {test_r2:.3f} indicates the proportion of variance explained by the model."
        })
    
    return plots

def generate_xgboost_plots(X_train, y_train, X_test, y_test, feature_names, is_classifier=False):
    """Generate plots for XGBoost model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: Names of features
        is_classifier: Whether this is a classification task
        
    Returns:
        List of plot dictionaries
    """
    if not xgboost_available:
        return [{
            "title": "XGBoost Not Available",
            "img_data": "",
            "interpretation": "XGBoost is not installed. Please install it with 'pip install xgboost'."
        }]
    
    plots = []
    
    # Set objective based on task type
    if is_classifier:
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric='logloss'
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric='rmse'
        )
    
    # Train model with evaluation
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    # Get predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Get evaluation history
    evals_result = model.evals_result()
    
    # Calculate metrics
    if is_classifier:
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # For probabilistic predictions
        y_pred_proba_test = model.predict_proba(X_test)[:, 1] if y_pred_test.ndim == 1 else model.predict_proba(X_test)
    else:
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
    
    # Plot 1: Feature Importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, importance_type='weight', ax=plt.gca(), title='XGBoost Feature Importance')
    plt.tight_layout()
    
    plots.append({
        "title": "Feature Importance",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the contribution of each feature to the model's predictions based on the number of times a feature is used in trees."
    })
    
    # Plot 2: Learning Curve
    plt.figure(figsize=(10, 6))
    
    # Get evaluation metric name
    eval_metric = list(evals_result['validation_0'].keys())[0]
    
    epochs = len(evals_result['validation_0'][eval_metric])
    x_axis = range(0, epochs)
    
    plt.plot(x_axis, evals_result['validation_0'][eval_metric], label='Train')
    plt.plot(x_axis, evals_result['validation_1'][eval_metric], label='Test')
    plt.xlabel('Iterations')
    plt.ylabel(eval_metric.upper())
    plt.title(f'XGBoost Learning Curve ({eval_metric.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Training History",
        "img_data": get_base64_plot(),
        "interpretation": f"Shows the model's learning progress over iterations. The gap between training and test curves indicates potential overfitting."
    })
    
    # Different plots based on task type
    if is_classifier:
        # ROC Curve for classification
        plt.figure(figsize=(10, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('XGBoost ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "ROC Curve",
            "img_data": get_base64_plot(),
            "interpretation": f"The ROC curve shows the trade-off between true positive rate and false positive rate. The AUC of {roc_auc:.3f} indicates the model's ability to distinguish between classes."
        })
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred_test)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('XGBoost Confusion Matrix')
        plt.colorbar()
        
        # Add text annotations
        threshold = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > threshold else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plots.append({
            "title": "Confusion Matrix",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the count of true positives, false positives, true negatives, and false negatives. This helps understand where the model makes correct predictions and where it makes errors."
        })
        
    else:
        # Residual Plot for regression
        plt.figure(figsize=(10, 6))
        residuals = y_test - y_pred_test
        plt.scatter(y_pred_test, residuals, alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('XGBoost Residual Plot')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Residual Plot",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the difference between actual and predicted values. Ideally, residuals should be randomly distributed around zero with no pattern."
        })
        
        # Actual vs Predicted Plot for regression
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.7)
        min_val = min(y_test.min(), y_pred_test.min())
        max_val = max(y_test.max(), y_pred_test.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'XGBoost: Actual vs. Predicted (Test R²={test_r2:.3f})')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Actual vs Predicted Values",
            "img_data": get_base64_plot(),
            "interpretation": f"Compares actual values against model predictions. Points should ideally fall along the diagonal red line. The test R² score of {test_r2:.3f} indicates the proportion of variance explained by the model."
        })
    
    return plots

def generate_linear_regression_plots(X_train, y_train, X_test, y_test, feature_names, is_classifier=False):
    """Generate plots for Linear Regression model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: Names of features
        is_classifier: Not used, Linear Regression is always a regression model
        
    Returns:
        List of plot dictionaries
    """
    plots = []
    
    # Simple Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Plot 1: Coefficients
    plt.figure(figsize=(10, 6))
    coef = model.coef_
    
    # Handle 1D or 2D coefficients
    if coef.ndim > 1:
        coef = coef[0]
        
    indices = np.argsort(np.abs(coef))
    plt.barh(np.array(feature_names)[indices], coef[indices])
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.title('Linear Regression Coefficients')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    
    plots.append({
        "title": "Feature Coefficients",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the importance and direction of each feature. Positive coefficients increase the prediction value; negative coefficients decrease it. The magnitude indicates importance."
    })
    
    # Plot 2: Residual Plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred_test
    plt.scatter(y_pred_test, residuals, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Linear Regression Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Residual Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the difference between actual and predicted values. Ideally, residuals should be randomly distributed around zero with no pattern, indicating the model captures the relationship well."
    })
    
    # Plot 3: Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.7)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Linear Regression: Actual vs. Predicted (Test R²={test_r2:.3f})')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Actual vs Predicted Values",
        "img_data": get_base64_plot(),
        "interpretation": f"Compares actual values against model predictions. Points should ideally fall along the diagonal red line. The test R² score of {test_r2:.3f} indicates the proportion of variance explained by the model."
    })
    
    # Plot 4: Residual Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Linear Regression Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Residual Distribution",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the distribution of residuals. In a good model, residuals should be normally distributed around zero, indicating the errors are random and not systematic."
    })
    
    # Plot 5: Scale-Location Plot (Sqrt of standardized residuals vs fitted values)
    plt.figure(figsize=(10, 6))
    standardized_residuals = residuals / np.std(residuals)
    sqrt_std_residuals = np.sqrt(np.abs(standardized_residuals))
    plt.scatter(y_pred_test, sqrt_std_residuals, alpha=0.7)
    plt.xlabel('Predicted Values')
    plt.ylabel('√|Standardized Residuals|')
    plt.title('Linear Regression Scale-Location Plot')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Scale-Location Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Helps assess if residuals have constant variance (homoscedasticity). A horizontal line with randomly spread points suggests residuals have constant variance."
    })
    
    return plots

def generate_logistic_regression_plots(X_train, y_train, X_test, y_test, feature_names, is_classifier=True):
    """Generate plots for Logistic Regression model
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_names: Names of features
        is_classifier: Not used, Logistic Regression is always a classifier
        
    Returns:
        List of plot dictionaries
    """
    plots = []
    
    # Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred_test, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred_test, average='binary', zero_division=0)
    
    # Plot 1: Coefficients
    plt.figure(figsize=(10, 6))
    coef = model.coef_[0]
    indices = np.argsort(np.abs(coef))
    plt.barh(np.array(feature_names)[indices], coef[indices])
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.title('Logistic Regression Coefficients')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    
    plots.append({
        "title": "Feature Coefficients",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the effect of each feature on the log-odds of the positive class. Positive coefficients increase probability; negative coefficients decrease it."
    })
    
    # Plot 2: ROC Curve
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic Regression ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "ROC Curve",
        "img_data": get_base64_plot(),
        "interpretation": f"The ROC curve shows the trade-off between true positive rate and false positive rate. The AUC of {roc_auc:.3f} indicates the model's ability to distinguish between classes."
    })
    
    # Plot 3: Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_test)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Logistic Regression Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > threshold else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plots.append({
        "title": "Confusion Matrix",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the count of true positives, false positives, true negatives, and false negatives. This helps understand where the model makes correct predictions and where it makes errors."
    })
    
    # Plot 4: Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    
    # Get precision-recall values at different thresholds
    precision_values = []
    recall_values = []
    thresholds = np.linspace(0, 1, 100)
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba_test >= threshold).astype(int)
        precision_values.append(precision_score(y_test, y_pred_thresh, zero_division=1))
        recall_values.append(recall_score(y_test, y_pred_thresh, zero_division=0))
    
    plt.plot(recall_values, precision_values)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Logistic Regression Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    # Mark the selected threshold's position
    default_threshold_precision = precision
    default_threshold_recall = recall
    plt.scatter([default_threshold_recall], [default_threshold_precision], 
                marker='o', color='red', s=100, label=f'Default Threshold (0.5)')
    plt.legend()
    
    plots.append({
        "title": "Precision-Recall Curve",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the trade-off between precision and recall at different classification thresholds. The red dot indicates the model's performance at the default threshold of 0.5."
    })
    
    # Plot 5: Decision Boundary (only for 2D data)
    if X_train.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        
        # Create a mesh grid
        x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
        y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
        h = 0.02  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Predict the function value for the whole grid
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        # Plot contour and training points
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu_r)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.RdBu_r)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title('Logistic Regression Decision Boundary')
        plt.colorbar()
        
        plots.append({
            "title": "Decision Boundary",
            "img_data": get_base64_plot(),
            "interpretation": "Visualizes the decision boundary of the model in feature space. The color represents the probability of the positive class, with the decision boundary at probability = 0.5."
        })
    
    return plots

def main():
    parser = argparse.ArgumentParser(description='Generate diagnostic plots for specific models')
    parser.add_argument('model', type=str, nargs='?', help='Name of the model (e.g., ridge_regression)')
    parser.add_argument('--output', type=str, default='static/diagnostic_plots',
                        help='Directory to save the plots')
    parser.add_argument('--list', action='store_true', help='List all available models')
    args = parser.parse_args()
    
    # Define available models and their generator functions
    available_models = {
        'ridge_regression': generate_ridge_regression_plots,
        'lasso_regression': generate_lasso_regression_plots,
        'random_forest': generate_random_forest_plots,
        'catboost': generate_catboost_plots,
        'lightgbm': generate_lightgbm_plots,
        'xgboost': generate_xgboost_plots,
        'gradient_boosting': generate_gradient_boosting_plots,
        'linear_regression': generate_linear_regression_plots,
        'logistic_regression': generate_logistic_regression_plots
    }
    
    # List available models if requested
    if args.list:
        print("Available models:")
        for i, model_name in enumerate(sorted(available_models.keys()), 1):
            print(f"{i}. {model_name}")
        return
    
    # Check if model is provided
    if args.model is None:
        print("Please specify a model name or use --list to see available models")
        return
    
    # Check if model is available
    if args.model.lower() not in available_models:
        print(f"Model '{args.model}' not implemented in this script.")
        print("Available models:", ', '.join(available_models.keys()))
        return
    
    # Create output directory
    model_dir = os.path.join(args.output, args.model.lower())
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate sample data - adapt based on model type
    data = generate_sample_data(args.model.lower())
    
    # Call the appropriate function
    model_function = available_models[args.model.lower()]
    plots = model_function(*data)
    
    # Save plots to files
    for i, plot in enumerate(plots):
        filename = os.path.join(model_dir, f"{i+1}_{plot['title'].replace(' ', '_').lower()}.png")
        
        # Convert and save the plot
        try:
            import base64
            import io
            from PIL import Image
            
            image_data = base64.b64decode(plot.get("img_data", ""))
            image = Image.open(io.BytesIO(image_data))
            image.save(filename)
            print(f"Saved: {filename}")
            
            # Create a JSON file with plot information
            json_filename = os.path.join(model_dir, f"{i+1}_{plot['title'].replace(' ', '_').lower()}.json")
            with open(json_filename, 'w') as f:
                json.dump({
                    'title': plot['title'],
                    'interpretation': plot['interpretation']
                }, f, indent=2)
                
        except Exception as e:
            print(f"Error saving {filename}: {e}")
    
    print(f"Successfully generated {len(plots)} plots for {args.model}")
    print(f"All plots saved to {model_dir}")

if __name__ == "__main__":
    main() 