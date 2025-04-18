import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import base64
from io import BytesIO
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import validation_curve

def get_base64_plot():
    """Convert the current matplotlib plot to a base64 string."""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

def generate_lasso_regression_plots(model, X, y, X_test=None, y_test=None, feature_names=None, alphas=None):
    """
    Generate diagnostic plots for Lasso Regression models.
    
    Parameters:
    -----------
    model : Lasso model object
        The fitted Lasso regression model
    X : array-like
        Training feature matrix
    y : array-like
        Training target variable
    X_test : array-like, optional
        Test feature matrix
    y_test : array-like, optional
        Test target variable
    feature_names : list, optional
        Names of the features
    alphas : array-like, optional
        Alpha values for regularization path plot
    
    Returns:
    --------
    plots : list of dict
        List of dictionaries containing:
        - 'title': Title of the plot
        - 'img_data': Base64 encoded image
        - 'interpretation': Interpretation of the plot
    """
    plots = []
    
    # Ensure data is in the right format
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Get predictions
    y_pred = model.predict(X)
    if X_test is not None and y_test is not None:
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        y_test_pred = model.predict(X_test)
    
    # Get feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    
    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.7)
    
    # Add y=x line
    min_val = min(min(y), min(y_pred))
    max_val = max(max(y), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    
    # Add R-squared value
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    plt.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    plots.append({
        'title': 'Actual vs Predicted Values',
        'img_data': get_base64_plot(),
        'interpretation': f'Shows how well the model predictions match actual values. Points should lie close to the diagonal line for a good model. The R² value of {r2:.4f} indicates the proportion of variance explained by the model, while RMSE of {rmse:.4f} measures the average prediction error.'
    })
    
    # 2. Residuals Plot
    plt.figure(figsize=(10, 6))
    residuals = y - y_pred
    
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    
    plots.append({
        'title': 'Residuals vs Predicted Values',
        'img_data': get_base64_plot(),
        'interpretation': 'Shows if residuals have a pattern related to the predicted values. Ideally, residuals should be randomly scattered around zero with no discernible pattern, which would indicate homoscedasticity and linearity.'
    })
    
    # 3. Sparse Coefficient Plot 
    plt.figure(figsize=(12, 8))
    
    # Get coefficients and determine which are non-zero
    coefs = model.coef_
    non_zero_mask = coefs != 0
    non_zero_count = np.sum(non_zero_mask)
    zero_count = len(coefs) - non_zero_count
    
    # Create DataFrame for plotting
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs
    })
    
    # Sort by absolute coefficient value
    coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
    
    # Plot non-zero coefficients
    colors = ['blue' if c != 0 else 'red' for c in coef_df['Coefficient']]
    plt.barh(np.arange(len(coef_df)), coef_df['Coefficient'], color=colors)
    plt.yticks(np.arange(len(coef_df)), coef_df['Feature'])
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Coefficient Value')
    plt.title(f'Lasso Regression Coefficients\n{non_zero_count} non-zero, {zero_count} zero coefficients')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Non-zero coefficients'),
                       Patch(facecolor='red', label='Zero coefficients')]
    plt.legend(handles=legend_elements, loc='best')
    
    plots.append({
        'title': 'Lasso Regression Coefficients',
        'img_data': get_base64_plot(),
        'interpretation': f'Shows the magnitude and direction of each feature\'s effect on the target variable. Lasso has selected {non_zero_count} features and eliminated {zero_count} features by setting their coefficients to exactly zero, demonstrating the feature selection capability of Lasso regression.'
    })
    
    # 4. Regularization Path
    if alphas is None:
        alphas = np.logspace(-4, 1, 100)
    
    plt.figure(figsize=(12, 8))
    
    from sklearn.linear_model import lasso_path
    _, coefs_path, _ = lasso_path(X, y, alphas=alphas, fit_intercept=True)
    
    # Plot paths
    for i, feature in enumerate(feature_names):
        plt.plot(alphas, coefs_path[i], label=feature, linewidth=2)
    
    plt.xscale('log')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Coefficient Value')
    plt.title('Lasso Regularization Path')
    
    # Highlight the chosen alpha
    plt.axvline(x=model.alpha, color='red', linestyle='--', label=f'Selected Alpha: {model.alpha}')
    
    # Add legend
    if len(feature_names) <= 10:
        plt.legend(loc='best')
    else:
        # If too many features, only show the legend for selected important features
        top_indices = np.argsort(np.abs(model.coef_))[-5:]  # Top 5 features
        handles, labels = plt.gca().get_legend_handles_labels()
        selected_handles = [handles[i] for i in top_indices] + [handles[-1]]  # Add alpha line
        selected_labels = [labels[i] for i in top_indices] + [labels[-1]]
        plt.legend(selected_handles, selected_labels, loc='best')
    
    plots.append({
        'title': 'Regularization Path',
        'img_data': get_base64_plot(),
        'interpretation': 'Shows how feature coefficients change with different regularization strengths (alpha values). Features drop to exactly zero as alpha increases, demonstrating Lasso\'s variable selection capability. The vertical line indicates the selected alpha value used in the model.'
    })
    
    # 5. Feature Selection Stability (if test data is available)
    if X_test is not None and y_test is not None:
        plt.figure(figsize=(12, 8))
        
        from sklearn.linear_model import Lasso
        
        # Number of bootstrap samples
        n_samples = min(50, len(X_test))
        
        # Matrix to store which features are selected in each sample
        feature_selection = np.zeros((n_samples, X.shape[1]))
        
        # Bootstrap sampling
        for i in range(n_samples):
            # Sample with replacement
            indices = np.random.choice(len(X_test), len(X_test), replace=True)
            X_boot, y_boot = X_test[indices], y_test[indices]
            
            # Fit Lasso
            lasso = Lasso(alpha=model.alpha)
            lasso.fit(X_boot, y_boot)
            
            # Store which features have non-zero coefficients
            feature_selection[i, :] = np.abs(lasso.coef_) > 0
        
        # Calculate selection frequency for each feature
        selection_freq = feature_selection.mean(axis=0)
        
        # Sort features by selection frequency
        sorted_indices = np.argsort(selection_freq)
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_freq = selection_freq[sorted_indices]
        
        # Plot feature selection stability
        plt.barh(np.arange(len(sorted_features)), sorted_freq, color='skyblue')
        plt.yticks(np.arange(len(sorted_features)), sorted_features)
        plt.xlabel('Selection Frequency')
        plt.title('Feature Selection Stability')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plots.append({
            'title': 'Feature Selection Stability',
            'img_data': get_base64_plot(),
            'interpretation': 'Shows how consistently each feature is selected across bootstrap samples. Features with high selection frequency are more stable and likely important to the model. Features with low frequency may be less reliable predictors or sensitive to small changes in the data.'
        })
    
    # 6. Validation Curve for Alpha Selection
    if X_test is not None and y_test is not None:
        plt.figure(figsize=(10, 6))
        
        from sklearn.linear_model import Lasso
        
        # Generate validation curve
        train_scores, test_scores = validation_curve(
            Lasso(), X, y, param_name="alpha", param_range=alphas,
            cv=min(5, len(X)), scoring="neg_mean_squared_error"
        )
        
        # Calculate mean and std for training scores
        train_mean = -train_scores.mean(axis=1)  # Negative MSE -> MSE
        train_std = train_scores.std(axis=1)
        
        # Calculate mean and std for test scores
        test_mean = -test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        
        # Plot validation curve
        plt.plot(alphas, train_mean, label="Training score", color="blue")
        plt.fill_between(alphas, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
        
        plt.plot(alphas, test_mean, label="Cross-validation score", color="green")
        plt.fill_between(alphas, test_mean - test_std, test_mean + test_std, alpha=0.1, color="green")
        
        # Highlight the chosen alpha
        plt.axvline(x=model.alpha, color='red', linestyle='--', label=f'Selected Alpha: {model.alpha}')
        
        plt.xscale("log")
        plt.xlabel("Alpha (log scale)")
        plt.ylabel("Mean Squared Error")
        plt.title("Validation Curve for Lasso Regression")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plots.append({
            'title': 'Validation Curve',
            'img_data': get_base64_plot(),
            'interpretation': 'Illustrates model performance (MSE) across different alpha values. The blue line shows training error, while the green line shows cross-validation error. The optimal alpha balances underfitting (high alpha, high bias) and overfitting (low alpha, high variance). The vertical line indicates the selected alpha value.'
        })
    
    # 7. Learning Curves
    if X_test is not None and y_test is not None:
        plt.figure(figsize=(10, 6))
        
        from sklearn.model_selection import learning_curve
        
        # Generate the learning curve data
        train_sizes, train_scores, test_scores = learning_curve(
            Lasso(alpha=model.alpha), X, y, cv=min(5, len(X)), 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="neg_mean_squared_error"
        )
        
        # Calculate mean and std for training scores
        train_mean = -train_scores.mean(axis=1)  # Negative MSE -> MSE
        train_std = train_scores.std(axis=1)
        
        # Calculate mean and std for test scores
        test_mean = -test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        
        # Plot learning curve
        plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training score")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
        
        plt.plot(train_sizes, test_mean, 'o-', color="green", label="Cross-validation score")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="green")
        
        plt.xlabel("Training Set Size")
        plt.ylabel("Mean Squared Error")
        plt.title(f"Learning Curves for Lasso Regression (alpha={model.alpha})")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plots.append({
            'title': 'Learning Curves',
            'img_data': get_base64_plot(),
            'interpretation': 'Shows how model performance changes with increasing training data. The gap between training error (blue) and cross-validation error (green) indicates whether the model would benefit from more data. A small gap suggests that the model has reached its capacity and more data may not help.'
        })
    
    return plots 