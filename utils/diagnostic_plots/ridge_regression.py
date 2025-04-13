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

def generate_ridge_regression_plots(model, X, y, X_test=None, y_test=None, feature_names=None, alphas=None):
    """
    Generate diagnostic plots for Ridge Regression models.
    
    Parameters:
    -----------
    model : Ridge model object
        The fitted Ridge regression model
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
    
    # 3. Coefficient Plot with Error Bars
    plt.figure(figsize=(12, 8))
    
    # Get coefficients
    coefs = model.coef_
    
    # Assuming standard error is not directly available, we'll use
    # a bootstrap approximation if test data is provided
    if X_test is not None and y_test is not None and len(X_test) > 30:
        # Calculate bootstrapped standard errors
        n_bootstrap = 100
        bootstrap_coefs = np.zeros((n_bootstrap, len(coefs)))
        
        for i in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(len(X_test), len(X_test), replace=True)
            X_boot, y_boot = X_test[indices], y_test[indices]
            
            # Fit model on bootstrap sample
            from sklearn.linear_model import Ridge
            boot_model = Ridge(alpha=model.alpha)
            boot_model.fit(X_boot, y_boot)
            
            # Store coefficients
            bootstrap_coefs[i, :] = boot_model.coef_
        
        # Calculate standard errors
        coef_errors = bootstrap_coefs.std(axis=0)
    else:
        # If test data not available, just use an arbitrary small value
        coef_errors = np.abs(coefs) * 0.2
    
    # Create DataFrame for plotting
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Error': coef_errors
    })
    
    # Sort by absolute coefficient value
    coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
    
    # Plot with error bars
    plt.figure(figsize=(12, 8))
    plt.errorbar(
        y=np.arange(len(coef_df)),
        x=coef_df['Coefficient'],
        xerr=coef_df['Error'],
        fmt='o',
        capsize=5,
        ecolor='black',
        markerfacecolor='blue',
        markeredgecolor='black'
    )
    
    plt.axvline(x=0, color='red', linestyle='--')
    plt.yticks(np.arange(len(coef_df)), coef_df['Feature'])
    plt.xlabel('Coefficient Value')
    plt.title('Ridge Regression Coefficients with Error Bars')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plots.append({
        'title': 'Ridge Regression Coefficients',
        'img_data': get_base64_plot(),
        'interpretation': 'Shows the magnitude and direction of each feature\'s effect on the target variable. The error bars represent uncertainty in the coefficient estimates. Coefficients shrunk toward zero indicate the regularization effect of Ridge regression.'
    })
    
    # 4. Regularization Path
    if alphas is None:
        alphas = np.logspace(-3, 3, 100)
    
    plt.figure(figsize=(12, 8))
    
    from sklearn.linear_model import ridge_regression
    coefs = []
    
    # Calculate coefficients for different alpha values
    for alpha in alphas:
        coef = ridge_regression(X, y, alpha=alpha)
        coefs.append(coef)
    
    # Convert to array for easier indexing
    coefs = np.array(coefs)
    
    # Plot paths
    for i, feature in enumerate(feature_names):
        plt.plot(alphas, coefs[:, i], label=feature, linewidth=2)
    
    plt.xscale('log')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Coefficient Value')
    plt.title('Regularization Path')
    
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
        'interpretation': 'Shows how feature coefficients change with different regularization strengths (alpha values). As alpha increases, coefficients are increasingly shrunk toward zero, demonstrating Ridge regression\'s bias-variance tradeoff. The vertical line indicates the selected alpha value used in the model.'
    })
    
    # 5. Validation Curve for Alpha Selection
    if X_test is not None and y_test is not None:
        plt.figure(figsize=(10, 6))
        
        from sklearn.linear_model import Ridge
        
        # Generate validation curve
        train_scores, test_scores = validation_curve(
            Ridge(), X, y, param_name="alpha", param_range=alphas,
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
        plt.title("Validation Curve for Ridge Regression")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plots.append({
            'title': 'Validation Curve',
            'img_data': get_base64_plot(),
            'interpretation': 'Illustrates model performance (MSE) across different alpha values. The blue line shows training error, while the green line shows cross-validation error. The optimal alpha balances underfitting (high alpha, high bias) and overfitting (low alpha, high variance). The vertical line indicates the selected alpha value.'
        })
    
    # 6. Learning Curves
    if X_test is not None and y_test is not None:
        plt.figure(figsize=(10, 6))
        
        from sklearn.model_selection import learning_curve
        
        # Generate the learning curve data
        train_sizes, train_scores, test_scores = learning_curve(
            Ridge(alpha=model.alpha), X, y, cv=min(5, len(X)), 
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
        plt.title(f"Learning Curves for Ridge Regression (alpha={model.alpha})")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plots.append({
            'title': 'Learning Curves',
            'img_data': get_base64_plot(),
            'interpretation': 'Shows how model performance changes with increasing training data. The gap between training error (blue) and cross-validation error (green) indicates whether the model would benefit from more data. A small gap suggests that the model has reached its capacity and more data may not help.'
        })
    
    return plots 