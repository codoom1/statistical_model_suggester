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

def generate_elastic_net_plots(model, X, y, X_test=None, y_test=None, feature_names=None, alphas=None, l1_ratios=None):
    """
    Generate diagnostic plots for Elastic Net Regression models.
    
    Parameters:
    -----------
    model : ElasticNet model object
        The fitted ElasticNet regression model
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
    l1_ratios : array-like, optional
        L1 ratio values for mixing parameter plot
    
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
    
    # Default parameter values if not provided
    if alphas is None:
        alphas = np.logspace(-4, 1, 50)
    if l1_ratios is None:
        l1_ratios = np.linspace(0.01, 0.99, 10)
    
    # Get current parameters
    try:
        current_alpha = model.alpha
        current_l1_ratio = model.l1_ratio
    except:
        current_alpha = 0.5
        current_l1_ratio = 0.5
    
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
    
    # 3. Coefficient Plot
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
    plt.title(f'Elastic Net Coefficients\n{non_zero_count} non-zero, {zero_count} zero coefficients')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Non-zero coefficients'),
                       Patch(facecolor='red', label='Zero coefficients')]
    plt.legend(handles=legend_elements, loc='best')
    
    plots.append({
        'title': 'Elastic Net Coefficients',
        'img_data': get_base64_plot(),
        'interpretation': f'Shows the magnitude and direction of each feature\'s effect on the target variable. Elastic Net has selected {non_zero_count} features and eliminated {zero_count} features by setting their coefficients to exactly zero. This demonstrates the combination of Ridge regularization (shrinking coefficients) and Lasso sparsity (feature selection).'
    })
    
    # 4. Regularization Path (Alpha path for fixed L1 ratio)
    plt.figure(figsize=(12, 8))
    
    # Use enet_path or manually compute coefficients for different alphas
    try:
        from sklearn.linear_model import enet_path
        _, coefs_path, _ = enet_path(X, y, l1_ratio=current_l1_ratio, alphas=alphas, fit_intercept=True)
        
        # Plot paths
        for i, feature in enumerate(feature_names):
            plt.plot(alphas, coefs_path[i], label=feature, linewidth=2)
    except:
        # Fallback if enet_path is not available
        from sklearn.linear_model import ElasticNet
        coefs_path = np.zeros((len(feature_names), len(alphas)))
        
        for i, alpha in enumerate(alphas):
            enet = ElasticNet(alpha=alpha, l1_ratio=current_l1_ratio)
            enet.fit(X, y)
            coefs_path[:, i] = enet.coef_
        
        # Plot paths
        for i, feature in enumerate(feature_names):
            plt.plot(alphas, coefs_path[i], label=feature, linewidth=2)
    
    plt.xscale('log')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Coefficient Value')
    plt.title(f'Elastic Net Regularization Path (L1 ratio={current_l1_ratio})')
    
    # Highlight the chosen alpha
    plt.axvline(x=current_alpha, color='red', linestyle='--', label=f'Selected Alpha: {current_alpha}')
    
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
        'interpretation': f'Shows how feature coefficients change with different regularization strengths (alpha values) at a fixed L1 ratio of {current_l1_ratio}. As alpha increases, coefficients shrink toward zero, with some becoming exactly zero. The vertical line indicates the selected alpha value used in the model.'
    })
    
    # 5. L1 Ratio Path (for fixed alpha)
    plt.figure(figsize=(12, 8))
    
    # Compute coefficients for different L1 ratios
    from sklearn.linear_model import ElasticNet
    l1_ratio_coefs = np.zeros((len(feature_names), len(l1_ratios)))
    
    for i, l1_ratio in enumerate(l1_ratios):
        enet = ElasticNet(alpha=current_alpha, l1_ratio=l1_ratio)
        enet.fit(X, y)
        l1_ratio_coefs[:, i] = enet.coef_
    
    # Plot paths
    for i, feature in enumerate(feature_names):
        plt.plot(l1_ratios, l1_ratio_coefs[i], label=feature, linewidth=2)
    
    plt.xlabel('L1 Ratio (0=Ridge, 1=Lasso)')
    plt.ylabel('Coefficient Value')
    plt.title(f'Coefficient Values vs L1 Ratio (Alpha={current_alpha})')
    
    # Highlight the chosen L1 ratio
    plt.axvline(x=current_l1_ratio, color='red', linestyle='--', label=f'Selected L1 Ratio: {current_l1_ratio}')
    
    # Add legend
    if len(feature_names) <= 10:
        plt.legend(loc='best')
    else:
        # If too many features, only show the legend for selected important features
        top_indices = np.argsort(np.abs(model.coef_))[-5:]  # Top 5 features
        handles, labels = plt.gca().get_legend_handles_labels()
        selected_handles = [handles[i] for i in top_indices] + [handles[-1]]  # Add L1 ratio line
        selected_labels = [labels[i] for i in top_indices] + [labels[-1]]
        plt.legend(selected_handles, selected_labels, loc='best')
    
    plots.append({
        'title': 'L1 Ratio Effect',
        'img_data': get_base64_plot(),
        'interpretation': f'Shows how feature coefficients change with different L1 ratios at a fixed alpha of {current_alpha}. As the L1 ratio increases from 0 (Ridge) to 1 (Lasso), the model transitions from shrinking coefficients toward zero to setting some exactly to zero. The vertical line indicates the selected L1 ratio used in the model.'
    })
    
    # 6. Validation Curve for Alpha Selection
    if X_test is not None and y_test is not None:
        plt.figure(figsize=(10, 6))
        
        from sklearn.linear_model import ElasticNet
        
        # Create a custom estimator with fixed L1 ratio
        class ElasticNetFixedL1(ElasticNet):
            def __init__(self, alpha=1.0):
                super().__init__(alpha=alpha, l1_ratio=current_l1_ratio)
        
        # Generate validation curve
        train_scores, test_scores = validation_curve(
            ElasticNetFixedL1(), X, y, param_name="alpha", param_range=alphas,
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
        plt.axvline(x=current_alpha, color='red', linestyle='--', label=f'Selected Alpha: {current_alpha}')
        
        plt.xscale("log")
        plt.xlabel("Alpha (log scale)")
        plt.ylabel("Mean Squared Error")
        plt.title(f"Validation Curve for Elastic Net (L1 ratio={current_l1_ratio})")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plots.append({
            'title': 'Validation Curve (Alpha)',
            'img_data': get_base64_plot(),
            'interpretation': f'Illustrates model performance (MSE) across different alpha values with a fixed L1 ratio of {current_l1_ratio}. The blue line shows training error, while the green line shows cross-validation error. The optimal alpha balances underfitting (high alpha, high bias) and overfitting (low alpha, high variance). The vertical line indicates the selected alpha value.'
        })
    
    # 7. 2D Heatmap (Alpha vs L1 Ratio)
    if X_test is not None and y_test is not None:
        plt.figure(figsize=(12, 10))
        
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import ElasticNet
        
        # Create grid for heatmap
        alphas_grid = np.logspace(-2, 1, 8)  # Fewer points for reasonable computation time
        l1_ratios_grid = np.linspace(0.1, 0.9, 9)
        
        # Compute MSE for each combination
        param_grid = {'alpha': alphas_grid, 'l1_ratio': l1_ratios_grid}
        
        # Use GridSearchCV to generate performance metrics
        grid = GridSearchCV(
            ElasticNet(), param_grid=param_grid,
            cv=min(3, len(X)), scoring='neg_mean_squared_error'
        )
        grid.fit(X, y)
        
        # Extract results
        results = pd.DataFrame(grid.cv_results_)
        
        # Compute mean test score (convert from negative to positive MSE)
        mse_scores = -results['mean_test_score'].values
        
        # Reshape for heatmap
        mse_matrix = mse_scores.reshape(len(alphas_grid), len(l1_ratios_grid)).T
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(mse_matrix, annot=True, fmt='.3g', 
                    xticklabels=[f'{a:.3g}' for a in alphas_grid],
                    yticklabels=[f'{l:.1f}' for l in l1_ratios_grid],
                    cmap='viridis_r')  # Reverse colormap so darker = better
        
        # Find the best combination
        best_idx = np.argmin(mse_scores)
        best_alpha = grid.cv_results_['param_alpha'][best_idx]
        best_l1 = grid.cv_results_['param_l1_ratio'][best_idx]
        
        # Mark the best combination
        best_alpha_idx = np.where(alphas_grid == best_alpha)[0][0]
        best_l1_idx = np.where(l1_ratios_grid == best_l1)[0][0]
        plt.plot(best_alpha_idx + 0.5, best_l1_idx + 0.5, 'r*', markersize=12)
        
        # Mark the current model parameters
        current_alpha_idx = np.abs(alphas_grid - current_alpha).argmin()
        current_l1_idx = np.abs(l1_ratios_grid - current_l1_ratio).argmin()
        plt.plot(current_alpha_idx + 0.5, current_l1_idx + 0.5, 'wx', markersize=12, mew=3)
        
        plt.xlabel('Alpha')
        plt.ylabel('L1 Ratio')
        plt.title('Mean Squared Error (MSE) for Alpha and L1 Ratio Combinations')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='r', markersize=10, label='Best Combination'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor='w', markeredgecolor='w', markersize=10, label='Current Model')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plots.append({
            'title': 'Alpha vs L1 Ratio Heatmap',
            'img_data': get_base64_plot(),
            'interpretation': f'Shows model performance (MSE) across different combinations of alpha and L1 ratio values. The red star indicates the best performing combination (α={best_alpha:.3g}, L1 ratio={best_l1:.1f}), while the white X shows the current model parameters (α={current_alpha:.3g}, L1 ratio={current_l1_ratio:.1f}). Darker colors indicate better performance (lower MSE).'
        })
    
    # 8. Learning Curves
    if X_test is not None and y_test is not None:
        plt.figure(figsize=(10, 6))
        
        from sklearn.model_selection import learning_curve
        
        # Generate the learning curve data
        train_sizes, train_scores, test_scores = learning_curve(
            ElasticNet(alpha=current_alpha, l1_ratio=current_l1_ratio), X, y, 
            cv=min(5, len(X)), train_sizes=np.linspace(0.1, 1.0, 10),
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
        plt.title(f"Learning Curves for Elastic Net (α={current_alpha}, L1 ratio={current_l1_ratio})")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plots.append({
            'title': 'Learning Curves',
            'img_data': get_base64_plot(),
            'interpretation': 'Shows how model performance changes with increasing training data. The gap between training error (blue) and cross-validation error (green) indicates whether the model would benefit from more data. A small gap suggests that the model has reached its capacity and more data may not help.'
        })
    
    return plots 