"""Regularized regression diagnostic plots (Ridge, Lasso, Elastic Net)."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_regularized_regression_plots(model, X=None, y=None, X_test=None, y_test=None, 
                                         feature_names=None, model_type=None):
    """Generate diagnostic plots for regularized regression models
    
    Args:
        model: Fitted regularized regression model (Ridge, Lasso, ElasticNet)
        X: Feature matrix
        y: Target variable
        X_test: Test data features (optional)
        y_test: Test data target (optional)
        feature_names: Names of features (optional)
        model_type: Type of model ('ridge', 'lasso', or 'elasticnet')
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Determine model type if not provided
    if model_type is None:
        if hasattr(model, '__class__') and hasattr(model.__class__, '__name__'):
            class_name = model.__class__.__name__.lower()
            if 'ridge' in class_name:
                model_type = 'ridge'
            elif 'lasso' in class_name:
                model_type = 'lasso'
            elif 'elastic' in class_name or 'elasticnet' in class_name:
                model_type = 'elasticnet'
            else:
                model_type = 'regularized'
        else:
            model_type = 'regularized'
    
    # Get feature names if not provided
    if feature_names is None and X is not None:
        if hasattr(X, 'columns'):  # If X is a DataFrame
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"Feature {i+1}" for i in range(X.shape[1] if X.ndim > 1 else 1)]
    
    # Plot 1: Coefficients with and without regularization (if OLS is available)
    if hasattr(model, 'coef_') and X is not None and y is not None:
        plt.figure(figsize=(12, max(6, len(feature_names) * 0.3)))
        
        # Get regularized coefficients
        reg_coefs = model.coef_
        if not isinstance(reg_coefs, np.ndarray):
            reg_coefs = np.array(reg_coefs)
        
        # Compute OLS coefficients for comparison
        try:
            from sklearn.linear_model import LinearRegression
            ols_model = LinearRegression()
            ols_model.fit(X, y)
            ols_coefs = ols_model.coef_
            
            # Handle intercept separately
            if hasattr(model, 'intercept_') and hasattr(ols_model, 'intercept_'):
                reg_intercept = model.intercept_
                ols_intercept = ols_model.intercept_
                
                # Add intercept to coefficient arrays
                reg_coefs = np.append(reg_coefs, reg_intercept)
                ols_coefs = np.append(ols_coefs, ols_intercept)
                feature_names_with_intercept = feature_names + ['Intercept']
            else:
                feature_names_with_intercept = feature_names
            
            # Create DataFrame for plotting
            coef_df = pd.DataFrame({
                'Feature': feature_names_with_intercept,
                'OLS': ols_coefs,
                f'{model_type.capitalize()}': reg_coefs
            })
            
            # Melt for seaborn plotting
            coef_df_melted = pd.melt(coef_df, id_vars='Feature', 
                                    var_name='Model', value_name='Coefficient')
            
            # Plot coefficients side by side
            sns.barplot(x='Coefficient', y='Feature', hue='Model', data=coef_df_melted)
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            plt.title(f'Coefficient Comparison: OLS vs {model_type.capitalize()}')
            plt.tight_layout()
            
            plots.append({
                "title": "Coefficient Comparison",
                "img_data": get_base64_plot(),
                "interpretation": f"Compares coefficients from OLS regression (no regularization) to {model_type.capitalize()} regression. {model_type.capitalize()} tends to shrink coefficients toward zero, with Lasso potentially eliminating some entirely. Large differences between OLS and regularized coefficients may indicate high multicollinearity or noise in the data."
            })
        except:
            # If OLS comparison fails, just plot regularized coefficients
            plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
            
            # Sort by coefficient magnitude for clarity
            sorted_idx = np.argsort(np.abs(reg_coefs))
            
            plt.barh(range(len(sorted_idx)), reg_coefs[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            plt.xlabel('Coefficient Value')
            plt.title(f'{model_type.capitalize()} Regression Coefficients')
            plt.tight_layout()
            
            plots.append({
                "title": f"{model_type.capitalize()} Coefficients",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows the coefficient values for {model_type.capitalize()} regression. The magnitude indicates each feature's importance in the model, with zero or near-zero coefficients suggesting features that have been effectively removed by regularization."
            })
    
    # Plot 2: Actual vs Predicted values
    if X is not None and y is not None and hasattr(model, 'predict'):
        plt.figure(figsize=(10, 8))
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Scatter plot
        plt.scatter(y, y_pred, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(min(y), min(y_pred))
        max_val = max(max(y), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.grid(True, alpha=0.3)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Add metrics annotation
        plt.annotate(f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plots.append({
            "title": "Actual vs Predicted Values",
            "img_data": get_base64_plot(),
            "interpretation": f"Shows how well the model's predictions match actual values. Points closer to the red diagonal line indicate better predictions. RMSE (Root Mean Squared Error) of {rmse:.3f} penalizes large errors, while MAE (Mean Absolute Error) of {mae:.3f} shows average error magnitude. R² of {r2:.3f} indicates the proportion of variance explained (closer to 1 is better)."
        })
        
        # Plot for test data if provided
        if X_test is not None and y_test is not None:
            plt.figure(figsize=(10, 8))
            
            # Get test predictions
            y_test_pred = model.predict(X_test)
            
            # Scatter plot
            plt.scatter(y_test, y_test_pred, alpha=0.6)
            
            # Add perfect prediction line
            min_val = min(min(y_test), min(y_test_pred))
            max_val = max(max(y_test), max(y_test_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            plt.xlabel('Actual Values (Test Set)')
            plt.ylabel('Predicted Values (Test Set)')
            plt.title('Test Set: Actual vs Predicted Values')
            plt.grid(True, alpha=0.3)
            
            # Calculate metrics for test set
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Add metrics annotation
            plt.annotate(f'RMSE: {test_rmse:.3f}\nMAE: {test_mae:.3f}\nR²: {test_r2:.3f}', 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            plots.append({
                "title": "Test Set Performance",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows model performance on unseen test data. Compare these metrics (RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}, R²: {test_r2:.3f}) with training metrics to assess overfitting. Similar performance suggests good generalization, while significantly worse test performance indicates potential overfitting."
            })
    
    # Plot 3: Residuals vs Predicted values
    if X is not None and y is not None and hasattr(model, 'predict'):
        plt.figure(figsize=(10, 6))
        
        # Get predictions and residuals
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Scatter plot
        plt.scatter(y_pred, residuals, alpha=0.6)
        
        # Add zero line
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add trend line
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smooth = lowess(residuals, y_pred, frac=0.6)
            plt.plot(smooth[:, 0], smooth[:, 1], 'g-', lw=2)
        except:
            pass
            
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Residuals vs Predicted",
            "img_data": get_base64_plot(),
            "interpretation": "Checks for patterns in residuals across the range of predictions. Ideally, residuals should be randomly scattered around zero (red line) with no clear pattern, indicating homoscedasticity. Funneling patterns suggest heteroscedasticity, while curves indicate non-linearity not captured by the model."
        })
        
        # Plot 4: Residual distribution
        plt.figure(figsize=(10, 6))
        
        # Histogram with KDE
        sns.histplot(residuals, kde=True, bins=30)
        
        # Add vertical line at zero
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        
        plots.append({
            "title": "Residual Distribution",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the distribution of residuals. Ideally, residuals should follow a normal distribution centered at zero (red line), indicating that errors are random and the model has captured the underlying patterns in the data. Skewness or multiple peaks may suggest model misspecification."
        })
    
    # Plot 5: Regularization path (coefficient values vs regularization strength)
    if model_type in ['lasso', 'elasticnet', 'ridge']:
        try:
            plt.figure(figsize=(12, 8))
            
            # Choose appropriate path function based on model type
            if model_type == 'lasso':
                from sklearn.linear_model import lasso_path
                alpha_range = np.logspace(-3, 2, 100)
                coefs, alphas, _ = lasso_path(X, y, alphas=alpha_range, fit_intercept=hasattr(model, 'fit_intercept') and model.fit_intercept)
            elif model_type == 'ridge':
                from sklearn.linear_model import ridge_regression
                alpha_range = np.logspace(-3, 3, 100)
                coefs = []
                for alpha in alpha_range:
                    coef = ridge_regression(X, y, alpha=alpha, fit_intercept=hasattr(model, 'fit_intercept') and model.fit_intercept)
                    coefs.append(coef)
                coefs = np.array(coefs).T
                alphas = alpha_range
            elif model_type == 'elasticnet':
                from sklearn.linear_model import enet_path
                alpha_range = np.logspace(-3, 2, 100)
                l1_ratio = model.l1_ratio if hasattr(model, 'l1_ratio') else 0.5
                coefs, alphas, _ = enet_path(X, y, l1_ratio=l1_ratio, alphas=alpha_range, fit_intercept=hasattr(model, 'fit_intercept') and model.fit_intercept)
            
            # Plot regularization path
            plt.figure(figsize=(12, 8))
            
            # For each feature, plot a line showing how its coefficient changes with regularization
            for i, feature in enumerate(feature_names):
                plt.semilogx(alphas, coefs[i], label=feature)
            
            # Mark the selected alpha value
            if hasattr(model, 'alpha'):
                plt.axvline(x=model.alpha, color='r', linestyle='--', alpha=0.7)
                plt.text(model.alpha, 0, f'Selected α: {model.alpha:.3g}', 
                        bbox=dict(facecolor='white', alpha=0.5),
                        horizontalalignment='center', verticalalignment='top')
            
            plt.xlabel('Alpha (regularization strength)')
            plt.ylabel('Coefficient Value')
            plt.title(f'{model_type.capitalize()} Path: Coefficients vs Regularization Strength')
            
            # Handle legend
            if len(feature_names) > 10:
                # Too many features, just show a few important ones
                top_features_idx = np.argsort(np.abs(model.coef_))[-5:]
                handles, labels = plt.gca().get_legend_handles_labels()
                plt.legend([handles[i] for i in top_features_idx], 
                          [labels[i] for i in top_features_idx], 
                          loc='best')
            else:
                plt.legend(loc='best')
            
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": f"{model_type.capitalize()} Path",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows how coefficient values change with regularization strength (alpha). As alpha increases, coefficients are shrunk toward zero. Features whose lines cross zero are completely eliminated at higher alpha values (for Lasso and ElasticNet). The vertical line indicates the selected alpha used in the model."
            })
        except:
            # Skip this plot if regularization path calculation fails
            pass
    
    # Plot 6: Cross-validation curve (if model has a CV version)
    cv_attribute = None
    if hasattr(model, 'alpha_') and hasattr(model, 'cv_alphas_') and hasattr(model, 'mse_path_'):
        # LassoCV or ElasticNetCV
        cv_attribute = {
            'alphas': model.cv_alphas_,
            'scores': model.mse_path_.mean(axis=1),
            'std': model.mse_path_.std(axis=1),
            'selected': model.alpha_,
            'metric': 'Mean Squared Error',
            'selected_idx': np.argmin(np.mean(model.mse_path_, axis=1))
        }
    elif hasattr(model, 'cv_values_'):
        # RidgeCV
        cv_attribute = {
            'alphas': model.alphas if hasattr(model, 'alphas') else np.logspace(-3, 3, len(model.cv_values_.mean(axis=0))),
            'scores': -model.cv_values_.mean(axis=0) if hasattr(model, 'cv_values_') else None,
            'std': model.cv_values_.std(axis=0) if hasattr(model, 'cv_values_') else None,
            'selected': model.alpha_ if hasattr(model, 'alpha_') else None,
            'metric': 'Mean Squared Error',
            'selected_idx': np.argmin(-model.cv_values_.mean(axis=0)) if hasattr(model, 'cv_values_') else None
        }
    
    if cv_attribute is not None and cv_attribute['scores'] is not None:
        plt.figure(figsize=(10, 6))
        
        # Plot scores with error bars
        plt.errorbar(cv_attribute['alphas'], cv_attribute['scores'], 
                    yerr=cv_attribute['std'], fmt='o-', alpha=0.8)
        
        # Mark the selected alpha
        if cv_attribute['selected'] is not None:
            min_score = cv_attribute['scores'][cv_attribute['selected_idx']]
            plt.plot(cv_attribute['selected'], min_score, 'ro', markersize=10)
            plt.annotate(f'Selected α: {cv_attribute["selected"]:.3g}',
                        xy=(cv_attribute['selected'], min_score),
                        xytext=(cv_attribute['selected'] * 2, min_score * 1.1),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        plt.xscale('log')
        plt.xlabel('Alpha (regularization strength)')
        plt.ylabel(cv_attribute['metric'])
        plt.title('Cross-Validation Performance vs Regularization Strength')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Cross-Validation Performance",
            "img_data": get_base64_plot(),
            "interpretation": f"Shows model performance across different regularization strengths during cross-validation. The optimal alpha (marked with a red circle) balances between underfitting (high alpha, high error) and overfitting (low alpha, high error). Error bars represent variation across cross-validation folds."
        })
    
    # Plot 7: Feature correlation heatmap
    if X is not None and feature_names is not None:
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        if hasattr(X, 'corr'):
            # If X is a DataFrame, use its corr method
            corr_matrix = X.corr()
        else:
            # Otherwise, convert to DataFrame first
            corr_matrix = pd.DataFrame(X, columns=feature_names).corr()
        
        # Plot heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                   annot=True if len(feature_names) < 15 else False, 
                   fmt='.2f', square=True, linewidths=.5)
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        plots.append({
            "title": "Feature Correlation Matrix",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the correlation between pairs of features. High correlations (near 1 or -1) indicate multicollinearity, which regularization helps address. Features with high correlation may have unstable coefficient estimates in standard regression but become more stable with regularization."
        })
    
    # Plot 8: Learning Curve
    if X is not None and y is not None:
        try:
            from sklearn.model_selection import learning_curve
            
            plt.figure(figsize=(10, 6))
            
            # Calculate learning curve
            n_samples = len(X)
            train_sizes = np.linspace(0.1, 1.0, min(10, n_samples // 5))
            train_sizes, train_scores, test_scores = learning_curve(
                model.__class__(**{k: v for k, v in model.get_params().items() 
                                if k != 'cv' and k != 'alphas' and k != 'gcv_mode'}), 
                X, y, train_sizes=train_sizes, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            
            # Convert MSE to positive values for plotting
            train_scores = -train_scores
            test_scores = -test_scores
            
            # Calculate means and standard deviations
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Plot learning curve
            plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training error")
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                           alpha=0.1, color="r")
            plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation error")
            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                           alpha=0.1, color="g")
            
            plt.title("Learning Curve")
            plt.xlabel("Training Examples")
            plt.ylabel("Mean Squared Error")
            plt.legend(loc="best")
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Learning Curve",
                "img_data": get_base64_plot(),
                "interpretation": "Shows how model error changes with increasing training data. A gap between training and validation errors indicates high variance (overfitting). If both curves plateau at high error, the model may have high bias (underfitting). Regularization helps reduce variance at the cost of some bias."
            })
        except:
            # Skip this plot if learning curve calculation fails
            pass
    
    return plots 