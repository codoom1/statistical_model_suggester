import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import base64
from io import BytesIO
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import io

def get_base64_plot():
    """Convert the current matplotlib plot to a base64 string."""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

def generate_bart_plots(model, X_train, y_train, X_test, y_test, feature_names=None, num_samples=100):
    """
    Generate diagnostic plots for Bayesian Additive Regression Trees (BART) model.
    
    Parameters:
    -----------
    model : BART model object
        The fitted BART model (from pybart, bartpy, or similar library)
    X_train : array-like
        Training data features
    y_train : array-like
        Training data target
    X_test : array-like
        Test data features
    y_test : array-like
        Test data target
    feature_names : list, optional
        Names of the features
    num_samples : int, optional
        Number of posterior samples to use for visualizations
    
    Returns:
    --------
    plots : list of dict
        List of dictionaries containing:
        - 'title': Title of the plot
        - 'img': Base64 encoded image
        - 'interpretation': Interpretation of the plot
    """
    if feature_names is None and hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()
    elif feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X_train.shape[1])]
    
    plots = []
    
    # Convert inputs to numpy arrays for consistency
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)
    if not isinstance(X_test, np.ndarray):
        X_test = np.array(X_test)
    if not isinstance(y_test, np.ndarray):
        y_test = np.array(y_test)
    
    # Get predictions and prediction intervals if available
    try:
        # Different BART implementations have different prediction methods
        # Try common formats
        if hasattr(model, 'predict'):
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
        elif hasattr(model, 'predict_numpy'):
            y_pred_test = model.predict_numpy(X_test)
            y_pred_train = model.predict_numpy(X_train)
        
        # Try to get prediction intervals if available
        has_intervals = False
        try:
            if hasattr(model, 'predict_intervals'):
                lower, upper = model.predict_intervals(X_test)
                has_intervals = True
            elif hasattr(model, 'predict_interval'):
                intervals = model.predict_interval(X_test)
                lower, upper = intervals[:, 0], intervals[:, 1]
                has_intervals = True
        except:
            has_intervals = False
    except:
        # If prediction methods fail, return empty plots list
        return plots
    
    # 1. Posterior Predictive Distribution
    try:
        plt.figure(figsize=(10, 6))
        
        # Try to get posterior samples if available
        try:
            if hasattr(model, 'get_samples'):
                samples = model.get_samples(X_test, num_samples=num_samples)
            elif hasattr(model, 'predict_posterior'):
                samples = model.predict_posterior(X_test, num_samples=num_samples)
            elif hasattr(model, 'partial_fit'):
                # For some implementations, we can get posterior samples this way
                samples = np.array([model.predict(X_test) for _ in range(num_samples)])
            
            # Plot posterior predictive for a few test points
            for i in range(min(5, X_test.shape[0])):
                sns.kdeplot(samples[:, i], label=f'Test point {i+1}')
                plt.axvline(y_test[i], color='red', linestyle='--')
            
            plt.title('Posterior Predictive Distribution')
            plt.xlabel('Predicted value')
            plt.ylabel('Density')
            plt.legend()
            
            plots.append({
                'title': 'Posterior Predictive Distribution',
                'img': get_base64_plot(),
                'interpretation': 'Shows the uncertainty in predictions for a few test points. Wider distributions indicate more uncertainty in the prediction. Red dotted lines show actual values.'
            })
        except:
            pass
    except:
        pass
    
    # 2. Actual vs. Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.6)
    
    # Add prediction intervals if available
    if has_intervals:
        # Sort by y_test for cleaner plotting
        idx = np.argsort(y_test)
        plt.fill_between(
            y_test[idx],
            lower[idx],
            upper[idx],
            alpha=0.2,
            color='gray',
            label='95% Prediction Interval'
        )
    
    # Add identity line
    min_val = min(np.min(y_test), np.min(y_pred_test))
    max_val = max(np.max(y_test), np.max(y_pred_test))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    plt.title(f'Actual vs. Predicted\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    if has_intervals:
        plt.legend()
    
    plots.append({
        'title': 'Actual vs. Predicted',
        'img': get_base64_plot(),
        'interpretation': f'Shows how well the BART model predictions match actual values. RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}. Closer to the identity line (red dashed) indicates better performance.'
    })
    
    # 3. Residuals Analysis
    residuals = y_test - y_pred_test
    
    plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2)
    
    # Residuals vs. Predicted
    ax1 = plt.subplot(gs[0, 0])
    ax1.scatter(y_pred_test, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_title('Residuals vs. Predicted')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Residuals')
    
    # Residuals distribution
    ax2 = plt.subplot(gs[0, 1])
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_title('Residuals Distribution')
    ax2.set_xlabel('Residuals')
    
    # QQ plot of residuals
    ax3 = plt.subplot(gs[1, 0])
    from scipy import stats
    stats.probplot(residuals, plot=ax3)
    ax3.set_title('QQ Plot of Residuals')
    
    # Residuals vs. Index
    ax4 = plt.subplot(gs[1, 1])
    ax4.scatter(range(len(residuals)), residuals, alpha=0.6)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_title('Residuals vs. Index')
    ax4.set_xlabel('Index')
    ax4.set_ylabel('Residuals')
    
    plt.tight_layout()
    
    plots.append({
        'title': 'Residuals Analysis',
        'img': get_base64_plot(),
        'interpretation': 'Residuals diagnostics reveal if model assumptions are met. Look for: (1) Residuals vs. Predicted: no pattern indicates homoscedasticity. (2) Residuals Distribution: bell-shaped suggests normality. (3) QQ Plot: points following straight line confirm normality. (4) Residuals vs. Index: no pattern confirms independence.'
    })
    
    # 4. Variable Importance
    try:
        # Try different approaches to get variable importance, depending on the BART implementation
        if hasattr(model, 'variable_importance') or hasattr(model, 'feature_importance'):
            # Direct method if available
            if hasattr(model, 'variable_importance'):
                importance = model.variable_importance()
            else:
                importance = model.feature_importance()
                
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Variable Importance')
            plt.tight_layout()
            
            plots.append({
                'title': 'Variable Importance',
                'img': get_base64_plot(),
                'interpretation': 'Shows the relative importance of each feature in the BART model. Features with higher values have more influence on predictions.'
            })
        else:
            # Use permutation importance as a fallback
            perm_importance = permutation_importance(
                estimator=lambda X: model.predict(X), 
                X=X_test, 
                y=y_test, 
                n_repeats=5, 
                random_state=42
            )
            
            plt.figure(figsize=(10, 6))
            sorted_idx = perm_importance.importances_mean.argsort()
            importances_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in sorted_idx],
                'Importance': perm_importance.importances_mean[sorted_idx]
            })
            
            sns.barplot(x='Importance', y='Feature', data=importances_df)
            plt.title('Permutation Feature Importance')
            plt.tight_layout()
            
            plots.append({
                'title': 'Permutation Feature Importance',
                'img': get_base64_plot(),
                'interpretation': 'Shows the importance of each feature based on the decrease in model performance when the feature is randomly permuted. Features with higher values have more influence on predictions.'
            })
    except:
        pass
    
    # 5. Tree Structure Visualization (if available)
    try:
        if hasattr(model, 'get_trees') or hasattr(model, 'trees'):
            plt.figure(figsize=(12, 8))
            
            if hasattr(model, 'get_trees'):
                trees = model.get_trees()
            else:
                trees = model.trees
            
            # Depending on implementation, visualize a few trees
            # Here we just show the variable split counts from the first few trees
            var_counts = np.zeros(len(feature_names))
            
            # Collect feature usage across trees
            num_trees_to_analyze = min(10, len(trees))
            for i in range(num_trees_to_analyze):
                try:
                    tree = trees[i]
                    # Extract feature usage - this is implementation specific
                    # This is a simplified approach that may need to be adapted
                    if hasattr(tree, 'get_feature_counts'):
                        counts = tree.get_feature_counts()
                        for feature_idx, count in counts.items():
                            var_counts[feature_idx] += count
                    elif hasattr(tree, 'nodes'):
                        for node in tree.nodes:
                            if hasattr(node, 'feature_idx') and node.feature_idx is not None:
                                var_counts[node.feature_idx] += 1
                except:
                    continue
            
            # Plot feature usage in trees
            feature_usage = pd.DataFrame({
                'Feature': feature_names,
                'Usage Count': var_counts
            })
            feature_usage = feature_usage.sort_values('Usage Count', ascending=False)
            
            sns.barplot(x='Usage Count', y='Feature', data=feature_usage)
            plt.title(f'Feature Usage in First {num_trees_to_analyze} Trees')
            plt.tight_layout()
            
            plots.append({
                'title': 'Tree Structure Analysis',
                'img': get_base64_plot(),
                'interpretation': f'Shows which features are most frequently used as split points in the first {num_trees_to_analyze} trees of the BART model. Features used more often have greater influence in the model structure.'
            })
    except:
        pass
    
    # 6. Partial Dependence Plots
    try:
        # Select top 4 most important features based on permutation importance
        perm_importance = permutation_importance(
            estimator=lambda X: model.predict(X), 
            X=X_test, 
            y=y_test, 
            n_repeats=5, 
            random_state=42
        )
        
        top_k = min(4, len(feature_names))
        top_features = np.argsort(perm_importance.importances_mean)[-top_k:]
        
        plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2)
        
        for i, feature_idx in enumerate(top_features):
            feature_name = feature_names[feature_idx]
            
            # Create feature grid
            feature_values = np.linspace(
                np.min(X_train[:, feature_idx]),
                np.max(X_train[:, feature_idx]),
                num=20
            )
            
            # Create data with varying target feature
            X_pd = np.tile(X_train.mean(axis=0), (len(feature_values), 1))
            X_pd[:, feature_idx] = feature_values
            
            # Predict using the model
            y_pd = model.predict(X_pd)
            
            # Plot partial dependence
            ax = plt.subplot(gs[i // 2, i % 2])
            ax.plot(feature_values, y_pd)
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Partial Dependence')
            ax.set_title(f'Partial Dependence for {feature_name}')
        
        plt.tight_layout()
        
        plots.append({
            'title': 'Partial Dependence Plots',
            'img': get_base64_plot(),
            'interpretation': 'Shows how the predicted value changes when one feature is varied while others remain constant at their average value. Reveals the marginal effect of each feature on the prediction, controlling for other features.'
        })
    except:
        pass
    
    # 7. Uncertainty across feature space (for a 2D slice)
    try:
        if X_train.shape[1] >= 2:
            # Select top 2 features from importance
            perm_importance = permutation_importance(
                estimator=lambda X: model.predict(X), 
                X=X_test, 
                y=y_test, 
                n_repeats=5, 
                random_state=42
            )
            
            top_2_features = np.argsort(perm_importance.importances_mean)[-2:]
            feat1, feat2 = top_2_features
            
            # Create a 2D grid for the top 2 features
            f1_values = np.linspace(
                np.min(X_train[:, feat1]),
                np.max(X_train[:, feat1]),
                num=20
            )
            
            f2_values = np.linspace(
                np.min(X_train[:, feat2]),
                np.max(X_train[:, feat2]),
                num=20
            )
            
            # Create meshgrid
            f1_grid, f2_grid = np.meshgrid(f1_values, f2_values)
            grid_points = np.column_stack([f1_grid.ravel(), f2_grid.ravel()])
            
            # Create full grid with average values for other features
            X_grid = np.tile(X_train.mean(axis=0), (len(grid_points), 1))
            X_grid[:, feat1] = grid_points[:, 0]
            X_grid[:, feat2] = grid_points[:, 1]
            
            # Get predictions and uncertainty
            try:
                # Different ways to get posterior samples depending on implementation
                if hasattr(model, 'get_samples'):
                    samples = model.get_samples(X_grid, num_samples=50)
                elif hasattr(model, 'predict_posterior'):
                    samples = model.predict_posterior(X_grid, num_samples=50)
                else:
                    # Fallback - use prediction intervals if available
                    lower, upper = model.predict_intervals(X_grid)
                    uncertainty = upper - lower
                    mean_pred = model.predict(X_grid)
                    has_samples = False
            
                # If we have samples, calculate uncertainty as standard deviation
                if 'samples' in locals():
                    uncertainty = np.std(samples, axis=0)
                    mean_pred = np.mean(samples, axis=0)
                    has_samples = True
                
                # Plot the uncertainty surface
                plt.figure(figsize=(12, 5))
                
                # Mean prediction plot
                plt.subplot(1, 2, 1)
                mean_surface = mean_pred.reshape(f1_grid.shape)
                plt.contourf(f1_grid, f2_grid, mean_surface, cmap='viridis')
                plt.colorbar(label='Predicted Value')
                plt.xlabel(feature_names[feat1])
                plt.ylabel(feature_names[feat2])
                plt.title('Mean Prediction Surface')
                
                # Uncertainty plot
                plt.subplot(1, 2, 2)
                uncertainty_surface = uncertainty.reshape(f1_grid.shape)
                plt.contourf(f1_grid, f2_grid, uncertainty_surface, cmap='plasma')
                plt.colorbar(label='Uncertainty (std dev)' if has_samples else 'Prediction Interval Width')
                plt.xlabel(feature_names[feat1])
                plt.ylabel(feature_names[feat2])
                plt.title('Prediction Uncertainty')
                
                plt.tight_layout()
                
                plots.append({
                    'title': 'Prediction Uncertainty Map',
                    'img': get_base64_plot(),
                    'interpretation': 'Visualizes the prediction mean (left) and uncertainty (right) across a 2D slice of the feature space. Brighter regions on the right indicate higher uncertainty in predictions. Helpful for identifying areas where the model is less confident in its predictions.'
                })
            except:
                pass
    except:
        pass
        
    return plots 