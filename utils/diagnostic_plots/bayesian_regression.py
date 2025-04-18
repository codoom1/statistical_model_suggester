"""Bayesian regression diagnostic plots."""
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

def generate_bayesian_regression_plots(model, X=None, y=None, feature_names=None, trace=None):
    """Generate diagnostic plots for Bayesian regression models
    
    Args:
        model: Fitted Bayesian regression model (PyMC3, Stan, etc.)
        X: Feature matrix
        y: Target variable
        feature_names: List of feature names
        trace: MCMC trace (samples from posterior distribution)
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Get feature names if not provided
    if feature_names is None and X is not None:
        if hasattr(X, 'columns'):  # If X is a DataFrame
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"X{i+1}" for i in range(X.shape[1] if X.ndim > 1 else 1)]
    
    # Try to extract trace if it's not provided but is in the model
    if trace is None:
        if hasattr(model, 'trace'):
            trace = model.trace
        elif hasattr(model, 'posterior_samples'):
            trace = model.posterior_samples
    
    # Plot 1: Coefficient posterior distributions
    if trace is not None:
        # Try to extract coefficient samples from the trace
        coef_samples = None
        intercept_samples = None
        
        # Handle different model formats
        if isinstance(trace, dict):
            # Find coefficient variables in the trace
            coef_keys = [k for k in trace.keys() if 'beta' in k.lower() or 'coef' in k.lower()]
            intercept_keys = [k for k in trace.keys() if 'intercept' in k.lower() or 'alpha' in k.lower()]
            
            if coef_keys:
                coef_samples = np.array([trace[k] for k in coef_keys])
                # Transpose to get [n_samples, n_features]
                if coef_samples.ndim > 1:
                    coef_samples = coef_samples.T
            
            if intercept_keys:
                intercept_samples = trace[intercept_keys[0]]
        
        elif hasattr(trace, 'get_values'):
            # PyMC3-style trace
            try:
                coef_samples = trace.get_values('beta', combine=True)
                intercept_samples = trace.get_values('alpha', combine=True)
            except:
                try:
                    # Try alternative naming
                    coef_samples = trace.get_values('coefs', combine=True)
                    intercept_samples = trace.get_values('intercept', combine=True)
                except:
                    pass
                
        # If we have coefficient samples, plot their distributions
        if coef_samples is not None:
            n_coefs = coef_samples.shape[1] if coef_samples.ndim > 1 else 1
            
            if n_coefs <= 10:  # Only plot if there aren't too many coefficients
                plt.figure(figsize=(12, 8))
                
                # Adjust feature_names to match coefficient count
                if feature_names and len(feature_names) != n_coefs:
                    feature_names = feature_names[:n_coefs]
                elif not feature_names:
                    feature_names = [f"X{i+1}" for i in range(n_coefs)]
                
                # Plot coefficient posteriors
                if n_coefs == 1:
                    sns.kdeplot(coef_samples, fill=True, alpha=0.5)
                    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                    plt.xlabel("Coefficient Value")
                    plt.title(f"Posterior Distribution for {feature_names[0]}")
                else:
                    for i in range(n_coefs):
                        sns.kdeplot(coef_samples[:, i], fill=True, alpha=0.5, label=feature_names[i])
                    
                    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                    plt.xlabel("Coefficient Value")
                    plt.title("Posterior Distributions for Coefficients")
                    plt.legend()
                
                plots.append({
                    "title": "Coefficient Posterior Distributions",
                    "img_data": get_base64_plot(),
                    "interpretation": "Shows the posterior distributions for the model coefficients. If the distribution doesn't overlap with zero, there's strong evidence that the predictor has a non-zero effect. The width of the distribution indicates uncertainty in the estimate."
                })
            
            # Plot intercept posterior if available
            if intercept_samples is not None:
                plt.figure(figsize=(10, 6))
                sns.kdeplot(intercept_samples, fill=True, alpha=0.5)
                plt.xlabel("Intercept Value")
                plt.title("Posterior Distribution for Intercept")
                
                plots.append({
                    "title": "Intercept Posterior Distribution",
                    "img_data": get_base64_plot(),
                    "interpretation": "Shows the posterior distribution for the model intercept (baseline value when all predictors are zero)."
                })
    
    # Plot 2: Predictive check (if X and y are provided)
    if X is not None and y is not None and hasattr(model, 'predict'):
        plt.figure(figsize=(10, 6))
        
        # Make predictions (mean of posterior predictive)
        try:
            y_pred = model.predict(X)
            
            # Scatter plot of actual vs predicted
            plt.scatter(y, y_pred, alpha=0.6)
            
            # Add reference line
            min_val = min(min(y), min(y_pred))
            max_val = max(max(y), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            plt.xlabel('Observed Values')
            plt.ylabel('Predicted Values')
            plt.title('Posterior Predictive Check')
            
            plots.append({
                "title": "Posterior Predictive Check",
                "img_data": get_base64_plot(),
                "interpretation": "Compares observed values to predictions from the model. Points should fall close to the diagonal line if the model is making accurate predictions."
            })
            
            # Also plot residuals
            plt.figure(figsize=(10, 6))
            residuals = y - y_pred
            
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Predicted Values')
            
            plots.append({
                "title": "Residuals vs Predicted",
                "img_data": get_base64_plot(),
                "interpretation": "Examines patterns in residuals. Ideally, residuals should be randomly distributed around zero across the range of predicted values, with no clear pattern."
            })
        except:
            # Skip if prediction fails
            pass
    
    # Plot 3: Trace plots for MCMC diagnostics
    if trace is not None:
        # Try to extract parameter names and their chain values
        param_names = []
        chain_values = []
        
        if isinstance(trace, dict):
            for key, value in trace.items():
                # Skip if the parameter has too many dimensions
                if np.array(value).ndim <= 1:
                    param_names.append(key)
                    chain_values.append(value)
        elif hasattr(trace, 'varnames'):
            # PyMC3-style
            for var in trace.varnames:
                try:
                    values = trace.get_values(var, combine=True)
                    if values.ndim <= 1:
                        param_names.append(var)
                        chain_values.append(values)
                except:
                    pass
        
        # Plot trace plots for a subset of parameters (max 6)
        if param_names and len(param_names) > 0:
            num_plots = min(6, len(param_names))
            fig, axes = plt.subplots(num_plots, 2, figsize=(15, 3*num_plots))
            
            # If only one parameter, axes won't be a 2D array
            if num_plots == 1:
                axes = np.array([axes])
            
            for i in range(num_plots):
                param = param_names[i]
                values = chain_values[i]
                
                # Trace plot
                axes[i, 0].plot(values)
                axes[i, 0].set_title(f'Trace for {param}')
                axes[i, 0].set_xlabel('Sample')
                axes[i, 0].set_ylabel('Value')
                
                # Density plot
                sns.kdeplot(values, ax=axes[i, 1], fill=True, alpha=0.5)
                axes[i, 1].set_title(f'Density for {param}')
                axes[i, 1].set_xlabel('Value')
            
            plt.tight_layout()
            
            plots.append({
                "title": "MCMC Trace Plots",
                "img_data": get_base64_plot(),
                "interpretation": "Diagnostic plots for MCMC sampling. The left panels show the trace of samples, which should look like 'hairy caterpillars' without trends or large jumps. The right panels show the posterior distributions."
            })
    
    # Plot 4: Posterior predictive distribution (if model supports it)
    if hasattr(model, 'sample_posterior_predictive') and X is not None:
        try:
            plt.figure(figsize=(12, 6))
            
            # Sample from posterior predictive distribution
            pred_samples = model.sample_posterior_predictive(X)
            
            # Plot histogram of a random prediction
            random_idx = np.random.randint(0, X.shape[0])
            if isinstance(pred_samples, dict) and 'y' in pred_samples:
                sample_values = pred_samples['y'][:, random_idx]
            else:
                # Assume it's already the array of samples
                sample_values = pred_samples[:, random_idx]
            
            # Plot histogram of posterior predictive for this observation
            sns.histplot(sample_values, kde=True, stat="density", alpha=0.6)
            
            # Add vertical line for observed value if y is provided
            if y is not None:
                plt.axvline(x=y[random_idx], color='r', linestyle='--', label='Observed Value')
                plt.legend()
            
            plt.title(f'Posterior Predictive Distribution for Observation {random_idx}')
            plt.xlabel('Predicted Value')
            
            plots.append({
                "title": "Posterior Predictive Distribution",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the distribution of possible predictions for a single observation, accounting for uncertainty in model parameters. The width of this distribution represents the predictive uncertainty."
            })
        except:
            # Skip if sampling fails
            pass
    
    # Plot 5: Coefficient summary with credible intervals
    if trace is not None and coef_samples is not None:
        plt.figure(figsize=(12, 8))
        
        # Calculate mean and credible intervals for coefficients
        n_coefs = coef_samples.shape[1] if coef_samples.ndim > 1 else 1
        
        if n_coefs == 1:
            means = [np.mean(coef_samples)]
            lower = [np.percentile(coef_samples, 2.5)]
            upper = [np.percentile(coef_samples, 97.5)]
        else:
            means = np.mean(coef_samples, axis=0)
            lower = np.percentile(coef_samples, 2.5, axis=0)
            upper = np.percentile(coef_samples, 97.5, axis=0)
        
        # Adjust feature_names to match coefficient count
        if feature_names and len(feature_names) != n_coefs:
            feature_names = feature_names[:n_coefs]
        elif not feature_names:
            feature_names = [f"X{i+1}" for i in range(n_coefs)]
        
        # Plot credible intervals
        y_pos = np.arange(n_coefs)
        
        plt.figure(figsize=(10, max(6, n_coefs * 0.5)))
        plt.errorbar(means, y_pos, xerr=[means - lower, upper - means], 
                    fmt='o', capsize=5, elinewidth=2, markeredgewidth=2)
        
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        plt.yticks(y_pos, feature_names)
        plt.xlabel('Coefficient Value')
        plt.title('Coefficient Estimates with 95% Credible Intervals')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Coefficient Credible Intervals",
            "img_data": get_base64_plot(),
            "interpretation": "Displays the estimated coefficients with their 95% credible intervals. If an interval doesn't cross zero (red line), there's strong evidence that the predictor has an effect. Wider intervals indicate greater uncertainty."
        })
    
    # Plot 6: Model comparison if available
    if hasattr(model, 'waic') or hasattr(model, 'loo'):
        try:
            plt.figure(figsize=(8, 6))
            
            metrics = {}
            
            if hasattr(model, 'waic'):
                waic = model.waic
                if isinstance(waic, tuple):
                    metrics['WAIC'] = waic[0]  # Usually the first element is the score
                else:
                    metrics['WAIC'] = waic
            
            if hasattr(model, 'loo'):
                loo = model.loo
                if isinstance(loo, tuple):
                    metrics['LOO'] = loo[0]
                else:
                    metrics['LOO'] = loo
                    
            if metrics:
                plt.bar(list(metrics.keys()), list(metrics.values()), alpha=0.7)
                plt.ylabel('Score Value')
                plt.title('Model Comparison Metrics')
                
                # Add text labels for values
                for k, v in metrics.items():
                    plt.annotate(f'{v:.2f}', 
                                xy=(k, v), 
                                xytext=(0, 5),
                                textcoords='offset points',
                                ha='center')
                
                plots.append({
                    "title": "Model Comparison Metrics",
                    "img_data": get_base64_plot(),
                    "interpretation": "Shows information criteria used for model comparison. Lower values of WAIC (Widely Applicable Information Criterion) and LOO (Leave-One-Out cross-validation) indicate better model fit while accounting for complexity."
                })
        except:
            # Skip if getting metrics fails
            pass
    
    return plots 