"""Bayesian Quantile Regression diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_bayesian_quantile_plots(trace, data=None, x_var=None, y_var=None, quantiles=None):
    """Generate diagnostic plots for Bayesian Quantile Regression model
    
    Args:
        trace: Dictionary with posterior samples for parameters
        data: Original data used for modeling (optional)
        x_var: Name of predictor variable (for scatter plots)
        y_var: Name of target variable (for scatter plots)
        quantiles: List of quantiles modeled (e.g., [0.1, 0.5, 0.9])
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Default quantiles if not provided
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    # Plot 1: Posterior distributions of parameters for different quantiles
    plt.figure(figsize=(12, 8))
    
    # Find parameters common to all quantiles
    all_params = list(trace.keys())
    common_params = []
    
    # Check if there are quantile-specific parameters
    quantile_specific_params = {}
    for param in all_params:
        # Check if the parameter is associated with a specific quantile
        for q in quantiles:
            q_str = str(q).replace(".", "_")
            if f"q{q_str}" in param or f"tau{q_str}" in param:
                if param not in quantile_specific_params:
                    quantile_specific_params[param] = q
                break
    
    # If we have data and specific parameters for different quantiles, we can plot them
    if quantile_specific_params:
        # Get parameter names without the quantile suffix
        base_params = set()
        for param in quantile_specific_params.keys():
            # Extract the base parameter name (without quantile suffix)
            for q in quantiles:
                q_str = str(q).replace(".", "_")
                if f"q{q_str}_" in param:
                    base_param = param.split(f"q{q_str}_")[1]
                    base_params.add(base_param)
                elif f"tau{q_str}_" in param:
                    base_param = param.split(f"tau{q_str}_")[1]
                    base_params.add(base_param)
        
        # Limit to a reasonable number for visualization
        base_params = list(base_params)
        if len(base_params) > 4:
            base_params = base_params[:4]
        
        # Plot posteriors for each base parameter across quantiles
        n_base = len(base_params)
        fig, axes = plt.subplots(n_base, 1, figsize=(10, 3*n_base))
        if n_base == 1:
            axes = [axes]
        
        for i, base_param in enumerate(base_params):
            ax = axes[i]
            
            for q in quantiles:
                q_str = str(q).replace(".", "_")
                q_param = None
                
                # Find the parameter for this quantile
                for param in quantile_specific_params:
                    if (f"q{q_str}_{base_param}" in param or 
                        f"q{q_str}" == param or
                        f"tau{q_str}_{base_param}" in param or 
                        f"tau{q_str}" == param):
                        q_param = param
                        break
                
                if q_param and q_param in trace:
                    # Plot the posterior for this quantile
                    sns.kdeplot(trace[q_param], ax=ax, label=f"τ={q}")
            
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.set_title(f"Posterior for {base_param} across quantiles")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
        
        plt.tight_layout()
        plots.append({
            "title": "Quantile Parameter Posteriors",
            "img_data": get_base64_plot(),
            "interpretation": "Shows posterior distributions of parameters across different quantiles. Differences between quantiles indicate quantile-specific effects. The vertical dashed red line represents zero. Parameters whose distributions are far from zero are likely significant."
        })
    else:
        # If we don't have quantile-specific parameters, plot regular posteriors
        # Identify parameters (excluding technical ones)
        params_to_plot = [p for p in all_params if not any(x in p.lower() for x in 
                                                      ['lp__', 'log_lik', 'nu', 'sigma'])]
        
        # Limit to a reasonable number for visualization
        if len(params_to_plot) > 6:
            params_to_plot = params_to_plot[:6]
        
        n_params = len(params_to_plot)
        rows = int(np.ceil(n_params / 2))
        
        for i, param in enumerate(params_to_plot):
            plt.subplot(rows, 2, i+1)
            
            # Plot kernel density estimate of posterior
            sns.kdeplot(trace[param], fill=True)
            
            # Add vertical line at 0 for reference
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            # Add mean and 95% credible interval
            mean_val = np.mean(trace[param])
            ci_low = np.percentile(trace[param], 2.5)
            ci_high = np.percentile(trace[param], 97.5)
            
            plt.title(f"{param}")
            plt.xlabel("Value")
            plt.ylabel("Density")
            
            # Add annotation with mean and CI
            plt.annotate(f"Mean: {mean_val:.3f}\n95% CI: [{ci_low:.3f}, {ci_high:.3f}]", 
                         xy=(0.05, 0.85), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        plots.append({
            "title": "Parameter Posteriors",
            "img_data": get_base64_plot(),
            "interpretation": "Shows posterior distributions of model parameters. The vertical dashed red line represents zero. Parameters whose distributions are far from zero are likely significant. The mean and 95% credible interval provide point and uncertainty estimates."
        })
    
    # Plot 2: If data is provided, plot quantile regression lines
    if data is not None and x_var is not None and y_var is not None:
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of the data
        plt.scatter(data[x_var], data[y_var], alpha=0.5, color='gray')
        
        # Sort x for smooth lines
        x_range = np.linspace(data[x_var].min(), data[x_var].max(), 100)
        
        # For each quantile, predict the quantile regression line
        colors = plt.cm.viridis(np.linspace(0, 1, len(quantiles)))
        
        for i, q in enumerate(quantiles):
            q_str = str(q).replace(".", "_")
            
            # Find intercept and slope for this quantile
            intercept_param = None
            slope_param = None
            
            for param in all_params:
                if (f"q{q_str}_intercept" in param or 
                    f"tau{q_str}_intercept" in param or 
                    f"intercept_q{q_str}" in param or
                    f"intercept_tau{q_str}" in param):
                    intercept_param = param
                    
                if (f"q{q_str}_{x_var}" in param or 
                    f"tau{q_str}_{x_var}" in param or
                    f"{x_var}_q{q_str}" in param or
                    f"{x_var}_tau{q_str}" in param):
                    slope_param = param
            
            # If we found both parameters, plot the line
            if intercept_param in trace and slope_param in trace:
                intercept = np.mean(trace[intercept_param])
                slope = np.mean(trace[slope_param])
                
                plt.plot(x_range, intercept + slope * x_range, 
                         label=f"τ={q}", color=colors[i], linewidth=2)
            else:
                # Try to find global parameters (not quantile-specific)
                intercept_param = next((p for p in all_params if p == "intercept" or p == "alpha"), None)
                slope_param = next((p for p in all_params if p == x_var or p == f"beta_{x_var}" or p == "beta"), None)
                
                if intercept_param in trace and slope_param in trace:
                    intercept = np.mean(trace[intercept_param])
                    slope = np.mean(trace[slope_param])
                    
                    # For different quantiles, adjust the intercept based on the error distribution
                    # This is a simplified approach, as Bayesian quantile regression would normally
                    # model the entire conditional distribution
                    if "sigma" in trace:
                        sigma = np.mean(trace["sigma"])
                        from scipy.stats import norm
                        quantile_adjustment = norm.ppf(q) * sigma
                        plt.plot(x_range, intercept + slope * x_range + quantile_adjustment, 
                                 label=f"τ={q}", color=colors[i], linewidth=2)
        
        plt.title("Quantile Regression Lines")
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Quantile Regression Lines",
            "img_data": get_base64_plot(),
            "interpretation": "Shows estimated regression lines for different quantiles overlaid on the data. Each line represents the estimated conditional quantile function. The spread between quantile lines indicates heterogeneity in the conditional distribution."
        })
    
    # Plot 3: Posterior predictive check with quantiles
    if data is not None and y_var is not None:
        plt.figure(figsize=(10, 6))
        
        # Create a histogram of the actual data
        sns.histplot(data[y_var], kde=True, stat="density", alpha=0.5, label="Observed Data")
        
        # If we have model-based predictions, we can overlay them
        if "sigma" in trace:
            # For simplicity, assuming a normal error model
            # This would need to be adapted for different error distributions
            intercept_param = next((p for p in all_params if p == "intercept" or p == "alpha"), None)
            sigma_param = next((p for p in all_params if p.lower() == "sigma"), None)
            
            if intercept_param in trace and sigma_param in trace:
                # Generate posterior predictive samples
                intercept_samples = trace[intercept_param]
                sigma_samples = trace[sigma_param]
                
                # Generate predictions (simplified, just using intercept)
                n_samples = min(500, len(intercept_samples))
                predictions = np.random.normal(
                    loc=intercept_samples[:n_samples, None], 
                    scale=sigma_samples[:n_samples, None],
                    size=(n_samples, 1000)
                )
                
                # Plot a few posterior predictive samples
                for i in range(min(10, n_samples)):
                    sns.kdeplot(predictions[i], alpha=0.1, color='blue')
                
                # Plot the average predictive distribution
                sns.kdeplot(predictions.flatten(), color='red', linewidth=2, 
                            label="Posterior Predictive")
        
        plt.title("Posterior Predictive Check")
        plt.xlabel(y_var)
        plt.ylabel("Density")
        plt.legend()
        
        plots.append({
            "title": "Posterior Predictive Check",
            "img_data": get_base64_plot(),
            "interpretation": "Compares the observed data distribution (histogram) with the posterior predictive distribution (red line). Good model fit is indicated by close alignment between the two distributions. Blue lines represent individual posterior draws."
        })
    
    # Plot 4: Asymmetric loss function visualization
    plt.figure(figsize=(10, 6))
    
    # Create a range of residual values
    residuals = np.linspace(-5, 5, 1000)
    
    # Plot asymmetric loss functions for different quantiles
    for i, q in enumerate(quantiles):
        # Calculate the quantile loss (or "check" function)
        loss = np.where(residuals >= 0, 
                         q * residuals, 
                         (q - 1) * residuals)
        
        plt.plot(residuals, loss, label=f"τ={q}", linewidth=2)
    
    # Add MSE loss for comparison
    plt.plot(residuals, residuals**2 / 10, 'k--', label="MSE (scaled)", linewidth=1)
    
    plt.title("Quantile Loss Functions")
    plt.xlabel("Residual (y - ŷ)")
    plt.ylabel("Loss")
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Quantile Loss Functions",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the asymmetric loss functions used in quantile regression. For a given quantile τ, positive errors are weighted by τ and negative errors by (τ-1). This asymmetry pulls the estimate toward the specified quantile. Compare with symmetric MSE loss (dashed line) used in standard regression."
    })
    
    # Plot 5: Trace plots for convergence diagnostics
    if len(all_params) > 0:
        plt.figure(figsize=(12, 8))
        
        # Select parameters to display (limit to a reasonable number)
        display_params = all_params[:min(6, len(all_params))]
        
        rows = int(np.ceil(len(display_params) / 2))
        
        for i, param in enumerate(display_params):
            plt.subplot(rows, 2, i+1)
            
            # Check if parameter has multiple chains or dimensions
            param_trace = trace[param]
            if len(param_trace.shape) > 1 and param_trace.shape[1] > 1:
                # If multidimensional, show first dimension
                plt.plot(param_trace[:, 0])
                plt.title(f"{param}[0] Trace")
            else:
                plt.plot(param_trace)
                plt.title(f"{param} Trace")
            
            plt.xlabel("Iteration")
            plt.ylabel("Value")
        
        plt.tight_layout()
        plots.append({
            "title": "Trace Plots",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the sampled values for each parameter across MCMC iterations. Good convergence is indicated by stationary, well-mixed chains without trends or patterns. Poor mixing or trends suggest convergence issues."
        })
    
    # Plot 6: Quantile-Quantile plot of the model
    if data is not None and y_var is not None:
        plt.figure(figsize=(10, 6))
        
        # Calculate empirical quantiles
        y_sorted = np.sort(data[y_var])
        p = np.linspace(0, 1, len(y_sorted))
        
        # Plot empirical quantiles
        plt.plot(p, y_sorted, 'o', alpha=0.5, label="Empirical Quantiles")
        
        # If we have model-based quantiles, plot them too
        if len(quantiles) >= 2:
            q_preds = {}
            
            # Try to find quantile-specific parameters
            for q in quantiles:
                q_str = str(q).replace(".", "_")
                
                intercept_param = None
                for param in all_params:
                    if (f"q{q_str}_intercept" in param or 
                        f"tau{q_str}_intercept" in param or 
                        f"intercept_q{q_str}" in param or
                        f"intercept_tau{q_str}" in param):
                        intercept_param = param
                        break
                
                if intercept_param in trace:
                    q_preds[q] = np.mean(trace[intercept_param])
            
            # If we found quantile predictions, plot a line connecting them
            if q_preds:
                q_vals = sorted(q_preds.keys())
                pred_quantiles = [q_preds[q] for q in q_vals]
                
                plt.plot(q_vals, pred_quantiles, 'r-', linewidth=2, label="Model Quantiles")
                
                # Add points at the specific quantiles
                plt.plot(q_vals, pred_quantiles, 'ro')
        
        plt.title("Quantile-Quantile Plot")
        plt.xlabel("Quantile Level (τ)")
        plt.ylabel(f"{y_var} Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plots.append({
            "title": "Quantile-Quantile Plot",
            "img_data": get_base64_plot(),
            "interpretation": "Compares empirical quantiles from the data (dots) with model-estimated quantiles (red line). Close alignment indicates good model fit across the distribution. Deviations reveal where the model may not capture the data distribution well."
        })
    
    return plots 