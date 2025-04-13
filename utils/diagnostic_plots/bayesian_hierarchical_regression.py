"""Bayesian Hierarchical Regression diagnostic plots."""
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

def generate_bayesian_hierarchical_plots(trace, model_info=None, data=None, group_var=None):
    """Generate diagnostic plots for Bayesian Hierarchical Regression model
    
    Args:
        trace: PyMC3/PyMC4 trace or dictionary with posterior samples
        model_info: Dictionary with model information
        data: Original data used for modeling (optional)
        group_var: Group variable name for hierarchical structure
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Check if trace is a dictionary-like object
    if not hasattr(trace, 'keys'):
        raise ValueError("Trace must be a dictionary-like object with parameter names as keys")

    # Get parameter names
    param_names = list(trace.keys())
    
    # Plot 1: Posterior distributions of fixed effects
    plt.figure(figsize=(12, 8))
    
    # Identify fixed effects (parameters without group variable)
    fixed_effects = [p for p in param_names if 'sigma' not in p.lower() and 'var' not in p.lower() 
                    and (group_var is None or group_var not in p)]
    
    # Limit to a reasonable number for visualization
    if len(fixed_effects) > 6:
        fixed_effects = fixed_effects[:6]
    
    n_fixed = len(fixed_effects)
    rows = int(np.ceil(n_fixed / 2))
    
    for i, param in enumerate(fixed_effects):
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
        "title": "Posterior Distributions",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the posterior distributions of fixed effect parameters. The vertical dashed red line represents zero. Parameters whose distributions are far from zero are likely significant. The mean and 95% credible interval provide point and uncertainty estimates."
    })
    
    # Plot 2: Forest plot of fixed effects
    plt.figure(figsize=(10, 6))
    
    # Extract means and credible intervals
    means = [np.mean(trace[param]) for param in fixed_effects]
    ci_lows = [np.percentile(trace[param], 2.5) for param in fixed_effects]
    ci_highs = [np.percentile(trace[param], 97.5) for param in fixed_effects]
    
    # Create forest plot
    y_pos = np.arange(len(fixed_effects))
    
    plt.errorbar(means, y_pos, xerr=[np.array(means)-np.array(ci_lows), np.array(ci_highs)-np.array(means)],
                 fmt='o', capsize=5, elinewidth=2, markeredgewidth=2)
    
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.yticks(y_pos, fixed_effects)
    plt.xlabel("Parameter Value")
    plt.title("Forest Plot of Fixed Effects")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plots.append({
        "title": "Forest Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Visualizes the parameter estimates with their 95% credible intervals. Parameters whose intervals don't cross zero (the red dashed line) are considered statistically significant. This plot helps identify the direction and magnitude of effects."
    })
    
    # Plot 3: Trace plots for convergence diagnostics
    if len(param_names) > 0:
        plt.figure(figsize=(12, 8))
        
        # Select parameters to display (limit to a reasonable number)
        display_params = param_names[:min(6, len(param_names))]
        
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
    
    # Plot 4: Group-level effects (if group variable is provided)
    if group_var is not None:
        # Try to identify group-level parameters
        group_params = [p for p in param_names if group_var in p]
        
        if group_params:
            plt.figure(figsize=(12, 8))
            
            # Identify a representative group parameter
            group_param = group_params[0]
            
            # Extract the group effects
            group_effects = trace[group_param]
            
            # If it's a 2D array, we assume first dimension is samples, second is groups
            if len(group_effects.shape) > 1:
                n_groups = group_effects.shape[1]
                
                # Calculate mean and CI for each group
                means = np.mean(group_effects, axis=0)
                ci_lows = np.percentile(group_effects, 2.5, axis=0)
                ci_highs = np.percentile(group_effects, 97.5, axis=0)
                
                # Limit to displaying 20 groups max
                if n_groups > 20:
                    # Take a sample of 20 groups
                    indices = np.linspace(0, n_groups-1, 20, dtype=int)
                    means = means[indices]
                    ci_lows = ci_lows[indices]
                    ci_highs = ci_highs[indices]
                    group_labels = [f"Group {i+1}" for i in indices]
                else:
                    group_labels = [f"Group {i+1}" for i in range(n_groups)]
                
                # Sort by mean for better visualization
                sorted_indices = np.argsort(means)
                means = means[sorted_indices]
                ci_lows = ci_lows[sorted_indices]
                ci_highs = ci_highs[sorted_indices]
                group_labels = [group_labels[i] for i in sorted_indices]
                
                # Create forest plot of group effects
                y_pos = np.arange(len(means))
                
                plt.errorbar(means, y_pos, xerr=[means-ci_lows, ci_highs-means],
                            fmt='o', capsize=5, elinewidth=1.5, markeredgewidth=1.5)
                
                plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                plt.yticks(y_pos, group_labels)
                plt.xlabel("Group Effect")
                plt.title(f"Group-level Effects for {group_param}")
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                
                plots.append({
                    "title": "Group-level Effects",
                    "img_data": get_base64_plot(),
                    "interpretation": "Shows the estimated effects for each group with 95% credible intervals. Groups are ordered by effect size. This plot reveals how groups differ from the population average and identifies outlier groups."
                })
    
    # Plot 5: Posterior predictive check
    if 'sigma' in param_names or 'sd' in param_names or any('sigma' in p.lower() for p in param_names):
        plt.figure(figsize=(10, 6))
        
        # Find the residual standard deviation parameter
        sigma_param = next((p for p in param_names if p.lower() == 'sigma' or p.lower() == 'sd'), 
                          next((p for p in param_names if 'sigma' in p.lower()), None))
        
        if sigma_param:
            # Generate posterior predictive samples
            sigma_samples = trace[sigma_param]
            
            # Generate random normal samples with these sigmas
            n_samples = min(100, len(sigma_samples))
            ppc_samples = np.random.normal(0, sigma_samples[:n_samples, None], (n_samples, 1000))
            
            # Plot density of predictive samples
            for i in range(min(20, n_samples)):
                sns.kdeplot(ppc_samples[i], alpha=0.1, color='blue')
                
            # Plot average predictive distribution
            sns.kdeplot(ppc_samples.flatten(), color='red', linewidth=2, label='Average')
            
            plt.title("Posterior Predictive Distribution")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()
            
            plots.append({
                "title": "Posterior Predictive Check",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the distribution of predicted values from the model. Blue lines represent individual posterior draws, while the red line shows the average. This helps assess model fit by comparing to the actual data distribution."
            })
    
    # Plot 6: Hierarchical Structure Visualization
    plt.figure(figsize=(10, 6))
    
    # Create a visualization of the hierarchical structure
    gs = GridSpec(3, 3, figure=plt.gcf())
    
    # Population level (fixed effects)
    ax_pop = plt.subplot(gs[0, 1])
    ax_pop.add_patch(plt.Circle((0.5, 0.5), 0.3, color='lightblue'))
    ax_pop.text(0.5, 0.5, "Population\nParameters", ha='center', va='center')
    ax_pop.set_xlim(0, 1)
    ax_pop.set_ylim(0, 1)
    ax_pop.axis('off')
    
    # Group level
    ax_groups = plt.subplot(gs[1, :])
    group_colors = plt.cm.tab10.colors
    n_groups_display = min(8, 5 if group_var else 0)
    
    for i in range(n_groups_display):
        x_pos = (i + 0.5) / n_groups_display
        ax_groups.add_patch(plt.Circle((x_pos, 0.5), 0.15, color=group_colors[i % len(group_colors)]))
        ax_groups.text(x_pos, 0.5, f"Group\n{i+1}", ha='center', va='center', fontsize=9)
        
        # Arrow from population to group
        plt.arrow(0.5, 0.2, x_pos - 0.5, 0.1, head_width=0.03, head_length=0.05, 
                  fc='black', ec='black', transform=plt.gcf().transFigure)
    
    ax_groups.set_xlim(0, 1)
    ax_groups.set_ylim(0, 1)
    ax_groups.axis('off')
    
    # Individual level
    ax_indiv = plt.subplot(gs[2, :])
    n_indiv_per_group = 3
    n_indiv_display = n_indiv_per_group * n_groups_display
    
    for i in range(n_indiv_display):
        group_idx = i // n_indiv_per_group
        x_pos = (i + 0.5) / n_indiv_display
        ax_indiv.add_patch(plt.Rectangle((x_pos - 0.03, 0.3), 0.06, 0.06, 
                                        color=group_colors[group_idx % len(group_colors)]))
        
        # Arrow from group to individual
        group_x = (group_idx + 0.5) / n_groups_display
        plt.arrow(group_x, 0.35, x_pos - group_x, -0.15, head_width=0.01, head_length=0.02, 
                  fc='black', ec='black', transform=plt.gcf().transFigure)
    
    ax_indiv.set_xlim(0, 1)
    ax_indiv.set_ylim(0, 1)
    ax_indiv.axis('off')
    
    plt.figtext(0.5, 0.95, "Hierarchical Model Structure", ha='center', fontsize=14)
    plt.figtext(0.15, 0.78, "Level 1: Population", fontsize=10)
    plt.figtext(0.15, 0.54, "Level 2: Groups", fontsize=10)
    plt.figtext(0.15, 0.28, "Level 3: Individuals", fontsize=10)
    
    plots.append({
        "title": "Hierarchical Structure",
        "img_data": get_base64_plot(),
        "interpretation": "Illustrates the hierarchical structure of the model, showing how parameters at the population level inform group-level parameters, which in turn inform individual-level predictions. This partial pooling allows information sharing across groups."
    })
    
    return plots 