"""Bayesian Model Averaging diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_bma_plots(model_results):
    """Generate diagnostic plots for Bayesian Model Averaging
    
    Args:
        model_results: Dictionary containing BMA results with keys:
            - models: List of model names or IDs
            - posterior_probs: Posterior probabilities for each model
            - coefficients: Coefficient estimates for each model
            - pip: Posterior inclusion probabilities for variables
            - post_mean: Posterior mean for each coefficient
            - post_sd: Posterior standard deviation for each coefficient
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Check if required keys are present
    required_keys = ['models', 'posterior_probs']
    if not all(key in model_results for key in required_keys):
        raise ValueError("model_results must contain 'models' and 'posterior_probs' keys")
    
    # Extract data
    models = model_results['models']
    posterior_probs = model_results['posterior_probs']
    
    # Plot 1: Posterior Model Probabilities
    plt.figure(figsize=(12, 6))
    
    # Sort models by posterior probability
    sorted_indices = np.argsort(posterior_probs)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_probs = [posterior_probs[i] for i in sorted_indices]
    
    # Limit to top 20 models for visibility
    display_limit = min(20, len(sorted_models))
    display_models = sorted_models[:display_limit]
    display_probs = sorted_probs[:display_limit]
    
    # Create bar chart
    plt.bar(range(len(display_models)), display_probs, color='lightblue')
    plt.xticks(range(len(display_models)), display_models, rotation=90)
    plt.xlabel('Model')
    plt.ylabel('Posterior Probability')
    plt.title('Posterior Model Probabilities')
    
    # Add cumulative probability line
    cumulative_probs = np.cumsum(display_probs)
    
    ax2 = plt.gca().twinx()
    ax2.plot(range(len(display_models)), cumulative_probs, 'ro-', linewidth=2)
    ax2.set_ylabel('Cumulative Probability', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim([0, min(1.05, max(cumulative_probs) * 1.1)])
    
    plt.tight_layout()
    plots.append({
        "title": "Model Posterior Probabilities",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the posterior probabilities of the top models. The red line indicates cumulative probability. Models with higher probabilities have stronger evidence. This plot helps identify the most important models in the BMA ensemble."
    })
    
    # Plot 2: Posterior Inclusion Probabilities (if available)
    if 'pip' in model_results:
        pip = model_results['pip']
        variable_names = list(pip.keys())
        pip_values = list(pip.values())
        
        # Sort by PIP for better visualization
        sorted_indices = np.argsort(pip_values)[::-1]
        sorted_vars = [variable_names[i] for i in sorted_indices]
        sorted_pips = [pip_values[i] for i in sorted_indices]
        
        plt.figure(figsize=(10, max(6, len(sorted_vars) * 0.3)))
        
        # Create horizontal bar chart
        plt.barh(range(len(sorted_vars)), sorted_pips, color='lightblue')
        plt.yticks(range(len(sorted_vars)), sorted_vars)
        plt.xlabel('Posterior Inclusion Probability')
        plt.title('Posterior Inclusion Probabilities')
        
        # Add vertical line at 0.5 for reference
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plots.append({
            "title": "Posterior Inclusion Probabilities",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the probability that each variable is included in the 'true' model. Variables with PIPs > 0.5 (red dashed line) have evidence supporting their inclusion. PIPs near 1 indicate strong evidence for a variable's importance, while PIPs near 0 suggest the variable may be irrelevant."
        })
    
    # Plot 3: Coefficient Posterior Distributions (if available)
    if all(k in model_results for k in ['post_mean', 'post_sd']):
        post_mean = model_results['post_mean']
        post_sd = model_results['post_sd']
        
        variable_names = list(post_mean.keys())
        
        # Limit to 12 variables for readability
        if len(variable_names) > 12:
            # If PIPs are available, use them to select top variables
            if 'pip' in model_results:
                pip = model_results['pip']
                top_vars = sorted(variable_names, key=lambda v: pip.get(v, 0), reverse=True)[:12]
            else:
                # Otherwise, select variables with largest absolute means
                top_vars = sorted(variable_names, key=lambda v: abs(post_mean[v]), reverse=True)[:12]
        else:
            top_vars = variable_names
        
        plt.figure(figsize=(12, 8))
        
        rows = int(np.ceil(len(top_vars) / 3))
        cols = min(3, len(top_vars))
        
        for i, var in enumerate(top_vars):
            plt.subplot(rows, cols, i+1)
            
            # Create x-range for normal distribution
            mean = post_mean[var]
            sd = post_sd[var]
            x = np.linspace(mean - 3.5*sd, mean + 3.5*sd, 1000)
            y = np.exp(-0.5 * ((x - mean) / sd)**2) / (sd * np.sqrt(2*np.pi))
            
            plt.plot(x, y, 'b-')
            plt.fill_between(x, y, alpha=0.3)
            
            # Add vertical line at 0 for reference
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            # Add mean and 95% credible interval
            ci_low = mean - 1.96*sd
            ci_high = mean + 1.96*sd
            
            plt.title(var)
            plt.xlabel("Coefficient Value")
            
            # Add annotation with mean and CI
            plt.annotate(f"Mean: {mean:.3f}\n95% CI: [{ci_low:.3f}, {ci_high:.3f}]", 
                         xy=(0.05, 0.85), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                         fontsize=8)
            
            # If PIPs are available, include them
            if 'pip' in model_results and var in model_results['pip']:
                pip_val = model_results['pip'][var]
                plt.annotate(f"PIP: {pip_val:.3f}", 
                             xy=(0.05, 0.7), xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                             fontsize=8)
        
        plt.tight_layout()
        plots.append({
            "title": "Coefficient Posterior Distributions",
            "img_data": get_base64_plot(),
            "interpretation": "Shows posterior distributions of coefficients across all models. The vertical red line at zero indicates no effect. Distributions centered away from zero with narrow spreads represent coefficients with consistent effects across models. The posterior inclusion probability (PIP) indicates how often the variable appears in selected models."
        })
    
    # Plot 4: Coefficient Heatmap (if coefficients across models are available)
    if 'coefficients' in model_results:
        coefficients = model_results['coefficients']
        
        # Check if it's a nested dictionary (model -> variable -> coefficient)
        if isinstance(next(iter(coefficients.values())), dict):
            plt.figure(figsize=(12, 10))
            
            # Limit to top models by posterior probability
            top_n_models = min(10, len(models))
            top_model_indices = sorted_indices[:top_n_models]
            top_models = [models[i] for i in top_model_indices]
            
            # Get all variable names from the coefficient dictionaries
            all_variables = set()
            for model, coefs in coefficients.items():
                all_variables.update(coefs.keys())
            
            # Filter variables if there are too many
            if len(all_variables) > 15:
                # If PIPs are available, use them to select top variables
                if 'pip' in model_results:
                    pip = model_results['pip']
                    top_vars = sorted(all_variables, key=lambda v: pip.get(v, 0), reverse=True)[:15]
                else:
                    # Otherwise just take first 15
                    top_vars = list(all_variables)[:15]
            else:
                top_vars = list(all_variables)
            
            # Create coefficient matrix for heatmap
            coef_matrix = np.zeros((len(top_vars), len(top_models)))
            
            for i, model in enumerate(top_models):
                model_coefs = coefficients[model]
                for j, var in enumerate(top_vars):
                    coef_matrix[j, i] = model_coefs.get(var, 0)
            
            # Create a custom colormap centered at zero
            colors = ["blue", "white", "red"]
            cmap = LinearSegmentedColormap.from_list("custom_divergent", colors)
            
            # Find max absolute value for symmetric colorbar
            vmax = np.max(np.abs(coef_matrix))
            
            # Create heatmap
            sns.heatmap(coef_matrix, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
                       xticklabels=top_models, yticklabels=top_vars, 
                       annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"label": "Coefficient Value"})
            
            plt.title("Coefficient Values Across Top Models")
            plt.tight_layout()
            
            plots.append({
                "title": "Coefficient Heatmap",
                "img_data": get_base64_plot(),
                "interpretation": "Shows how coefficient values vary across different models. Blue cells indicate negative relationships, red cells indicate positive relationships, and white cells indicate coefficients near zero or variables not included in that model. This visualization helps identify consistent patterns across models."
            })
    
    # Plot 5: PIP vs Posterior Mean (if both are available)
    if all(k in model_results for k in ['pip', 'post_mean']):
        pip = model_results['pip']
        post_mean = model_results['post_mean']
        
        # Get variables that have both PIP and posterior mean
        common_vars = set(pip.keys()) & set(post_mean.keys())
        
        if common_vars:
            plt.figure(figsize=(10, 8))
            
            # Extract data
            x_vals = [abs(post_mean[var]) for var in common_vars]
            y_vals = [pip[var] for var in common_vars]
            var_names = list(common_vars)
            
            # Create scatter plot
            plt.scatter(x_vals, y_vals, alpha=0.7)
            
            # Add variable labels
            for i, var in enumerate(var_names):
                plt.annotate(var, (x_vals[i], y_vals[i]), 
                            fontsize=8, alpha=0.8,
                            xytext=(5, 5), textcoords='offset points')
            
            # Add reference lines
            plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
            
            plt.xlabel("Absolute Posterior Mean")
            plt.ylabel("Posterior Inclusion Probability")
            plt.title("Posterior Inclusion Probability vs Effect Size")
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "PIP vs Effect Size",
                "img_data": get_base64_plot(),
                "interpretation": "Compares the importance of variables (PIP) with the magnitude of their effects (absolute posterior mean). Variables in the upper right are both important and have large effects. Variables with high PIP but small effects may have consistent but minor impacts. The horizontal red line at PIP=0.5 indicates evidence favoring inclusion."
            })
    
    # Plot 6: Model Space Visualization
    if len(models) <= 20:  # Only create this for a reasonable number of models
        plt.figure(figsize=(12, 8))
        
        # Create network representation of model space
        n_models = len(models)
        
        # Generate positions in a circle
        angles = np.linspace(0, 2*np.pi, n_models, endpoint=False)
        pos_x = np.cos(angles)
        pos_y = np.sin(angles)
        
        # Plot nodes (models)
        sizes = [p * 1000 for p in posterior_probs]  # Scale by posterior probability
        plt.scatter(pos_x, pos_y, s=sizes, c=posterior_probs, cmap='viridis', 
                   alpha=0.7, edgecolors='black')
        
        # Add model labels
        for i, model in enumerate(models):
            plt.annotate(model, (pos_x[i], pos_y[i]), 
                        fontsize=9, ha='center', va='center')
        
        # Draw edges between similar models
        if 'coefficients' in model_results:
            coefficients = model_results['coefficients']
            
            if isinstance(next(iter(coefficients.values())), dict):
                # For each pair of models, calculate similarity
                for i in range(n_models):
                    for j in range(i+1, n_models):
                        model_i = models[i]
                        model_j = models[j]
                        
                        # Get sets of variables in each model
                        vars_i = set(coefficients[model_i].keys())
                        vars_j = set(coefficients[model_j].keys())
                        
                        # Calculate Jaccard similarity (intersection over union)
                        intersection = len(vars_i & vars_j)
                        union = len(vars_i | vars_j)
                        
                        if union > 0:
                            similarity = intersection / union
                            
                            # Draw edge if models are similar
                            if similarity > 0.7:  # Threshold for showing edge
                                plt.plot([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]], 
                                        'k-', alpha=similarity*0.5, linewidth=similarity*2)
        
        plt.title("Model Space Visualization")
        plt.axis('equal')
        plt.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(posterior_probs)
        cbar = plt.colorbar(sm)
        cbar.set_label('Posterior Probability')
        
        plots.append({
            "title": "Model Space Visualization",
            "img_data": get_base64_plot(),
            "interpretation": "Visualizes the model space as a network. Each node represents a model with size proportional to its posterior probability. Edges connect models that share similar variables. This plot helps visualize clusters of related models and identify the dominant model structures in the BMA ensemble."
        })
    
    return plots 