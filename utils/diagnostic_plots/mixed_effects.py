"""Mixed effects model diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
from scipy import stats

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_mixed_effects_plots(model, data=None):
    """Generate diagnostic plots for mixed effects models
    
    Args:
        model: Fitted mixed effects model (statsmodels or similar)
        data: Original data used to fit the model (optional)
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Get residuals if available
    if hasattr(model, 'resid'):
        residuals = model.resid
    else:
        residuals = None
    
    # Get fitted values if available
    if hasattr(model, 'fittedvalues'):
        fitted = model.fittedvalues
    else:
        fitted = None
        
    # Only proceed with plots if we have residuals and fitted values
    if residuals is not None and fitted is not None:
        # Plot 1: Residuals vs Fitted Values
        plt.figure(figsize=(10, 6))
        plt.scatter(fitted, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add smoothed trend line
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smooth = lowess(residuals, fitted, frac=0.2)
            plt.plot(smooth[:, 0], smooth[:, 1], 'r-', lw=2)
        except:
            # If lowess fails, we'll skip the trend line
            pass
            
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted Values')
        
        plots.append({
            "title": "Residuals vs Fitted",
            "img_data": get_base64_plot(),
            "interpretation": "Checks the assumption of linearity and homoscedasticity. Residuals should be randomly scattered around the horizontal line at zero with no clear pattern. Funneling patterns suggest heteroscedasticity, while curves suggest non-linearity."
        })
        
        # Plot 2: Normal Q-Q Plot of Residuals
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Normal Q-Q Plot')
        
        plots.append({
            "title": "Normal Q-Q Plot",
            "img_data": get_base64_plot(),
            "interpretation": "Assesses whether residuals follow a normal distribution. Points should follow the diagonal reference line. Deviations suggest non-normality, which may violate model assumptions."
        })
        
        # Plot 3: Scale-Location Plot (Standardized Residuals vs Fitted Values)
        plt.figure(figsize=(10, 6))
        standardized_residuals = residuals / np.std(residuals)
        plt.scatter(fitted, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
        
        # Add smoothed trend line
        try:
            smooth = lowess(np.sqrt(np.abs(standardized_residuals)), fitted, frac=0.2)
            plt.plot(smooth[:, 0], smooth[:, 1], 'r-', lw=2)
        except:
            pass
            
        plt.xlabel('Fitted Values')
        plt.ylabel('√|Standardized Residuals|')
        plt.title('Scale-Location Plot')
        
        plots.append({
            "title": "Scale-Location Plot",
            "img_data": get_base64_plot(),
            "interpretation": "Another check for homoscedasticity. The spread of residuals should be even across the range of fitted values. Any trend indicates changing variance with the response level."
        })
        
        # Plot 4: Random Effects Distribution
        if hasattr(model, 'random_effects'):
            plt.figure(figsize=(10, 8))
            random_effects = model.random_effects
            
            if isinstance(random_effects, dict):
                num_groups = len(random_effects)
                plt.subplot(num_groups, 1, 1)
                
                for i, (group_name, effects) in enumerate(random_effects.items()):
                    plt.subplot(num_groups, 1, i+1)
                    
                    if isinstance(effects, np.ndarray):
                        plt.hist(effects, bins=20, alpha=0.7)
                    elif isinstance(effects, pd.DataFrame):
                        for col in effects.columns:
                            plt.hist(effects[col], bins=20, alpha=0.7, label=col)
                        if len(effects.columns) > 1:
                            plt.legend()
                    
                    plt.title(f'Random Effects Distribution: {group_name}')
                    plt.xlabel('Effect Size')
                    plt.ylabel('Frequency')
            else:
                # If it's not a dictionary, try plotting directly
                plt.hist(random_effects, bins=20)
                plt.title('Random Effects Distribution')
                plt.xlabel('Effect Size')
                plt.ylabel('Frequency')
                
            plt.tight_layout()
            
            plots.append({
                "title": "Random Effects Distribution",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the distribution of random effects. Should approximately follow a normal distribution centered near zero. Extreme outliers may indicate groups that differ substantially from the overall pattern."
            })
    
    # Plot 5: Model Coefficients (Fixed Effects)
    if hasattr(model, 'params') and hasattr(model, 'bse'):
        plt.figure(figsize=(12, 8))
        
        coefs = model.params
        errors = model.bse
        
        # Filter out random effects and intercept if needed
        if hasattr(model, 'fe_params'):
            coefs = model.fe_params
            if hasattr(model, 'bse_fe'):
                errors = model.bse_fe
        
        # Sort coefficients by magnitude for better visualization
        if len(coefs) > 0:
            coef_names = coefs.index if hasattr(coefs, 'index') else [f'Param {i}' for i in range(len(coefs))]
            coef_values = coefs.values if hasattr(coefs, 'values') else coefs
            error_values = errors.values if hasattr(errors, 'values') else errors
            
            # Sort by absolute coefficient value
            if len(coef_values) > 1:
                sorted_indices = np.argsort(np.abs(coef_values))
                coef_names = [coef_names[i] for i in sorted_indices]
                coef_values = coef_values[sorted_indices]
                error_values = error_values[sorted_indices]
            
            plt.figure(figsize=(10, max(6, len(coef_names) * 0.3)))
            y_pos = np.arange(len(coef_names))
            
            plt.barh(y_pos, coef_values, xerr=error_values, align='center', alpha=0.7)
            plt.yticks(y_pos, coef_names)
            plt.xlabel('Coefficient Value')
            plt.title('Fixed Effects Coefficients with Standard Errors')
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plots.append({
                "title": "Fixed Effects Coefficients",
                "img_data": get_base64_plot(),
                "interpretation": "Displays the estimated fixed effects with standard errors. Coefficients that don't cross zero are statistically significant. The magnitude indicates the strength of the relationship between predictors and the outcome."
            })
    
    # Plot 6: Residuals by Group (if grouping information is available)
    if data is not None and hasattr(model, 'groups') and residuals is not None:
        plt.figure(figsize=(12, 8))
        
        # Get group information
        groups = model.groups
        unique_groups = np.unique(groups)
        
        # Limit to at most 10 groups to prevent overcrowding
        if len(unique_groups) > 10:
            # Take first 10 groups
            unique_groups = unique_groups[:10]
        
        # Box plot of residuals by group
        group_residuals = []
        group_labels = []
        
        for group in unique_groups:
            group_mask = (groups == group)
            group_residuals.append(residuals[group_mask])
            group_labels.append(f'Group {group}')
        
        plt.boxplot(group_residuals, labels=group_labels)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.ylabel('Residuals')
        plt.title('Residuals by Group')
        plt.xticks(rotation=45 if len(unique_groups) > 4 else 0)
        plt.tight_layout()
        
        plots.append({
            "title": "Residuals by Group",
            "img_data": get_base64_plot(),
            "interpretation": "Examines how residuals vary across different groups. Consistent distributions centered at zero indicate that the model is adequately accounting for group-level variation. Systematic differences between groups may suggest missing interaction terms."
        })
    
    return plots 