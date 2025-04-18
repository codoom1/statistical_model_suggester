import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import base64
from io import BytesIO
import scipy.stats as stats
from matplotlib.patches import Patch

def get_base64_plot():
    """Convert the current matplotlib plot to a base64 string."""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

def generate_repeated_measures_anova_plots(data, dv, within_factors, subject_id, model_results=None):
    """
    Generate diagnostic plots for Repeated Measures ANOVA models.
    
    Parameters:
    -----------
    data : pandas DataFrame
        The dataset containing the dependent variable, within-subjects factors, and subject ID
    dv : str
        Name of the dependent variable
    within_factors : list of str
        Names of the within-subjects factors
    subject_id : str
        Name of the subject identifier column
    model_results : object, optional
        Results object from a statistical package (optional, used for additional plots if available)
    
    Returns:
    --------
    plots : list of dict
        List of dictionaries containing:
        - 'title': Title of the plot
        - 'img_data': Base64 encoded image
        - 'interpretation': Interpretation of the plot
    """
    plots = []
    
    # Convert within_factors to a list if it's a single string
    if isinstance(within_factors, str):
        within_factors = [within_factors]
    
    # 1. Profile Plot 
    plt.figure(figsize=(12, 6))
    
    if len(within_factors) == 1:
        # Simple profile plot with one within-subjects factor
        factor = within_factors[0]
        
        # Calculate means and standard errors
        factor_means = data.groupby(factor)[dv].mean()
        factor_se = data.groupby(factor)[dv].sem()
        
        # Plot means with error bars
        plt.errorbar(factor_means.index, factor_means.values, 
                    yerr=factor_se.values, marker='o', linestyle='-', 
                    capsize=5, ecolor='black', markerfacecolor='blue')
        
        plt.xlabel(factor)
        plt.ylabel(f'Mean {dv}')
        plt.title(f'Profile Plot of {dv} by {factor}')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plots.append({
            'title': 'Profile Plot',
            'img_data': get_base64_plot(),
            'interpretation': f'Shows how the mean of {dv} changes across different levels of {factor}. Error bars represent standard errors. The profile plot helps visualize the main effect of the within-subjects factor.'
        })
        
        # Also create a boxplot for distribution visualization
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=factor, y=dv, data=data)
        plt.title(f'Boxplot of {dv} by {factor}')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plots.append({
            'title': 'Boxplot by Factor Level',
            'img_data': get_base64_plot(),
            'interpretation': f'Shows the distribution of {dv} across different levels of {factor}. Boxplots display the median, quartiles, and potential outliers, providing insights into the data distribution at each factor level.'
        })
        
    elif len(within_factors) == 2:
        # Profile plot with two within-subjects factors
        factor1, factor2 = within_factors
        
        # Calculate means for each combination
        grouped_means = data.groupby([factor1, factor2])[dv].mean().reset_index()
        pivot_means = grouped_means.pivot(index=factor1, columns=factor2, values=dv)
        
        # Create line plot with different colors for each level of factor2
        pivot_means.plot(marker='o', linestyle='-', grid=True, figsize=(12, 6))
        
        plt.xlabel(factor1)
        plt.ylabel(f'Mean {dv}')
        plt.title(f'Profile Plot of {dv} by {factor1} and {factor2}')
        plt.legend(title=factor2)
        
        plots.append({
            'title': 'Interaction Profile Plot',
            'img_data': get_base64_plot(),
            'interpretation': f'Shows how the mean of {dv} changes across levels of {factor1}, with separate lines for each level of {factor2}. Non-parallel lines suggest an interaction between the two factors.'
        })
    
    # 2. Individual Trajectory Plot
    plt.figure(figsize=(10, 6))
    
    if len(within_factors) == 1:
        factor = within_factors[0]
        
        # Get unique subjects and factor levels
        subjects = data[subject_id].unique()
        factor_levels = data[factor].unique()
        
        # Plot individual trajectories
        for subj in subjects[:min(20, len(subjects))]:  # Limit to 20 subjects if there are many
            subj_data = data[data[subject_id] == subj]
            plt.plot(subj_data[factor], subj_data[dv], marker='o', linestyle='-', alpha=0.3)
        
        # Add mean trajectory
        factor_means = data.groupby(factor)[dv].mean()
        plt.plot(factor_means.index, factor_means.values, marker='o', linestyle='-', 
                color='red', linewidth=3, label='Group Mean')
        
        plt.xlabel(factor)
        plt.ylabel(dv)
        plt.title(f'Individual Trajectories of {dv} by {factor}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plots.append({
            'title': 'Individual Trajectories',
            'img_data': get_base64_plot(),
            'interpretation': 'Shows individual subject responses (thin lines) along with the group mean (thick red line). This plot helps visualize individual variability and potential outliers in the repeated measures design.'
        })
    
    # 3. Normality Check of Residuals
    plt.figure(figsize=(12, 6))
    
    # Calculate residuals - for repeated measures, we can use deviation from subject means
    subject_means = data.groupby(subject_id)[dv].transform('mean')
    overall_mean = data[dv].mean()
    condition_means = data.groupby(within_factors)[dv].transform('mean')
    
    # Estimate residuals (may not be exact without the full model)
    residuals = data[dv] - subject_means - (condition_means - overall_mean)
    
    # Create QQ plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--')
    ax1.set_xlabel('Residuals')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of Residuals')
    
    # QQ plot
    stats.probplot(residuals, plot=ax2)
    ax2.set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    
    plots.append({
        'title': 'Residual Normality Check',
        'img_data': get_base64_plot(),
        'interpretation': 'Checks the normality assumption of residuals. The histogram should approximate a normal distribution, and points in the Q-Q plot should follow the diagonal line for normally distributed residuals. Deviations suggest the normality assumption may be violated.'
    })
    
    # 4. Sphericity Test Visualization (if available)
    if model_results is not None and hasattr(model_results, 'mauchly'):
        try:
            # Extract Mauchly's test results
            mauchly_w = model_results.mauchly.W
            mauchly_p = model_results.mauchly.p_value
            sphericity_assumed = mauchly_p > 0.05
            
            plt.figure(figsize=(8, 6))
            
            # Create bar chart for Mauchly's W
            plt.bar(['Mauchly\'s W'], [mauchly_w], color='blue')
            plt.axhline(y=1, color='red', linestyle='--', label='Perfect Sphericity (W=1)')
            
            plt.ylim(0, 1.1)
            plt.ylabel('Mauchly\'s W')
            plt.title(f'Mauchly\'s Test of Sphericity\nW={mauchly_w:.3f}, p={mauchly_p:.4f}')
            
            # Add text indicating interpretation
            if sphericity_assumed:
                plt.text(0, mauchly_w + 0.05, 'Sphericity Assumed (p > 0.05)', 
                        ha='center', color='green')
            else:
                plt.text(0, mauchly_w + 0.05, 'Sphericity Violated (p < 0.05)', 
                        ha='center', color='red')
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            plots.append({
                'title': 'Sphericity Test',
                'img_data': get_base64_plot(),
                'interpretation': f'Visualizes Mauchly\'s test of sphericity. Mauchly\'s W = {mauchly_w:.3f} with p = {mauchly_p:.4f}. {"Sphericity is not violated (p > 0.05), so no correction is needed." if sphericity_assumed else "Sphericity is violated (p < 0.05), suggesting that a correction (e.g., Greenhouse-Geisser or Huynh-Feldt) should be applied to the degrees of freedom."}'
            })
        except:
            pass
    
    # 5. Covariance Structure Heatmap
    plt.figure(figsize=(8, 6))
    
    # Reshape data to wide format for covariance calculation
    if len(within_factors) == 1:
        factor = within_factors[0]
        try:
            # Pivot the data to get wide format
            wide_data = data.pivot(index=subject_id, columns=factor, values=dv)
            
            # Compute covariance matrix
            cov_matrix = wide_data.cov()
            
            # Plot heatmap
            sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Covariance Structure of Repeated Measures')
            
            plots.append({
                'title': 'Covariance Structure',
                'img_data': get_base64_plot(),
                'interpretation': 'Shows the covariance matrix between different time points or conditions. For sphericity to hold, these covariances should be approximately equal. Large differences suggest violation of the sphericity assumption.'
            })
        except:
            # If pivot fails (e.g., due to duplicate entries), skip this plot
            pass
    
    # 6. Effect Size Plot
    if model_results is not None and hasattr(model_results, 'anova_table'):
        try:
            # Extract effect sizes
            anova_table = model_results.anova_table
            effect_sizes = anova_table['np2'].values  # Partial eta-squared values
            effect_names = anova_table.index.values
            
            plt.figure(figsize=(10, 6))
            
            # Plot bar chart of effect sizes
            bars = plt.bar(effect_names, effect_sizes, color='skyblue')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.axhline(y=0.01, color='red', linestyle='--', label='Small Effect (0.01)')
            plt.axhline(y=0.06, color='orange', linestyle='--', label='Medium Effect (0.06)')
            plt.axhline(y=0.14, color='green', linestyle='--', label='Large Effect (0.14)')
            
            plt.xlabel('Effects')
            plt.ylabel('Partial Eta Squared')
            plt.title('Effect Sizes (Partial Eta Squared)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            plots.append({
                'title': 'Effect Size Plot',
                'img_data': get_base64_plot(),
                'interpretation': 'Shows the effect sizes (Partial Eta Squared) for each factor and interaction. Larger values indicate stronger effects. Guidelines suggest values of 0.01, 0.06, and 0.14 represent small, medium, and large effect sizes, respectively.'
            })
        except:
            pass
    
    # 7. Pairwise Comparison Plot (if available)
    if len(within_factors) == 1 and len(data[within_factors[0]].unique()) > 2:
        factor = within_factors[0]
        levels = sorted(data[factor].unique())
        level_means = data.groupby(factor)[dv].mean()
        
        # Create all pairwise combinations
        pairs = [(levels[i], levels[j]) for i in range(len(levels)) for j in range(i+1, len(levels))]
        
        # Calculate mean differences
        mean_diffs = [level_means[pair[1]] - level_means[pair[0]] for pair in pairs]
        
        # Create labels for the pairs
        pair_labels = [f"{pair[0]} vs {pair[1]}" for pair in pairs]
        
        plt.figure(figsize=(max(8, len(pairs)*0.8), 6))
        
        # Create bar chart of mean differences
        colors = ['green' if diff > 0 else 'red' for diff in mean_diffs]
        bars = plt.bar(pair_labels, mean_diffs, color=colors)
        
        # Add a horizontal line at zero
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + 0.01 if height > 0 else height - 0.01,
                    f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.xlabel('Pairwise Comparisons')
        plt.ylabel(f'Mean Difference in {dv}')
        plt.title('Pairwise Mean Differences')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        legend_elements = [
            Patch(facecolor='green', label='Positive Difference'),
            Patch(facecolor='red', label='Negative Difference')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        plots.append({
            'title': 'Pairwise Comparisons',
            'img_data': get_base64_plot(),
            'interpretation': 'Shows the mean differences between all pairs of factor levels. Green bars indicate positive differences (second level higher than first), while red bars indicate negative differences. This plot helps identify which specific level comparisons contribute to the overall effect.'
        })
    
    return plots 