"""Analysis of Covariance (ANCOVA) diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_ancova_plots(model=None, data=None, dv=None, group_var=None, 
                         covariates=None, adjusted_means=None, summary=None):
    """Generate diagnostic plots for Analysis of Covariance (ANCOVA) models
    
    Args:
        model: Fitted ANCOVA model (can be statsmodels or other framework)
        data: DataFrame containing the variables
        dv: Name of dependent variable column in data
        group_var: Name of categorical independent variable column in data
        covariates: List of covariate column names in data
        adjusted_means: Pre-computed adjusted means if available
        summary: Model summary object if available
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Check if we have necessary data
    if data is None or dv is None or group_var is None:
        return plots
    
    # Ensure covariates is a list
    if covariates is None:
        covariates = []
    elif not isinstance(covariates, list):
        covariates = [covariates]
    
    # Get unique groups
    groups = data[group_var].unique()
    
    # Plot 1: Covariate Relationship by Group
    if len(covariates) > 0:
        for covariate in covariates:
            plt.figure(figsize=(10, 6))
            
            # Scatter plot with regression lines for each group
            sns.scatterplot(x=covariate, y=dv, hue=group_var, data=data, alpha=0.6)
            sns.regplot(x=covariate, y=dv, data=data, scatter=False, ci=None, 
                       line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 2},
                       label='Overall')
            
            for group in groups:
                group_data = data[data[group_var] == group]
                sns.regplot(x=covariate, y=dv, data=group_data, scatter=False, ci=None,
                           label=f'{group} trend')
            
            plt.title(f'Relationship between {dv} and {covariate} by {group_var}')
            plt.xlabel(covariate)
            plt.ylabel(dv)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": f"Covariate Relationship: {covariate}",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows the relationship between the dependent variable ({dv}) and the covariate ({covariate}) across different groups. Parallel regression lines suggest the homogeneity of regression slopes assumption is met (no interaction between covariate and group). The dashed black line shows the overall relationship ignoring groups."
            })
    
    # Plot 2: Adjusted vs Raw Group Means
    if adjusted_means is not None or (model is not None and hasattr(model, 'get_marginal_effects')):
        plt.figure(figsize=(10, 6))
        
        # If adjusted means were not provided but can be computed from model
        if adjusted_means is None and model is not None:
            try:
                # Try to get adjusted means from model
                if hasattr(model, 'get_marginal_effects'):
                    adjusted_means = model.get_marginal_effects()
                elif hasattr(model, 'emmeans_'):
                    adjusted_means = model.emmeans_
            except:
                # If we can't get adjusted means, we'll calculate raw means only
                pass
        
        # Calculate raw means by group
        raw_means = data.groupby(group_var)[dv].mean()
        
        # Prepare data for plotting
        plot_data = []
        
        for group in groups:
            # Add raw mean
            plot_data.append({
                'Group': str(group),
                'Mean': raw_means[group],
                'Type': 'Raw'
            })
            
            # Add adjusted mean if available
            if adjusted_means is not None:
                # Handle different possible formats of adjusted_means
                if isinstance(adjusted_means, pd.DataFrame):
                    if group_var in adjusted_means.columns and 'mean' in adjusted_means.columns:
                        adj_mean = adjusted_means[adjusted_means[group_var] == group]['mean'].values[0]
                    elif str(group) in adjusted_means.index:
                        adj_mean = adjusted_means.loc[str(group)].values[0]
                    else:
                        continue
                elif isinstance(adjusted_means, dict) and str(group) in adjusted_means:
                    adj_mean = adjusted_means[str(group)]
                else:
                    continue
                    
                plot_data.append({
                    'Group': str(group),
                    'Mean': adj_mean,
                    'Type': 'Adjusted'
                })
        
        # Create DataFrame and plot
        plot_df = pd.DataFrame(plot_data)
        
        # Create bar plot
        sns.barplot(x='Group', y='Mean', hue='Type', data=plot_df)
        plt.title(f'Raw vs Adjusted Means by {group_var}')
        plt.xlabel(group_var)
        plt.ylabel(f'Mean {dv}')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Raw vs Adjusted Means",
            "img_data": get_base64_plot(),
            "interpretation": "Compares the raw group means with the covariate-adjusted means. Differences between raw and adjusted means indicate the impact of controlling for covariates. If groups differ in covariate values, the adjustment can substantially change the interpretation of group differences."
        })
    
    # Plot 3: Residual Diagnostics
    if model is not None and hasattr(model, 'resid'):
        # Create residual plots
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 2, figure=fig)
        
        # Get residuals
        residuals = model.resid
        
        # 1. Residuals vs Fitted
        ax1 = fig.add_subplot(gs[0, 0])
        fitted_values = model.fittedvalues
        ax1.scatter(fitted_values, residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        
        # Add a lowess smoothed line
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smooth = lowess(residuals, fitted_values)
            ax1.plot(smooth[:, 0], smooth[:, 1], 'k-', lw=2)
        except:
            pass
            
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted')
        ax1.grid(True, alpha=0.3)
        
        # 2. QQ Plot
        ax2 = fig.add_subplot(gs[0, 1])
        from scipy import stats
        stats.probplot(residuals, plot=ax2)
        ax2.set_title('Normal Q-Q Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Scale-Location Plot
        ax3 = fig.add_subplot(gs[1, 0])
        sqrt_abs_resid = np.sqrt(np.abs(residuals))
        ax3.scatter(fitted_values, sqrt_abs_resid, alpha=0.6)
        
        # Add a lowess smoothed line
        try:
            smooth = lowess(sqrt_abs_resid, fitted_values)
            ax3.plot(smooth[:, 0], smooth[:, 1], 'k-', lw=2)
        except:
            pass
            
        ax3.set_xlabel('Fitted Values')
        ax3.set_ylabel('√|Residuals|')
        ax3.set_title('Scale-Location Plot')
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals by Group
        ax4 = fig.add_subplot(gs[1, 1])
        # Create a DataFrame with residuals and group
        resid_data = pd.DataFrame({
            'Residuals': residuals,
            'Group': data[group_var].values
        })
        
        # Plot boxplots of residuals by group
        sns.boxplot(x='Group', y='Residuals', data=resid_data, ax=ax4)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_title(f'Residuals by {group_var}')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plots.append({
            "title": "Residual Diagnostics",
            "img_data": get_base64_plot(),
            "interpretation": "Four diagnostic plots: (1) Residuals vs Fitted - should show random scatter around zero with no pattern; (2) Normal Q-Q - residuals should follow the diagonal line for normality; (3) Scale-Location - should show even spread of residuals (homoscedasticity); (4) Residuals by Group - should show similar distributions across groups. Violations suggest model assumptions may not be met."
        })
    
    # Plot 4: Pairwise Comparisons
    if adjusted_means is not None:
        try:
            plt.figure(figsize=(10, 8))
            
            # Convert adjusted_means to a format suitable for pairwise comparisons
            if isinstance(adjusted_means, pd.DataFrame):
                # Extract group values and means
                if group_var in adjusted_means.columns and 'mean' in adjusted_means.columns:
                    adj_groups = adjusted_means[group_var].values
                    adj_values = adjusted_means['mean'].values
                else:
                    adj_groups = adjusted_means.index
                    adj_values = adjusted_means.iloc[:, 0].values
            elif isinstance(adjusted_means, dict):
                adj_groups = list(adjusted_means.keys())
                adj_values = list(adjusted_means.values())
            else:
                raise ValueError("Adjusted means format not recognized")
                
            # Create data for pairwise comparison
            compare_data = []
            for i, group in enumerate(adj_groups):
                compare_data.extend([adj_values[i]] * 10)  # Replicate for stability
                
            compare_groups = []
            for group in adj_groups:
                compare_groups.extend([str(group)] * 10)
                
            # Perform Tukey's HSD test
            tukey = pairwise_tukeyhsd(compare_data, compare_groups, alpha=0.05)
            
            # Plot the results
            from matplotlib.lines import Line2D
            
            # Extract results
            result_df = pd.DataFrame(data=tukey._results_table.data[1:], 
                                     columns=tukey._results_table.data[0])
            
            # Sort by group1, group2
            result_df = result_df.sort_values(by=['group1', 'group2'])
            
            # Plot each pairwise comparison
            plt.figure(figsize=(12, len(result_df) * 0.5 + 2))
            
            # Plot confidence intervals
            for i, row in enumerate(result_df.itertuples()):
                # Extract values
                group1 = row.group1
                group2 = row.group2
                diff = float(row.meandiff)
                lower = float(row.lower)
                upper = float(row.upper)
                reject = row.reject == 'True'
                
                # Plot confidence interval
                plt.plot([lower, upper], [i, i], 'o-', color='red' if reject else 'blue')
                plt.axvline(x=0, color='black', linestyle='--')
                
                # Add group labels
                plt.text(-0.1, i, f"{group1} - {group2}", ha='right', va='center')
                
                # Add p-value annotation
                p_value = float(row.p_adj)
                p_text = f"p = {p_value:.4f}" if p_value >= 0.0001 else "p < 0.0001"
                plt.text(upper + 0.1, i, p_text, ha='left', va='center')
            
            # Set axis limits and labels
            plt.ylim(-1, len(result_df))
            plt.xlim(min(result_df['lower'].astype(float).min() - 1, -1), 
                     max(result_df['upper'].astype(float).max() + 1, 1))
            
            plt.xlabel('Mean Difference')
            plt.title("Tukey's HSD Pairwise Comparisons of Adjusted Means")
            
            # Add legend
            legend_elements = [
                Line2D([0], [0], marker='o', color='red', label='Significant Difference'),
                Line2D([0], [0], marker='o', color='blue', label='Non-significant Difference')
            ]
            plt.legend(handles=legend_elements, loc='best')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plots.append({
                "title": "Pairwise Comparisons",
                "img_data": get_base64_plot(),
                "interpretation": "Shows Tukey's HSD pairwise comparisons of adjusted group means. Confidence intervals in red cross zero indicate statistically significant differences between group pairs. The p-values show the significance level of each comparison after adjustment for multiple testing."
            })
        except Exception as e:
            # If pairwise comparison plot fails, we'll skip it
            pass
    
    # Plot 5: ANCOVA Model Summary
    if summary is not None:
        try:
            plt.figure(figsize=(10, 8))
            plt.axis('off')
            
            # Extract relevant information from the summary
            if hasattr(summary, 'tables'):
                table_data = summary.tables[0].data
            elif isinstance(summary, pd.DataFrame):
                table_data = summary.values
            elif isinstance(summary, str):
                # Try to parse string summary into rows
                table_data = [line.split() for line in summary.split('\n') if line.strip()]
            else:
                # Fallback for other summary formats
                table_data = [['Term', 'Sum Sq', 'df', 'F value', 'Pr(>F)']]
                
                # Extract parameters from the model if possible
                if hasattr(model, 'params'):
                    for i, param in enumerate(model.params):
                        name = model.model.exog_names[i] if hasattr(model.model, 'exog_names') else f"Param {i}"
                        p_value = model.pvalues[i] if hasattr(model, 'pvalues') else np.nan
                        table_data.append([name, np.nan, np.nan, np.nan, p_value])
            
            # Create a table
            table = plt.table(
                cellText=table_data[1:],
                colLabels=table_data[0],
                cellLoc='center',
                loc='center',
                edges='horizontal'
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Color significant p-values
            if len(table_data[0]) >= 5:  # If there's a p-value column
                p_val_col = table_data[0].index('Pr(>F)') if 'Pr(>F)' in table_data[0] else 4
                
                for i in range(len(table_data) - 1):
                    cell = table_data[i + 1][p_val_col]
                    
                    # Convert cell to float if it's a string representation of a number
                    try:
                        if isinstance(cell, str):
                            # Handle scientific notation or special formats
                            if 'e' in cell.lower() or cell.startswith('<'):
                                p_value = float(cell.replace('<', '').replace('>', '')) \
                                        if cell.replace('<', '').replace('>', '') else 0.0001
                            else:
                                p_value = float(cell)
                        else:
                            p_value = float(cell)
                            
                        # Color based on significance
                        if p_value < 0.001:
                            table[(i + 1, p_val_col)].set_facecolor('lightcoral')
                        elif p_value < 0.01:
                            table[(i + 1, p_val_col)].set_facecolor('lightpink')
                        elif p_value < 0.05:
                            table[(i + 1, p_val_col)].set_facecolor('lightyellow')
                    except (ValueError, TypeError):
                        # Skip if we can't convert to float
                        pass
            
            plt.title('ANCOVA Model Summary')
            
            plots.append({
                "title": "ANCOVA Model Summary",
                "img_data": get_base64_plot(),
                "interpretation": "Displays the ANCOVA model results table. F-values test the significance of each term in the model. The p-values (Pr(>F)) indicate statistical significance, with values less than 0.05 traditionally considered significant. The colored cells highlight significant effects at different levels (red: p<0.001, pink: p<0.01, yellow: p<0.05)."
            })
        except:
            # If summary plot fails, we'll skip it
            pass
    
    return plots 