"""Structural Equation Modeling diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
import networkx as nx
import matplotlib.patches as patches

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_sem_plots(model, data=None, observed_vars=None, latent_vars=None, 
                     paths=None, model_fit=None, standardized=True):
    """Generate diagnostic plots for Structural Equation Models
    
    Args:
        model: Fitted SEM model (can be any framework: lavaan, semopy, etc.)
        data: DataFrame with observed variables
        observed_vars: List of observed variable names
        latent_vars: List of latent variable names 
        paths: List of tuples with path relationships (from, to, coefficient)
        model_fit: Dictionary with model fit indices
        standardized: Whether to use standardized coefficients
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Try to infer model information if not provided explicitly
    if hasattr(model, 'get_paths') and paths is None:
        # Try semopy-style interface
        try:
            paths = model.get_paths()
        except:
            pass
    
    if hasattr(model, 'coef_') and paths is None:
        # Try sklearn-style interface
        try:
            paths = []
            for i, var_from in enumerate(model.feature_names_in_):
                for j, var_to in enumerate(model.feature_names_out_):
                    if model.coef_[j, i] != 0:
                        paths.append((var_from, var_to, model.coef_[j, i]))
        except:
            pass
    
    # For lavaan-style models, need special handling
    if 'lavaan' in str(type(model)).lower() and paths is None:
        try:
            # Try to extract parameter estimates
            import rpy2.robjects as ro
            from rpy2.robjects.packages import importr
            lavaan = importr('lavaan')
            
            # Get parameter estimates
            param_estimates = ro.r['parameterEstimates'](model, standardized=standardized)
            
            # Convert to pandas DataFrame
            param_df = pd.DataFrame({
                'lhs': param_estimates.rx2('lhs'),
                'op': param_estimates.rx2('op'),
                'rhs': param_estimates.rx2('rhs'),
                'est': param_estimates.rx2('est'),
                'std.all': param_estimates.rx2('std.all') if standardized else param_estimates.rx2('est')
            })
            
            # Extract paths
            paths = []
            for _, row in param_df.iterrows():
                if row['op'] in ['~', '=~']:  # Regression or measurement model
                    paths.append((row['rhs'], row['lhs'], row['std.all'] if standardized else row['est']))
            
            # Extract observed and latent variables
            observed_vars = list(set(param_df[param_df['op'] == '=~']['rhs']))
            latent_vars = list(set(param_df[param_df['op'] == '=~']['lhs']))
            
            # Get model fit indices
            fit_indices = ro.r['fitMeasures'](model)
            model_fit = {
                'chisq': fit_indices.rx2('chisq')[0],
                'df': fit_indices.rx2('df')[0],
                'pvalue': fit_indices.rx2('pvalue')[0],
                'cfi': fit_indices.rx2('cfi')[0],
                'tli': fit_indices.rx2('tli')[0],
                'rmsea': fit_indices.rx2('rmsea')[0],
                'srmr': fit_indices.rx2('srmr')[0]
            }
        except Exception as e:
            print(f"Could not extract lavaan model information: {e}")
    
    # For semopy-style models
    if 'semopy' in str(type(model)).lower() and model_fit is None:
        try:
            fit_res = model.calc_fit()
            model_fit = {
                'chisq': fit_res.loc['Chi-square', 'Value'],
                'df': fit_res.loc['Degrees of freedom', 'Value'],
                'pvalue': fit_res.loc['p-value', 'Value'],
                'cfi': fit_res.loc['CFI', 'Value'],
                'tli': fit_res.loc['TLI', 'Value'],
                'rmsea': fit_res.loc['RMSEA', 'Value'],
                'srmr': fit_res.loc['SRMR', 'Value']
            }
        except Exception as e:
            print(f"Could not extract semopy model fit: {e}")
    
    # Plot 1: Path Diagram
    if paths is not None:
        plt.figure(figsize=(12, 10))
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Get all variables
        all_vars = set()
        for src, dst, _ in paths:
            all_vars.add(src)
            all_vars.add(dst)
        
        # If latent variables are not explicitly provided, try to infer
        if latent_vars is None:
            # Heuristic: variables that are only on the left side of paths are likely latent
            from_vars = set([p[0] for p in paths])
            to_vars = set([p[1] for p in paths])
            potential_latent = to_vars - from_vars
            
            # Separate observed and latent
            if observed_vars is None:
                observed_vars = list(all_vars - potential_latent)
            latent_vars = list(potential_latent)
        
        # If observed variables are not explicitly provided
        if observed_vars is None:
            observed_vars = list(all_vars - set(latent_vars) if latent_vars else all_vars)
        
        # Add nodes to graph
        for var in all_vars:
            is_latent = var in latent_vars if latent_vars else False
            G.add_node(var, latent=is_latent)
        
        # Add edges with attributes
        for src, dst, coef in paths:
            G.add_edge(src, dst, weight=coef)
        
        # Create layout for the graph
        # Hierarchical layout may work better for SEM models
        pos = nx.spring_layout(G, seed=42)
        
        # Draw latent variables as ellipses
        for node, (x, y) in pos.items():
            if G.nodes[node].get('latent', False):
                # Draw ellipse
                ellipse = patches.Ellipse((x, y), width=0.15, height=0.1, 
                                       fill=True, facecolor='lightblue', edgecolor='black',
                                       zorder=1)
                plt.gca().add_patch(ellipse)
                plt.text(x, y, node, horizontalalignment='center', verticalalignment='center',
                       fontsize=10, zorder=2)
            else:
                # Draw rectangle for observed variables
                rect = patches.Rectangle((x-0.07, y-0.05), width=0.14, height=0.1,
                                      fill=True, facecolor='lightgreen', edgecolor='black',
                                      zorder=1)
                plt.gca().add_patch(rect)
                plt.text(x, y, node, horizontalalignment='center', verticalalignment='center',
                       fontsize=10, zorder=2)
        
        # Draw edges with weights as labels
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 0)
            
            # Edge color based on weight sign
            if weight > 0:
                edge_color = 'blue'
            elif weight < 0:
                edge_color = 'red'
            else:
                edge_color = 'gray'
            
            # Draw the edge
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2.0,
                                alpha=min(0.9, abs(weight) + 0.2),  # Stronger weights are more visible
                                edge_color=edge_color, 
                                arrows=True, arrowstyle='->', arrowsize=15)
            
            # Add weight as label
            edge_x = (pos[u][0] + pos[v][0]) / 2
            edge_y = (pos[u][1] + pos[v][1]) / 2
            
            # Offset the label slightly
            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]
            edge_len = np.sqrt(dx*dx + dy*dy)
            
            if edge_len > 0:
                offset_x = -dy / edge_len * 0.03
                offset_y = dx / edge_len * 0.03
            else:
                offset_x = offset_y = 0
                
            plt.text(edge_x + offset_x, edge_y + offset_y, f"{weight:.2f}", 
                   fontsize=9, ha='center', va='center',
                   bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2',
                           alpha=0.8))
        
        # Remove axis
        plt.axis('off')
        plt.title('Structural Equation Model Path Diagram')
        
        plots.append({
            "title": "SEM Path Diagram",
            "img_data": get_base64_plot(),
            "interpretation": "Visualizes the structural equation model with paths between variables. Blue arrows indicate positive relationships, red arrows negative ones. Latent variables are shown as ellipses and observed variables as rectangles. Numbers on arrows represent standardized path coefficients."
        })
    
    # Plot 2: Factor Loadings (for measurement model)
    if paths is not None and latent_vars is not None:
        # Extract factor loadings (paths from observed to latent variables)
        factor_loadings = {}
        
        for src, dst, coef in paths:
            # Check if this is a measurement relationship (observed -> latent)
            if dst in latent_vars and src in observed_vars:
                if dst not in factor_loadings:
                    factor_loadings[dst] = []
                factor_loadings[dst].append((src, coef))
        
        # Plot factor loadings for each latent variable
        for latent_var, loadings in factor_loadings.items():
            plt.figure(figsize=(10, 6))
            
            # Sort loadings by magnitude
            loadings.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Separate variables and coefficients
            vars_names = [loading[0] for loading in loadings]
            coefficients = [loading[1] for loading in loadings]
            
            # Create bar chart
            colors = ['blue' if coef > 0 else 'red' for coef in coefficients]
            plt.barh(range(len(vars_names)), [abs(c) for c in coefficients], color=colors, alpha=0.6)
            plt.yticks(range(len(vars_names)), vars_names)
            
            # Add coefficient values as labels
            for i, coef in enumerate(coefficients):
                plt.text(max(0.01, abs(coef) - 0.05), i, f"{coef:.3f}", 
                       va='center', ha='right', color='black', fontweight='bold')
            
            plt.xlabel('Standardized Factor Loading')
            plt.title(f'Factor Loadings for {latent_var}')
            plt.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, 
                      label='0.7 (Strong loading)')
            plt.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5,
                      label='0.3 (Weak loading)')
            plt.legend()
            plt.tight_layout()
            
            plots.append({
                "title": f"Factor Loadings: {latent_var}",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows the factor loadings for the latent variable '{latent_var}'. Loadings above 0.7 indicate strong relationships, while those below 0.3 suggest weaker connections. Blue bars represent positive loadings, red bars negative ones."
            })
    
    # Plot 3: Model Fit Indices
    if model_fit is not None:
        plt.figure(figsize=(12, 8))
        
        # Create table-like visualization of fit indices
        fit_metrics = [
            ('Chi-square', model_fit.get('chisq', np.nan), '< 3×df'),
            ('Degrees of freedom', model_fit.get('df', np.nan), 'Higher is better'),
            ('p-value', model_fit.get('pvalue', np.nan), '> 0.05'),
            ('CFI (Comparative Fit Index)', model_fit.get('cfi', np.nan), '> 0.95'),
            ('TLI (Tucker-Lewis Index)', model_fit.get('tli', np.nan), '> 0.95'), 
            ('RMSEA', model_fit.get('rmsea', np.nan), '< 0.05'),
            ('SRMR', model_fit.get('srmr', np.nan), '< 0.08')
        ]
        
        # Create color coding based on cutoffs
        colors = []
        for metric, value, _ in fit_metrics:
            if metric == 'Chi-square' and 'df' in model_fit:
                # Chi-square should be less than 3 times df
                if value < 3 * model_fit['df']:
                    colors.append('green')
                else:
                    colors.append('red')
            elif metric == 'p-value':
                if value > 0.05:
                    colors.append('green')
                else:
                    colors.append('red')
            elif metric in ['CFI (Comparative Fit Index)', 'TLI (Tucker-Lewis Index)']:
                if value > 0.95:
                    colors.append('green')
                elif value > 0.9:
                    colors.append('orange')
                else:
                    colors.append('red')
            elif metric == 'RMSEA':
                if value < 0.05:
                    colors.append('green')
                elif value < 0.08:
                    colors.append('orange')
                else:
                    colors.append('red')
            elif metric == 'SRMR':
                if value < 0.08:
                    colors.append('green')
                else:
                    colors.append('red')
            else:
                colors.append('blue')
        
        # Plot the data
        cell_text = [[metric, f"{value:.4f}" if isinstance(value, (int, float)) else str(value), cutoff] 
                    for (metric, value, cutoff) in fit_metrics]
        
        # Create table
        plt.axis('off')
        table = plt.table(cellText=cell_text,
                        colLabels=['Fit Metric', 'Value', 'Recommended Cutoff'],
                        cellLoc='center',
                        loc='center',
                        cellColours=[(color, 'white', 'white') for color in colors])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        plt.title('Model Fit Indices')
        plt.tight_layout()
        
        plots.append({
            "title": "Model Fit Indices",
            "img_data": get_base64_plot(),
            "interpretation": "Displays the key model fit indices and their interpretation. Green values indicate good fit, orange indicates acceptable fit, and red indicates poor fit according to common guidelines. A well-fitting model should have CFI/TLI > 0.95, RMSEA < 0.05, and SRMR < 0.08."
        })
    
    # Plot 4: Residual Correlation Matrix
    if hasattr(model, 'residuals') and data is not None:
        try:
            # Get residual correlation matrix
            residuals = model.residuals()
            
            plt.figure(figsize=(12, 10))
            
            # Plot heatmap of residual correlations
            sns.heatmap(residuals, cmap='coolwarm', center=0, 
                      vmin=-0.1, vmax=0.1, annot=True, fmt='.2f')
            
            plt.title('Residual Correlation Matrix')
            plt.tight_layout()
            
            plots.append({
                "title": "Residual Correlation Matrix",
                "img_data": get_base64_plot(),
                "interpretation": "Shows residual correlations after model fitting. Large residual values (> 0.1 or < -0.1) indicate relationships not adequately captured by the model. Such residuals suggest the need for model refinement, possibly by adding paths where large residuals exist."
            })
        except Exception as e:
            print(f"Could not calculate residual matrix: {e}")
    
    # Plot 5: Variable Correlations and Model-Implied Correlations
    if data is not None and observed_vars is not None:
        try:
            # Get observed data correlations
            observed_data = data[observed_vars]
            observed_corr = observed_data.corr()
            
            # Get model-implied correlations if available
            model_implied_corr = None
            if hasattr(model, 'implied_cov'):
                # semopy-style interface
                try:
                    imp_cov = model.implied_cov()
                    # Convert to correlation matrix
                    D = np.sqrt(np.diag(imp_cov))
                    D_inv = np.zeros_like(D)
                    D_inv[D > 0] = 1 / D[D > 0]
                    model_implied_corr = np.diag(D_inv) @ imp_cov @ np.diag(D_inv)
                except:
                    pass
            
            # Plot observed correlations
            plt.figure(figsize=(12, 10))
            sns.heatmap(observed_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Observed Variable Correlations')
            plt.tight_layout()
            
            plots.append({
                "title": "Observed Correlations",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the empirical correlations between observed variables in the dataset. Strong correlations (positive or negative) suggest relationships that should be captured by the structural model."
            })
            
            # Plot model-implied correlations if available
            if model_implied_corr is not None:
                plt.figure(figsize=(12, 10))
                sns.heatmap(model_implied_corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
                plt.title('Model-Implied Correlations')
                plt.tight_layout()
                
                plots.append({
                    "title": "Model-Implied Correlations",
                    "img_data": get_base64_plot(),
                    "interpretation": "Shows the correlations between variables as implied by the fitted model. Compare with the observed correlations to assess model fit. Large discrepancies indicate relationships not adequately captured by the model."
                })
                
                # Plot correlation differences
                plt.figure(figsize=(12, 10))
                diff = observed_corr.values - model_implied_corr
                sns.heatmap(diff, annot=True, fmt='.2f', cmap='coolwarm', 
                          center=0, vmin=-0.3, vmax=0.3)
                plt.title('Observed - Model-Implied Correlations')
                plt.tight_layout()
                
                plots.append({
                    "title": "Correlation Differences",
                    "img_data": get_base64_plot(),
                    "interpretation": "Shows the differences between observed and model-implied correlations. Large values (> 0.1 or < -0.1) indicate relationships not adequately captured by the model. These differences can help identify areas for model improvement."
                })
        except Exception as e:
            print(f"Could not calculate correlation matrices: {e}")
    
    # Plot 6: Modification Indices
    if hasattr(model, 'modification_indices'):
        try:
            # Get modification indices
            mod_indices = model.modification_indices()
            
            # Convert to DataFrame if it's not already
            if not isinstance(mod_indices, pd.DataFrame):
                mod_indices = pd.DataFrame(mod_indices)
            
            # Sort by index value
            mod_indices_sorted = mod_indices.sort_values(by=mod_indices.columns[2], ascending=False)
            
            # Take top 10 modifications
            top_mods = mod_indices_sorted.head(10)
            
            plt.figure(figsize=(12, 6))
            
            # Create a bar chart of modification indices
            y_pos = range(len(top_mods))
            mod_vals = top_mods.iloc[:, 2].values
            labels = [f"{top_mods.iloc[i, 0]} - {top_mods.iloc[i, 1]}" for i in range(len(top_mods))]
            
            plt.barh(y_pos, mod_vals, color='skyblue')
            plt.yticks(y_pos, labels)
            plt.xlabel('Modification Index')
            plt.title('Top 10 Modification Indices')
            
            # Add a reference line for significance
            plt.axvline(x=3.84, color='red', linestyle='--', 
                      label='Significant (χ²₁ = 3.84, p = 0.05)')
            plt.legend()
            plt.tight_layout()
            
            plots.append({
                "title": "Modification Indices",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the top 10 potential model modifications ordered by their impact. Modification indices indicate how much the model χ² would decrease if a particular parameter were freed. Values above 3.84 (red line) suggest statistically significant improvements, but modifications should be theoretically justified."
            })
        except Exception as e:
            print(f"Could not calculate modification indices: {e}")
    
    return plots 