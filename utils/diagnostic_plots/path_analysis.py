"""Path Analysis diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.patches as patches
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

def generate_path_analysis_plots(model=None, data=None, paths=None, effects=None, 
                             model_fit=None, variable_names=None):
    """Generate diagnostic plots for Path Analysis models
    
    Args:
        model: Fitted path analysis model (can be from any framework)
        data: DataFrame with variables used in the model
        paths: List of tuples with path relationships (from, to, coefficient)
        effects: Dictionary with direct, indirect and total effects
        model_fit: Dictionary with model fit indices (chi-square, CFI, RMSEA, etc.)
        variable_names: List of variable names used in the model
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Extract path information from model if not provided
    if paths is None and model is not None:
        # Try to extract paths from different frameworks
        try:
            # Try for lavaan models (R)
            if 'lavaan' in str(type(model)).lower():
                import rpy2.robjects as ro
                from rpy2.robjects.packages import importr
                lavaan = importr('lavaan')
                
                # Extract parameter estimates
                param_estimates = ro.r['parameterEstimates'](model)
                paths = []
                
                # Convert to Python data structure
                for i in range(len(param_estimates.rx2('lhs'))):
                    lhs = param_estimates.rx2('lhs')[i]
                    op = param_estimates.rx2('op')[i]
                    rhs = param_estimates.rx2('rhs')[i]
                    est = param_estimates.rx2('est')[i]
                    
                    # Only include paths (not variances/covariances)
                    if op == '~':
                        paths.append((rhs, lhs, est))
                        
            # Try for statsmodels
            elif hasattr(model, 'params'):
                paths = []
                for i, coef in enumerate(model.params[1:]):  # Skip intercept
                    if abs(coef) > 0:
                        var_from = variable_names[i] if variable_names else f"X{i+1}"
                        var_to = 'Y'
                        paths.append((var_from, var_to, coef))
        except Exception as e:
            print(f"Could not extract path information from model: {e}")
    
    # Plot 1: Path Diagram
    if paths is not None:
        plt.figure(figsize=(12, 10))
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Extract all variables
        all_vars = set()
        for src, dst, _ in paths:
            all_vars.add(src)
            all_vars.add(dst)
        
        # Add nodes
        for var in all_vars:
            G.add_node(var)
        
        # Add edges with weights
        for src, dst, coef in paths:
            G.add_edge(src, dst, weight=coef)
        
        # Create layout (hierarchical layout works well for path diagrams)
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', 
                            alpha=0.8, node_shape='o', edgecolors='black')
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # Draw edges with different colors and widths based on coefficient values
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            
            # Determine edge color and width based on coefficient value
            if weight > 0:
                edge_color = 'blue'
            else:
                edge_color = 'red'
                
            edge_width = 2.0 * abs(weight)
            
            # Draw the edge
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=edge_width,
                                alpha=0.7, edge_color=edge_color, 
                                arrows=True, arrowstyle='->', arrowsize=15)
                                
            # Add coefficient label
            edge_x = (pos[u][0] + pos[v][0]) / 2.0
            edge_y = (pos[u][1] + pos[v][1]) / 2.0
            
            # Calculate offset for label to avoid overlap with the edge
            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]
            edge_len = np.sqrt(dx*dx + dy*dy)
            
            if edge_len > 0:
                offset_x = -dy / edge_len * 0.1
                offset_y = dx / edge_len * 0.1
            else:
                offset_x = offset_y = 0
                
            plt.text(edge_x + offset_x, edge_y + offset_y, f"{weight:.2f}", 
                   fontsize=10, fontweight='bold', ha='center', va='center',
                   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2',
                           alpha=0.8))
        
        plt.axis('off')  # Hide axis
        plt.title('Path Analysis Diagram', fontsize=14, fontweight='bold')
        
        plots.append({
            "title": "Path Diagram",
            "img_data": get_base64_plot(),
            "interpretation": "Visualization of the path analysis model showing relationships between variables. Blue arrows indicate positive relationships, red arrows indicate negative relationships. Numbers on arrows show path coefficients."
        })
    
    # Plot 2: Standardized Effects
    if effects is not None or paths is not None:
        # If effects not provided but paths available, calculate direct effects from paths
        if effects is None:
            effects = {'direct': {}}
            for src, dst, coef in paths:
                if dst not in effects['direct']:
                    effects['direct'][dst] = {}
                effects['direct'][dst][src] = coef
        
        plt.figure(figsize=(14, 10))
        
        # Get all effect types and variables
        effect_types = []
        all_to_vars = set()
        all_from_vars = set()
        
        # Collect information
        for effect_type, effect_dict in effects.items():
            effect_types.append(effect_type)
            for to_var, from_dict in effect_dict.items():
                all_to_vars.add(to_var)
                for from_var in from_dict.keys():
                    all_from_vars.add(from_var)
        
        # Sort variable names
        all_to_vars = sorted(all_to_vars)
        all_from_vars = sorted(all_from_vars)
        
        # Create a subplot for each effect type
        n_types = len(effect_types)
        fig, axes = plt.subplots(1, n_types, figsize=(15, 5 * n_types))
        
        # Handle single subplot case
        if n_types == 1:
            axes = [axes]
        
        # Plot each type of effect
        for i, effect_type in enumerate(effect_types):
            # Create a matrix of effects
            effect_matrix = np.zeros((len(all_to_vars), len(all_from_vars)))
            
            # Fill in known effects
            effect_dict = effects[effect_type]
            for j, to_var in enumerate(all_to_vars):
                if to_var in effect_dict:
                    for k, from_var in enumerate(all_from_vars):
                        if from_var in effect_dict[to_var]:
                            effect_matrix[j, k] = effect_dict[to_var][from_var]
            
            # Create heatmap
            ax = axes[i]
            im = sns.heatmap(effect_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                          center=0, vmin=-1, vmax=1, cbar=True, ax=ax,
                          xticklabels=all_from_vars, yticklabels=all_to_vars)
            
            # Set title and labels
            ax.set_title(f"{effect_type.capitalize()} Effects", fontsize=12, fontweight='bold')
            ax.set_xlabel("From Variable", fontsize=11)
            ax.set_ylabel("To Variable", fontsize=11)
        
        plt.tight_layout()
        
        effect_types_str = ", ".join(effect_types)
        plots.append({
            "title": "Standardized Effects",
            "img_data": get_base64_plot(),
            "interpretation": f"Heatmap showing {effect_types_str} between variables. Values range from -1 to 1, with blue indicating positive effects and red indicating negative effects. The intensity of color represents the magnitude of the effect."
        })
    
    # Plot 3: Model Fit Indices
    if model_fit is not None:
        plt.figure(figsize=(10, 6))
        
        # Common fit indices and their recommended cutoffs
        fit_indices = [
            ('Chi-square', model_fit.get('chisq', np.nan), '< 3×df'),
            ('Degrees of freedom', model_fit.get('df', np.nan), '-'),
            ('p-value', model_fit.get('pvalue', np.nan), '> 0.05'),
            ('CFI', model_fit.get('cfi', np.nan), '≥ 0.95'),
            ('TLI', model_fit.get('tli', np.nan), '≥ 0.95'),
            ('RMSEA', model_fit.get('rmsea', np.nan), '≤ 0.06'),
            ('SRMR', model_fit.get('srmr', np.nan), '≤ 0.08'),
            ('GFI', model_fit.get('gfi', np.nan), '≥ 0.90'),
            ('AGFI', model_fit.get('agfi', np.nan), '≥ 0.90')
        ]
        
        # Filter out NaN values
        fit_indices = [(name, value, cutoff) for name, value, cutoff in fit_indices 
                    if not (isinstance(value, float) and np.isnan(value))]
        
        # Create colors based on fit thresholds
        colors = []
        for name, value, cutoff in fit_indices:
            if name == 'Chi-square' and 'df' in model_fit:
                # Chi-square should be less than 3 times df
                colors.append('green' if value < 3 * model_fit['df'] else 'red')
            elif name == 'p-value':
                colors.append('green' if value > 0.05 else 'red')
            elif name in ['CFI', 'TLI', 'GFI', 'AGFI']:
                if value >= 0.95:
                    colors.append('green')
                elif value >= 0.90:
                    colors.append('orange')
                else:
                    colors.append('red')
            elif name == 'RMSEA':
                if value <= 0.05:
                    colors.append('green')
                elif value <= 0.08:
                    colors.append('orange')
                else:
                    colors.append('red')
            elif name == 'SRMR':
                if value <= 0.08:
                    colors.append('green')
                else:
                    colors.append('red')
            else:
                colors.append('gray')
        
        # Create table content
        cell_text = [[name, f"{value:.3f}" if isinstance(value, (int, float)) else str(value), cutoff] 
                   for name, value, cutoff in fit_indices]
        
        # Create table
        plt.axis('off')
        table = plt.table(cellText=cell_text,
                        colLabels=['Fit Index', 'Value', 'Good Fit Cutoff'],
                        cellLoc='center',
                        loc='center',
                        cellColours=[(color, 'white', 'white') for color in colors])
        
        # Format table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        plt.title('Model Fit Indices', fontsize=14, fontweight='bold')
        
        plots.append({
            "title": "Model Fit Indices",
            "img_data": get_base64_plot(),
            "interpretation": "Table showing model fit indices and their values. Green indicates good fit, orange indicates acceptable fit, and red indicates poor fit according to common cutoff criteria."
        })
    
    # Plot 4: Residual Correlation Heatmap
    if model is not None and hasattr(model, 'residuals') and data is not None:
        try:
            # Extract residuals
            residuals = model.residuals()
            
            plt.figure(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(residuals, annot=True, fmt=".2f", cmap="coolwarm", 
                      center=0, vmin=-0.3, vmax=0.3, cbar=True)
            
            plt.title('Residual Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            plots.append({
                "title": "Residual Correlations",
                "img_data": get_base64_plot(),
                "interpretation": "Heatmap showing correlations between residuals. Large residual correlations (> 0.1 or < -0.1) suggest relationships not captured by the model, indicating possible areas for model improvement."
            })
        except Exception as e:
            print(f"Could not generate residual correlation plot: {e}")
    
    # Plot 5: Modification Indices (if available)
    if model is not None and hasattr(model, 'modification_indices'):
        try:
            # Get modification indices
            mod_indices = model.modification_indices()
            
            # Convert to DataFrame for easier manipulation
            if not isinstance(mod_indices, pd.DataFrame):
                mod_indices = pd.DataFrame(mod_indices)
            
            # Sort by modification index value (descending)
            mod_indices = mod_indices.sort_values(mod_indices.columns[2], ascending=False)
            
            # Take top 10 modification indices
            top_indices = mod_indices.head(10)
            
            plt.figure(figsize=(12, 6))
            
            # Create bar chart of top modification indices
            y_pos = range(len(top_indices))
            index_values = top_indices.iloc[:, 2].values
            
            # Create labels from variable pairs
            labels = [f"{row[0]} - {row[1]}" for _, row in top_indices.iterrows()]
            
            plt.barh(y_pos, index_values, color='skyblue')
            plt.yticks(y_pos, labels)
            plt.xlabel('Modification Index')
            plt.title('Top 10 Modification Indices', fontsize=14, fontweight='bold')
            
            # Add reference line at 3.84 (chi-square critical value for df=1 at p=0.05)
            plt.axvline(x=3.84, color='red', linestyle='--', 
                      label='χ² critical value (df=1, p=0.05)')
            plt.legend()
            
            plt.tight_layout()
            
            plots.append({
                "title": "Modification Indices",
                "img_data": get_base64_plot(),
                "interpretation": "Bar chart showing the top 10 modification indices. Modification indices indicate how much the chi-square value would decrease if a parameter was freed. Values above 3.84 (red line) are statistically significant at p=0.05."
            })
        except Exception as e:
            print(f"Could not generate modification indices plot: {e}")
    
    # Plot 6: Path Coefficients (Bar chart)
    if paths is not None:
        plt.figure(figsize=(12, 6))
        
        # Get coefficients and create labels
        coeffs = [coef for _, _, coef in paths]
        labels = [f"{src} → {dst}" for src, dst, _ in paths]
        
        # Sort by absolute coefficient value
        sorted_indices = np.argsort(np.abs(coeffs))[::-1]
        sorted_coeffs = [coeffs[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]
        
        # Set colors based on coefficient sign
        colors = ['blue' if c > 0 else 'red' for c in sorted_coeffs]
        
        # Create bar chart
        y_pos = range(len(sorted_coeffs))
        plt.barh(y_pos, sorted_coeffs, color=colors, alpha=0.7)
        plt.yticks(y_pos, sorted_labels)
        
        # Add zero reference line
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.xlabel('Path Coefficient')
        plt.title('Path Coefficients (Sorted by Magnitude)', fontsize=14, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        
        plots.append({
            "title": "Path Coefficients",
            "img_data": get_base64_plot(),
            "interpretation": "Bar chart showing path coefficients sorted by absolute magnitude. Blue bars represent positive coefficients, and red bars represent negative coefficients. Longer bars indicate stronger relationships between variables."
        })
    
    return plots 