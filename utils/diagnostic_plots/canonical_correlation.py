"""Canonical Correlation Analysis diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_canonical_correlation_plots(model=None, X=None, Y=None, 
                                       X_names=None, Y_names=None,
                                       canonical_correlations=None,
                                       x_loadings=None, y_loadings=None,
                                       x_scores=None, y_scores=None,
                                       x_cross_loadings=None, y_cross_loadings=None,
                                       n_pairs=None, p_values=None):
    """Generate diagnostic plots for Canonical Correlation Analysis.
    
    Args:
        model: Fitted canonical correlation model (from statsmodels or other)
        X: Matrix of predictors (first set of variables)
        Y: Matrix of criterion variables (second set of variables)
        X_names: Names of X variables
        Y_names: Names of Y variables
        canonical_correlations: List of canonical correlation coefficients
        x_loadings: Loadings of X variables on canonical variates
        y_loadings: Loadings of Y variables on canonical variates
        x_scores: X canonical scores
        y_scores: Y canonical scores
        x_cross_loadings: Cross-loadings for X variables
        y_cross_loadings: Cross-loadings for Y variables
        n_pairs: Number of canonical pairs to analyze
        p_values: P-values for canonical correlations
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Check if we have necessary data
    if X is None or Y is None:
        return plots
    
    # Convert X and Y to numpy arrays if they're pandas DataFrames
    if isinstance(X, pd.DataFrame):
        if X_names is None:
            X_names = X.columns.tolist()
        X = X.values
    
    if isinstance(Y, pd.DataFrame):
        if Y_names is None:
            Y_names = Y.columns.tolist()
        Y = Y.values
    
    # If names are not provided, create generic names
    if X_names is None:
        X_names = [f'X{i+1}' for i in range(X.shape[1])]
    
    if Y_names is None:
        Y_names = [f'Y{i+1}' for i in range(Y.shape[1])]
    
    # If we have a model object, try to extract canonical correlations and other values
    if model is not None:
        try:
            # Extract canonical correlations from the model
            if hasattr(model, 'cancorr'):
                canonical_correlations = model.cancorr
            elif hasattr(model, 'canonical_correlations'):
                canonical_correlations = model.canonical_correlations
            elif hasattr(model, 'corr'):
                canonical_correlations = model.corr
                
            # Try to extract loadings
            if hasattr(model, 'x_loadings') and x_loadings is None:
                x_loadings = model.x_loadings
            if hasattr(model, 'y_loadings') and y_loadings is None:
                y_loadings = model.y_loadings
                
            # Try to extract scores
            if hasattr(model, 'x_scores') and x_scores is None:
                x_scores = model.x_scores
            if hasattr(model, 'y_scores') and y_scores is None:
                y_scores = model.y_scores
                
            # Try to extract cross-loadings
            if hasattr(model, 'x_cross_loadings') and x_cross_loadings is None:
                x_cross_loadings = model.x_cross_loadings
            if hasattr(model, 'y_cross_loadings') and y_cross_loadings is None:
                y_cross_loadings = model.y_cross_loadings
                
            # Try to extract p-values
            if hasattr(model, 'pvals') and p_values is None:
                p_values = model.pvals
        except:
            # If extraction fails, we'll use the explicitly provided values
            pass
    
    # Determine number of canonical pairs to analyze
    if n_pairs is None:
        if canonical_correlations is not None:
            n_pairs = len(canonical_correlations)
        else:
            n_pairs = min(X.shape[1], Y.shape[1])
    
    # If not provided, calculate canonical correlation using a simple method
    if canonical_correlations is None:
        # Calculate covariance matrices
        C_xx = np.cov(X, rowvar=False)
        C_yy = np.cov(Y, rowvar=False)
        C_xy = np.cov(X, Y, rowvar=False)[:X.shape[1], X.shape[1]:]
        
        # Calculate canonical correlations using SVD
        try:
            # This is a simplified approach
            C_xx_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C_xx))
            C_yy_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C_yy))
            
            K = C_xx_inv_sqrt.dot(C_xy).dot(C_yy_inv_sqrt)
            U, canonical_correlations, Vt = np.linalg.svd(K, full_matrices=False)
            
            # Ensure we don't have more pairs than min(X.shape[1], Y.shape[1])
            n_pairs = min(n_pairs, min(X.shape[1], Y.shape[1]))
            canonical_correlations = canonical_correlations[:n_pairs]
            
            # Calculate loadings and scores if not provided
            if x_loadings is None:
                x_loadings = C_xx_inv_sqrt.dot(U)[:, :n_pairs]
            
            if y_loadings is None:
                y_loadings = C_yy_inv_sqrt.dot(Vt.T)[:, :n_pairs]
            
            if x_scores is None:
                x_scores = X.dot(x_loadings)
            
            if y_scores is None:
                y_scores = Y.dot(y_loadings)
                
        except:
            # If calculation fails, we'll skip plots that require canonical correlations
            canonical_correlations = None
            
    # Plot 1: Canonical Correlation Scree Plot
    if canonical_correlations is not None:
        plt.figure(figsize=(10, 6))
        
        # Squared canonical correlations represent explained variance
        plt.bar(range(1, len(canonical_correlations) + 1), 
              canonical_correlations ** 2,
              color='steelblue')
        
        plt.plot(range(1, len(canonical_correlations) + 1), 
               canonical_correlations ** 2,
               'o-', color='darkred')
        
        plt.xlabel('Canonical Dimension')
        plt.ylabel('Squared Canonical Correlation')
        plt.title('Canonical Correlation Scree Plot')
        plt.xticks(range(1, len(canonical_correlations) + 1))
        plt.grid(True, alpha=0.3)
        
        # Add significance markers if p-values are available
        if p_values is not None:
            for i, p in enumerate(p_values):
                if i < len(canonical_correlations):
                    marker = '*' if p < 0.05 else ''
                    if p < 0.001:
                        marker = '***'
                    elif p < 0.01:
                        marker = '**'
                    
                    if marker:
                        plt.text(i + 1, canonical_correlations[i] ** 2,
                               marker, ha='center', va='bottom')
        
        plt.tight_layout()
        
        plots.append({
            "title": "Canonical Correlation Scree Plot",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the squared canonical correlations for each canonical dimension. Higher values indicate stronger relationships between the corresponding canonical variates. The scree plot helps identify how many canonical dimensions are meaningful. Asterisks indicate statistical significance (* p<0.05, ** p<0.01, *** p<0.001)."
        })
    
    # Plot 2: Canonical Loadings Heatmap
    if x_loadings is not None and y_loadings is not None:
        # Plot X loadings
        plt.figure(figsize=(12, max(6, 0.5 * len(X_names))))
        
        # Ensure we're only showing the first n_pairs loadings
        x_load_plot = x_loadings[:, :n_pairs]
        
        # Create a DataFrame for better heatmap
        x_load_df = pd.DataFrame(x_load_plot, 
                               index=X_names, 
                               columns=[f'CV{i+1}' for i in range(n_pairs)])
        
        # Create heatmap
        ax = sns.heatmap(x_load_df, annot=True, cmap='coolwarm', center=0,
                       linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('X Variable Loadings on Canonical Variates')
        plt.tight_layout()
        
        plots.append({
            "title": "X Canonical Loadings",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the correlation between each X variable and its canonical variates. Values range from -1 to 1, with colors indicating the strength and direction of the relationship. Higher absolute values indicate variables that contribute more strongly to the canonical variate."
        })
        
        # Plot Y loadings
        plt.figure(figsize=(12, max(6, 0.5 * len(Y_names))))
        
        # Ensure we're only showing the first n_pairs loadings
        y_load_plot = y_loadings[:, :n_pairs]
        
        # Create a DataFrame for better heatmap
        y_load_df = pd.DataFrame(y_load_plot, 
                               index=Y_names, 
                               columns=[f'CV{i+1}' for i in range(n_pairs)])
        
        # Create heatmap
        ax = sns.heatmap(y_load_df, annot=True, cmap='coolwarm', center=0,
                       linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Y Variable Loadings on Canonical Variates')
        plt.tight_layout()
        
        plots.append({
            "title": "Y Canonical Loadings",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the correlation between each Y variable and its canonical variates. Values range from -1 to 1, with colors indicating the strength and direction of the relationship. Higher absolute values indicate variables that contribute more strongly to the canonical variate."
        })
    
    # Plot 3: Canonical Variate Scatter Plots
    if x_scores is not None and y_scores is not None and canonical_correlations is not None:
        # Create scatter plots for the first few canonical dimensions
        for i in range(min(3, n_pairs)):
            plt.figure(figsize=(8, 8))
            
            # Calculate correlation for annotation
            corr_val = canonical_correlations[i] if i < len(canonical_correlations) else 0
            
            # Scatter plot of canonical variates
            plt.scatter(x_scores[:, i], y_scores[:, i], alpha=0.7)
            
            # Add regression line
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()
            
            # Make axis limits the same
            plot_min = min(xmin, ymin)
            plot_max = max(xmax, ymax)
            
            plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--')
            
            plt.xlim(plot_min, plot_max)
            plt.ylim(plot_min, plot_max)
            
            plt.xlabel(f'X Canonical Variate {i+1}')
            plt.ylabel(f'Y Canonical Variate {i+1}')
            plt.title(f'Canonical Correlation {i+1}: {corr_val:.3f}')
            plt.grid(True, alpha=0.3)
            
            # Add correlation value in the corner
            plt.text(0.05, 0.95, f'r = {corr_val:.3f}', 
                   transform=plt.gca().transAxes,
                   fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            
            plt.tight_layout()
            
            plots.append({
                "title": f"Canonical Variate Scatter Plot {i+1}",
                "img_data": get_base64_plot(),
                "interpretation": f"Displays the relationship between the {i+1}st pair of canonical variates. Each point represents an observation, with coordinates determined by the canonical scores. The correlation between these scores is the canonical correlation (r = {corr_val:.3f}). Points close to the diagonal line indicate a strong linear relationship between the X and Y canonical variates."
            })
    
    # Plot 4: Structure Correlations (Cross-Loadings) Visualization
    if x_cross_loadings is not None and y_cross_loadings is not None:
        # Ensure we're only looking at the first few dimensions
        n_display = min(3, n_pairs)
        
        for i in range(n_display):
            plt.figure(figsize=(12, max(6, 0.3 * (len(X_names) + len(Y_names)))))
            
            # Combine X and Y cross-loadings for this dimension
            combined_loadings = np.zeros(len(X_names) + len(Y_names))
            combined_loadings[:len(X_names)] = x_cross_loadings[:, i]
            combined_loadings[len(X_names):] = y_cross_loadings[:, i]
            
            combined_names = X_names + Y_names
            
            # Sort by absolute value for better visualization
            sort_idx = np.argsort(np.abs(combined_loadings))[::-1]
            sorted_loadings = combined_loadings[sort_idx]
            sorted_names = [combined_names[j] for j in sort_idx]
            
            # Create horizontal bar chart
            bars = plt.barh(range(len(sorted_names)), sorted_loadings, height=0.7)
            
            # Color the bars based on the variable set (X or Y)
            for j, bar in enumerate(bars):
                if sort_idx[j] < len(X_names):
                    bar.set_color('steelblue')
                else:
                    bar.set_color('indianred')
            
            # Add labels
            plt.yticks(range(len(sorted_names)), sorted_names)
            plt.xlabel('Cross-Loading (Structure Correlation)')
            plt.title(f'Structure Correlations for Canonical Dimension {i+1}')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.grid(True, alpha=0.3)
            
            # Add a legend
            plt.legend([plt.Rectangle((0,0),1,1,color='steelblue'), 
                       plt.Rectangle((0,0),1,1,color='indianred')],
                     ['X Variables', 'Y Variables'], loc='best')
            
            plt.tight_layout()
            
            plots.append({
                "title": f"Structure Correlations for Dimension {i+1}",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows the correlation of each original variable with the {i+1}st pair of canonical variates. Also known as structure correlations or cross-loadings, these values indicate how strongly each original variable is related to the canonical dimensions. Variables with higher absolute values are more important for interpreting the canonical relationship."
            })
    elif x_loadings is not None and y_loadings is not None and canonical_correlations is not None:
        # Calculate cross-loadings if not provided directly
        try:
            # Simplified approach to calculate cross-loadings
            x_cross_loadings = x_loadings.dot(np.diag(canonical_correlations[:n_pairs]))
            y_cross_loadings = y_loadings.dot(np.diag(canonical_correlations[:n_pairs]))
            
            # Ensure we're only looking at the first few dimensions
            n_display = min(3, n_pairs)
            
            for i in range(n_display):
                plt.figure(figsize=(12, max(6, 0.3 * (len(X_names) + len(Y_names)))))
                
                # Combine X and Y cross-loadings for this dimension
                combined_loadings = np.zeros(len(X_names) + len(Y_names))
                combined_loadings[:len(X_names)] = x_cross_loadings[:, i]
                combined_loadings[len(X_names):] = y_cross_loadings[:, i]
                
                combined_names = X_names + Y_names
                
                # Sort by absolute value for better visualization
                sort_idx = np.argsort(np.abs(combined_loadings))[::-1]
                sorted_loadings = combined_loadings[sort_idx]
                sorted_names = [combined_names[j] for j in sort_idx]
                
                # Create horizontal bar chart
                bars = plt.barh(range(len(sorted_names)), sorted_loadings, height=0.7)
                
                # Color the bars based on the variable set (X or Y)
                for j, bar in enumerate(bars):
                    if sort_idx[j] < len(X_names):
                        bar.set_color('steelblue')
                    else:
                        bar.set_color('indianred')
                
                # Add labels
                plt.yticks(range(len(sorted_names)), sorted_names)
                plt.xlabel('Cross-Loading (Structure Correlation)')
                plt.title(f'Structure Correlations for Canonical Dimension {i+1}')
                plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                plt.grid(True, alpha=0.3)
                
                # Add a legend
                plt.legend([plt.Rectangle((0,0),1,1,color='steelblue'), 
                           plt.Rectangle((0,0),1,1,color='indianred')],
                         ['X Variables', 'Y Variables'], loc='best')
                
                plt.tight_layout()
                
                plots.append({
                    "title": f"Structure Correlations for Dimension {i+1}",
                    "img_data": get_base64_plot(),
                    "interpretation": f"Shows the correlation of each original variable with the {i+1}st pair of canonical variates. Also known as structure correlations or cross-loadings, these values indicate how strongly each original variable is related to the canonical dimensions. Variables with higher absolute values are more important for interpreting the canonical relationship."
                })
        except:
            # Skip if we can't calculate cross-loadings
            pass
    
    # Plot 5: Redundancy Analysis
    if x_loadings is not None and y_loadings is not None and canonical_correlations is not None:
        try:
            plt.figure(figsize=(10, 6))
            
            # Calculate redundancy indices
            # Redundancy is the amount of variance in one set explained by the canonical variates of the other set
            x_variance_extracted = np.mean(x_loadings ** 2, axis=0)
            y_variance_extracted = np.mean(y_loadings ** 2, axis=0)
            
            # Redundancy indices
            x_redundancy = x_variance_extracted * (canonical_correlations ** 2)
            y_redundancy = y_variance_extracted * (canonical_correlations ** 2)
            
            # Calculate cumulative redundancy
            x_cum_redundancy = np.cumsum(x_redundancy)
            y_cum_redundancy = np.cumsum(y_redundancy)
            
            # Plot the redundancy indices
            width = 0.35
            indices = np.arange(n_pairs)
            
            plt.bar(indices - width/2, x_redundancy, width, label='X Redundancy', color='steelblue')
            plt.bar(indices + width/2, y_redundancy, width, label='Y Redundancy', color='indianred')
            
            # Add a line for cumulative redundancy
            plt.plot(indices, x_cum_redundancy, 'o-', color='darkblue', label='X Cumulative')
            plt.plot(indices, y_cum_redundancy, 's-', color='darkred', label='Y Cumulative')
            
            plt.xlabel('Canonical Dimension')
            plt.ylabel('Proportion of Variance')
            plt.title('Redundancy Analysis')
            plt.xticks(indices, [f'{i+1}' for i in indices])
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plots.append({
                "title": "Redundancy Analysis",
                "img_data": get_base64_plot(),
                "interpretation": "Shows the proportion of variance in one set of variables that can be explained by the canonical variates of the other set. X redundancy indicates how much variance in X variables is explained by Y canonical variates, and vice versa. Higher values indicate stronger predictive relationships between the two sets of variables. The cumulative lines show the total redundancy captured by including successive dimensions."
            })
        except:
            # Skip if redundancy calculation fails
            pass
    
    # Plot 6: Biplot of First Two Canonical Dimensions
    if x_loadings is not None and y_loadings is not None and x_scores is not None and y_scores is not None:
        try:
            plt.figure(figsize=(12, 10))
            
            # Standardize scores for better visualization
            norm_x_scores = x_scores[:, :2] / np.std(x_scores[:, :2], axis=0)
            norm_y_scores = y_scores[:, :2] / np.std(y_scores[:, :2], axis=0)
            
            # Plot scores
            plt.scatter(norm_x_scores[:, 0], norm_x_scores[:, 1], 
                      alpha=0.6, label='X Scores', color='steelblue')
            
            # Scale loadings for better visualization
            scale = 5
            x_load_scaled = x_loadings[:, :2] * scale
            y_load_scaled = y_loadings[:, :2] * scale
            
            # Plot X loadings
            for i, (name, x, y) in enumerate(zip(X_names, x_load_scaled[:, 0], x_load_scaled[:, 1])):
                plt.arrow(0, 0, x, y, color='darkblue', alpha=0.8, head_width=0.1)
                plt.text(x * 1.1, y * 1.1, name, color='darkblue', ha='center', va='center')
            
            # Plot Y loadings
            for i, (name, x, y) in enumerate(zip(Y_names, y_load_scaled[:, 0], y_load_scaled[:, 1])):
                plt.arrow(0, 0, x, y, color='darkred', alpha=0.8, head_width=0.1)
                plt.text(x * 1.1, y * 1.1, name, color='darkred', ha='center', va='center')
            
            # Add circles for reference
            circle1 = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
            circle2 = plt.Circle((0, 0), 2, fill=False, color='gray', linestyle='--')
            circle3 = plt.Circle((0, 0), 3, fill=False, color='gray', linestyle='--')
            plt.gca().add_patch(circle1)
            plt.gca().add_patch(circle2)
            plt.gca().add_patch(circle3)
            
            # Adjust axes to be equal and centered
            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
            
            # Set limits to include both scores and loadings
            max_range = max(
                np.max(np.abs(norm_x_scores[:, :2])),
                np.max(np.abs(x_load_scaled)),
                np.max(np.abs(y_load_scaled))
            ) * 1.2
            
            plt.xlim(-max_range, max_range)
            plt.ylim(-max_range, max_range)
            
            plt.xlabel('First Canonical Dimension')
            plt.ylabel('Second Canonical Dimension')
            plt.title('Canonical Correlation Biplot')
            plt.grid(True, alpha=0.3)
            
            # Add a legend for variable types
            plt.legend([
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=10),
                plt.Line2D([0], [0], color='darkblue', lw=2),
                plt.Line2D([0], [0], color='darkred', lw=2)
            ], ['Observations', 'X Variables', 'Y Variables'], loc='best')
            
            plt.tight_layout()
            
            plots.append({
                "title": "Canonical Correlation Biplot",
                "img_data": get_base64_plot(),
                "interpretation": "Biplot visualizing the relationship between variables and canonical dimensions. Blue arrows represent X variables, red arrows represent Y variables, and blue points represent observations. Variables with arrows pointing in similar directions are positively correlated, while those pointing in opposite directions are negatively correlated. The length of each arrow indicates how strongly the variable contributes to the canonical dimensions."
            })
        except:
            # Skip if biplot creation fails
            pass
    
    # Plot 7: Canonical Weights Comparison for first two dimensions
    if x_loadings is not None and y_loadings is not None and n_pairs >= 2:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, max(6, 0.3 * max(len(X_names), len(Y_names)))))
            
            # Sort X loadings by first dimension
            x_sort_idx = np.argsort(np.abs(x_loadings[:, 0]))[::-1]
            sorted_x_names = [X_names[i] for i in x_sort_idx]
            sorted_x_loadings_dim1 = x_loadings[x_sort_idx, 0]
            sorted_x_loadings_dim2 = x_loadings[x_sort_idx, 1]
            
            # Create horizontal bar chart for X loadings
            y_pos = np.arange(len(sorted_x_names))
            axes[0].barh(y_pos - 0.2, sorted_x_loadings_dim1, height=0.4, label='Dimension 1', color='steelblue')
            axes[0].barh(y_pos + 0.2, sorted_x_loadings_dim2, height=0.4, label='Dimension 2', color='lightsteelblue')
            
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels(sorted_x_names)
            axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            axes[0].set_xlabel('Canonical Weights')
            axes[0].set_title('X Variables Canonical Weights')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Sort Y loadings by first dimension
            y_sort_idx = np.argsort(np.abs(y_loadings[:, 0]))[::-1]
            sorted_y_names = [Y_names[i] for i in y_sort_idx]
            sorted_y_loadings_dim1 = y_loadings[y_sort_idx, 0]
            sorted_y_loadings_dim2 = y_loadings[y_sort_idx, 1]
            
            # Create horizontal bar chart for Y loadings
            y_pos = np.arange(len(sorted_y_names))
            axes[1].barh(y_pos - 0.2, sorted_y_loadings_dim1, height=0.4, label='Dimension 1', color='indianred')
            axes[1].barh(y_pos + 0.2, sorted_y_loadings_dim2, height=0.4, label='Dimension 2', color='lightcoral')
            
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(sorted_y_names)
            axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            axes[1].set_xlabel('Canonical Weights')
            axes[1].set_title('Y Variables Canonical Weights')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plots.append({
                "title": "Canonical Weights Comparison",
                "img_data": get_base64_plot(),
                "interpretation": "Compares the canonical weights (loadings) of variables on the first two canonical dimensions. The left panel shows weights for X variables, and the right panel shows weights for Y variables. Variables with larger absolute weights contribute more to the canonical correlation. Variables with similar patterns of weights across dimensions tend to capture similar information."
            })
        except:
            # Skip if weights comparison fails
            pass
            
    return plots 