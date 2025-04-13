"""Factor analysis diagnostic plots."""
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

def generate_factor_analysis_plots(model, X=None, feature_names=None, n_factors=None):
    """Generate diagnostic plots for factor analysis
    
    Args:
        model: Fitted factor analysis model (sklearn, statsmodels, or similar)
        X: Original feature matrix (optional)
        feature_names: List of feature names (optional)
        n_factors: Number of factors in the model (optional)
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Extract loadings if available
    loadings = None
    var_explained = None
    components = None
    
    # Try to get loadings from different model types
    if hasattr(model, 'loadings_'):
        loadings = model.loadings_
    elif hasattr(model, 'components_'):
        components = model.components_
        loadings = components.T  # Transpose for proper orientation
    elif hasattr(model, 'loadings'):
        if callable(model.loadings):
            loadings = model.loadings()
        else:
            loadings = model.loadings
    
    # Try to get explained variance from different model types
    if hasattr(model, 'explained_variance_ratio_'):
        var_explained = model.explained_variance_ratio_
    elif hasattr(model, 'eigenvalues'):
        if callable(model.eigenvalues):
            eigenvalues = model.eigenvalues()
        else:
            eigenvalues = model.eigenvalues
        total = sum(eigenvalues)
        var_explained = [ev / total for ev in eigenvalues]
    
    # If n_factors wasn't provided, try to determine from the model
    if n_factors is None:
        if hasattr(model, 'n_components'):
            n_factors = model.n_components
        elif hasattr(model, 'n_factors'):
            n_factors = model.n_factors
        elif loadings is not None:
            n_factors = loadings.shape[1]
        elif var_explained is not None:
            n_factors = len(var_explained)
    
    # Plot 1: Scree plot (if explained variance ratio is available)
    if var_explained is not None:
        plt.figure(figsize=(10, 6))
        
        # Number of components
        num_vars = len(var_explained)
        x = range(1, num_vars + 1)
        
        # Create scree plot
        plt.plot(x, var_explained, 'o-', linewidth=2, color='blue')
        plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5)  # Common threshold line
        
        # Calculate cumulative explained variance
        cumulative = np.cumsum(var_explained)
        plt.plot(x, cumulative, 'o-', linewidth=2, color='green')
        
        # Highlight selected number of factors
        if n_factors:
            plt.axvline(x=n_factors, color='red', alpha=0.3)
            plt.annotate(f'Selected factors: {n_factors}', 
                         xy=(n_factors, 0.05), 
                         xytext=(n_factors+0.2, 0.15),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        plt.title('Scree Plot with Cumulative Explained Variance')
        plt.xlabel('Factor Number')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True, alpha=0.3)
        plt.xticks(x)
        plt.legend(['Individual', 'Cumulative'])
        
        plots.append({
            "title": "Scree Plot",
            "img_data": get_base64_plot(),
            "interpretation": "The scree plot shows the explained variance for each factor. Look for the 'elbow' point where the curve flattens, which suggests an optimal number of factors. Factors with eigenvalues > 1 or explaining > 10% variance (red line) are often considered significant."
        })
    
    # Plot 2: Factor Loadings Heatmap
    if loadings is not None:
        plt.figure(figsize=(12, 10))
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(loadings.shape[0])]
        
        # Ensure loadings is a numpy array
        loadings_array = np.array(loadings)
        
        # Create a DataFrame for better heatmap labels
        loadings_df = pd.DataFrame(
            loadings_array,
            index=feature_names[:loadings_array.shape[0]],
            columns=[f"Factor {i+1}" for i in range(loadings_array.shape[1])]
        )
        
        # Custom colormap for better visualization of positive/negative loadings
        colors = ['blue', 'white', 'red']
        cmap = LinearSegmentedColormap.from_list('factor_loadings', colors, N=100)
        
        # Create heatmap
        sns.heatmap(loadings_df, annot=True, cmap=cmap, center=0, 
                   vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
        
        plt.title('Factor Loadings Heatmap')
        plt.tight_layout()
        
        plots.append({
            "title": "Factor Loadings Heatmap",
            "img_data": get_base64_plot(),
            "interpretation": "The heatmap shows how strongly each variable loads onto each factor. Red indicates positive loadings, blue indicates negative loadings. Strong loadings (>0.4 or <-0.4) suggest that a variable contributes significantly to a factor."
        })
    
    # Plot 3: Factor Correlation Heatmap (if X is provided)
    if X is not None and hasattr(model, 'transform'):
        plt.figure(figsize=(10, 8))
        
        # Transform the data to get factor scores
        try:
            factor_scores = model.transform(X)
            
            # Calculate correlations between factors
            factors_df = pd.DataFrame(
                factor_scores,
                columns=[f"Factor {i+1}" for i in range(factor_scores.shape[1])]
            )
            
            corr_matrix = factors_df.corr()
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                      annot=True, fmt=".2f", square=True, linewidths=.5)
            
            plt.title('Factor Correlation Matrix')
            plt.tight_layout()
            
            plots.append({
                "title": "Factor Correlation Matrix",
                "img_data": get_base64_plot(),
                "interpretation": "Shows correlations between extracted factors. In an ideal factor analysis, factors should be uncorrelated (orthogonal), though this isn't always the case with real-world data. High correlations suggest that factors might be measuring related constructs."
            })
        except:
            # Skip if transform fails
            pass
    
    # Plot 4: Communalities (variance explained by factors for each variable)
    if loadings is not None:
        plt.figure(figsize=(12, 6))
        
        # Calculate communalities (sum of squared loadings for each variable)
        communalities = np.sum(loadings**2, axis=1)
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(len(communalities))]
        
        # Create bar chart
        y_pos = np.arange(len(communalities))
        plt.barh(y_pos, communalities, align='center')
        plt.yticks(y_pos, feature_names)
        
        # Add a reference line at 0.5 (common threshold)
        plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
        
        plt.title('Communalities: Variance Explained by Factors')
        plt.xlabel('Communality')
        plt.ylabel('Variables')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plots.append({
            "title": "Communalities",
            "img_data": get_base64_plot(),
            "interpretation": "Communalities represent how much of each variable's variance is explained by the extracted factors. Variables with low communalities (<0.5, red line) are not well-represented by the factor solution and might be candidates for removal."
        })
    
    # Plot 5: Factor Scatter Plot (first two factors, if X is provided)
    if X is not None and hasattr(model, 'transform') and n_factors >= 2:
        plt.figure(figsize=(10, 8))
        
        try:
            # Transform the data to get factor scores
            factor_scores = model.transform(X)
            
            # Scatter plot of first two factors
            plt.scatter(factor_scores[:, 0], factor_scores[:, 1], alpha=0.6)
            plt.xlabel('Factor 1')
            plt.ylabel('Factor 2')
            plt.title('Factor Scores: Factor 1 vs Factor 2')
            plt.grid(True, alpha=0.3)
            
            # Add zero lines
            plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
            
            plots.append({
                "title": "Factor Scores Plot",
                "img_data": get_base64_plot(),
                "interpretation": "This plot shows how observations score on the first two factors, revealing potential clusters or patterns. The quadrants can often be interpreted based on the loadings of variables onto these factors."
            })
        except:
            # Skip if transform fails
            pass
    
    # Plot 6: Proportion of Variance Explained by Each Factor
    if var_explained is not None:
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        x = range(1, len(var_explained) + 1)
        plt.bar(x, var_explained, alpha=0.7)
        
        # Add cumulative line
        cumulative = np.cumsum(var_explained)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(x, cumulative, 'r-', marker='o')
        ax2.set_ylabel('Cumulative Proportion', color='r')
        ax2.tick_params(axis='y', colors='r')
        
        # Highlight selected number of factors
        if n_factors:
            plt.axvline(x=n_factors + 0.5, color='green', linestyle='--', alpha=0.7)
        
        plt.title('Proportion of Variance Explained by Each Factor')
        plt.xlabel('Factor Number')
        plt.ylabel('Proportion of Variance')
        plt.xticks(x)
        plt.grid(True, alpha=0.3)
        
        # Annotate total variance explained
        if n_factors:
            total_var = sum(var_explained[:n_factors])
            plt.annotate(f'Total variance explained: {total_var:.2%}', 
                        xy=(0.5, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plots.append({
            "title": "Variance Explained by Factors",
            "img_data": get_base64_plot(),
            "interpretation": f"Shows the proportion of variance explained by each factor (bars) and cumulatively (red line). A good factor solution typically explains at least 60-70% of the total variance."
        })
    
    return plots 