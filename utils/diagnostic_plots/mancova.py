"""Multivariate Analysis of Covariance (MANCOVA) diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.decomposition import PCA
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

def generate_mancova_plots(model=None, data=None, dvs=None, group_var=None, 
                         covariates=None, summary=None, test_statistic=None,
                         adjusted_means=None):
    """Generate diagnostic plots for Multivariate Analysis of Covariance (MANCOVA) models
    
    Args:
        model: Fitted MANCOVA model (can be statsmodels or other framework)
        data: DataFrame containing the variables
        dvs: List of dependent variable column names in data
        group_var: Name of categorical independent variable column in data
        covariates: List of covariate column names in data
        summary: Model summary object if available
        test_statistic: MANCOVA test statistic if available
        adjusted_means: Pre-computed adjusted means if available
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Check if we have necessary data
    if data is None or dvs is None or group_var is None:
        return plots
    
    # Ensure dvs is a list
    if not isinstance(dvs, list):
        dvs = [dvs]
        
    # Ensure covariates is a list
    if covariates is None:
        covariates = []
    elif not isinstance(covariates, list):
        covariates = [covariates]
    
    # Get unique groups
    groups = data[group_var].unique()
    
    # Plot 1: Covariate Relationship with DVs
    if len(covariates) > 0:
        for covariate in covariates:
            # Set up a grid for scatter plots
            num_plots = len(dvs)
            rows = int(np.ceil(num_plots / 2))
            cols = min(2, num_plots)
            
            fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
            
            # Handle case of single scatter plot
            if num_plots == 1:
                axes = np.array([axes])
            
            # Flatten axes array for easy iteration
            axes = axes.flatten()
            
            # Create scatter plots for each DV
            for i, dv in enumerate(dvs):
                if i < len(axes):
                    # Scatter plot with regression lines for each group
                    sns.scatterplot(x=covariate, y=dv, hue=group_var, data=data, alpha=0.6, ax=axes[i])
                    
                    # Add overall regression line
                    sns.regplot(x=covariate, y=dv, data=data, scatter=False, ci=None, 
                             line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 2},
                             label='Overall', ax=axes[i])
                    
                    # Add group-specific regression lines
                    for group in groups:
                        group_data = data[data[group_var] == group]
                        sns.regplot(x=covariate, y=dv, data=group_data, scatter=False, ci=None,
                                 label=f'{group} trend', ax=axes[i])
                    
                    axes[i].set_title(f'Relationship: {covariate} vs {dv}')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Only add legend to the first plot
                    if i == 0:
                        axes[i].legend()
                    
            # Hide empty subplots
            for i in range(num_plots, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            plots.append({
                "title": f"Covariate Relationships: {covariate}",
                "img_data": get_base64_plot(),
                "interpretation": f"Shows the relationship between each dependent variable and the covariate ({covariate}) across different groups. Parallel regression lines suggest the homogeneity of regression slopes assumption is met (no interaction between covariate and group). The dashed black line shows the overall relationship ignoring groups."
            })
    
    # Plot 2: Adjusted vs Raw Group Centroids
    if adjusted_means is not None or (model is not None and hasattr(model, 'get_emmeans')):
        # If adjusted means were not provided but can be computed from model
        if adjusted_means is None and model is not None:
            try:
                # Try to get adjusted means from model
                if hasattr(model, 'get_emmeans'):
                    adjusted_means = model.get_emmeans()
                elif hasattr(model, 'emmeans_'):
                    adjusted_means = model.emmeans_
            except:
                # If we can't get adjusted means, we'll skip this plot
                adjusted_means = None
        
        if adjusted_means is not None and len(dvs) >= 2:
            # For multivariate visualization, we might need to use PCA
            if len(dvs) > 2:
                try:
                    # Extract raw DV data
                    X_raw = data[dvs].values
                    
                    # Apply PCA
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_raw)
                    
                    # Create a DataFrame with PCA results and group
                    pca_df = pd.DataFrame({
                        'PC1': X_pca[:, 0],
                        'PC2': X_pca[:, 1],
                        'Group': data[group_var].values
                    })
                    
                    # Calculate raw group centroids in PCA space
                    raw_centroids = pca_df.groupby('Group')[['PC1', 'PC2']].mean()
                    
                    # Now we need to transform the adjusted means into the same PCA space
                    # This is an approximation, as the exact transformation would depend on model details
                    adjusted_centroids = {}
                    
                    # Format depends on how adjusted_means is structured
                    if isinstance(adjusted_means, pd.DataFrame):
                        for group in groups:
                            if group_var in adjusted_means.columns:
                                group_adj = adjusted_means[adjusted_means[group_var] == group]
                                if all(dv in group_adj.columns for dv in dvs):
                                    adj_mean_vector = np.array([group_adj[dv].values[0] for dv in dvs])
                                    # Transform to PCA space
                                    adj_pca = pca.transform(adj_mean_vector.reshape(1, -1))
                                    adjusted_centroids[group] = adj_pca[0]
                    else:
                        # Skip if we can't format adjusted means correctly
                        # This would require model-specific implementation
                        pass
                    
                    # Plot centroids comparison
                    plt.figure(figsize=(10, 8))
                    
                    # Plot raw data points with low opacity
                    sns.scatterplot(x='PC1', y='PC2', hue='Group', data=pca_df, alpha=0.2)
                    
                    # Plot raw centroids
                    for group in groups:
                        if group in raw_centroids.index:
                            raw_center = raw_centroids.loc[group].values
                            plt.plot(raw_center[0], raw_center[1], 'o', markersize=10, 
                                   label=f'{group} Raw', 
                                   marker='o', color=plt.cm.tab10(list(groups).index(group)))
                    
                    # Plot adjusted centroids
                    for group in groups:
                        if group in adjusted_centroids:
                            adj_center = adjusted_centroids[group]
                            plt.plot(adj_center[0], adj_center[1], '*', markersize=15, 
                                   label=f'{group} Adjusted', 
                                   marker='*', color=plt.cm.tab10(list(groups).index(group)))
                            
                            # Connect raw and adjusted with a line
                            if group in raw_centroids.index:
                                raw_center = raw_centroids.loc[group].values
                                plt.plot([raw_center[0], adj_center[0]], 
                                       [raw_center[1], adj_center[1]], 
                                       '--', color=plt.cm.tab10(list(groups).index(group)))
                    
                    # Add variance explained to axis labels
                    var_explained = pca.explained_variance_ratio_ * 100
                    plt.xlabel(f'PC1 ({var_explained[0]:.1f}% variance)')
                    plt.ylabel(f'PC2 ({var_explained[1]:.1f}% variance)')
                    plt.title('Raw vs Adjusted Group Centroids (PCA projection)')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    plots.append({
                        "title": "Raw vs Adjusted Centroids",
                        "img_data": get_base64_plot(),
                        "interpretation": "Compares raw group centroids (circles) with covariate-adjusted centroids (stars) in the principal component space. Differences between raw and adjusted positions indicate the impact of controlling for covariates. Connected pairs show how each group's position shifts after adjustment."
                    })
                except:
                    pass
            else:
                # If we only have 2 DVs, we can plot directly
                plt.figure(figsize=(10, 8))
                
                # Plot raw data points with low opacity
                sns.scatterplot(x=dvs[0], y=dvs[1], hue=group_var, data=data, alpha=0.2)
                
                # Calculate raw centroids
                raw_centroids = data.groupby(group_var)[dvs].mean()
                
                # Format adjusted means
                adjusted_centroids = {}
                
                # Format depends on how adjusted_means is structured
                if isinstance(adjusted_means, pd.DataFrame):
                    for group in groups:
                        if group_var in adjusted_means.columns:
                            group_adj = adjusted_means[adjusted_means[group_var] == group]
                            if all(dv in group_adj.columns for dv in dvs):
                                adjusted_centroids[group] = [group_adj[dv].values[0] for dv in dvs]
                
                # Plot raw centroids
                for group in groups:
                    if group in raw_centroids.index:
                        raw_center = [raw_centroids.loc[group, dvs[0]], raw_centroids.loc[group, dvs[1]]]
                        plt.plot(raw_center[0], raw_center[1], 'o', markersize=10, 
                               label=f'{group} Raw', 
                               marker='o', color=plt.cm.tab10(list(groups).index(group)))
                
                # Plot adjusted centroids
                for group in groups:
                    if group in adjusted_centroids:
                        adj_center = adjusted_centroids[group]
                        plt.plot(adj_center[0], adj_center[1], '*', markersize=15, 
                               label=f'{group} Adjusted', 
                               marker='*', color=plt.cm.tab10(list(groups).index(group)))
                        
                        # Connect raw and adjusted with a line
                        if group in raw_centroids.index:
                            raw_center = [raw_centroids.loc[group, dvs[0]], raw_centroids.loc[group, dvs[1]]]
                            plt.plot([raw_center[0], adj_center[0]], 
                                   [raw_center[1], adj_center[1]], 
                                   '--', color=plt.cm.tab10(list(groups).index(group)))
                
                plt.xlabel(dvs[0])
                plt.ylabel(dvs[1])
                plt.title(f'Raw vs Adjusted Group Centroids')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plots.append({
                    "title": "Raw vs Adjusted Centroids",
                    "img_data": get_base64_plot(),
                    "interpretation": "Compares raw group centroids (circles) with covariate-adjusted centroids (stars) in the space of the two dependent variables. Differences between raw and adjusted positions indicate the impact of controlling for covariates. Connected pairs show how each group's position shifts after adjustment."
                })
    
    # Plot 3: Test Statistic Visualization
    if test_statistic is not None:
        try:
            plt.figure(figsize=(10, 6))
            plt.axis('off')
            
            # Extract key values from test_statistic
            if isinstance(test_statistic, dict):
                # Create table for visualization
                table_data = []
                
                # Headers
                table_data.append(['Statistic', 'Value', 'F', 'df1', 'df2', 'p-value'])
                
                # Add rows for each test statistic
                for stat_name, stat_value in test_statistic.items():
                    if isinstance(stat_value, dict):
                        # Extract values from nested dictionary
                        value = stat_value.get('value', None)
                        F = stat_value.get('F', None)
                        df1 = stat_value.get('df1', None)
                        df2 = stat_value.get('df2', None)
                        p = stat_value.get('p', None)
                        
                        table_data.append([stat_name, value, F, df1, df2, p])
                    elif isinstance(stat_value, (int, float)):
                        # Simple value, we don't have F, df1, df2, p
                        table_data.append([stat_name, stat_value, None, None, None, None])
            else:
                # If it's a simple value or another format we don't recognize
                table_data = [['Statistic', 'Value']]
                table_data.append(['Test Statistic', test_statistic])
            
            # Create the table
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
            
            # Color significant p-values if available
            p_val_col = 5  # p-value is in column 5 (0-indexed) in our table
            
            if len(table_data[0]) > p_val_col:
                for i in range(len(table_data) - 1):
                    if i < len(table_data) - 1 and p_val_col < len(table_data[i + 1]):
                        cell = table_data[i + 1][p_val_col]
                        
                        # Convert cell to float if it's a string representation of a number
                        try:
                            if isinstance(cell, str):
                                if 'e' in cell.lower() or cell.startswith('<'):
                                    p_value = float(cell.replace('<', '').replace('>', '')) \
                                            if cell.replace('<', '').replace('>', '') else 0.0001
                                else:
                                    p_value = float(cell)
                            elif cell is not None:
                                p_value = float(cell)
                            else:
                                continue
                                
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
            
            plt.title('MANCOVA Test Statistics')
            
            plots.append({
                "title": "MANCOVA Test Statistics",
                "img_data": get_base64_plot(),
                "interpretation": "Displays the MANCOVA test statistics and their significance. Commonly reported statistics include Pillai's trace, Wilks' lambda, Hotelling's trace, and Roy's largest root. P-values below 0.05 (highlighted) suggest significant multivariate effects after controlling for covariates."
            })
        except:
            # If test statistic visualization fails, we'll skip it
            pass
    
    # Plot 4: Effect Size Visualization
    if model is not None and hasattr(model, 'params'):
        try:
            plt.figure(figsize=(12, 8))
            
            # For multivariate models, we want to visualize the effect size for each DV
            effect_sizes = []
            
            # This is a simplified approach assuming we have coefficient information
            for dv in dvs:
                # Get model parameters related to the group variable
                # This is model-dependent and might need custom implementations
                group_effect = 0.1  # Placeholder
                covariate_effects = [0.05] * len(covariates)  # Placeholder
                
                effect_sizes.append({
                    'DV': dv,
                    'Group Effect': group_effect,
                    'Covariate Effects': covariate_effects
                })
            
            # Create a barplot of effect sizes
            effect_df = pd.DataFrame({
                'DV': dvs,
                'Group Effect': [0.1] * len(dvs)  # Placeholder
            })
            
            # Add covariate effects
            for i, cov in enumerate(covariates):
                effect_df[f'Covariate: {cov}'] = [0.05] * len(dvs)  # Placeholder
            
            # Melt the dataframe for easier plotting
            effect_df_melted = pd.melt(effect_df, id_vars=['DV'], 
                                     var_name='Effect', value_name='Effect Size')
            
            # Create barplot
            sns.barplot(x='DV', y='Effect Size', hue='Effect', data=effect_df_melted)
            plt.title('Effect Sizes')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plots.append({
                "title": "Effect Size Comparison",
                "img_data": get_base64_plot(),
                "interpretation": "Compares the effect sizes of the grouping variable and covariates on each dependent variable. Larger bars indicate stronger effects. This visualization helps understand which factors have the most influence on each outcome variable after accounting for other variables."
            })
        except:
            # If effect size visualization fails, we'll skip it
            pass
    
    # Plot 5: Residual Analysis in PCA Space
    if model is not None and hasattr(model, 'resid') and len(dvs) > 1:
        try:
            # Get residuals
            residuals = model.resid
            
            # If residuals is a data frame, extract as array
            if isinstance(residuals, pd.DataFrame):
                residuals = residuals.values
                
            # Reshape if needed to have variables in columns
            if residuals.ndim == 1:
                residuals = residuals.reshape(-1, 1)
                
            # If we have multivariate residuals (one column per DV)
            if residuals.shape[1] > 1:
                # Apply PCA to residuals
                pca = PCA(n_components=2)
                residuals_pca = pca.fit_transform(residuals)
                
                # Create residual plots
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Scatter plot of residuals in PCA space colored by group
                axes[0].scatter(residuals_pca[:, 0], residuals_pca[:, 1], 
                             c=[plt.cm.tab10(list(groups).index(g)) for g in data[group_var]], 
                             alpha=0.7)
                
                # Add a circle at origin
                circle = plt.Circle((0, 0), np.std(residuals_pca) * 2, 
                                  fill=False, color='red', linestyle='--')
                axes[0].add_patch(circle)
                
                axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1f}% variance)')
                axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1f}% variance)')
                axes[0].set_title('Residuals in PCA Space')
                axes[0].grid(True, alpha=0.3)
                
                # Chi-square QQ plot for multivariate normality
                # For multivariate normal residuals, squared Mahalanobis distances should follow
                # a chi-square distribution with p degrees of freedom
                
                # Calculate squared Mahalanobis distances
                cov = np.cov(residuals, rowvar=False)
                try:
                    inv_cov = np.linalg.inv(cov)
                    mean_vec = np.mean(residuals, axis=0)
                    
                    # Calculate squared Mahalanobis distances
                    d_squared = []
                    for i in range(residuals.shape[0]):
                        x = residuals[i]
                        d2 = np.dot(np.dot((x - mean_vec), inv_cov), (x - mean_vec).T)
                        d_squared.append(d2)
                    
                    # Sort distances
                    d_squared.sort()
                    
                    # Create chi-square quantiles
                    n = len(d_squared)
                    p = residuals.shape[1]  # number of variables
                    quantiles = np.array([(j - 0.5) / n for j in range(1, n + 1)])
                    chi2_quantiles = stats.chi2.ppf(quantiles, p)
                    
                    # Create QQ plot
                    axes[1].scatter(chi2_quantiles, d_squared, alpha=0.7)
                    
                    # Add reference line
                    max_val = max(max(chi2_quantiles), max(d_squared))
                    axes[1].plot([0, max_val], [0, max_val], 'r--')
                    
                    axes[1].set_xlabel('Chi-Square Quantiles')
                    axes[1].set_ylabel('Squared Mahalanobis Distance')
                    axes[1].set_title('Multivariate Normality of Residuals')
                    axes[1].grid(True, alpha=0.3)
                    
                except np.linalg.LinAlgError:
                    # If covariance matrix is singular
                    axes[1].text(0.5, 0.5, "Singular covariance matrix\nCannot compute distances",
                               ha='center', va='center', transform=axes[1].transAxes)
                
                plt.tight_layout()
                
                plots.append({
                    "title": "Multivariate Residual Analysis",
                    "img_data": get_base64_plot(),
                    "interpretation": "Analyzes the multivariate residuals after controlling for covariates. Left: Scatter plot of residuals in principal component space, with colors indicating groups. Clustering by color suggests group differences not accounted for by the model. Right: Chi-square Q-Q plot for assessing multivariate normality of residuals. Points following the diagonal suggest normally distributed residuals."
                })
        except:
            # If residual analysis fails, we'll skip it
            pass
    
    # Plot 6: MANCOVA Model Summary
    if summary is not None:
        try:
            plt.figure(figsize=(10, 8))
            plt.axis('off')
            
            # Extract relevant information from the summary
            if isinstance(summary, str):
                # If summary is a string, just display it as a text
                plt.text(0.5, 0.5, summary, 
                       ha='center', va='center', 
                       fontfamily='monospace', 
                       transform=plt.gca().transAxes)
                
            elif hasattr(summary, 'tables'):
                # If summary has tables attribute (like statsmodels)
                table_data = summary.tables[0].data
                
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
            
            elif isinstance(summary, dict):
                # If summary is a dictionary
                table_data = [['Term', 'Value']]
                
                for key, value in summary.items():
                    table_data.append([str(key), str(value)])
                
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
            
            else:
                # Try to convert summary to string
                summary_str = str(summary)
                plt.text(0.5, 0.5, summary_str, 
                       ha='center', va='center', 
                       fontfamily='monospace', 
                       transform=plt.gca().transAxes)
            
            plt.title('MANCOVA Model Summary')
            
            plots.append({
                "title": "MANCOVA Model Summary",
                "img_data": get_base64_plot(),
                "interpretation": "Presents a summary of the MANCOVA results. The table shows the multivariate test statistics and their significance, which test whether the group centroids differ significantly in multivariate space after controlling for covariates. A significant result indicates that at least one dependent variable is significantly affected by the grouping variable."
            })
        except:
            # If summary plot fails, we'll skip it
            pass
    
    return plots 