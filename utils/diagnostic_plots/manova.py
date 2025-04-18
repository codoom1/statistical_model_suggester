"""Multivariate Analysis of Variance (MANOVA) diagnostic plots."""
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

def generate_manova_plots(model=None, data=None, dvs=None, group_var=None, 
                          summary=None, test_statistic=None):
    """Generate diagnostic plots for Multivariate Analysis of Variance (MANOVA) models
    
    Args:
        model: Fitted MANOVA model (can be statsmodels or other framework)
        data: DataFrame containing the variables
        dvs: List of dependent variable column names in data
        group_var: Name of categorical independent variable column in data
        summary: Model summary object if available
        test_statistic: MANOVA test statistic if available (Pillai's trace, Wilk's lambda, etc.)
        
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
    
    # Get unique groups
    groups = data[group_var].unique()
    
    # Plot 1: Group Centroids in DV Space
    if len(dvs) >= 2:
        # If we have more than 2 DVs, we need to use PCA for visualization
        if len(dvs) > 2:
            # Extract DV data
            X = data[dvs].values
            
            # Apply PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            # Create a DataFrame with PCA results and group
            pca_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Group': data[group_var].values
            })
            
            # Plot in PCA space
            plt.figure(figsize=(10, 8))
            
            # Plot points
            sns.scatterplot(x='PC1', y='PC2', hue='Group', data=pca_df, alpha=0.7)
            
            # Plot group centroids
            centroids = pca_df.groupby('Group')[['PC1', 'PC2']].mean()
            
            # Add ellipses for each group (95% confidence)
            for group in groups:
                group_data = pca_df[pca_df['Group'] == group]
                cov = np.cov(group_data['PC1'], group_data['PC2'])
                
                # Calculate eigenvalues and eigenvectors
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                
                # Sort by eigenvalue in decreasing order
                idx = np.argsort(eigenvals)[::-1]
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]
                
                # Convert to angle in degrees
                theta = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                
                # Width and height of ellipse (95% confidence)
                width, height = 2 * np.sqrt(eigenvals) * 1.96
                
                # Center of ellipse
                center = centroids.loc[group].values
                
                # Create ellipse patch
                ellipse = patches.Ellipse(center, width, height, 
                                          angle=theta, fill=False, 
                                          edgecolor=plt.cm.tab10(list(groups).index(group)), 
                                          linewidth=2)
                plt.gca().add_patch(ellipse)
                
                # Add group label at centroid
                plt.text(center[0], center[1], str(group), 
                        ha='center', va='center', 
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            
            # Add variance explained
            var_explained = pca.explained_variance_ratio_ * 100
            plt.xlabel(f'PC1 ({var_explained[0]:.1f}% variance)')
            plt.ylabel(f'PC2 ({var_explained[1]:.1f}% variance)')
            plt.title('Group Centroids in Principal Component Space')
            
        else:
            # If we only have 2 DVs, we can plot directly
            plt.figure(figsize=(10, 8))
            
            # Plot points
            sns.scatterplot(x=dvs[0], y=dvs[1], hue=group_var, data=data, alpha=0.7)
            
            # Calculate group centroids
            centroids = data.groupby(group_var)[dvs].mean()
            
            # Add ellipses for each group (95% confidence)
            for group in groups:
                group_data = data[data[group_var] == group]
                
                # Get covariance matrix for this group's DVs
                cov = np.cov(group_data[dvs[0]], group_data[dvs[1]])
                
                # Calculate eigenvalues and eigenvectors
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                
                # Sort by eigenvalue in decreasing order
                idx = np.argsort(eigenvals)[::-1]
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]
                
                # Convert to angle in degrees
                theta = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                
                # Width and height of ellipse (95% confidence)
                width, height = 2 * np.sqrt(eigenvals) * 1.96
                
                # Center of ellipse
                center = centroids.loc[group].values
                
                # Create ellipse patch
                ellipse = patches.Ellipse(center, width, height, 
                                          angle=theta, fill=False, 
                                          edgecolor=plt.cm.tab10(list(groups).index(group)), 
                                          linewidth=2)
                plt.gca().add_patch(ellipse)
                
                # Add group label at centroid
                plt.text(center[0], center[1], str(group), 
                        ha='center', va='center', 
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            
            plt.xlabel(dvs[0])
            plt.ylabel(dvs[1])
            plt.title(f'Group Centroids in {dvs[0]} vs {dvs[1]} Space')
        
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Group Centroids Visualization",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the distribution of groups in the dependent variable space. Each point represents an observation, and ellipses show 95% confidence regions for each group. Greater separation between group centroids indicates stronger multivariate differences between groups. Overlapping ellipses suggest groups may not be statistically distinct."
        })
    
    # Plot 2: Univariate Boxplots for Each DV
    if len(dvs) > 0:
        # Set up a grid for boxplots
        num_plots = len(dvs)
        rows = int(np.ceil(num_plots / 2))
        cols = min(2, num_plots)
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        
        # Handle case of single boxplot
        if num_plots == 1:
            axes = np.array([axes])
        
        # Flatten axes array for easy iteration
        axes = axes.flatten()
        
        # Create boxplots for each DV
        for i, dv in enumerate(dvs):
            if i < len(axes):
                sns.boxplot(x=group_var, y=dv, data=data, ax=axes[i])
                axes[i].set_title(f'Distribution of {dv} by Group')
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        plots.append({
            "title": "Univariate Distributions",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the distribution of each dependent variable across groups. These boxplots help identify univariate differences between groups that contribute to multivariate differences. Outliers, differences in medians, and variations in spread provide insights into how groups differ on individual variables."
        })
    
    # Plot 3: Correlation Heatmap by Group
    if len(dvs) > 1:
        # Set up grid for correlation heatmaps
        num_groups = len(groups)
        rows = int(np.ceil((num_groups + 1) / 2))  # +1 for overall correlation
        cols = min(2, num_groups + 1)
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
        
        # Handle case of single plot
        if rows * cols == 1:
            axes = np.array([axes])
        
        # Flatten axes array for easy iteration
        axes = axes.flatten()
        
        # Overall correlation
        overall_corr = data[dvs].corr()
        sns.heatmap(overall_corr, annot=True, fmt=".2f", cmap="coolwarm",
                   vmin=-1, vmax=1, center=0, ax=axes[0])
        axes[0].set_title("Overall Correlation Between DVs")
        
        # Correlation by group
        for i, group in enumerate(groups):
            if i + 1 < len(axes):
                group_data = data[data[group_var] == group][dvs]
                group_corr = group_data.corr()
                
                sns.heatmap(group_corr, annot=True, fmt=".2f", cmap="coolwarm",
                           vmin=-1, vmax=1, center=0, ax=axes[i + 1])
                axes[i + 1].set_title(f"Correlation for Group: {group}")
        
        # Hide empty subplots
        for i in range(num_groups + 1, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        plots.append({
            "title": "Correlation Structures",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the correlation structure among dependent variables overall and within each group. Differences in correlation patterns between groups suggest that the relationships between variables change across groups, which is an important aspect of multivariate analysis. Similar correlation structures suggest that while means may differ, the underlying relationships are consistent."
        })
    
    # Plot 4: Multivariate Normality Assessments
    if len(dvs) > 1:
        # Create squared Mahalanobis distances for each group
        fig, axes = plt.subplots(1, len(groups), figsize=(15, 5))
        
        # Handle case of single group
        if len(groups) == 1:
            axes = np.array([axes])
        
        for i, group in enumerate(groups):
            group_data = data[data[group_var] == group][dvs]
            
            # Calculate Mahalanobis distances
            group_mean = group_data.mean().values
            group_cov = group_data.cov().values
            
            # Calculate inverse covariance matrix
            try:
                inv_cov = np.linalg.inv(group_cov)
                
                # Calculate squared Mahalanobis distances
                d_squared = []
                
                for _, row in group_data.iterrows():
                    x = row.values
                    delta = x - group_mean
                    d2 = np.dot(np.dot(delta, inv_cov), delta)
                    d_squared.append(d2)
                
                # QQ plot of chi-square quantiles vs sorted squared distances
                # For multivariate normal data, should follow a chi-square distribution
                # with degrees of freedom equal to the number of variables
                n = len(d_squared)
                df = len(dvs)
                quantiles = np.array([(j - 0.5) / n for j in range(1, n + 1)])
                chi2_quantiles = stats.chi2.ppf(quantiles, df)
                
                # Sort distances
                d_squared.sort()
                
                # Create QQ plot
                axes[i].scatter(chi2_quantiles, d_squared, alpha=0.7)
                
                # Add reference line
                max_val = max(max(chi2_quantiles), max(d_squared))
                axes[i].plot([0, max_val], [0, max_val], 'r--')
                
                axes[i].set_title(f"Chi-Square Q-Q Plot: Group {group}")
                axes[i].set_xlabel("Chi-Square Quantiles")
                axes[i].set_ylabel("Squared Mahalanobis Distance")
                axes[i].grid(True, alpha=0.3)
            except np.linalg.LinAlgError:
                # If covariance matrix is singular
                axes[i].text(0.5, 0.5, "Singular covariance matrix\nCannot compute distances",
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"Group {group}: Error")
        
        plt.tight_layout()
        
        plots.append({
            "title": "Multivariate Normality",
            "img_data": get_base64_plot(),
            "interpretation": "Assesses multivariate normality within each group using chi-square Q-Q plots of squared Mahalanobis distances. Points following the diagonal line suggest multivariate normality, which is an assumption of MANOVA. Systematic deviations indicate potential violations of the multivariate normality assumption."
        })
    
    # Plot 5: Canonical Discriminant Analysis (if possible)
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        # Extract DV data and group labels
        X = data[dvs].values
        y = data[group_var].values
        
        # Fit LDA for visualization
        lda = LinearDiscriminantAnalysis(n_components=min(2, len(groups) - 1))
        X_lda = lda.fit_transform(X, y)
        
        # Create a DataFrame with LDA results and group
        lda_df = pd.DataFrame({
            'LD1': X_lda[:, 0],
            'LD2': X_lda[:, 1] if X_lda.shape[1] > 1 else np.zeros(X_lda.shape[0]),
            'Group': y
        })
        
        # Plot in LDA space
        plt.figure(figsize=(10, 8))
        
        # Plot points
        if X_lda.shape[1] > 1:
            sns.scatterplot(x='LD1', y='LD2', hue='Group', data=lda_df, alpha=0.7)
        else:
            sns.scatterplot(x='LD1', y=[0] * len(lda_df), hue='Group', data=lda_df, alpha=0.7)
        
        # Plot group centroids
        centroids = lda_df.groupby('Group')[['LD1', 'LD2']].mean()
        
        for group in groups:
            center = centroids.loc[group].values
            plt.plot(center[0], center[1], 'o', markersize=10, 
                    marker='*', color='black')
            plt.text(center[0], center[1], str(group), 
                    ha='center', va='center', color='white',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))
        
        if X_lda.shape[1] > 1:
            plt.xlabel('Linear Discriminant 1')
            plt.ylabel('Linear Discriminant 2')
        else:
            plt.xlabel('Linear Discriminant 1')
            plt.yticks([])
            
        plt.title('Canonical Discriminant Analysis')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Canonical Discriminant Analysis",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the separation of groups along canonical discriminant functions (linear combinations of the dependent variables that maximize between-group differences). Greater separation along these axes indicates stronger multivariate differences between groups. The discriminant functions represent the dimensions along which the groups differ most."
        })
    except:
        # If LDA fails, we'll skip this plot
        pass
    
    # Plot 6: Test Statistic Visualization
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
            elif hasattr(test_statistic, 'items'):
                # If it's like a dict but not a dict
                table_data = [['Statistic', 'Value', 'F', 'df1', 'df2', 'p-value']]
                
                for stat_name, stat_value in test_statistic.items():
                    if hasattr(stat_value, 'get'):
                        # Extract values if possible
                        value = stat_value.get('value', None)
                        F = stat_value.get('F', None)
                        df1 = stat_value.get('df1', None)
                        df2 = stat_value.get('df2', None)
                        p = stat_value.get('p', None)
                        
                        table_data.append([stat_name, value, F, df1, df2, p])
                    else:
                        # Simple value
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
            
            plt.title('MANOVA Test Statistics')
            
            plots.append({
                "title": "MANOVA Test Statistics",
                "img_data": get_base64_plot(),
                "interpretation": "Displays the MANOVA test statistics and their significance. Commonly reported statistics include Pillai's trace, Wilks' lambda, Hotelling's trace, and Roy's largest root. P-values below 0.05 (highlighted) suggest significant multivariate effects. Different statistics may lead to different conclusions in some cases, with Pillai's trace being the most robust for general use."
            })
        except:
            # If test statistic visualization fails, we'll skip it
            pass
    
    # Plot 7: MANOVA Model Summary
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
            
            plt.title('MANOVA Model Summary')
            
            plots.append({
                "title": "MANOVA Model Summary",
                "img_data": get_base64_plot(),
                "interpretation": "Presents a summary of the MANOVA results. The table shows the multivariate test statistics and their significance, which test whether the group centroids differ significantly in multivariate space. A significant result indicates that at least one dependent variable is significantly affected by the grouping variable."
            })
        except:
            # If summary plot fails, we'll skip it
            pass
    
    return plots 