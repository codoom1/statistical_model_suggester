"""Kruskal-Wallis Test diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
import seaborn as sns
import io
import base64
from itertools import combinations

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_kruskal_wallis_plots(groups, group_names=None):
    """Generate diagnostic plots for Kruskal-Wallis Test
    
    Args:
        groups: List of arrays containing the data for each group
        group_names: Names of the groups (list of strings)
        
    Returns:
        List of dictionaries with plot information
    """
    if group_names is None:
        group_names = [f'Group {i+1}' for i in range(len(groups))]
    
    # Ensure we have the right number of group names
    if len(group_names) != len(groups):
        group_names = [f'Group {i+1}' for i in range(len(groups))]
    
    # Perform Kruskal-Wallis test
    stat, p_value = kruskal(*groups)
    
    plots = []
    
    # Plot 1: Box plot comparing all groups
    plt.figure(figsize=(12, 8))
    plt.boxplot(groups, labels=group_names)
    plt.ylabel('Values')
    plt.title(f'Box Plot Comparison\nKruskal-Wallis H: {stat:.4f}, p-value: {p_value:.4f}')
    plt.xticks(rotation=45 if len(group_names) > 4 else 0)
    plt.tight_layout()
    plots.append({
        "title": "Box Plot Comparison",
        "img_data": get_base64_plot(),
        "interpretation": "This box plot compares the distributions of all groups. " +
                         "The Kruskal-Wallis test is a non-parametric alternative to one-way ANOVA and " +
                         "tests if samples originate from the same distribution. Look for differences " +
                         "in median (horizontal line in box) and overall positioning of the boxes."
    })
    
    # Plot 2: Violin plot
    plt.figure(figsize=(12, 8))
    parts = plt.violinplot(groups, showmedians=True)
    # Set colors for violin plots
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(f'C{i}')
        pc.set_alpha(0.7)
    
    plt.xticks(np.arange(1, len(groups) + 1), group_names, rotation=45 if len(group_names) > 4 else 0)
    plt.ylabel('Values')
    plt.title('Violin Plot of Groups')
    plt.tight_layout()
    plots.append({
        "title": "Violin Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Violin plots show the full distribution shape of each group. " +
                         "The width at each point indicates the density of data points at that value. " +
                         "The Kruskal-Wallis test examines if at least one distribution stochastically " +
                         "dominates at least one other distribution, which may be visible in these shapes."
    })
    
    # Plot 3: Strip plot (jittered points)
    plt.figure(figsize=(12, 8))
    
    # Convert to dataframe for seaborn
    data_list = []
    for i, group in enumerate(groups):
        df = pd.DataFrame({'value': group, 'group': group_names[i]})
        data_list.append(df)
    df = pd.concat(data_list)
    
    sns.stripplot(x='group', y='value', data=df, jitter=True, alpha=0.7)
    plt.title('Strip Plot of All Data Points')
    plt.xlabel('')
    plt.xticks(rotation=45 if len(group_names) > 4 else 0)
    plt.tight_layout()
    plots.append({
        "title": "Strip Plot",
        "img_data": get_base64_plot(),
        "interpretation": "This strip plot shows all individual data points with a small amount of " +
                         "random horizontal 'jitter' to avoid overplotting. This helps visualize the " +
                         "actual data distribution and identify potential outliers or patterns that might " +
                         "influence the Kruskal-Wallis test results."
    })
    
    # Plot 4: Ranked data visualization
    plt.figure(figsize=(12, 8))
    
    # Combine and rank all data
    all_data = np.concatenate(groups)
    ranks = np.argsort(np.argsort(all_data)) + 1
    
    # Split ranks back to their original groups
    start_idx = 0
    ranked_groups = []
    for group in groups:
        end_idx = start_idx + len(group)
        ranked_groups.append(ranks[start_idx:end_idx])
        start_idx = end_idx
    
    # Plot the rank distributions
    plt.boxplot(ranked_groups, labels=group_names)
    plt.ylabel('Rank')
    plt.title('Rank Distribution by Group')
    plt.xticks(rotation=45 if len(group_names) > 4 else 0)
    plt.tight_layout()
    plots.append({
        "title": "Rank Distribution",
        "img_data": get_base64_plot(),
        "interpretation": "The Kruskal-Wallis test is based on the ranks of the data points rather than " +
                         "their actual values. This plot shows how the ranks are distributed across groups. " +
                         "If all groups come from identical distributions, the rank distributions should be similar. " +
                         "Differences in median ranks suggest that some groups tend to have systematically " +
                         "higher or lower values than others."
    })
    
    # Plot 5: Empirical Cumulative Distribution Function (ECDF)
    plt.figure(figsize=(12, 8))
    
    for i, group in enumerate(groups):
        # Sort values and compute ECDF
        x = np.sort(group)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where='post', label=group_names[i])
    
    plt.xlabel('Values')
    plt.ylabel('Cumulative Probability')
    plt.title('Empirical Cumulative Distribution Function (ECDF)')
    plt.legend()
    plt.tight_layout()
    plots.append({
        "title": "ECDF Comparison",
        "img_data": get_base64_plot(),
        "interpretation": "The ECDF shows the proportion of data points that are less than or equal to " +
                         "a given value for each group. For the Kruskal-Wallis test, separation between " +
                         "the curves suggests differences in the distributions. Groups whose curves are " +
                         "consistently to the right have stochastically larger values."
    })
    
    # Plot 6: Density plot
    plt.figure(figsize=(12, 8))
    
    for i, group in enumerate(groups):
        sns.kdeplot(group, label=group_names[i])
    
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Kernel Density Estimate')
    plt.legend()
    plt.tight_layout()
    plots.append({
        "title": "Density Plot",
        "img_data": get_base64_plot(),
        "interpretation": "This density plot shows the estimated probability distribution of each group. " +
                         "It helps visualize differences in the shape and position of each distribution. " +
                         "The Kruskal-Wallis test will be significant when at least one distribution is " +
                         "shifted relative to the others, which would appear as a horizontal displacement."
    })
    
    # If there are more than 2 groups and the Kruskal-Wallis test is significant,
    # add a pairwise comparison plot
    if len(groups) > 2 and p_value < 0.05:
        plt.figure(figsize=(12, 10))
        
        # Perform pairwise Mann-Whitney U tests
        pair_results = []
        for (i, group_i), (j, group_j) in combinations(enumerate(groups), 2):
            u_stat, p = mannwhitneyu(group_i, group_j, alternative='two-sided')
            pair_results.append({
                'pair': f"{group_names[i]} vs {group_names[j]}",
                'p_value': p,
                'u_stat': u_stat,
                'significant': p < 0.05,  # Using 0.05 as default threshold
                'group1_idx': i,
                'group2_idx': j
            })
        
        # Create a matrix for the heatmap
        n_groups = len(groups)
        heatmap_matrix = np.zeros((n_groups, n_groups))
        annot_matrix = np.empty((n_groups, n_groups), dtype=object)
        
        # Set diagonal to NaN
        for i in range(n_groups):
            heatmap_matrix[i, i] = np.nan
            annot_matrix[i, i] = ""
            
        # Fill matrices
        for result in pair_results:
            i, j = result['group1_idx'], result['group2_idx']
            heatmap_matrix[i, j] = -np.log10(result['p_value'])  # -log10(p) for better visualization
            heatmap_matrix[j, i] = -np.log10(result['p_value'])  # Mirror for full matrix
            
            if result['significant']:
                sig = '**' if result['p_value'] < 0.01 else '*'
            else:
                sig = 'ns'
            
            annot_matrix[i, j] = f"p={result['p_value']:.4f} {sig}"
            annot_matrix[j, i] = f"p={result['p_value']:.4f} {sig}"
        
        # Create the heatmap
        mask = np.isnan(heatmap_matrix)
        sns.heatmap(heatmap_matrix, annot=annot_matrix, mask=mask, 
                    cmap="YlOrRd", xticklabels=group_names, yticklabels=group_names,
                    fmt="")
        plt.title('Pairwise Comparison P-values (-log10 scale)')
        plt.tight_layout()
        plots.append({
            "title": "Pairwise Comparisons",
            "img_data": get_base64_plot(),
            "interpretation": "This heatmap shows the results of pairwise Mann-Whitney U tests between all groups. " +
                             "The color intensity represents the significance level (-log10 of p-value), with " +
                             "darker colors indicating more significant differences. Cells marked with * (p < 0.05) " +
                             "or ** (p < 0.01) show statistically significant differences between those groups. " +
                             "This helps identify which specific group differences are contributing to the " +
                             "overall Kruskal-Wallis result."
        })
    
    return plots 