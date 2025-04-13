"""Mann-Whitney U Test diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
import seaborn as sns
import io
import base64

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_mann_whitney_plots(group1, group2, group_names=None):
    """Generate diagnostic plots for Mann-Whitney U Test
    
    Args:
        group1: First group data (numpy array)
        group2: Second group data (numpy array)
        group_names: Names of the groups (list of two strings)
        
    Returns:
        List of dictionaries with plot information
    """
    if group_names is None:
        group_names = ['Group 1', 'Group 2']
    
    # Perform Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    
    plots = []
    
    # Plot 1: Box plot comparing groups
    plt.figure(figsize=(10, 6))
    plt.boxplot([group1, group2], labels=group_names)
    plt.ylabel('Values')
    plt.title(f'Box Plot Comparison\nMann-Whitney U: {u_stat:.1f}, p-value: {p_value:.4f}')
    plots.append({
        "title": "Box Plot Comparison",
        "img_data": get_base64_plot(),
        "interpretation": "This box plot compares the distributions of the two groups. " +
                         "The Mann-Whitney U test compares the ranks of values, not the means. " +
                         "Look for differences in median (horizontal line in box) and overall " +
                         "distribution position rather than mean values."
    })
    
    # Plot 2: Violin plot
    plt.figure(figsize=(10, 6))
    plt.violinplot([group1, group2], showmedians=True)
    plt.xticks([1, 2], group_names)
    plt.ylabel('Values')
    plt.title('Violin Plot of Groups')
    plots.append({
        "title": "Violin Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Violin plots show the full distribution shape of each group. " +
                         "The width at each point indicates the density of data points at that value. " +
                         "The Mann-Whitney U test examines if one distribution is stochastically greater " +
                         "than the other, which may be visible in these distributions."
    })
    
    # Plot 3: Histogram comparison
    plt.figure(figsize=(12, 6))
    plt.hist([group1, group2], alpha=0.7, label=group_names)
    plt.axvline(np.median(group1), color='blue', linestyle='dashed', linewidth=1, label=f'Median of {group_names[0]}')
    plt.axvline(np.median(group2), color='orange', linestyle='dashed', linewidth=1, label=f'Median of {group_names[1]}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram Comparison of Groups')
    plt.legend()
    plots.append({
        "title": "Histogram Comparison",
        "img_data": get_base64_plot(),
        "interpretation": "This histogram shows the frequency distribution of values in both groups. " +
                         "The dashed lines represent the median of each group. The Mann-Whitney U test " +
                         "assesses whether values in one group tend to be larger than in the other group, " +
                         "which may be visible as a shift in the distributions."
    })
    
    # Plot 4: Empirical Cumulative Distribution Function (ECDF)
    plt.figure(figsize=(10, 6))
    
    # Sort values and compute ECDF for Group 1
    x1 = np.sort(group1)
    y1 = np.arange(1, len(x1) + 1) / len(x1)
    
    # Sort values and compute ECDF for Group 2
    x2 = np.sort(group2)
    y2 = np.arange(1, len(x2) + 1) / len(x2)
    
    plt.step(x1, y1, where='post', label=group_names[0])
    plt.step(x2, y2, where='post', label=group_names[1])
    plt.xlabel('Values')
    plt.ylabel('Cumulative Probability')
    plt.title('Empirical Cumulative Distribution Function (ECDF)')
    plt.legend()
    plots.append({
        "title": "ECDF Comparison",
        "img_data": get_base64_plot(),
        "interpretation": "The ECDF shows the proportion of data points that are less than or equal to " +
                         "a given value. For the Mann-Whitney U test, a shift in the ECDF of one group " +
                         "relative to the other suggests a stochastic difference between groups. " +
                         "The greater the horizontal separation between curves, the stronger the evidence " +
                         "that the groups differ."
    })
    
    # Plot 5: Rank comparison visualization
    plt.figure(figsize=(12, 6))
    
    # Combine, sort, and rank the data
    combined = np.concatenate([group1, group2])
    ranks = np.argsort(np.argsort(combined)) + 1  # Compute ranks
    
    # Split ranks back into groups
    group1_ranks = ranks[:len(group1)]
    group2_ranks = ranks[len(group1):]
    
    # Create rank comparison plot
    plt.scatter(np.arange(len(group1)), group1_ranks, label=group_names[0], alpha=0.7)
    plt.scatter(np.arange(len(group2)) + 0.1, group2_ranks, label=group_names[1], alpha=0.7)
    plt.xlabel('Data Point Index')
    plt.ylabel('Rank')
    plt.title('Rank Comparison Between Groups')
    plt.legend()
    plots.append({
        "title": "Rank Comparison",
        "img_data": get_base64_plot(),
        "interpretation": "This plot shows the ranks of each data point when both groups are combined. " +
                         "The Mann-Whitney U test is based on these ranks. If one group tends to have " +
                         "higher ranks than the other, it suggests a systematic difference between the groups. " +
                         "The U statistic is essentially a measure of the overlap between these rank distributions."
    })
    
    # Plot 6: Density plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(group1, label=group_names[0])
    sns.kdeplot(group2, label=group_names[1])
    plt.axvline(np.median(group1), color='blue', linestyle='dashed', linewidth=1, label=f'Median of {group_names[0]}')
    plt.axvline(np.median(group2), color='orange', linestyle='dashed', linewidth=1, label=f'Median of {group_names[1]}')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Kernel Density Estimate')
    plt.legend()
    plots.append({
        "title": "Density Plot",
        "img_data": get_base64_plot(),
        "interpretation": "This density plot shows the estimated probability distribution of each group. " +
                         "It smooths out the data to visualize the underlying distribution. The dashed lines " +
                         "represent the median of each group. The Mann-Whitney U test is significant when " +
                         "one distribution is shifted relative to the other, which would appear as a horizontal " +
                         "displacement between the curves."
    })
    
    return plots 