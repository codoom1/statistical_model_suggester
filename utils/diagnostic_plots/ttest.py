"""T-test diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_ind
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

def generate_ttest_plots(group1, group2, group_names=None):
    """Generate diagnostic plots for t-test
    
    Args:
        group1: First group data (numpy array)
        group2: Second group data (numpy array)
        group_names: Names of the groups (list of two strings)
        
    Returns:
        List of dictionaries with plot information
    """
    if group_names is None:
        group_names = ['Group 1', 'Group 2']
    
    # Perform t-test
    t_stat, p_value = ttest_ind(group1, group2)
    
    plots = []
    
    # Plot 1: Box plot comparing groups
    plt.figure(figsize=(10, 6))
    plt.boxplot([group1, group2], labels=group_names)
    plt.ylabel('Values')
    plt.title(f'Box Plot Comparison\nT-statistic: {t_stat:.4f}, p-value: {p_value:.4f}')
    plots.append({
        "title": "Box Plot Comparison",
        "img_data": get_base64_plot(),
        "interpretation": "This box plot compares the distributions of the two groups. " +
                         "The line in each box represents the median, the boxes represent the interquartile range (IQR), " +
                         "and the whiskers extend to the most extreme non-outlier points. " +
                         "Compare the medians and spreads to visually assess group differences."
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
                         "The white dot represents the median. This helps visualize differences in " +
                         "distribution shape, central tendency, and spread between groups."
    })
    
    # Plot 3: Q-Q plot for normality check of group 1
    plt.figure(figsize=(10, 6))
    stats.probplot(group1, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot for {group_names[0]}')
    plots.append({
        "title": f"Q-Q Plot for {group_names[0]}",
        "img_data": get_base64_plot(),
        "interpretation": "This Q-Q plot checks if the first group follows a normal distribution. " +
                         "Points should follow the reference line closely if the data is normally distributed. " +
                         "Departures from the line indicate non-normality, which may affect the validity " +
                         "of the t-test results as t-tests assume normality."
    })
    
    # Plot 4: Q-Q plot for normality check of group 2
    plt.figure(figsize=(10, 6))
    stats.probplot(group2, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot for {group_names[1]}')
    plots.append({
        "title": f"Q-Q Plot for {group_names[1]}",
        "img_data": get_base64_plot(),
        "interpretation": "This Q-Q plot checks if the second group follows a normal distribution. " +
                         "Points should follow the reference line closely if the data is normally distributed. " +
                         "Departures from the line indicate non-normality, which may affect the validity " +
                         "of the t-test results as t-tests assume normality."
    })
    
    # Plot 5: Histogram comparison
    plt.figure(figsize=(12, 6))
    plt.hist([group1, group2], alpha=0.7, label=group_names)
    plt.axvline(np.mean(group1), color='blue', linestyle='dashed', linewidth=1, label=f'Mean of {group_names[0]}')
    plt.axvline(np.mean(group2), color='orange', linestyle='dashed', linewidth=1, label=f'Mean of {group_names[1]}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram Comparison of Groups')
    plt.legend()
    plots.append({
        "title": "Histogram Comparison",
        "img_data": get_base64_plot(),
        "interpretation": "This histogram shows the frequency distribution of values in both groups. " +
                         "The dashed lines represent the mean of each group. Compare the central tendency " +
                         "and spread of the distributions. Overlapping distributions with similar means " +
                         "may indicate a non-significant difference between groups."
    })
    
    # Plot 6: Mean and confidence intervals
    plt.figure(figsize=(8, 6))
    means = [np.mean(group1), np.mean(group2)]
    std_errors = [np.std(group1, ddof=1)/np.sqrt(len(group1)), 
                 np.std(group2, ddof=1)/np.sqrt(len(group2))]
    
    # 95% confidence intervals (approximately 1.96 * SE)
    conf_intervals = [1.96 * se for se in std_errors]
    
    plt.errorbar([0, 1], means, yerr=conf_intervals, fmt='o', capsize=10, elinewidth=2, markeredgewidth=2)
    plt.xlim(-0.5, 1.5)
    plt.xticks([0, 1], group_names)
    plt.ylabel('Mean Value')
    plt.title('Group Means with 95% Confidence Intervals')
    plots.append({
        "title": "Mean and Confidence Intervals",
        "img_data": get_base64_plot(),
        "interpretation": "This plot shows the mean of each group with its 95% confidence interval. " +
                         "If the confidence intervals overlap substantially, the difference between groups " +
                         "may not be statistically significant. Non-overlapping intervals suggest a significant " +
                         "difference, although this is not a formal statistical test by itself."
    })
    
    return plots 