"""Chi-Square test diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
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

def generate_chi_square_plots(observed_table, row_labels=None, col_labels=None):
    """Generate diagnostic plots for Chi-Square test
    
    Args:
        observed_table: Contingency table (numpy array or pandas DataFrame)
        row_labels: Labels for rows (list of strings)
        col_labels: Labels for columns (list of strings)
        
    Returns:
        List of dictionaries with plot information
    """
    # Convert to numpy array if DataFrame
    if isinstance(observed_table, pd.DataFrame):
        if row_labels is None:
            row_labels = observed_table.index.tolist()
        if col_labels is None:
            col_labels = observed_table.columns.tolist()
        observed_table = observed_table.values
    
    # Assign default labels if none provided
    if row_labels is None:
        row_labels = [f'Row {i+1}' for i in range(observed_table.shape[0])]
    if col_labels is None:
        col_labels = [f'Col {i+1}' for i in range(observed_table.shape[1])]
    
    # Perform Chi-Square test
    chi2, p_value, dof, expected = chi2_contingency(observed_table)
    
    plots = []
    
    # Plot 1: Observed counts heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(observed_table, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=col_labels, yticklabels=row_labels)
    plt.title(f'Observed Counts\nChi-square: {chi2:.4f}, p-value: {p_value:.4f}, df: {dof}')
    plt.tight_layout()
    plots.append({
        "title": "Observed Counts Heatmap",
        "img_data": get_base64_plot(),
        "interpretation": "This heatmap displays the observed counts in each cell of the contingency table. " +
                         "The color intensity represents the magnitude of the count, with darker colors indicating " +
                         "higher counts. The numbers in each cell show the exact count."
    })
    
    # Plot 2: Expected counts heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(expected, annot=True, fmt='.1f', cmap='YlGnBu',
                xticklabels=col_labels, yticklabels=row_labels)
    plt.title('Expected Counts (Under Independence)')
    plt.tight_layout()
    plots.append({
        "title": "Expected Counts Heatmap",
        "img_data": get_base64_plot(),
        "interpretation": "This heatmap shows the expected counts in each cell if the null hypothesis (independence) " +
                         "were true. Expected counts are calculated based on the row and column totals. Compare " +
                         "this to the observed counts to see where the major differences occur."
    })
    
    # Plot 3: Residuals (standardized difference between observed and expected)
    residuals = (observed_table - expected) / np.sqrt(expected)
    plt.figure(figsize=(10, 8))
    sns.heatmap(residuals, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                xticklabels=col_labels, yticklabels=row_labels)
    plt.title('Standardized Residuals')
    plt.tight_layout()
    plots.append({
        "title": "Standardized Residuals",
        "img_data": get_base64_plot(),
        "interpretation": "Standardized residuals show the difference between observed and expected counts, " +
                         "scaled by the square root of the expected count. Values further from zero indicate " +
                         "cells that contribute more to the chi-square statistic. As a rule of thumb, " +
                         "values above 2 or below -2 indicate significant deviations from independence."
    })
    
    # Plot 4: Grouped bar chart
    plt.figure(figsize=(12, 8))
    df = pd.DataFrame(observed_table, index=row_labels, columns=col_labels)
    df.plot(kind='bar', figsize=(12, 8))
    plt.title('Grouped Bar Chart of Observed Counts')
    plt.xlabel('Row Category')
    plt.ylabel('Count')
    plt.legend(title='Column Category')
    plt.tight_layout()
    plots.append({
        "title": "Grouped Bar Chart",
        "img_data": get_base64_plot(),
        "interpretation": "This grouped bar chart shows the observed counts for each combination of row and column " +
                         "categories. It provides a visual way to compare counts across different categories and " +
                         "identify patterns or differences that may contribute to the chi-square result."
    })
    
    # Plot 5: Contribution to chi-square
    contribution = (observed_table - expected)**2 / expected
    plt.figure(figsize=(10, 8))
    sns.heatmap(contribution, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=col_labels, yticklabels=row_labels)
    plt.title('Contribution to Chi-square Statistic')
    plt.tight_layout()
    plots.append({
        "title": "Chi-square Contribution",
        "img_data": get_base64_plot(),
        "interpretation": "This heatmap shows how much each cell contributes to the overall chi-square statistic. " +
                         "Higher values (darker colors) indicate cells that have a larger influence on the test result. " +
                         "This helps identify which specific combinations of categories are most responsible for " +
                         "rejecting or failing to reject the null hypothesis."
    })
    
    # Plot 6: Mosaic plot (if matplotlib version supports it)
    try:
        from matplotlib.pyplot import subplots
        from statsmodels.graphics.mosaicplot import mosaic
        
        plt.figure(figsize=(12, 8))
        # Convert contingency table to format needed for mosaic plot
        contingency_data = {}
        for i, row in enumerate(row_labels):
            for j, col in enumerate(col_labels):
                contingency_data[(row, col)] = observed_table[i, j]
        
        mosaic(contingency_data, title='Mosaic Plot of Contingency Table')
        plt.tight_layout()
        plots.append({
            "title": "Mosaic Plot",
            "img_data": get_base64_plot(),
            "interpretation": "The mosaic plot displays the contingency table as a set of rectangles whose areas are " +
                             "proportional to the cell counts. This visualization helps identify patterns of association " +
                             "between the row and column variables. In the case of independence, the rectangles within " +
                             "each row (or column) would have the same proportions as the column (or row) totals."
        })
    except (ImportError, AttributeError) as e:
        # If mosaic plot is not available, create a stacked bar chart instead
        plt.figure(figsize=(12, 8))
        df_pct = df.div(df.sum(axis=1), axis=0) * 100
        df_pct.plot(kind='bar', stacked=True, figsize=(12, 8))
        plt.title('Stacked Bar Chart of Proportions')
        plt.xlabel('Row Category')
        plt.ylabel('Percentage')
        plt.legend(title='Column Category')
        plt.tight_layout()
        plots.append({
            "title": "Stacked Bar Chart",
            "img_data": get_base64_plot(),
            "interpretation": "This stacked bar chart shows the proportion of each column category within each row category. " +
                             "If the variables are independent, the proportions (and thus the colored segments) should be " +
                             "similar across all row categories. Differences in the pattern suggest an association between " +
                             "the row and column variables."
        })
    
    return plots 