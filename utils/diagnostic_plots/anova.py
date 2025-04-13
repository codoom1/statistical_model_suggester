"""ANOVA diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.graphics.gofplots import qqplot
import io
import base64
from scipy import stats

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_anova_plots(data, value_col, group_col):
    """Generate diagnostic plots for ANOVA
    
    Args:
        data: Pandas DataFrame
        value_col: Name of the value column (dependent variable)
        group_col: Name of the grouping column (factor)
        
    Returns:
        List of dictionaries with plot information
    """
    # Run ANOVA to get residuals
    formula = f"{value_col} ~ C({group_col})"
    model = ols(formula, data=data).fit()
    residuals = model.resid
    fitted = model.fittedvalues
    
    plots = []
    
    # Plot 1: Box Plot of Groups
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=group_col, y=value_col, data=data)
    plt.title('Box Plot by Group')
    plots.append({
        "title": "Box Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Compares the distribution of the dependent variable across groups. Look for differences in medians, spread, and potential outliers. Similar variances across groups support the homogeneity of variance assumption."
    })
    
    # Plot 2: Residuals vs Fitted
    plt.figure(figsize=(10, 6))
    plt.scatter(fitted, residuals)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plots.append({
        "title": "Residuals vs Fitted",
        "img_data": get_base64_plot(),
        "interpretation": "Checks for homogeneity of variance and linearity. Residuals should be evenly spread around the horizontal line with no systematic patterns. Fan-shaped patterns indicate heteroscedasticity (violation of equal variance assumption)."
    })
    
    # Plot 3: Q-Q Plot
    plt.figure(figsize=(10, 6))
    qqplot(residuals, line='45', fit=True)
    plt.title('Normal Q-Q Plot')
    plots.append({
        "title": "Q-Q Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Assesses normality of residuals. Points should follow the diagonal line closely if residuals are normally distributed. Deviations suggest non-normality which may affect p-values in smaller samples."
    })
    
    # Plot 4: Means Plot with Error Bars
    plt.figure(figsize=(10, 6))
    means = data.groupby(group_col)[value_col].mean()
    ci = data.groupby(group_col)[value_col].sem() * 1.96  # 95% CI based on SEM
    
    # Convert to plotting format
    mean_df = pd.DataFrame({
        'group': means.index,
        'mean': means.values,
        'ci_lower': means.values - ci.values,
        'ci_upper': means.values + ci.values
    })
    
    # Plot means and CIs
    plt.errorbar(mean_df['group'], mean_df['mean'], 
                 yerr=ci.values, fmt='o', capsize=5)
    
    plt.xlabel(group_col)
    plt.ylabel(f'Mean {value_col}')
    plt.title('Group Means with 95% CI')
    plots.append({
        "title": "Group Means",
        "img_data": get_base64_plot(),
        "interpretation": "Visualizes mean differences between groups with 95% confidence intervals. Non-overlapping intervals suggest statistically significant differences. The pattern and magnitude of differences inform about the effect being studied."
    })
    
    return plots 