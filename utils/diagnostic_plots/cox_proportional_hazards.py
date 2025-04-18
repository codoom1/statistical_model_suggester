"""Cox Proportional Hazards diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import proportional_hazard_test
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_cox_ph_plots(data, duration_col, event_col, features):
    """Generate diagnostic plots for Cox Proportional Hazards model
    
    Args:
        data: DataFrame containing survival data
        duration_col: Column name for time
        event_col: Column name for event (1 = event occurred, 0 = censored)
        features: List of feature column names
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Copy data to avoid modifying the original
    df = data.copy()
    
    # Fit the Cox model
    cph = CoxPHFitter()
    cph.fit(df, duration_col=duration_col, event_col=event_col)
    
    # Plot 1: Hazard Ratios
    plt.figure(figsize=(10, 6))
    cph.plot()
    plt.title('Hazard Ratios with 95% Confidence Intervals')
    plots.append({
        "title": "Hazard Ratios",
        "img_data": get_base64_plot(),
        "interpretation": "Visualizes the hazard ratios for each covariate with confidence intervals. Values greater than 1 indicate increased risk, while values less than 1 indicate decreased risk. The confidence intervals crossing 1 suggest the variable may not be statistically significant."
    })
    
    # Plot 2: Survival Function
    plt.figure(figsize=(10, 6))
    # Create sample data for different feature values
    if len(features) > 0:
        feature = features[0]
        sample_low = df.copy()
        sample_low[feature] = df[feature].quantile(0.25)
        
        sample_high = df.copy()
        sample_high[feature] = df[feature].quantile(0.75)
        
        cph.plot_partial_effects_on_outcome(feature, [df[feature].quantile(0.25), df[feature].quantile(0.75)], 
                                          cmap='coolwarm')
        plt.title(f'Survival Curves Stratified by {feature}')
    else:
        # If no features, just plot the baseline survival function
        cph.plot_baseline_survival()
        plt.title('Baseline Survival Function')
    
    plots.append({
        "title": "Survival Function",
        "img_data": get_base64_plot(),
        "interpretation": "Shows how the survival probability changes over time. For stratified plots, the separation between curves indicates the effect of the variable on survival, with wider separation suggesting stronger effects."
    })
    
    # Plot 3: Schoenfeld Residuals (Test of Proportional Hazards Assumption)
    plt.figure(figsize=(10, 6))
    try:
        results = proportional_hazard_test(cph, df, time_transform='rank')
        for i, (variable, series) in enumerate(results.schoenfeld_scaled.iteritems()):
            plt.subplot(min(3, len(features)), np.ceil(len(features) / 3), i + 1)
            plt.scatter(results.schoenfeld_scaled.index, series)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title(f'Scaled Schoenfeld Residuals for {variable}')
            if i >= 5:  # Limit to 6 plots to avoid overcrowding
                break
                
        plt.tight_layout()
        plots.append({
            "title": "Schoenfeld Residuals",
            "img_data": get_base64_plot(),
            "interpretation": "Tests the proportional hazards assumption. Residuals should be randomly scattered around zero with no time trend. Significant patterns may indicate violation of the proportional hazards assumption."
        })
    except Exception as e:
        # In case of error in computing Schoenfeld residuals
        plt.text(0.5, 0.5, f"Could not compute Schoenfeld residuals: {str(e)}", 
                 horizontalalignment='center', fontsize=12)
        plt.axis('off')
        plots.append({
            "title": "Schoenfeld Residuals",
            "img_data": get_base64_plot(),
            "interpretation": "Tests the proportional hazards assumption. Residuals should be randomly scattered around zero with no time trend. Significant patterns may indicate violation of the proportional hazards assumption."
        })
    
    # Plot 4: Martingale Residuals
    plt.figure(figsize=(10, 6))
    # Calculate martingale residuals
    df['martingale'] = cph.compute_residuals(df, 'martingale')
    
    # Plot against linear predictor
    df['linear_pred'] = cph.predict_partial_hazard(df)
    plt.scatter(df['linear_pred'], df['martingale'])
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Linear Predictor')
    plt.ylabel('Martingale Residuals')
    plt.title('Martingale Residuals vs Linear Predictor')
    
    plots.append({
        "title": "Martingale Residuals",
        "img_data": get_base64_plot(),
        "interpretation": "Assesses model fit. Residuals should be randomly scattered around zero. Patterns may indicate non-linearity or model misspecification. Values close to 1 represent unexpected survivors, while values close to -1 represent unexpected deaths."
    })
    
    # Plot 5: Deviance Residuals
    plt.figure(figsize=(10, 6))
    # Calculate deviance residuals
    df['deviance'] = cph.compute_residuals(df, 'deviance')
    
    # Plot against linear predictor
    plt.scatter(df['linear_pred'], df['deviance'])
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Linear Predictor')
    plt.ylabel('Deviance Residuals')
    plt.title('Deviance Residuals vs Linear Predictor')
    
    plots.append({
        "title": "Deviance Residuals",
        "img_data": get_base64_plot(),
        "interpretation": "Symmetrized version of martingale residuals, useful for detecting outliers. Points far from zero may indicate observations that are not well-fitted by the model."
    })
    
    # Plot 6: Log-Log Plot (Another test of proportional hazards)
    plt.figure(figsize=(10, 6))
    if len(features) > 0:
        feature = features[0]
        if df[feature].nunique() > 2 and df[feature].nunique() <= 5:
            # For categorical with few categories
            cph.plot_partial_effects_on_outcome(feature, df[feature].unique(), 
                                              cmap='coolwarm', log_y=True)
            plt.title(f'Log-Log Survival Plot Stratified by {feature}')
        else:
            # For continuous or many categories, use quartiles
            cph.plot_partial_effects_on_outcome(feature, 
                                              [df[feature].quantile(q) for q in [0.25, 0.5, 0.75]], 
                                              cmap='coolwarm', log_y=True)
            plt.title(f'Log-Log Survival Plot Stratified by {feature} Quartiles')
    else:
        # If no features, just note that we can't create this plot
        plt.text(0.5, 0.5, "Log-Log plot requires categorical features", 
                 horizontalalignment='center', fontsize=12)
        plt.axis('off')
    
    plots.append({
        "title": "Log-Log Survival Plot",
        "img_data": get_base64_plot(),
        "interpretation": "Tests the proportional hazards assumption. Parallel lines suggest the proportional hazards assumption is met. Crossing or diverging lines indicate violation of this assumption."
    })
    
    return plots 