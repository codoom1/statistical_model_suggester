"""Kaplan-Meier survival analysis diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import pandas as pd
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

def generate_kaplan_meier_plots(data, duration_col, event_col, group_col=None):
    """Generate diagnostic plots for Kaplan-Meier survival analysis
    
    Args:
        data: DataFrame containing survival data
        duration_col: Column name for time
        event_col: Column name for event (1 = event occurred, 0 = censored)
        group_col: Optional column name for grouping variable
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Copy data to avoid modifying the original
    df = data.copy()
    
    # Plot 1: Survival Function
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    
    if group_col is not None and group_col in df.columns:
        # Stratified by group
        groups = df[group_col].unique()
        for group in groups:
            mask = df[group_col] == group
            kmf.fit(df[mask][duration_col], df[mask][event_col], label=f'{group_col}={group}')
            kmf.plot_survival_function()
        
        plt.title(f'Kaplan-Meier Survival Function Stratified by {group_col}')
        
        # Run log-rank test for group differences
        if len(groups) == 2:
            # For two groups
            g1 = groups[0]
            g2 = groups[1]
            results = logrank_test(
                df[df[group_col] == g1][duration_col], 
                df[df[group_col] == g2][duration_col],
                df[df[group_col] == g1][event_col],
                df[df[group_col] == g2][event_col]
            )
            p_value = results.p_value
            plt.text(0.05, 0.05, f'Log-rank test p-value: {p_value:.4f}', 
                     transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    else:
        # Overall survival
        kmf.fit(df[duration_col], df[event_col])
        kmf.plot_survival_function()
        plt.title('Kaplan-Meier Overall Survival Function')
    
    plt.ylabel('Survival Probability')
    plt.xlabel('Time')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Survival Function",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the probability of surviving beyond a given time. Each step represents an event time. Censored observations are marked with '+'. For stratified plots, separation between curves indicates survival differences between groups, and the log-rank test assesses statistical significance."
    })
    
    # Plot 2: Cumulative Hazard Function
    plt.figure(figsize=(10, 6))
    
    if group_col is not None and group_col in df.columns:
        # Stratified by group
        for group in df[group_col].unique():
            mask = df[group_col] == group
            kmf.fit(df[mask][duration_col], df[mask][event_col], label=f'{group_col}={group}')
            kmf.plot_cumulative_density()
        
        plt.title(f'Cumulative Incidence Function Stratified by {group_col}')
    else:
        # Overall cumulative hazard
        kmf.fit(df[duration_col], df[event_col])
        kmf.plot_cumulative_density()
        plt.title('Cumulative Incidence Function')
    
    plt.ylabel('Cumulative Probability of Event')
    plt.xlabel('Time')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Cumulative Incidence",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the cumulative probability of experiencing the event by a given time point (1 - survival probability). Steeper slopes indicate higher hazard (risk) of the event occurring at that time."
    })
    
    # Plot 3: Survival Function with Confidence Intervals
    plt.figure(figsize=(10, 6))
    
    if group_col is not None and group_col in df.columns:
        # Use first group for demonstration
        group = df[group_col].unique()[0]
        mask = df[group_col] == group
        kmf.fit(df[mask][duration_col], df[mask][event_col], label=f'{group_col}={group}')
        kmf.plot_survival_function(ci_show=True)
        
        plt.title(f'Survival Function with 95% CI for {group_col}={group}')
    else:
        # Overall survival with CI
        kmf.fit(df[duration_col], df[event_col])
        kmf.plot_survival_function(ci_show=True)
        plt.title('Survival Function with 95% Confidence Intervals')
    
    plt.ylabel('Survival Probability')
    plt.xlabel('Time')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Survival with Confidence Intervals",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the survival function with 95% confidence intervals. Wider intervals indicate greater uncertainty, typically seen at later times when fewer subjects remain at risk. The shaded region represents the confidence band."
    })
    
    # Plot 4: At-Risk Table
    if group_col is not None and group_col in df.columns:
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        
        # Get unique groups
        groups = df[group_col].unique()
        
        # Set color cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # Plot survival functions
        for i, group in enumerate(groups):
            mask = df[group_col] == group
            kmf = KaplanMeierFitter()
            kmf.fit(df[mask][duration_col], df[mask][event_col], label=f'{group_col}={group}')
            kmf.plot_survival_function(ax=ax, ci_show=False, color=color_cycle[i % len(color_cycle)])
        
        # Add at-risk table below
        max_time = df[duration_col].max()
        time_points = np.linspace(0, max_time, num=min(5, int(max_time)) + 1)
        
        # Create an at-risk table as text
        at_risk_data = []
        
        for group in groups:
            mask = df[group_col] == group
            group_df = df[mask]
            at_risk = []
            
            for t in time_points:
                count = sum(group_df[duration_col] >= t)
                at_risk.append(count)
            
            at_risk_data.append(at_risk)
        
        # Add text table below plot
        for i, t in enumerate(time_points):
            plt.text(t, -0.1, f"t={t:.1f}", ha='center', transform=ax.get_xaxis_transform())
            
            for j, group in enumerate(groups):
                plt.text(t, -0.15 - 0.05 * j, str(at_risk_data[j][i]), 
                         ha='center', transform=ax.get_xaxis_transform(), 
                         color=color_cycle[j % len(color_cycle)])
        
        # Add group labels
        for j, group in enumerate(groups):
            plt.text(-0.02, -0.15 - 0.05 * j, f"{group_col}={group}: ", 
                     ha='right', transform=ax.get_xaxis_transform(), 
                     color=color_cycle[j % len(color_cycle)])
        
        plt.text(-0.02, -0.1, "At risk: ", ha='right', transform=ax.get_xaxis_transform())
        
        # Adjust layout
        plt.subplots_adjust(bottom=0.2)
        plt.title(f'Survival Function with At-Risk Table Stratified by {group_col}')
        plt.ylabel('Survival Probability')
        plt.xlabel('Time')
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Survival with At-Risk Table",
            "img_data": get_base64_plot(),
            "interpretation": "Displays survival curves with a table showing the number of subjects still at risk (not yet experienced the event or been censored) at various time points. The decreasing numbers reflect subjects who have either experienced the event or been censored."
        })
    
    # Plot 5: Survival Probability at Specific Time Points (for multiple groups)
    if group_col is not None and group_col in df.columns and len(df[group_col].unique()) > 1:
        plt.figure(figsize=(10, 6))
        
        time_points = [
            df[duration_col].quantile(0.25),
            df[duration_col].median(),
            df[duration_col].quantile(0.75)
        ]
        
        survival_at_timepoints = []
        groups = df[group_col].unique()
        
        for group in groups:
            mask = df[group_col] == group
            kmf = KaplanMeierFitter()
            kmf.fit(df[mask][duration_col], df[mask][event_col])
            
            survival_probs = []
            for t in time_points:
                try:
                    survival_probs.append(kmf.predict(t))
                except:
                    survival_probs.append(np.nan)
            
            survival_at_timepoints.append(survival_probs)
        
        # Create bar plot
        bar_data = pd.DataFrame(
            survival_at_timepoints, 
            index=groups,
            columns=[f"t={t:.1f}" for t in time_points]
        )
        
        bar_data.plot(kind='bar', ax=plt.gca())
        plt.title('Survival Probability at Specific Time Points')
        plt.ylabel('Survival Probability')
        plt.xlabel(group_col)
        plt.ylim(0, 1)
        plt.legend(title='Time Points')
        
        plots.append({
            "title": "Survival Probability Comparison",
            "img_data": get_base64_plot(),
            "interpretation": "Compares survival probabilities across groups at specific time points (25th, 50th, and 75th percentiles of follow-up time). This allows for direct comparison of survival rates at clinically relevant time points."
        })
    
    # Plot 6: Event Occurrence Overview
    plt.figure(figsize=(10, 6))
    
    # Sort by time
    df_sorted = df.sort_values(by=duration_col).reset_index(drop=True)
    
    # Calculate cumulative events
    event_times = df_sorted[df_sorted[event_col] == 1][duration_col]
    cumulative_events = np.arange(1, len(event_times) + 1)
    
    plt.step(event_times, cumulative_events, where='post')
    plt.title('Cumulative Number of Events Over Time')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Number of Events')
    plt.grid(True, alpha=0.3)
    
    plots.append({
        "title": "Cumulative Events",
        "img_data": get_base64_plot(),
        "interpretation": "Shows the total number of events (e.g., deaths, failures) that have occurred up to each time point. Steeper slopes indicate time periods with higher event rates. Plateaus indicate periods with few or no events."
    })
    
    return plots 