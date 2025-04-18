"""Survival analysis diagnostic plots (Cox Regression, Kaplan-Meier)."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_survival_analysis_plots(model, X=None, time=None, event=None, 
                                    groups=None, group_names=None, feature_names=None):
    """Generate diagnostic plots for survival analysis models (Cox PH, Kaplan-Meier)
    
    Args:
        model: Fitted survival model (Cox PH, Kaplan-Meier)
        X: Feature matrix for Cox regression
        time: Survival/censoring times
        event: Event indicator (1=event, 0=censored)
        groups: Group indicator for stratified analysis
        group_names: Names of groups for stratified analysis
        feature_names: Names of features for Cox regression
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Get feature names if not provided
    if feature_names is None and X is not None:
        if hasattr(X, 'columns'):  # If X is a DataFrame
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"X{i+1}" for i in range(X.shape[1] if X.ndim > 1 else 1)]
    
    # Check if we have time and event data
    has_surv_data = time is not None and event is not None
            
    # Plot 1: Kaplan-Meier Survival Curves (overall and by group if provided)
    if has_surv_data:
        plt.figure(figsize=(12, 7))
        
        # If no groups are provided, create a single group
        if groups is None:
            # Plot overall KM curve
            times = np.sort(np.unique(time))
            surv_prob = []
            
            # Simple KM estimation
            n_total = len(time)
            n_at_risk = n_total
            
            for t in times:
                n_events = np.sum((time == t) & (event == 1))
                
                if n_at_risk > 0:
                    surv_prob.append(1 - n_events / n_at_risk)
                else:
                    surv_prob.append(surv_prob[-1] if surv_prob else 1.0)
                
                n_at_risk -= np.sum(time == t)
            
            # Convert to cumulative product
            surv_prob = np.cumprod([1.0] + surv_prob)[1:]
            
            plt.step(times, surv_prob, where='post', label='Overall', linewidth=2)
            
            # Add censoring marks
            cens_times = time[event == 0]
            if len(cens_times) > 0:
                # Find the survival probability at each censoring time
                cens_probs = []
                for t in cens_times:
                    idx = np.searchsorted(times, t)
                    prob = surv_prob[idx-1] if idx > 0 and idx <= len(surv_prob) else 1.0
                    cens_probs.append(prob)
                
                plt.plot(cens_times, cens_probs, 'k+', markersize=8, label='Censored')
        
        else:
            # Plot KM curves for each group
            unique_groups = np.unique(groups)
            
            if group_names is None or len(group_names) != len(unique_groups):
                group_names = [f"Group {i+1}" for i in range(len(unique_groups))]
            
            for i, group in enumerate(unique_groups):
                group_mask = (groups == group)
                group_time = time[group_mask]
                group_event = event[group_mask]
                
                # Sort by time
                sort_idx = np.argsort(group_time)
                group_time = group_time[sort_idx]
                group_event = group_event[sort_idx]
                
                times = np.unique(group_time)
                surv_prob = []
                
                # Simple KM estimation
                n_total = len(group_time)
                n_at_risk = n_total
                
                for t in times:
                    n_events = np.sum((group_time == t) & (group_event == 1))
                    
                    if n_at_risk > 0:
                        surv_prob.append(1 - n_events / n_at_risk)
                    else:
                        surv_prob.append(surv_prob[-1] if surv_prob else 1.0)
                    
                    n_at_risk -= np.sum(group_time == t)
                
                # Convert to cumulative product
                surv_prob = np.cumprod([1.0] + surv_prob)[1:]
                
                plt.step(times, surv_prob, where='post', label=group_names[i], linewidth=2)
                
                # Add censoring marks
                cens_times = group_time[group_event == 0]
                if len(cens_times) > 0:
                    # Find the survival probability at each censoring time
                    cens_probs = []
                    for t in cens_times:
                        idx = np.searchsorted(times, t)
                        prob = surv_prob[idx-1] if idx > 0 and idx <= len(surv_prob) else 1.0
                        cens_probs.append(prob)
                    
                    plt.plot(cens_times, cens_probs, 'k+', markersize=8, alpha=0.7)
        
        plt.title('Kaplan-Meier Survival Curve')
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.legend()
        
        plots.append({
            "title": "Kaplan-Meier Survival Curve",
            "img_data": get_base64_plot(),
            "interpretation": "The Kaplan-Meier curve shows the probability of surviving over time. Each step down represents events occurring at that time point. '+' marks indicate censored observations. If groups are shown, non-overlapping curves suggest different survival experiences between groups."
        })
    
    # Plot 2: Log-log plot for Cox PH assumption (if multiple groups)
    if has_surv_data and groups is not None:
        plt.figure(figsize=(12, 7))
        
        unique_groups = np.unique(groups)
        if group_names is None or len(group_names) != len(unique_groups):
            group_names = [f"Group {i+1}" for i in range(len(unique_groups))]
        
        for i, group in enumerate(unique_groups):
            group_mask = (groups == group)
            group_time = time[group_mask]
            group_event = event[group_mask]
            
            # Sort by time
            sort_idx = np.argsort(group_time)
            group_time = group_time[sort_idx]
            group_event = group_event[sort_idx]
            
            times = np.unique(group_time)
            surv_prob = []
            
            # Simple KM estimation
            n_total = len(group_time)
            n_at_risk = n_total
            
            for t in times:
                n_events = np.sum((group_time == t) & (group_event == 1))
                
                if n_at_risk > 0:
                    surv_prob.append(1 - n_events / n_at_risk)
                else:
                    surv_prob.append(surv_prob[-1] if surv_prob else 1.0)
                
                n_at_risk -= np.sum(group_time == t)
            
            # Convert to cumulative product
            surv_prob = np.cumprod([1.0] + surv_prob)[1:]
            
            # Calculate log(-log(S(t))) 
            # Handle potential numerical issues
            valid_idx = (surv_prob > 0) & (surv_prob < 1)
            log_log_surv = np.log(-np.log(surv_prob[valid_idx]))
            valid_times = times[valid_idx]
            
            if len(valid_times) > 0:
                plt.plot(np.log(valid_times), log_log_surv, 'o-', label=group_names[i], alpha=0.7)
        
        plt.title('Log-Log Plot for Proportional Hazards Assumption')
        plt.xlabel('Log(Time)')
        plt.ylabel('Log(-Log(Survival Probability))')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plots.append({
            "title": "Log-Log Survival Plot",
            "img_data": get_base64_plot(),
            "interpretation": "This plot helps assess the proportional hazards assumption. If curves for different groups are approximately parallel, the assumption is supported. Converging or diverging lines suggest the assumption may be violated, indicating time-varying effects."
        })
    
    # Plot 3: Cox PH model coefficients with confidence intervals
    if hasattr(model, 'params') and hasattr(model, 'confidence_intervals') and feature_names:
        try:
            plt.figure(figsize=(12, max(6, len(feature_names) * 0.5)))
            
            # Extract coefficients and CIs
            coefs = model.params
            conf_int = model.confidence_intervals()
            
            if hasattr(coefs, 'index'):
                coef_names = coefs.index.tolist()
            else:
                coef_names = feature_names[:len(coefs)]
            
            # Create horizontal error bar plot
            y_pos = np.arange(len(coefs))
            plt.errorbar(coefs, y_pos, xerr=[coefs - conf_int.iloc[:, 0], conf_int.iloc[:, 1] - coefs], 
                        fmt='o', capsize=5, elinewidth=2, markeredgewidth=2, markersize=8)
            
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
            plt.yticks(y_pos, coef_names)
            plt.xlabel('Coefficient Value (Log Hazard Ratio)')
            plt.title('Cox PH Model Coefficients with 95% Confidence Intervals')
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Cox Regression Coefficients",
                "img_data": get_base64_plot(),
                "interpretation": "Shows coefficient estimates (log hazard ratios) with confidence intervals. Positive values indicate higher hazard (worse survival) as the predictor increases. If an interval doesn't cross zero (red line), the predictor is significantly associated with survival outcome."
            })
            
            # Plot 4: Hazard ratios (exponentiated coefficients)
            plt.figure(figsize=(12, max(6, len(feature_names) * 0.5)))
            
            # Calculate hazard ratios and CIs
            hr = np.exp(coefs)
            hr_lower = np.exp(conf_int.iloc[:, 0])
            hr_upper = np.exp(conf_int.iloc[:, 1])
            
            # Create horizontal error bar plot
            plt.errorbar(hr, y_pos, xerr=[hr - hr_lower, hr_upper - hr], 
                        fmt='o', capsize=5, elinewidth=2, markeredgewidth=2, markersize=8)
            
            plt.axvline(x=1, color='r', linestyle='--', alpha=0.7)
            plt.yticks(y_pos, coef_names)
            plt.xlabel('Hazard Ratio')
            plt.title('Cox PH Model Hazard Ratios with 95% Confidence Intervals')
            plt.grid(True, alpha=0.3)
            plt.xscale('log')  # Log scale often helps visualize hazard ratios
            
            plots.append({
                "title": "Hazard Ratios",
                "img_data": get_base64_plot(),
                "interpretation": "Shows hazard ratios with confidence intervals. A hazard ratio > 1 indicates higher risk, while < 1 indicates lower risk as the predictor increases. If an interval doesn't cross 1 (red line), the predictor is significantly associated with survival outcome."
            })
        except:
            # Skip if this fails
            pass
    
    # Plot 5: Martingale residuals (for Cox model)
    if hasattr(model, 'predict_partial_hazard') and X is not None and has_surv_data:
        try:
            plt.figure(figsize=(12, 6))
            
            # Calculate linear predictor (risk score)
            risk_score = model.predict_partial_hazard(X).values
            
            # Calculate expected number of events
            expected_events = np.zeros_like(time, dtype=float)
            unique_times = np.sort(np.unique(time))
            
            for t in unique_times:
                at_risk = time >= t
                events_at_t = (time == t) & (event == 1)
                
                if np.sum(at_risk) > 0:
                    haz_contrib = events_at_t.sum() * risk_score[at_risk] / risk_score[at_risk].sum()
                    expected_events[at_risk] += haz_contrib
            
            # Martingale residuals
            martingale_resid = event - expected_events
            
            # Plot residuals against risk score
            plt.scatter(np.log(risk_score), martingale_resid, alpha=0.6)
            
            # Add smoothed trend line
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smooth = lowess(martingale_resid, np.log(risk_score), frac=0.6)
                plt.plot(smooth[:, 0], smooth[:, 1], 'r-', linewidth=2)
            except:
                pass
                
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('Log Risk Score (Linear Predictor)')
            plt.ylabel('Martingale Residual')
            plt.title('Martingale Residuals vs Log Risk Score')
            plt.grid(True, alpha=0.3)
            
            plots.append({
                "title": "Martingale Residuals",
                "img_data": get_base64_plot(),
                "interpretation": "Martingale residuals help assess model fit. The residuals should be randomly distributed around zero with no clear pattern. Systematic trends suggest potential non-linearity or missing predictors. Positive residuals indicate more observed events than expected, while negative values indicate fewer events than expected."
            })
        except:
            # Skip if this fails
            pass
    
    # Plot 6: Schoenfeld residuals for proportional hazards assumption (for Cox model)
    if hasattr(model, 'schoenfeld_residuals') and hasattr(model, 'concordance_index'):
        try:
            # Get Schoenfeld residuals
            schoenfeld = model.schoenfeld_residuals
            
            if schoenfeld is not None:
                for col in schoenfeld.columns:
                    plt.figure(figsize=(10, 6))
                    
                    # Get residuals for the current covariate
                    resid = schoenfeld[col]
                    times = schoenfeld.index
                    
                    plt.scatter(times, resid, alpha=0.6)
                    
                    # Add smoothed trend line
                    try:
                        from statsmodels.nonparametric.smoothers_lowess import lowess
                        smooth = lowess(resid, times, frac=0.6)
                        plt.plot(smooth[:, 0], smooth[:, 1], 'r-', linewidth=2)
                    except:
                        pass
                        
                    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                    plt.xlabel('Time')
                    plt.ylabel(f'Schoenfeld Residual for {col}')
                    plt.title(f'Schoenfeld Residuals vs Time for {col}')
                    plt.grid(True, alpha=0.3)
                    
                    plots.append({
                        "title": f"Schoenfeld Residuals for {col}",
                        "img_data": get_base64_plot(),
                        "interpretation": f"Schoenfeld residuals for {col} test the proportional hazards assumption. A horizontal trend (flat red line) supports the assumption. An upward or downward trend suggests a time-varying effect, indicating the proportional hazards assumption may be violated for this variable."
                    })
        except:
            # Skip if this fails
            pass
    
    # Plot 7: Cumulative hazard function for model checking
    if has_surv_data and groups is not None:
        plt.figure(figsize=(12, 7))
        
        unique_groups = np.unique(groups)
        if group_names is None or len(group_names) != len(unique_groups):
            group_names = [f"Group {i+1}" for i in range(len(unique_groups))]
        
        for i, group in enumerate(unique_groups):
            group_mask = (groups == group)
            group_time = time[group_mask]
            group_event = event[group_mask]
            
            # Sort by time
            sort_idx = np.argsort(group_time)
            group_time = group_time[sort_idx]
            group_event = group_event[sort_idx]
            
            times = np.unique(group_time)
            cum_hazard = []
            
            # Simple Nelson-Aalen estimation
            n_total = len(group_time)
            n_at_risk = n_total
            cum_haz = 0
            
            for t in times:
                n_events = np.sum((group_time == t) & (group_event == 1))
                
                if n_at_risk > 0:
                    cum_haz += n_events / n_at_risk
                
                cum_hazard.append(cum_haz)
                n_at_risk -= np.sum(group_time == t)
            
            plt.step(times, cum_hazard, where='post', label=group_names[i], linewidth=2)
        
        plt.title('Nelson-Aalen Cumulative Hazard Estimate')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Hazard')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plots.append({
            "title": "Cumulative Hazard Function",
            "img_data": get_base64_plot(),
            "interpretation": "The cumulative hazard function shows the accumulated risk over time. Steeper slopes indicate higher hazard rates during those time periods. For groups, non-proportional curves may indicate violation of the proportional hazards assumption."
        })
    
    # Plot 8: Model performance metrics (if available)
    if hasattr(model, 'concordance_index') or hasattr(model, 'score'):
        plt.figure(figsize=(8, 6))
        
        # Get metrics
        metrics = {}
        
        if hasattr(model, 'concordance_index'):
            metrics['C-index'] = model.concordance_index
        elif hasattr(model, 'score'):
            try:
                metrics['C-index'] = model.score(X, time, event)
            except:
                pass
        
        if hasattr(model, 'log_likelihood_'):
            metrics['Log-likelihood'] = model.log_likelihood_
        
        if hasattr(model, 'AIC_'):
            metrics['AIC'] = model.AIC_
        
        if metrics:
            # Create simple bar chart
            plt.bar(range(len(metrics)), list(metrics.values()), alpha=0.7)
            plt.xticks(range(len(metrics)), list(metrics.keys()))
            plt.ylabel('Value')
            plt.title('Model Performance Metrics')
            
            # Add text labels
            for i, (k, v) in enumerate(metrics.items()):
                plt.text(i, v * 0.5, f'{v:.3f}', ha='center')
            
            plots.append({
                "title": "Model Performance",
                "img_data": get_base64_plot(),
                "interpretation": "Shows key performance metrics. The C-index (concordance index) measures discriminatory power; values range from 0.5 (random prediction) to 1.0 (perfect prediction). Values around 0.7+ typically indicate good performance. Other metrics like log-likelihood and AIC are useful for model comparison."
            })
    
    return plots 