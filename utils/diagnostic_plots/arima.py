"""ARIMA (AutoRegressive Integrated Moving Average) diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.ticker as ticker

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def generate_arima_plots(model=None, data=None, time_index=None, predictions=None, 
                      residuals=None, forecast=None, forecast_index=None, 
                      order=None, seasonal_order=None):
    """Generate diagnostic plots for ARIMA models
    
    Args:
        model: Fitted ARIMA model (can be from any framework)
        data: Original time series data
        time_index: Time indices for the data
        predictions: In-sample predictions from the model
        residuals: Model residuals
        forecast: Out-of-sample forecasts
        forecast_index: Time indices for the forecast
        order: ARIMA model order (p, d, q)
        seasonal_order: Seasonal ARIMA components (P, D, Q, s)
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Extract data from model if not provided
    if model is not None:
        # Try to extract from a statsmodels ARIMA model
        if hasattr(model, 'fittedvalues') and predictions is None:
            predictions = model.fittedvalues
            
        if hasattr(model, 'resid') and residuals is None:
            residuals = model.resid
            
        if hasattr(model, 'data') and data is None:
            data = model.data.orig_endog
            
        if hasattr(model, 'model') and order is None:
            if hasattr(model.model, 'order'):
                order = model.model.order
                
        if hasattr(model, 'model') and seasonal_order is None:
            if hasattr(model.model, 'seasonal_order'):
                seasonal_order = model.model.seasonal_order
    
    # Plot 1: Original Time Series with Fitted Values
    if data is not None:
        plt.figure(figsize=(12, 6))
        
        # Create time index if not provided
        if time_index is None:
            time_index = np.arange(len(data))
        
        # Plot original data
        plt.plot(time_index, data, 'b-', label='Original Data', alpha=0.7)
        
        # Plot fitted values if available
        if predictions is not None:
            # Make sure predictions align with data
            if len(predictions) < len(data):
                # Handle differencing in ARIMA (first few values will be NaN)
                padding = len(data) - len(predictions)
                pred_index = time_index[padding:]
            else:
                pred_index = time_index
                
            plt.plot(pred_index, predictions, 'r--', label='Fitted Values', linewidth=2)
        
        # Add forecast if available
        if forecast is not None:
            if forecast_index is None:
                # Create forecast index by extending time_index
                last_index = time_index[-1]
                if isinstance(last_index, (int, float)):
                    step = 1 if len(time_index) <= 1 else time_index[-1] - time_index[-2]
                    forecast_index = np.arange(last_index + step, 
                                            last_index + step + len(forecast) * step, 
                                            step)
                else:  # Assume datetime-like index
                    forecast_index = pd.date_range(start=last_index, periods=len(forecast) + 1)[1:]
            
            plt.plot(forecast_index, forecast, 'g--', label='Forecast', linewidth=2)
            
            # Add forecast confidence intervals if available
            if hasattr(model, 'get_forecast') and hasattr(model.get_forecast(), 'conf_int'):
                forecast_obj = model.get_forecast(steps=len(forecast))
                conf_int = forecast_obj.conf_int()
                plt.fill_between(forecast_index, 
                               conf_int.iloc[:, 0], 
                               conf_int.iloc[:, 1], 
                               color='g', alpha=0.1)
        
        plt.title('Time Series with Fitted Values', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add model order annotation if available
        model_info = ''
        if order is not None:
            model_info += f"ARIMA{order}"
        if seasonal_order is not None:
            model_info += f" x {seasonal_order} (seasonal)"
            
        if model_info:
            plt.annotate(model_info, xy=(0.01, 0.01), xycoords='axes fraction', 
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
        plt.tight_layout()
        
        plots.append({
            "title": "Time Series Plot",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the original time series data (blue) with the model's fitted values (red dashed). Any forecast values are shown in green dashed line. Closeness of the fitted line to the original data indicates how well the model captures the time series patterns."
        })
    
    # Plot 2: Residuals Analysis
    if residuals is not None:
        # Standardize residuals for better analysis
        std_residuals = (residuals - residuals.mean()) / residuals.std() if hasattr(residuals, 'std') else (residuals - np.mean(residuals)) / np.std(residuals)
        
        # Create a grid of residual plots
        fig = plt.figure(figsize=(14, 12))
        gs = GridSpec(3, 2, figure=fig)
        
        # Plot 2a: Residuals Time Series
        ax1 = fig.add_subplot(gs[0, :])
        
        if time_index is not None and len(time_index) == len(residuals):
            res_index = time_index
        else:
            res_index = np.arange(len(residuals))
            
        ax1.plot(res_index, residuals, 'b-')
        ax1.axhline(y=0, color='r', linestyle='--')
        
        # Add horizontal lines at ±2 std dev for visual reference
        if hasattr(residuals, 'std'):
            std_dev = residuals.std()
        else:
            std_dev = np.std(residuals)
            
        ax1.axhline(y=2*std_dev, color='r', linestyle='--', alpha=0.3)
        ax1.axhline(y=-2*std_dev, color='r', linestyle='--', alpha=0.3)
        
        ax1.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Residual')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2b: Residual Histogram
        ax2 = fig.add_subplot(gs[1, 0])
        sns.histplot(residuals, kde=True, ax=ax2)
        
        # Add normal distribution overlay
        if hasattr(residuals, 'mean') and hasattr(residuals, 'std'):
            x = np.linspace(residuals.min(), residuals.max(), 100)
            y = np.exp(-(x - residuals.mean())**2 / (2 * residuals.std()**2)) / (residuals.std() * np.sqrt(2 * np.pi))
            ax2.plot(x, y * len(residuals) * (residuals.max() - residuals.min()) / 30, 'r-')
        
        ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Frequency')
        
        # Plot 2c: Residual QQ Plot
        ax3 = fig.add_subplot(gs[1, 1])
        
        from scipy import stats
        stats.probplot(std_residuals, dist="norm", plot=ax3)
        ax3.set_title('Residual QQ Plot', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 2d: Residual ACF
        ax4 = fig.add_subplot(gs[2, 0])
        try:
            plot_acf(residuals, ax=ax4, lags=min(40, len(residuals)//2 - 1), alpha=0.05)
            ax4.set_title('Residual ACF', fontsize=12, fontweight='bold')
        except:
            # Fallback if plot_acf fails
            residual_acf = acf(residuals, nlags=min(40, len(residuals)//2 - 1))
            ax4.bar(range(len(residual_acf)), residual_acf)
            ax4.axhline(y=0, color='r', linestyle='--')
            
            # Add confidence intervals (approx. 95%)
            ci = 1.96 / np.sqrt(len(residuals))
            ax4.axhline(y=ci, color='r', linestyle='--', alpha=0.3)
            ax4.axhline(y=-ci, color='r', linestyle='--', alpha=0.3)
            
            ax4.set_title('Residual ACF', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Lag')
            ax4.set_ylabel('Autocorrelation')
        
        # Plot 2e: Residual PACF
        ax5 = fig.add_subplot(gs[2, 1])
        try:
            plot_pacf(residuals, ax=ax5, lags=min(40, len(residuals)//2 - 1), alpha=0.05, method='ywm')
            ax5.set_title('Residual PACF', fontsize=12, fontweight='bold')
        except:
            # Fallback if plot_pacf fails
            residual_pacf = pacf(residuals, nlags=min(40, len(residuals)//2 - 1), method='ywunbiased')
            ax5.bar(range(len(residual_pacf)), residual_pacf)
            ax5.axhline(y=0, color='r', linestyle='--')
            
            # Add confidence intervals (approx. 95%)
            ci = 1.96 / np.sqrt(len(residuals))
            ax5.axhline(y=ci, color='r', linestyle='--', alpha=0.3)
            ax5.axhline(y=-ci, color='r', linestyle='--', alpha=0.3)
            
            ax5.set_title('Residual PACF', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Lag')
            ax5.set_ylabel('Partial Autocorrelation')
            
        plt.tight_layout()
        
        plots.append({
            "title": "Residual Analysis",
            "img_data": get_base64_plot(),
            "interpretation": "Multiple plots examining model residuals. For a good model: (1) residuals should fluctuate randomly around zero with no patterns, (2) follow a normal distribution, (3) fall along the straight line in the QQ plot, and (4) have no significant autocorrelation in ACF/PACF plots. Significant spikes in the ACF/PACF may indicate that the model isn't capturing all patterns in the data."
        })
    
    # Plot 3: ACF and PACF of Original Data
    if data is not None:
        plt.figure(figsize=(12, 10))
        
        # Create two subplots
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        
        try:
            # Plot ACF of original data
            plot_acf(data, ax=ax1, lags=min(40, len(data)//2 - 1), alpha=0.05)
            ax1.set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
        except:
            # Fallback
            data_acf = acf(data, nlags=min(40, len(data)//2 - 1))
            ax1.bar(range(len(data_acf)), data_acf)
            ax1.axhline(y=0, color='r', linestyle='--')
            
            # Add confidence intervals (approx. 95%)
            ci = 1.96 / np.sqrt(len(data))
            ax1.axhline(y=ci, color='r', linestyle='--', alpha=0.3)
            ax1.axhline(y=-ci, color='r', linestyle='--', alpha=0.3)
            
            ax1.set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Lag')
            ax1.set_ylabel('Autocorrelation')
            
        try:
            # Plot PACF of original data
            plot_pacf(data, ax=ax2, lags=min(40, len(data)//2 - 1), alpha=0.05, method='ywm')
            ax2.set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
        except:
            # Fallback
            data_pacf = pacf(data, nlags=min(40, len(data)//2 - 1), method='ywunbiased')
            ax2.bar(range(len(data_pacf)), data_pacf)
            ax2.axhline(y=0, color='r', linestyle='--')
            
            # Add confidence intervals (approx. 95%)
            ci = 1.96 / np.sqrt(len(data))
            ax2.axhline(y=ci, color='r', linestyle='--', alpha=0.3)
            ax2.axhline(y=-ci, color='r', linestyle='--', alpha=0.3)
            
            ax2.set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Lag')
            ax2.set_ylabel('Partial Autocorrelation')
            
        plt.tight_layout()
        
        # Add model order annotation
        model_info = ''
        if order is not None:
            p, d, q = order
            ar_info = f"AR: {p} lag{'s' if p != 1 else ''}" if p > 0 else ""
            i_info = f"I: order {d}" if d > 0 else ""
            ma_info = f"MA: {q} lag{'s' if q != 1 else ''}" if q > 0 else ""
            
            model_components = [comp for comp in [ar_info, i_info, ma_info] if comp]
            model_info = ", ".join(model_components)
            
        # Provide interpretation hint if order is available
        hint = ""
        if order is not None:
            p, d, q = order
            
            if any(x > 0 for x in [p, d, q]):
                hint = "In this model: "
                
                if p > 0:
                    hint += f"AR({p}) uses {p} past values for prediction"
                    if d > 0 or q > 0:
                        hint += ", "
                    
                if d > 0:
                    hint += f"I({d}) applies {d}-order differencing"
                    if q > 0:
                        hint += ", "
                        
                if q > 0:
                    hint += f"MA({q}) uses {q} past error terms"
                    
                hint += "."
        
        plots.append({
            "title": "ACF and PACF Analysis",
            "img_data": get_base64_plot(),
            "interpretation": "Shows autocorrelation (ACF) and partial autocorrelation (PACF) of the original time series. These help identify appropriate ARIMA model orders. Significant spikes in ACF suggest MA terms, while significant spikes in PACF suggest AR terms. " + (model_info + ". " if model_info else "") + hint
        })
    
    # Plot 4: Forecast Plot
    if forecast is not None:
        plt.figure(figsize=(12, 6))
        
        # Create time indices for both data and forecast
        if time_index is None:
            time_index = np.arange(len(data)) if data is not None else None
            
        if forecast_index is None and time_index is not None:
            # Create forecast index by extending time_index
            last_index = time_index[-1]
            if isinstance(last_index, (int, float)):
                step = 1 if len(time_index) <= 1 else time_index[-1] - time_index[-2]
                forecast_index = np.arange(last_index + step, 
                                        last_index + step + len(forecast) * step, 
                                        step)
            else:  # Assume datetime-like index
                forecast_index = pd.date_range(start=last_index, periods=len(forecast) + 1)[1:]
                
        # Plot original data if available
        if data is not None and time_index is not None:
            plt.plot(time_index, data, 'b-', label='Historical Data', alpha=0.7)
            
            # Add vertical line at the forecast start
            plt.axvline(x=time_index[-1], color='k', linestyle='--', alpha=0.5)
            
        # Plot forecast
        plt.plot(forecast_index, forecast, 'g-', label='Forecast', linewidth=2)
        
        # Add confidence intervals if available
        if hasattr(model, 'get_forecast') and hasattr(model.get_forecast(), 'conf_int'):
            forecast_obj = model.get_forecast(steps=len(forecast))
            conf_int = forecast_obj.conf_int()
            
            plt.fill_between(forecast_index, 
                           conf_int.iloc[:, 0], 
                           conf_int.iloc[:, 1], 
                           color='g', alpha=0.2,
                           label='95% Confidence Interval')
                           
        plt.title('Forecast Plot', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis for better readability if using date labels
        if isinstance(time_index, pd.DatetimeIndex) or isinstance(forecast_index, pd.DatetimeIndex):
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(10))
            plt.gcf().autofmt_xdate()
            
        plt.tight_layout()
        
        plots.append({
            "title": "Forecast Plot",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the model's forecast (green) beyond the historical data (blue). The dashed vertical line marks the boundary between historical data and forecasts. The shaded area (if present) shows the 95% confidence interval, indicating uncertainty in the forecast."
        })
    
    # Plot 5: Model Information
    if order is not None or seasonal_order is not None:
        plt.figure(figsize=(8, 6))
        
        # Create model information table
        model_info = []
        model_info.append(['Component', 'Value', 'Interpretation'])
        
        # Add ARIMA components
        if order is not None:
            p, d, q = order
            model_info.append(['AR order (p)', str(p), 'Number of autoregressive lags'])
            model_info.append(['Integration order (d)', str(d), 'Number of differencing operations'])
            model_info.append(['MA order (q)', str(q), 'Number of moving average lags'])
            
        # Add seasonal components
        if seasonal_order is not None:
            P, D, Q, s = seasonal_order
            model_info.append(['Seasonal AR (P)', str(P), 'Number of seasonal AR terms'])
            model_info.append(['Seasonal differencing (D)', str(D), 'Order of seasonal differencing'])
            model_info.append(['Seasonal MA (Q)', str(Q), 'Number of seasonal MA terms'])
            model_info.append(['Seasonal period (s)', str(s), 'Number of observations per seasonal cycle'])
            
        # Add model fit metrics if available
        if hasattr(model, 'aic'):
            model_info.append(['AIC', f"{model.aic:.2f}", 'Akaike Information Criterion (lower is better)'])
            
        if hasattr(model, 'bic'):
            model_info.append(['BIC', f"{model.bic:.2f}", 'Bayesian Information Criterion (lower is better)'])
            
        if hasattr(model, 'mse'):
            model_info.append(['MSE', f"{model.mse:.4f}", 'Mean Squared Error'])
            
        # Create table
        plt.axis('off')
        
        tbl = plt.table(cellText=model_info[1:], colLabels=model_info[0],
                       loc='center', cellLoc='center')
        
        # Format table
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.2, 1.5)
        
        # Add header color
        for j in range(len(model_info[0])):
            tbl[(0, j)].set_facecolor('#4285f4')
            tbl[(0, j)].set_text_props(weight='bold', color='white')
            
        # Color rows for ARIMA and seasonal components
        if order is not None and seasonal_order is not None:
            # ARIMA parameters
            for i in range(1, 4):
                tbl[(i, 0)].set_facecolor('#e6f2ff')
                tbl[(i, 1)].set_facecolor('#e6f2ff')
                tbl[(i, 2)].set_facecolor('#e6f2ff')
                
            # Seasonal parameters
            for i in range(4, 8):
                tbl[(i, 0)].set_facecolor('#fff2e6')
                tbl[(i, 1)].set_facecolor('#fff2e6')
                tbl[(i, 2)].set_facecolor('#fff2e6')
                
        plt.title('ARIMA Model Information', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plots.append({
            "title": "Model Information",
            "img_data": get_base64_plot(),
            "interpretation": "Summarizes the ARIMA model configuration. AR terms capture the relationship with past values, MA terms model the influence of past errors, and differencing (I) removes trends to make the series stationary. Lower AIC/BIC values indicate better model fit."
        })
    
    return plots 