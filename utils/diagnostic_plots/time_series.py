"""Time series model diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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

def generate_time_series_plots(model, y=None, y_pred=None, residuals=None, 
                              train_pred=None, test_pred=None, test_actual=None,
                              forecast=None, forecast_index=None, index=None,
                              interval=None):
    """Generate diagnostic plots for time series models
    
    Args:
        model: Fitted time series model (ARIMA, SARIMA, etc.)
        y: Original time series data
        y_pred: Predicted values from model
        residuals: Model residuals
        train_pred: Predictions on training data
        test_pred: Predictions on test data
        test_actual: Actual values for test data
        forecast: Future forecasted values
        forecast_index: Index for forecast values
        index: Time index for original data
        interval: Prediction intervals for forecasts
        
    Returns:
        List of dictionaries with plot information
    """
    plots = []
    
    # Try to extract residuals if not provided
    if residuals is None and hasattr(model, 'resid'):
        residuals = model.resid
    
    # Try to extract predictions if not provided
    if y_pred is None and hasattr(model, 'fittedvalues'):
        y_pred = model.fittedvalues
    
    # Create time index if not provided
    if index is None and y is not None:
        if hasattr(y, 'index'):
            index = y.index
        else:
            index = np.arange(len(y))
    
    # Create forecast index if not provided
    if forecast_index is None and forecast is not None:
        if hasattr(index, 'max'):
            # Create future dates
            last_date = index.max()
            if hasattr(last_date, 'freq'):
                forecast_index = pd.date_range(start=last_date + last_date.freq, 
                                             periods=len(forecast),
                                             freq=last_date.freq)
            else:
                forecast_index = np.arange(len(index), len(index) + len(forecast))
        else:
            forecast_index = np.arange(len(index), len(index) + len(forecast))
    
    # Plot 1: Original Series and Fitted Values
    if y is not None and (y_pred is not None or train_pred is not None):
        plt.figure(figsize=(12, 6))
        
        # Plot original data
        plt.plot(index, y, 'b-', label='Original Series', alpha=0.7)
        
        # Plot fitted values
        if y_pred is not None:
            plt.plot(index, y_pred, 'r-', label='Fitted Values', alpha=0.7)
        
        # Or plot train/test predictions if available
        if train_pred is not None:
            split_point = len(train_pred)
            plt.plot(index[:split_point], train_pred, 'r-', label='In-Sample Predictions', alpha=0.7)
            
            if test_pred is not None and test_actual is not None:
                plt.plot(index[split_point:split_point+len(test_pred)], test_pred, 'g-', 
                        label='Out-of-Sample Predictions', alpha=0.7)
                plt.plot(index[split_point:split_point+len(test_actual)], test_actual, 'b-', 
                        label='Test Data', alpha=0.7)
        
        # Add forecast if available
        if forecast is not None and forecast_index is not None:
            plt.plot(forecast_index, forecast, 'g--', label='Forecast', alpha=0.7)
            
            # Add prediction intervals if available
            if interval is not None:
                lower = interval[0]
                upper = interval[1]
                plt.fill_between(forecast_index, lower, upper, color='g', alpha=0.2, 
                               label='Prediction Interval')
        
        plt.title('Time Series and Model Fit')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Time Series and Model Fit",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the original time series data and the model's fitted values. A good model should track the original series closely, capturing its patterns without overfitting to noise."
        })
    
    # Plot 2: Residual Analysis
    if residuals is not None:
        # Create a 2x2 subplot layout for residual analysis
        fig = plt.figure(figsize=(14, 12))
        gs = GridSpec(2, 2, figure=fig)
        
        # Plot 2.1: Residuals over time
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(index if index is not None else np.arange(len(residuals)), 
                residuals, 'o-', markersize=3, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax1.set_title('Residuals Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Residual')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2.2: Residual Histogram
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(residuals, kde=True, ax=ax2, alpha=0.6)
        ax2.set_title('Residual Distribution')
        ax2.set_xlabel('Residual')
        
        # Plot 2.3: ACF of Residuals
        ax3 = fig.add_subplot(gs[1, 0])
        plot_acf(residuals, ax=ax3, lags=40, alpha=0.05)
        ax3.set_title('Autocorrelation of Residuals (ACF)')
        
        # Plot 2.4: Residual Q-Q Plot
        ax4 = fig.add_subplot(gs[1, 1])
        qq = ax4.plot(np.sort(np.random.normal(0, 1, len(residuals))), 
                     np.sort(residuals), 'o', alpha=0.6)
        
        # Add reference line
        min_val = min(np.min(residuals), -3)
        max_val = max(np.max(residuals), 3)
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        ax4.set_title('Normal Q-Q Plot of Residuals')
        ax4.set_xlabel('Theoretical Quantiles')
        ax4.set_ylabel('Sample Quantiles')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plots.append({
            "title": "Residual Analysis",
            "img_data": get_base64_plot(),
            "interpretation": "Four diagnostic plots for residuals: (1) Residuals over time should show no pattern, (2) Histogram should approximate a normal distribution, (3) ACF should show no significant autocorrelation (spikes should stay within blue bands), and (4) Q-Q plot should follow the diagonal line if residuals are normally distributed."
        })
    
    # Plot 3: Seasonal Decomposition (if available in model or can be computed)
    if hasattr(model, 'seasonal') or (y is not None and len(y) >= 4):
        try:
            # Try to get decomposition from model or compute it
            trend = None
            seasonal = None
            
            if hasattr(model, 'trend'):
                trend = model.trend
            if hasattr(model, 'seasonal'):
                seasonal = model.seasonal
            
            # If not available in model, try to compute for simple cases
            if (trend is None or seasonal is None) and y is not None:
                # Simple moving average for trend
                window_size = min(12, len(y) // 4)
                if window_size % 2 == 0:
                    window_size += 1  # Make it odd
                
                if window_size >= 3:
                    y_array = np.array(y) if not isinstance(y, np.ndarray) else y
                    trend = pd.Series(y_array).rolling(window=window_size, center=True).mean()
                    
                    # Extract approximate seasonality (very simplified)
                    if hasattr(y, 'index') and hasattr(y.index, 'month'):
                        # For monthly data
                        seasonal_groups = pd.Series(y_array).groupby(y.index.month).mean()
                        seasonal = np.array([seasonal_groups[m] for m in y.index.month])
                    
                    if trend is not None:
                        plt.figure(figsize=(12, 10))
                        
                        # Original data
                        plt.subplot(311)
                        plt.plot(index, y, label='Original')
                        plt.title('Original Time Series')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        # Trend
                        plt.subplot(312)
                        plt.plot(index, trend, label='Trend', color='red')
                        plt.title('Trend Component')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        # Seasonal and Residual
                        plt.subplot(313)
                        if seasonal is not None:
                            plt.plot(index, seasonal, label='Seasonal', color='green')
                            plt.title('Seasonal Component')
                        else:
                            plt.plot(index, y - trend, label='Remainder', color='purple')
                            plt.title('Remainder Component (Seasonal + Residual)')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        
                        plots.append({
                            "title": "Time Series Decomposition",
                            "img_data": get_base64_plot(),
                            "interpretation": "Decomposes the time series into its trend and other components. The trend shows the long-term progression, while the remainder captures seasonal patterns and residual noise."
                        })
        except:
            # Skip decomposition if it fails
            pass
    
    # Plot 4: Autocorrelation and Partial Autocorrelation Functions
    if y is not None:
        plt.figure(figsize=(12, 10))
        
        # ACF
        plt.subplot(211)
        plot_acf(y, lags=40, alpha=0.05, ax=plt.gca())
        plt.title('Autocorrelation Function (ACF)')
        
        # PACF
        plt.subplot(212)
        plot_pacf(y, lags=40, alpha=0.05, ax=plt.gca())
        plt.title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        
        plots.append({
            "title": "Autocorrelation Functions",
            "img_data": get_base64_plot(),
            "interpretation": "ACF and PACF help identify appropriate orders for ARIMA models. For AR(p) processes, PACF cuts off after lag p, while for MA(q) processes, ACF cuts off after lag q. Significant spikes at seasonal lags suggest seasonal patterns."
        })
    
    # Plot 5: Forecast Evaluation (if test data and predictions are available)
    if test_actual is not None and test_pred is not None:
        plt.figure(figsize=(12, 6))
        
        # Scatter plot of actual vs predicted
        plt.scatter(test_actual, test_pred, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(min(test_actual), min(test_pred))
        max_val = max(max(test_actual), max(test_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Forecast Evaluation: Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Calculate forecast error metrics
        mae = np.mean(np.abs(test_actual - test_pred))
        rmse = np.sqrt(np.mean((test_actual - test_pred) ** 2))
        mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100 if np.all(test_actual != 0) else np.nan
        
        # Add metric annotations
        plt.annotate(f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.2f}%', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plots.append({
            "title": "Forecast Evaluation",
            "img_data": get_base64_plot(),
            "interpretation": f"Compares predicted values against actual values for the test period. Points closer to the diagonal red line indicate better predictions. MAE (Mean Absolute Error): {mae:.4f}, RMSE (Root Mean Squared Error): {rmse:.4f}, MAPE (Mean Absolute Percentage Error): {mape:.2f}%."
        })
    
    # Plot 6: Forecast with Prediction Intervals (if available)
    if forecast is not None and forecast_index is not None:
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        if y is not None:
            plt.plot(index, y, 'b-', label='Historical Data', alpha=0.7)
        
        # Plot forecast
        plt.plot(forecast_index, forecast, 'g-', label='Forecast', linewidth=2)
        
        # Add prediction intervals if available
        if interval is not None:
            lower = interval[0]
            upper = interval[1]
            plt.fill_between(forecast_index, lower, upper, color='g', alpha=0.2, 
                           label='95% Prediction Interval')
        
        plt.title('Forecast with Prediction Intervals')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plots.append({
            "title": "Forecast with Prediction Intervals",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the model's forecast with uncertainty represented by prediction intervals. Wider intervals indicate higher uncertainty in the forecast. The historical data is shown for context."
        })
    
    # Plot 7: Forecast Error Distribution (if test data available)
    if test_actual is not None and test_pred is not None:
        plt.figure(figsize=(10, 6))
        
        # Calculate forecast errors
        errors = test_actual - test_pred
        
        # Plot error distribution
        sns.histplot(errors, kde=True, alpha=0.6)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        plt.title('Forecast Error Distribution')
        plt.xlabel('Forecast Error')
        plt.ylabel('Frequency')
        
        # Add summary statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        plt.annotate(f'Mean Error: {mean_error:.4f}\nStd Dev: {std_error:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plots.append({
            "title": "Forecast Error Distribution",
            "img_data": get_base64_plot(),
            "interpretation": f"Shows the distribution of forecast errors. Ideally, errors should be normally distributed around zero (red line), indicating unbiased forecasts. Mean Error: {mean_error:.4f}, Standard Deviation: {std_error:.4f}."
        })
    
    return plots 