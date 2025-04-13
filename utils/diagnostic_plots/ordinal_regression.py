"""Ordinal regression diagnostic plots."""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import io
import base64
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression

def get_base64_plot():
    """Convert current matplotlib plot to base64 string"""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode('utf-8')

def save_plot(plot_name):
    """Save the current plot to the appropriate directory"""
    # Create directory if it doesn't exist
    plot_dir = Path("static/diagnostic_plots/ordinal_regression")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(plot_dir / f"{plot_name}.png", dpi=100, bbox_inches='tight')
    plt.close()

def generate_ordinal_regression_plots(X, y):
    """Generate diagnostic plots for Ordinal regression
    
    Args:
        X: Features (numpy array)
        y: Target variable (numpy array) - ordinal categorical data
        
    Returns:
        List of dictionaries with plot information
    """
    # Debug info
    print("DEBUG: y data type:", type(y))
    if hasattr(y, 'dtype'):
        print("DEBUG: y dtype:", y.dtype)
    elif hasattr(y, 'dtypes'):
        print("DEBUG: y dtypes:", y.dtypes)
    print("DEBUG: First 5 y values:", y[:5] if hasattr(y, '__getitem__') else "Cannot slice")
    
    # Convert y to array if it's a pandas Series
    if hasattr(y, 'values'):
        y_array = y.values
    else:
        y_array = np.array(y)
    
    print("DEBUG: y_array type:", type(y_array))
    print("DEBUG: y_array dtype:", y_array.dtype)
    print("DEBUG: y_array shape:", y_array.shape)
    print("DEBUG: Unique values in y_array:", np.unique(y_array))
    
    # If y appears to be continuous, discretize it into ordinal categories
    if y_array.dtype.kind in ['f', 'i'] and len(np.unique(y_array)) > 10:
        print("DEBUG: Converting continuous target to ordinal categories...")
        # Use quantiles to create balanced categories
        n_bins = 5  # Number of ordinal categories
        bins = np.percentile(y_array, np.linspace(0, 100, n_bins+1))
        print("DEBUG: Bins:", bins)
        binned_y = np.digitize(y_array, bins[:-1])
        print("DEBUG: Binned y (before adjustment):", np.unique(binned_y))
        y_array = binned_y - 1  # Start from 0
        print("DEBUG: Created ordinal categories:", np.unique(y_array))
    
    # Force conversion to integer type for classification
    y_array = y_array.astype(int)
    print("DEBUG: Final y_array:", np.unique(y_array))
    
    # Get unique ordered categories
    categories = np.sort(np.unique(y_array))
    n_categories = len(categories)
    
    # Check if X has a constant column and remove it if needed
    if hasattr(X, 'columns') and 'const' in X.columns:
        X_model = X.drop('const', axis=1)
    elif isinstance(X, np.ndarray) and X.shape[1] > 1 and np.isclose(np.std(X[:, 0]), 0):
        X_model = X[:, 1:]
    else:
        X_model = X
    
    plots = []
    
    try:
        # Use scikit-learn's LogisticRegression as a substitute
        # This is not a true ordinal regression, but can serve for diagnostics
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        model.fit(X_model, y_array)
        
        # Get predicted probabilities for each category
        predicted_probs = model.predict_proba(X_model)
        predicted_y = model.predict(X_model)
        
        # Plot 1: Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_array, predicted_y)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=categories, yticklabels=categories)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plots.append({
            "title": "Confusion Matrix",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the count of actual vs. predicted categories. Diagonal elements represent correct predictions."
        })
        save_plot("confusion_matrix")
        
        # Plot 2: Classification Accuracy by Category
        plt.figure(figsize=(10, 6))
        accuracy_by_cat = []
        for i, cat in enumerate(categories):
            mask = (y_array == cat)
            if np.sum(mask) > 0:  # Avoid division by zero
                accuracy = np.mean(predicted_y[mask] == y_array[mask])
                accuracy_by_cat.append((cat, accuracy))
        
        # Convert to DataFrame for plotting
        acc_df = pd.DataFrame(accuracy_by_cat, columns=['Category', 'Accuracy'])
        plt.bar(acc_df['Category'].astype(str), acc_df['Accuracy'])
        plt.ylim(0, 1.1)
        for i, acc in enumerate(acc_df['Accuracy']):
            plt.text(i, acc + 0.05, f'{acc:.2f}', ha='center')
        
        plt.xlabel('Category')
        plt.ylabel('Accuracy')
        plt.title('Classification Accuracy by Category')
        plots.append({
            "title": "Classification Accuracy by Category",
            "img_data": get_base64_plot(),
            "interpretation": "Shows how well the model predicts each category. Lower accuracy for specific categories suggests they are harder to predict."
        })
        save_plot("accuracy_by_category")
        
        # Plot 3: Coefficient plot (coefficients for each category)
        plt.figure(figsize=(12, 8))
        coefs = model.coef_
        feature_names = X_model.columns if hasattr(X_model, 'columns') else [f'X{i}' for i in range(X_model.shape[1])]
        
        # Plot coefficients for each category
        for i, cat in enumerate(model.classes_):
            if i >= len(coefs):  # Skip reference class in some models
                continue
            plt.barh(np.arange(len(feature_names)) + i*0.1, coefs[i], height=0.1, label=f'Category {cat}')
        
        plt.yticks(np.arange(len(feature_names)), feature_names)
        plt.axvline(x=0, color='r', linestyle='-')
        plt.xlabel('Coefficient Value')
        plt.title('Coefficient Plot by Category')
        plt.legend()
        plots.append({
            "title": "Coefficient Plot",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the impact of each feature on the probability of each category. Positive values increase the probability, negative values decrease it."
        })
        save_plot("coefficient_plot")
        
        # Plot 4: Observed vs Predicted Categories
        plt.figure(figsize=(10, 8))
        observed_counts = np.bincount(y_array.astype(int), minlength=len(categories))
        predicted_counts = np.bincount(predicted_y.astype(int), minlength=len(categories))
        
        # Make sure we only include categories that actually appear in the data
        x_pos = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x_pos - width/2, observed_counts[:len(categories)], width, label='Observed')
        plt.bar(x_pos + width/2, predicted_counts[:len(categories)], width, label='Predicted')
        
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.title('Observed vs Predicted Category Counts')
        plt.xticks(x_pos, categories)
        plt.legend()
        plots.append({
            "title": "Observed vs Predicted Category Counts",
            "img_data": get_base64_plot(),
            "interpretation": "Compares the distribution of observed vs predicted categories. Large differences suggest model bias toward certain categories."
        })
        save_plot("observed_vs_predicted")
        
        # Plot 5: Proportional Odds Assumption Check using separate binary logits
        plt.figure(figsize=(12, 8))
        
        binary_coefs = []
        binary_names = []
        
        # For each cutpoint, create a binary logit model
        for cutpoint in range(len(categories)-1):
            # Convert to binary: 1 if y > cutpoint, 0 otherwise
            binary_y = (y_array > categories[cutpoint]).astype(int)
            try:
                logit_model = sm.Logit(binary_y, sm.add_constant(X_model)).fit(disp=False)
                coefs = logit_model.params[1:]  # Skip intercept
                for i, coef in enumerate(coefs):
                    binary_coefs.append(coef)
                    name = feature_names[i] if i < len(feature_names) else f'X{i}'
                    binary_names.append(f"{name}_cut{cutpoint}")
            except Exception as e:
                print(f"Skipping cutpoint {cutpoint} due to error: {e}")
                continue
        
        # Convert to array for plotting
        if binary_coefs:
            binary_coefs = np.array(binary_coefs)
            y_pos = np.arange(len(binary_coefs))
            
            plt.barh(y_pos, binary_coefs)
            plt.yticks(y_pos, binary_names)
            plt.xlabel('Coefficient value')
            plt.title('Proportional Odds Assumption Check')
            plots.append({
                "title": "Proportional Odds Assumption Check",
                "img_data": get_base64_plot(),
                "interpretation": "Checks the proportional odds assumption. If coefficients for each predictor are similar across cutpoints, the assumption is valid. Large differences suggest a violation."
            })
            save_plot("proportional_odds_check")
        
        # Plot 6: Prediction Probability Distribution
        plt.figure(figsize=(12, 8))
        for i, cat in enumerate(categories):
            # Get predicted probabilities for this category
            if i < predicted_probs.shape[1]:
                probs = predicted_probs[:, i]
                plt.hist(probs, bins=20, alpha=0.5, label=f'Category {cat}')
        
        plt.xlabel('Predicted probability')
        plt.ylabel('Count')
        plt.title('Prediction Probability Distribution by Category')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plots.append({
            "title": "Prediction Probability Distribution",
            "img_data": get_base64_plot(),
            "interpretation": "Shows the distribution of predicted probabilities for each category. Well-separated distributions indicate the model can distinguish categories effectively."
        })
        save_plot("prediction_probability")
            
        return plots
        
    except Exception as e:
        import traceback
        error_message = f"Could not generate ordinal regression plots: {str(e)}\n\n"
        error_message += traceback.format_exc()
        print(f"Error in ordinal regression diagnostics:\n{error_message}")
        
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, error_message, 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=10, wrap=True)
        plt.axis('off')
        error_img = get_base64_plot()
        save_plot("error_plot")
        
        return [{
            "title": "Error in Ordinal Regression Diagnostics",
            "img_data": error_img,
            "interpretation": f"An error occurred while generating the plots: {str(e)}"
        }] 