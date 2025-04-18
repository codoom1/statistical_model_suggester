#!/usr/bin/env python
"""
Comprehensive diagnostic plot generator for all statistical models.

This script generates and saves diagnostic plots for all model types available
in the utils/diagnostic_plots directory. It can generate plots for a specific model
or for all models.
"""
import os
import sys
import json
import argparse
import importlib
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path so we can import our utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def save_plot_from_base64(img_data, filename):
    """Convert base64 image to file and save it
    
    Args:
        img_data: Base64 encoded image string
        filename: Path where to save the image
    """
    if not img_data:
        print(f"Warning: Empty image data for {filename}")
        return
    
    try:
        import base64
        import io
        from PIL import Image
        
        image_data = base64.b64decode(img_data)
        image = Image.open(io.BytesIO(image_data))
        image.save(filename)
        print(f"Saved: {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def generate_sample_data(model_type):
    """Generate sample data for demonstration
    
    Args:
        model_type: Type of model to generate data for
        
    Returns:
        Tuple of data appropriate for the model type
    """
    np.random.seed(42)  # For reproducibility
    
    # Tree-based and boosting models
    if model_type in ['decision_trees', 'xgboost', 'catboost', 'lightgbm', 'gradient_boosting', 'random_forest']:
        # Sample data for tree-based models
        n = 200
        # Features with different distributions
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.exponential(1, n)
        X3 = np.random.uniform(-1, 1, n)
        X4 = np.random.binomial(1, 0.5, n)
        X5 = np.random.poisson(3, n)
        X = np.column_stack([X1, X2, X3, X4, X5])
        
        # Complex non-linear relationship with interaction
        y = (0.8 * X1 + 0.2 * X2**2 + 0.3 * X3 * X4 + 0.4 * np.log1p(X5) + 
             0.5 * np.sin(X1) + np.random.normal(0, 0.5, n))
        
        # Split into train/test
        train_idx = np.random.choice(np.arange(n), size=int(0.7*n), replace=False)
        test_idx = np.array(list(set(range(n)) - set(train_idx)))
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # For classification variant
        if 'class' in model_type or model_type in ['lightgbm_classification']:
            # Convert to binary classification problem
            y_binary = (y > np.median(y)).astype(int)
            y_train = y_binary[train_idx]
            y_test = y_binary[test_idx]
            return X_train, y_train, X_test, y_test, ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"], True
        else:
            # Regression problem
            return X_train, y_train, X_test, y_test, ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"], False
    
    # Linear models
    elif model_type in ['linear_regression', 'ridge_regression', 'lasso_regression', 'elastic_net_regression']:
        # Linear regression sample data
        n = 100
        X = np.random.normal(0, 1, (n, 3))
        # y = intercept + X1 + 2*X2 + 0.5*X3 + noise
        y = 2 + X[:, 0] + 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 1, n)
        
        # Split into train/test
        train_idx = np.random.choice(np.arange(n), size=int(0.7*n), replace=False)
        test_idx = np.array(list(set(range(n)) - set(train_idx)))
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        return X_train, y_train, X_test, y_test, ["Feature 1", "Feature 2", "Feature 3"], False
    
    # Classification models
    elif model_type in ['logistic_regression', 'svm', 'naive_bayes', 'k_nearest_neighbors', 'neural_network']:
        # Classification sample data
        n = 200
        # Generate two features
        X = np.random.normal(0, 1, (n, 2))
        
        # Create a non-linear decision boundary
        y = (np.sin(X[:, 0]) + np.cos(X[:, 1]) > 0).astype(int)
        
        # Split into train/test
        train_idx = np.random.choice(np.arange(n), size=int(0.7*n), replace=False)
        test_idx = np.array(list(set(range(n)) - set(train_idx)))
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        return X_train, y_train, X_test, y_test, ["Feature 1", "Feature 2"], True
    
    # Statistical tests
    elif model_type in ['ttest', 'mann_whitney', 'chi_square', 'kruskal_wallis', 'anova']:
        if model_type == 'ttest' or model_type == 'mann_whitney':
            # Two-sample test data
            group1 = np.random.normal(10, 2, 30)
            group2 = np.random.normal(12, 2, 30)
            return group1, group2, ['Group A', 'Group B']
        
        elif model_type == 'chi_square':
            # Chi-square sample data - 2x3 contingency table
            observed = np.array([
                [45, 30, 25],  # Treatment
                [25, 35, 40]   # Control
            ])
            row_labels = ['Treatment', 'Control']
            col_labels = ['Success', 'Partial', 'Failure']
            return observed, row_labels, col_labels
        
        elif model_type == 'kruskal_wallis':
            # Kruskal-Wallis test data
            group1 = np.random.normal(50, 10, 25)
            group2 = np.random.normal(55, 8, 25)
            group3 = np.random.gamma(shape=7, scale=1.5, size=25) + 45
            return [group1, group2, group3], ['Group A', 'Group B', 'Group C']
        
        elif model_type == 'anova':
            # ANOVA sample data
            data = pd.DataFrame({
                'group': np.repeat(['A', 'B', 'C'], 30),
                'value': np.concatenate([
                    np.random.normal(10, 2, 30),
                    np.random.normal(12, 2, 30),
                    np.random.normal(15, 2, 30)
                ])
            })
            return data, 'value', 'group'
    
    # Time series models
    elif model_type in ['time_series', 'arima']:
        # Time series sample data
        n = 200
        time = np.arange(n)
        
        # Trend component
        trend = 0.05 * time
        
        # Seasonal component (period=20)
        seasonal = 2 * np.sin(2 * np.pi * time / 20)
        
        # Autoregressive component (AR(1) with coefficient 0.8)
        ar = np.zeros(n)
        ar[0] = np.random.normal(0, 1)
        for t in range(1, n):
            ar[t] = 0.8 * ar[t-1] + np.random.normal(0, 0.2)
        
        # Combine components
        y = trend + seasonal + ar + np.random.normal(0, 0.5, n)
        
        return time, y
    
    # Dimensionality reduction
    elif model_type in ['pca', 'factor_analysis', 'multidimensional_scaling']:
        # Generate correlated data
        n = 100
        t = np.linspace(0, 2*np.pi, n)
        x1 = np.sin(t) + np.random.normal(0, 0.1, n)
        x2 = np.cos(t) + np.random.normal(0, 0.1, n)
        x3 = x1 + x2 + np.random.normal(0, 0.1, n)
        X = np.column_stack([x1, x2, x3, np.random.normal(0, 1, (n, 2))])
        feature_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
        return X, feature_names
    
    # Survival analysis
    elif model_type in ['survival_analysis', 'kaplan_meier', 'cox_proportional_hazards']:
        n = 200
        
        # Generate covariates
        X = np.random.normal(0, 1, (n, 3))
        
        # Generate survival times based on covariates
        true_betas = np.array([0.5, -0.3, 0.7])
        baseline_hazard = 0.1
        linear_pred = X @ true_betas
        scale = 1.0 / np.exp(linear_pred)
        
        # Generate Weibull distributed survival times
        shape = 1.5
        survival_times = np.random.weibull(shape, n) * scale
        
        # Generate censoring times - some subjects will have event, others will be censored
        censoring_times = np.random.exponential(10, n)
        
        # Observed time is the minimum of survival time and censoring time
        observed_times = np.minimum(survival_times, censoring_times)
        
        # Event indicator: 1 if the event is observed, 0 if censored
        event = (survival_times <= censoring_times).astype(int)
        
        # Create groups for KM plots
        groups = np.zeros(n, dtype=int)
        groups[X[:, 0] > 0] = 1  # Based on first covariate
        
        # For Cox PH
        if model_type == 'cox_proportional_hazards':
            return X, observed_times, event, ["Feature 1", "Feature 2", "Feature 3"]
        
        # For Kaplan-Meier
        return observed_times, event, groups, ["Group 0", "Group 1"]
    
    # Default case
    print(f"Warning: No specific sample data generator for model_type={model_type}, using generic data")
    X = np.random.normal(0, 1, (100, 2))
    y = np.random.normal(0, 1, 100)
    return X, y

def get_all_plot_modules():
    """Get all available plot modules from the diagnostic_plots directory"""
    
    plot_dir = Path(__file__).parent / 'diagnostic_plots'
    modules = []
    
    for file_path in plot_dir.glob('*.py'):
        if file_path.name.startswith('__') or file_path.name in ['debug.py', 'update_template.py']:
            continue
        
        module_name = file_path.stem
        modules.append(module_name)
    
    print(f"Found {len(modules)} available plot modules: {', '.join(modules)}")
    return modules

def generate_plots_for_model(model_name, output_dir):
    """Generate plots for a specific model
    
    Args:
        model_name: Name of the model (should match a Python module in diagnostic_plots)
        output_dir: Directory to save the plots
    """
    # Create output directory for this model
    model_dir = os.path.join(output_dir, model_name.lower().replace(' ', '_'))
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if module_name is directly in the diagnostic_plots directory
    diagnostic_plots_dir = Path(__file__).parent / 'diagnostic_plots'
    available_modules = [f.stem for f in diagnostic_plots_dir.glob('*.py') 
                         if not f.name.startswith('__') and f.name not in ['debug.py', 'update_template.py']]
    
    # Print available modules for debugging
    print(f"Available modules in diagnostic_plots: {available_modules}")
    
    try:
        # Try direct file path check first
        if model_name.lower() in available_modules:
            module_name = f"utils.diagnostic_plots.{model_name.lower()}"
            try:
                plot_module = importlib.import_module(module_name)
                print(f"Successfully imported {module_name}")
            except ImportError as e:
                print(f"Error importing {module_name}: {e}")
                # Fallback to the original approach
                raise ImportError(f"Could not import {module_name}")
        else:
            # Import the appropriate plot generation module
            module_name = f"utils.diagnostic_plots.{model_name}"
            try:
                plot_module = importlib.import_module(module_name)
            except ImportError:
                print(f"Module {module_name} not found. Trying alternative names...")
                
                # Try common naming variations
                variations = [
                    model_name,
                    model_name.replace(' ', '_'),
                    model_name.replace('-', '_'),
                    model_name.lower(),
                    model_name.lower().replace(' ', '_'),
                    model_name.lower().replace('-', '_')
                ]
                
                for var in variations:
                    try:
                        module_name = f"utils.diagnostic_plots.{var}"
                        print(f"Trying import: {module_name}")
                        plot_module = importlib.import_module(module_name)
                        print(f"Successfully imported {module_name}")
                        break
                    except ImportError as e:
                        print(f"Import failed for {module_name}: {e}")
                        continue
                else:
                    print(f"Could not find a module for {model_name}")
                    return
        
        # Find the appropriate function to call
        plot_function = None
        for name in dir(plot_module):
            if name.startswith('generate_') and name.endswith('_plots'):
                plot_function = getattr(plot_module, name)
                break
        
        if not plot_function:
            print(f"Could not find a plot generation function in {module_name}")
            return
            
        print(f"Generating plots for {model_name}")
        
        # Generate sample data appropriate for this model
        data = generate_sample_data(model_name.lower())
        
        # Get function signature to understand what parameters it expects
        import inspect
        sig = inspect.signature(plot_function)
        param_names = list(sig.parameters.keys())
        
        # Handle different function signatures based on the first parameter
        plots = None
        
        # Try to match the function signature with our available data
        if len(param_names) >= 1:
            # CatBoost and other boosting models-style signatures with many optional parameters
            if param_names[0] == 'model' and 'X_train' in param_names and 'y_train' in param_names:
                if isinstance(data, tuple) and len(data) >= 6 and isinstance(data[4], list):
                    # Unpack the data
                    X_train, y_train, X_test, y_test, feature_names, is_classifier = data
                    
                    # Call the function with the appropriate arguments
                    plots = plot_function(
                        model=None,  # We don't have a trained model
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        feature_names=feature_names,
                        is_classifier=is_classifier
                    )
                else:
                    print(f"Data format does not match expected signature for {model_name}")
            
            # Simple X, y format like linear_regression and logistic_regression
            elif len(param_names) == 2 and param_names[0] == 'X' and param_names[1] == 'y':
                if isinstance(data, tuple) and len(data) >= 4:
                    # Use training data for these models
                    X_train, y_train = data[0], data[1]
                    plots = plot_function(X_train, y_train)
                else:
                    print(f"Data format does not match expected signature for {model_name}")
            
            # Statistical test models with different signatures
            elif model_name.lower() in ['ttest', 'mann_whitney']:
                if isinstance(data, tuple) and len(data) == 3:
                    group1, group2, group_names = data
                    plots = plot_function(group1, group2, group_names)
                
            elif model_name.lower() == 'chi_square':
                if isinstance(data, tuple) and len(data) == 3:
                    observed, row_labels, col_labels = data
                    plots = plot_function(observed, row_labels, col_labels)
                
            elif model_name.lower() == 'kruskal_wallis':
                if isinstance(data, tuple) and len(data) == 2:
                    groups, group_names = data
                    plots = plot_function(groups, group_names)
                
            elif model_name.lower() == 'anova':
                if isinstance(data, tuple) and len(data) == 3:
                    data_df, value_col, group_col = data
                    plots = plot_function(data_df, value_col, group_col)
                
            # Time series models
            elif model_name.lower() in ['time_series', 'arima']:
                if isinstance(data, tuple) and len(data) == 2:
                    time, values = data
                    plots = plot_function(time, values)
                
            # Dimensionality reduction
            elif model_name.lower() in ['pca', 'factor_analysis', 'multidimensional_scaling']:
                if isinstance(data, tuple) and len(data) == 2:
                    X, feature_names = data
                    plots = plot_function(X, feature_names)
                
            # Survival analysis
            elif model_name.lower() == 'cox_proportional_hazards':
                if isinstance(data, tuple) and len(data) == 4:
                    X, times, events, feature_names = data
                    plots = plot_function(X, times, events, feature_names)
                
            elif model_name.lower() == 'kaplan_meier':
                if isinstance(data, tuple) and len(data) == 4:
                    times, events, groups, group_names = data
                    plots = plot_function(times, events, groups, group_names)
            
            # Default case - try to use the input args directly
            else:
                try:
                    plots = plot_function(*data)
                except Exception as e:
                    print(f"Error calling plot function: {e}")
                    print(f"Function expects: {param_names}")
                    print(f"Data provided: {data}")
        
        # If plots is still None, fallback to minimal args
        if plots is None:
            try:
                if len(param_names) == 1 and param_names[0] == 'model':
                    plots = plot_function(None)
                else:
                    print(f"Could not match function signature for {model_name}")
                    return
            except Exception as e:
                print(f"Error with fallback approach: {e}")
                return
        
        # Save plots to files
        for i, plot in enumerate(plots):
            filename = os.path.join(model_dir, f"{i+1}_{plot['title'].replace(' ', '_').lower()}.png")
            save_plot_from_base64(plot.get("img_data", ""), filename)
            
            # Create a JSON file with plot information (title and interpretation)
            json_filename = os.path.join(model_dir, f"{i+1}_{plot['title'].replace(' ', '_').lower()}.json")
            with open(json_filename, 'w') as f:
                json.dump({
                    'title': plot['title'],
                    'interpretation': plot['interpretation']
                }, f, indent=2)
        
        print(f"Successfully generated {len(plots)} plots for {model_name}")
        
    except Exception as e:
        print(f"Error generating plots for {model_name}: {str(e)}")
        print(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description='Generate diagnostic plots for statistical models')
    parser.add_argument('--output', type=str, default='static/diagnostic_plots',
                        help='Directory to save the plots')
    parser.add_argument('--model', type=str, nargs='*',
                        help='Generate plots only for specific model(s) (optional)')
    parser.add_argument('--list', action='store_true',
                        help='List all available models and exit')
    args = parser.parse_args()
    
    # Get all available plot modules
    all_modules = get_all_plot_modules()
    
    # Clean up module names for display
    model_names = [module.replace('_', ' ').title() for module in all_modules]
    
    if args.list:
        print("Available models for plot generation:")
        for i, name in enumerate(sorted(model_names), 1):
            print(f"{i}. {name}")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model:
        # Generate plots only for the specified models
        for model in args.model:
            # Try to match the model name to available modules
            matching = [m for m in all_modules if model.lower() in m.lower()]
            
            if matching:
                for match in matching:
                    generate_plots_for_model(match, args.output)
            else:
                print(f"Model '{model}' not found. Available models: {', '.join(all_modules)}")
    else:
        # Generate plots for all models
        for module in tqdm(all_modules, desc="Generating plots"):
            generate_plots_for_model(module, args.output)
    
    print(f"All plots generated and saved to {args.output}")

if __name__ == "__main__":
    main() 