#!/usr/bin/env python
"""
Standalone diagnostic plot generator for a specific model.

This script can generate plots for a specific model type without
depending on other diagnostic plot modules.
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
    
    # Default case
    print(f"Using generic data for model_type={model_type}")
    X = np.random.normal(0, 1, (100, 2))
    y = np.random.normal(0, 1, 100)
    return X, y

def generate_plots_for_model(model_name, output_dir, module_path=None):
    """Generate plots for a specific model
    
    Args:
        model_name: Name of the model
        output_dir: Directory to save the plots
        module_path: Optional direct path to the module file
    """
    # Create output directory for this model
    model_dir = os.path.join(output_dir, model_name.lower().replace(' ', '_'))
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Try direct import of module
        if module_path:
            # Add directory to path
            module_dir = os.path.dirname(module_path)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            
            # Import module directly
            spec = importlib.util.spec_from_file_location(model_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the plot function
            plot_function = None
            for name in dir(module):
                if name.startswith('generate_') and name.endswith('_plots'):
                    plot_function = getattr(module, name)
                    break
            
            if not plot_function:
                print(f"Could not find a plot generation function in {module_path}")
                return
        else:
            # Import the appropriate plot generation module
            module_name = f"utils.diagnostic_plots.{model_name.lower()}"
            
            try:
                plot_module = importlib.import_module(module_name)
            except ImportError as e:
                print(f"Error importing {module_name}: {e}")
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
        
        # Handle different function signatures
        plots = None
        
        # Try to match the function signature with our available data
        if len(param_names) >= 1:
            # CatBoost and other boosting models-style signatures
            if param_names[0] == 'model' and 'X_train' in param_names:
                if isinstance(data, tuple) and len(data) >= 6:
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
                
            # Simple X, y format like linear_regression
            elif len(param_names) == 2 and param_names[0] == 'X' and param_names[1] == 'y':
                if isinstance(data, tuple) and len(data) >= 2:
                    X, y = data[0], data[1]
                    plots = plot_function(X, y)
            
            # Default case - try to use the input args directly
            else:
                try:
                    plots = plot_function(*data)
                except Exception as e:
                    print(f"Error calling plot function: {e}")
                    print(f"Function expects: {param_names}")
                    # Try with None for model
                    if len(param_names) == 1 and param_names[0] == 'model':
                        plots = plot_function(None)
        
        # Save plots to files
        if plots:
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
        else:
            print(f"No plots were generated for {model_name}")
        
    except Exception as e:
        print(f"Error generating plots for {model_name}: {str(e)}")
        print(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(description='Generate diagnostic plots for a specific statistical model')
    parser.add_argument('model', type=str, help='Name of the model (e.g., ridge_regression)')
    parser.add_argument('--output', type=str, default='static/diagnostic_plots',
                        help='Directory to save the plots')
    parser.add_argument('--module_path', type=str, 
                        help='Optional direct path to the module file')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots for the specified model
    generate_plots_for_model(args.model, args.output, args.module_path)
    
    print(f"All plots generated and saved to {args.output}")

if __name__ == "__main__":
    main() 