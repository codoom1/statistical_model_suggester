#!/usr/bin/env python
"""
Generate diagnostic plots for CatBoost and LightGBM models.
This is a simplified version of the generate_diagnostics.py script.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import base64
import io
from pathlib import Path

from utils.diagnostic_plots.catboost import generate_catboost_plots
from utils.diagnostic_plots.lightgbm import generate_lightgbm_plots

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
        image_data = base64.b64decode(img_data)
        image = Image.open(io.BytesIO(image_data))
        image.save(filename)
        print(f"Saved: {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")

def generate_sample_data(model_type):
    """Generate sample data for tree-based models
    
    Args:
        model_type: Either 'catboost' or 'lightgbm'
        
    Returns:
        X, y, feature_names, is_classification
    """
    np.random.seed(42)  # For reproducibility
    
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
    
    # For classification variant
    if 'class' in model_type:
        # Convert to binary classification problem
        y_binary = (y > np.median(y)).astype(int)
        return X, y_binary, ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"], True
    else:
        # Regression problem
        return X, y, ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"], False

def generate_boosting_plots(output_dir='static/diagnostic_plots'):
    """Generate plots for CatBoost and LightGBM models
    
    Args:
        output_dir: Directory to save the plots
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate CatBoost plots
    catboost_dir = output_dir / 'catboost'
    catboost_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Generating CatBoost plots")
        # Generate appropriate sample data for CatBoost
        X, y, feature_names, is_classification = generate_sample_data('catboost')
        plots = generate_catboost_plots(
            model=None,  # No model, just using the data
            X_train=X[:150],
            y_train=y[:150],
            X_test=X[150:],
            y_test=y[150:],
            feature_names=feature_names,
            is_classifier=is_classification
        )
        
        # Save plots
        for i, plot in enumerate(plots):
            plot_title = plot.get('title', f'Plot_{i}')
            img_data = plot.get('img_data', '')
            interpretation = plot.get('interpretation', 'No interpretation available')
            
            filename = f"{i+1}_{plot_title.replace(' ', '_').lower()}.png"
            filepath = catboost_dir / filename
            
            if img_data:
                image_data = base64.b64decode(img_data)
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                print(f"Saved: {filepath}")
                
                # Save interpretation
                json_filename = filepath.with_suffix('.json')
                with open(json_filename, 'w') as f:
                    json.dump({
                        'title': plot_title,
                        'interpretation': interpretation
                    }, f, indent=2)
        
        print(f"Generated {len(plots)} CatBoost plots")
    except Exception as e:
        print(f"Error generating CatBoost plots: {e}")
    
    # Generate LightGBM plots
    lightgbm_dir = output_dir / 'lightgbm'
    lightgbm_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Generating LightGBM plots")
        # Generate appropriate sample data for LightGBM
        X, y, feature_names, is_classification = generate_sample_data('lightgbm')
        plots = generate_lightgbm_plots(
            model=None,  # No model, just using the data
            X_train=X[:150],
            y_train=y[:150],
            X_test=X[150:],
            y_test=y[150:],
            feature_names=feature_names,
            is_classifier=is_classification
        )
        
        # Save plots
        for i, plot in enumerate(plots):
            plot_title = plot.get('title', f'Plot_{i}')
            img_data = plot.get('img_data', '')
            interpretation = plot.get('interpretation', 'No interpretation available')
            
            filename = f"{i+1}_{plot_title.replace(' ', '_').lower()}.png"
            filepath = lightgbm_dir / filename
            
            if img_data:
                image_data = base64.b64decode(img_data)
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                print(f"Saved: {filepath}")
                
                # Save interpretation
                json_filename = filepath.with_suffix('.json')
                with open(json_filename, 'w') as f:
                    json.dump({
                        'title': plot_title,
                        'interpretation': interpretation
                    }, f, indent=2)
        
        print(f"Generated {len(plots)} LightGBM plots")
    except Exception as e:
        print(f"Error generating LightGBM plots: {e}")
    
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    from PIL import Image
    generate_boosting_plots() 