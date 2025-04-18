#!/usr/bin/env python
"""
Generate diagnostic plots for all model types and update the application to use them.
This script:
1. Updates requirements.txt to include necessary dependencies
2. Generates and saves diagnostic plots for all model types
3. Updates the model_interpretation.html template to use the saved images
"""
import os
import sys
import argparse
import json
from pathlib import Path

# Import our utility modules
sys.path.append('.')
from utils.update_requirements import update_requirements, install_requirements
from utils.generate_diagnostics import generate_plots_for_model, read_model_database
from utils.diagnostic_plots.update_template import update_template

def main():
    parser = argparse.ArgumentParser(description='Generate diagnostic plots and update the application')
    parser.add_argument('--database', type=str, default='model_database.json',
                        help='Path to the model database JSON file')
    parser.add_argument('--output', type=str, default='static/diagnostic_plots',
                        help='Directory to save the plots')
    parser.add_argument('--requirements', type=str, default='requirements.txt',
                        help='Path to the requirements.txt file')
    parser.add_argument('--template', type=str, default='templates/model_interpretation.html',
                        help='Path to the model_interpretation.html template')
    parser.add_argument('--skip-install', action='store_true',
                        help='Skip installing dependencies')
    parser.add_argument('--model', type=str, 
                        help='Process only this model (optional)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("STATISTICAL MODEL DIAGNOSTIC PLOTS GENERATOR")
    print("=" * 80)
    
    # Step 1: Update requirements.txt
    print("\n1. Updating requirements.txt...")
    added_packages = update_requirements(args.requirements)
    
    # Install dependencies if needed
    if added_packages and not args.skip_install:
        response = input("Would you like to install the added packages now? (y/n) ")
        if response.lower() in ['y', 'yes']:
            install_requirements(args.requirements)
    
    # Step 2: Generate and save diagnostic plots
    print("\n2. Generating diagnostic plots...")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read model database
    model_database = read_model_database(args.database)
    
    if args.model:
        # Generate plots only for the specified model
        if args.model in model_database:
            print(f"Generating plots for {args.model}")
            generate_plots_for_model(args.model, model_database[args.model], args.output)
        else:
            print(f"Model '{args.model}' not found in database")
    else:
        # Generate plots for all relevant models
        models_to_process = [
            # Linear Regression models
            "Linear Regression",
            "Multiple Linear Regression",
            "Hierarchical Linear Regression",
            # Logistic Regression models
            "Logistic Regression",
            "Multinomial Logistic Regression",
            # ANOVA models
            "One-way ANOVA",
            "Two-way ANOVA",
            "Repeated Measures ANOVA",
            # Random Forest models
            "Random Forest",
            "Random Forest Classifier",
            "Random Forest Regressor",
            # PCA models
            "Principal Component Analysis"
        ]
        
        for model_name in models_to_process:
            if model_name in model_database:
                print(f"Generating plots for {model_name}")
                generate_plots_for_model(model_name, model_database[model_name], args.output)
            else:
                print(f"Model '{model_name}' not found in database, skipping...")
    
    # Step 3: Update model_interpretation.html template
    print("\n3. Updating model_interpretation.html template...")
    if os.path.exists(args.template):
        update_template(args.template)
    else:
        print(f"Warning: Template file {args.template} not found, skipping update")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC PLOTS GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nDiagnostic plots have been saved to: {args.output}")
    print("The model_interpretation.html template has been updated to use the saved plots")
    print("\nTo use the diagnostic plots:")
    print("1. Make sure the static/diagnostic_plots directory is accessible to your Flask app")
    print("2. Restart your Flask app to apply the template changes")
    print("3. Visit the model interpretation page for each model to see the plots")

if __name__ == "__main__":
    main() 