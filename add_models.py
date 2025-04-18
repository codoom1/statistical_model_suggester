#!/usr/bin/env python
"""
Add PCA and ANOVA models to the database and generate diagnostic plots for them.
"""
import json
import os
import subprocess
from pathlib import Path

# Models to add
MODELS_TO_ADD = [
    "One-way ANOVA", 
    "Two-way ANOVA", 
    "Repeated Measures ANOVA", 
    "Principal Component Analysis"
]

def add_models_to_database():
    # Load the existing database
    with open("model_database.json", "r") as f:
        database = json.load(f)
    
    # Count models before adding
    initial_count = len(database)
    
    # Add models if they don't exist
    added_models = []
    for model in MODELS_TO_ADD:
        if model not in database:
            database[model] = {
                "description": f"Sample {model} model for diagnostics",
                "synthetic_data": {}
            }
            added_models.append(model)
    
    # Save the updated database
    with open("model_database.json", "w") as f:
        json.dump(database, f, indent=2)
    
    print(f"Added {len(added_models)} new models to database: {added_models}")
    print(f"Database now contains {len(database)} models (was {initial_count})")
    
    return added_models

def generate_plots_for_models(models):
    # Create output directories
    output_dir = Path("static/diagnostic_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating diagnostic plots...\n")
    
    # Generate plots for each model
    for model in models:
        model_dir = output_dir / model.replace(" ", "_").lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating plots for {model}...")
        
        # Run the diagnostic plots generator script
        cmd = ["python", "utils/generate_diagnostics.py", "--model", model]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print the output
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")
        
        print(f"Plots saved to {model_dir}")

if __name__ == "__main__":
    print("Adding models to database...")
    added_models = add_models_to_database()
    
    # Always regenerate plots regardless if models were added
    generate_plots_for_models(MODELS_TO_ADD) 