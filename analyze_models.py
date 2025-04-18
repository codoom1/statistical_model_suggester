#!/usr/bin/env python3
import json

# Load model database
with open('model_database.json', 'r') as f:
    models = json.load(f)

print(f'Total number of models: {len(models)}')

# Collect all unique fields
fields = set()
for model_name, model in models.items():
    fields.update(model.keys())

print(f'All fields found: {sorted(list(fields))}')

# Define models that legitimately do not have a dependent variable
clustering_models = [
    'Principal Component Analysis', 'K-Means Clustering', 'Hierarchical_Clustering',
    'Gaussian_Mixture_Model', 'DBSCAN', 'Spectral_Clustering', 'Agglomerative_Clustering',
    'Mean_Shift', 'Affinity_Propagation', 'OPTICS'
]

# Check for missing or empty fields
missing_fields = {}
for model_name, model in models.items():
    for field in fields:
        # Skip dependent_variable check for clustering models
        if field == 'dependent_variable' and model_name in clustering_models:
            continue
            
        if field not in model or not model[field]:
            if field not in missing_fields:
                missing_fields[field] = []
            missing_fields[field].append(model_name)

# Print findings
for field, models_missing in missing_fields.items():
    print(f'\nField "{field}" is missing in {len(models_missing)} models:')
    if len(models_missing) <= 5:
        for model in models_missing:
            print(f'  - {model}')
    else:
        for model in models_missing[:5]:
            print(f'  - {model}')
        print(f'  - ...and {len(models_missing) - 5} more')

# Check for implementation completeness
for model_name, model in models.items():
    if 'implementation' in model:
        for lang, impl in model['implementation'].items():
            if not impl.get('code') or not impl.get('documentation'):
                print(f'\nModel "{model_name}" has incomplete implementation for {lang}:')
                if not impl.get('code'):
                    print(f'  - Missing code for {lang}')
                if not impl.get('documentation'):
                    print(f'  - Missing documentation for {lang}') 