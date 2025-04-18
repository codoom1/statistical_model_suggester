#!/usr/bin/env python3
import json
import os

def verify_model_fields(models):
    """Verify that all models have the required fields with valid values."""
    required_fields = {
        'description': str,
        'use_cases': list,
        'analysis_goals': list,
        'dependent_variable': list,
        'independent_variables': list,
        'sample_size': list,
        'missing_data': list,
        'data_distribution': list,
        'relationship_type': list,
        'implementation': dict
    }
    
    # Define models that legitimately do not need a dependent variable
    clustering_models = [
        'Principal Component Analysis', 'K-Means Clustering', 'Hierarchical_Clustering',
        'Gaussian_Mixture_Model', 'DBSCAN', 'Spectral_Clustering', 'Agglomerative_Clustering',
        'Mean_Shift', 'Affinity_Propagation', 'OPTICS'
    ]
    
    issues = {}
    
    for model_name, model in models.items():
        model_issues = []
        
        # Check that all required fields exist
        for field, field_type in required_fields.items():
            # Skip dependent_variable check for clustering models
            if field == 'dependent_variable' and model_name in clustering_models:
                continue
                
            if field not in model:
                model_issues.append(f"Missing field: {field}")
            elif not isinstance(model[field], field_type):
                model_issues.append(f"Field {field} has incorrect type: {type(model[field]).__name__}, expected {field_type.__name__}")
            elif field_type == list and not model[field] and field != 'dependent_variable':
                model_issues.append(f"Field {field} is empty")
        
        # Check implementation field
        if 'implementation' in model:
            impl = model['implementation']
            for lang in ['python', 'r', 'spss', 'sas', 'stata']:
                if lang not in impl:
                    model_issues.append(f"Missing implementation for {lang}")
                elif not isinstance(impl[lang], dict):
                    model_issues.append(f"Implementation for {lang} has incorrect type: {type(impl[lang]).__name__}, expected dict")
                else:
                    if 'code' not in impl[lang] or not impl[lang]['code']:
                        model_issues.append(f"Missing code for {lang}")
                    if 'documentation' not in impl[lang] or not impl[lang]['documentation']:
                        model_issues.append(f"Missing documentation for {lang}")
        
        if model_issues:
            issues[model_name] = model_issues
    
    return issues

def main():
    # Check if model_database.json exists
    if not os.path.exists('model_database.json'):
        print("Error: model_database.json not found.")
        return
    
    # Load model database
    with open('model_database.json', 'r') as f:
        try:
            models = json.load(f)
        except json.JSONDecodeError:
            print("Error: model_database.json is not valid JSON.")
            return
    
    print(f"Loaded {len(models)} models from model_database.json")
    
    # Verify model fields
    issues = verify_model_fields(models)
    
    if issues:
        print("\nIssues found in the model database:")
        for model_name, model_issues in issues.items():
            print(f"\n{model_name}:")
            for issue in model_issues:
                print(f"  - {issue}")
        print(f"\nTotal models with issues: {len(issues)}")
    else:
        print("\nAll models have the required fields with valid values.")
    
    # Count and display model categories
    analysis_goals = {}
    dependent_vars = {}
    for model_name, model in models.items():
        for goal in model.get('analysis_goals', []):
            analysis_goals[goal] = analysis_goals.get(goal, 0) + 1
        
        for dep_var in model.get('dependent_variable', []):
            dependent_vars[dep_var] = dependent_vars.get(dep_var, 0) + 1
    
    print("\nModels by analysis goal:")
    for goal, count in sorted(analysis_goals.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {goal}: {count} models")
    
    print("\nModels by dependent variable type:")
    for var, count in sorted(dependent_vars.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {var}: {count} models")

if __name__ == "__main__":
    main() 