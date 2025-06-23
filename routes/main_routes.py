from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from flask_login import login_required, current_user
from models import db, User, Analysis, get_model_details
from datetime import datetime
import json
import os
import random
from collections import OrderedDict
main = Blueprint('main', __name__)
# Path for history file (legacy support)
HISTORY_FILE = 'history.json'
# -----------------------------------------------------------------------------
# MODEL GROUPING FOR NAVIGATION DROPDOWN
# -----------------------------------------------------------------------------
# Defines the categorization of models for the main navigation dropdown.
# - Keys: The display name of the group (string).
# - Values: A list of model name strings that belong to this group.
#           These names MUST exactly match the keys in MODEL_DATABASE (model_database.json).
# - OrderedDict: Used to ensure the groups appear in the dropdown in the defined order.
# -----------------------------------------------------------------------------
# Define Model Groups (Order matters for the dropdown)
MODEL_GROUPS = OrderedDict([
    ('Classical Statistical Tests', [
        'T test',
        'Chi-Square Test',
        'Mann-Whitney U Test',
        'Kruskal-Wallis Test',
        'Analysis of Variance (ANOVA)',
        'Analysis of Covariance (ANCOVA)',
        'Repeated Measures ANOVA'
        # Add other relevant test models here if they exist
    ]),
    ('Regression Models', [
        'Linear Regression',
        'Multiple Linear Regression',
        'Logistic Regression',
        'Multinomial Logistic Regression',
        'Ordinal Regression',
        'Poisson Regression',
        'Ridge Regression',
        'Lasso Regression',
        'Elastic Net Regression',
        'Quantile Regression',
        'Stepwise Regression',
        'Generalized Linear Model (GLM)',
        'Generalized Additive Model (GAM)',
        'Kernel Regression',
        'Polynomial Regression',
        'Bayesian Linear Regression', # Consider if Bayesian models get their own group
        'Bayesian Quantile Regression' # Or are subtypes here
    ]),
    ('Time Series Models', [
        'ARIMA',
        'Exponential Smoothing',
        'Prophet',
        'Vector Autoregression (VAR)'
        # Add other time series models
    ]),
    ('Multivariate Analysis', [
        'Principal Component Analysis (PCA)',
        'Factor Analysis',
        'K-Means clustering',
        'Discriminant Analysis',
        'Canonical Correlation',
        'Multidimensional Scaling',
        'Multivariate Analysis of Covariance (MANCOVA)',
        'Multivariate Analysis of Variance (MANOVA)',
        'Analysis of Covariance (ANCOVA)',
        'Analysis of Variance (ANOVA)'
        # Add other multivariate models (K-Means, DBSCAN etc. could fit here or ML)
    ]),
    ('Machine Learning Models', [
        'Decision Trees',
        'Random Forest',
        'Gradient Boosting',
        'XGBoost',
        'LightGBM',
        'CatBoost',
        'Support Vector Machines (SVM)',
        'K-Nearest Neighbors (KNN)',
        'Naive Bayes classifier',
        'Neural Networks',
        'K-Means',
        'Hierarchical Clustering',
        'DBSCAN'
        # Add other ML models
    ]),
    ('Mixed and Hierarchical Models', [
        'Mixed Effects Model',
        'Hierarchical Linear Model',
        'Multilevel Model',
        'Bayesian Hierarchical Regression'
    ]),
    ('Structural Models', [
        'Structural Equation Modeling (SEM)',
        'Path Analysis'
    ]),
    ('Survival Models', [
        'Cox Proportional Hazards Model',
        'Kaplan-Meier Curve'
    ]),
    ('Bayesian Models', [
        'Bayesian Linear Regression',
        'Bayesian Hierarchical Regression',
        'Bayesian Model Averaging',
        'Bayesian Quantile Regression',
        'Bayesian Additive Regression Trees (BART)'
     ]),
     #(optional) 'Deep Learning Models', [
      #  'Convolutional Neural Networks (CNN)',
      #  'Recurrent Neural Networks (RNN)',
      #  'Long Short-Term Memory (LSTM)',
      #  'Gated Recurrent Units (GRU)',
      #  'Transformer Models'
     #])
])
# Make model groups available to all templates
@main.context_processor
def inject_model_groups():
    return dict(model_groups=MODEL_GROUPS)
def load_history():
    """Load analysis history from JSON file (legacy support)"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []
def save_history(history):
    """Save analysis history to JSON file (legacy support)"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
def add_to_history(research_question, recommended_model, analysis_goal, dependent_variable_type,
                   independent_variables, sample_size, missing_data, data_distribution, relationship_type,
                   variables_correlated='unknown'):
    """Add analysis to history file (legacy support)"""
    history = load_history()
    history.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'research_question': research_question,
        'analysis_goal': analysis_goal,
        'dependent_variable': dependent_variable_type,
        'independent_variables': independent_variables,
        'sample_size': sample_size,
        'missing_data': missing_data,
        'data_distribution': data_distribution,
        'relationship_type': relationship_type,
        'variables_correlated': variables_correlated,
        'recommended_model': recommended_model
    })
    save_history(history)
def get_model_recommendation(analysis_goal, dependent_variable, independent_variables,
                            sample_size, missing_data, data_distribution, relationship_type,
                            variables_correlated):
    """Get model recommendation based on input parameters"""
    # Get the model database from app config
    MODEL_DATABASE = current_app.config.get('MODEL_DATABASE', {})
    # Convert sample_size to integer if it's a string
    try:
        sample_size = int(sample_size)
    except (ValueError, TypeError):
        sample_size = 50  # Default to medium sample size if not provided
    # Categorize sample size
    if sample_size < 30:
        size_category = 'small'
    elif sample_size < 100:
        size_category = 'medium'
    else:
        size_category = 'large'
    # Define model families for diversity in recommendations
    model_families = {
        'classical': ['Linear Regression', 'Logistic Regression', 'Poisson Regression', 'Ridge Regression',
                     'Lasso Regression', 'Elastic Net Regression'],
        'tree_based': ['Decision Trees', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost'],
        'bayesian': ['Bayesian Linear Regression', 'Bayesian Hierarchical Regression', 'Bayesian Model Averaging',
                    'Bayesian Quantile Regression', 'Bayesian Additive Regression Trees'],
        'hierarchical': ['Mixed Effects Model', 'Hierarchical Linear Model', 'Multilevel Model'],
        'neural_network': ['Neural Networks'],
        'nonparametric': ['Support Vector Machines', 'K-Nearest Neighbors', 'Kernel_Regression'],
        'dimensionality_reduction': ['Principal Component Analysis', 'Factor Analysis', 'Multidimensional Scaling'],
        'clustering': ['Cluster Analysis', 'K-Means', 'Hierarchical Clustering', 'DBSCAN', 'Gaussian Mixture Models'],
        'time_series': ['ARIMA', 'Exponential Smoothing', 'Prophet'],
        'hypothesis_testing': ['T_test', 'Chi_Square_Test', 'Mann_Whitney_U_Test', 'Kruskal_Wallis_Test',
                              'Analysis of Variance (ANOVA)', 'Analysis of Covariance (ANCOVA)']
    }
    # Build a reverse lookup of model to family
    model_to_family = {}
    for family, models in model_families.items():
        for model in models:
            model_to_family[model] = family
    # Define clustering models (these don't require a dependent variable)
    clustering_models = ['Cluster Analysis', 'K-Means', 'Hierarchical Clustering', 'DBSCAN',
                         'Gaussian Mixture Models', 'Principal Component Analysis', 'Factor Analysis']
    # For clustering analysis, ensure we have a default dependent variable if not provided
    if analysis_goal == 'cluster' and not dependent_variable:
        dependent_variable = 'continuous'  # A sensible default for clustering
    # Score models based on compatibility
    model_scores = {}
    for model_name, model in MODEL_DATABASE.items():
        score = 0
        current_app.logger.debug(f"SCORING {model_name}: Starting score = {score}")
        # Check analysis goal compatibility - heavily weighted
        if analysis_goal in model.get('analysis_goals', []):
            score += 3
            current_app.logger.debug(f"  + Analysis goal match: +3 → {score}")
        else:
            # If analysis goal doesn't match, this model is less relevant
            current_app.logger.debug(f"  × Skipping {model_name}: analysis goal mismatch")
            continue  # Skip models that don't match the primary analysis goal
        # Special handling for clustering models when the goal is 'cluster'
        is_clustering_model = model_name in clustering_models or 'cluster' in analysis_goal.lower()
        # Check dependent variable compatibility - heavily weighted
        # Skip this check for clustering models when the goal is 'cluster'
        if is_clustering_model and analysis_goal == 'cluster':
            # Clustering models get a bonus instead of being checked for dependent variable
            score += 3
            current_app.logger.debug(f"  + Clustering model bonus: +3 → {score}")
        elif dependent_variable in model.get('dependent_variable', []):
            score += 3
            current_app.logger.debug(f"  + Dependent variable match: +3 → {score}")
        else:
            # If dependent variable type doesn't match, this model is less relevant
            current_app.logger.debug(f"  × Skipping {model_name}: dependent variable mismatch")
            continue  # Skip models that don't match the dependent variable type
        # Check relationship type compatibility - important factor
        if relationship_type in model.get('relationship_type', []):
            score += 2
            current_app.logger.debug(f"  + Relationship type match: +2 → {score}")
        elif relationship_type == 'linear' and 'non_linear' in model.get('relationship_type', []):
            # Non-linear models can handle linear relationships too
            score += 1.5
            current_app.logger.debug(f"  + Non-linear compatibility: +1.5 → {score}")
        # Check independent variable compatibility
        independent_var_score = 0
        for var in independent_variables:
            if var in model.get('independent_variables', []):
                independent_var_score += 1
        if independent_variables and independent_var_score == len(independent_variables):
            score += 2
            current_app.logger.debug(f"  + Independent variable match: +2 → {score}")
        elif independent_variables and independent_var_score > 0:
            score += independent_var_score / len(independent_variables)
            current_app.logger.debug(f"  + Independent variable compatibility: +{score} → {score}")
        # Special bonus for regularization models with many continuous variables
        # These models excel at handling correlated predictors
        if model_name in ['Elastic Net Regression', 'Ridge Regression', 'Lasso Regression'] and \
           'continuous' in independent_variables and \
           relationship_type == 'linear':
            # Only apply regularization bonus when variables are explicitly correlated
            if variables_correlated == 'yes':
                score += 3.5
                current_app.logger.debug(f"  + Regularization bonus: +3.5 → {score}")
            elif variables_correlated == 'unknown':
                score += 0.75
                current_app.logger.debug(f"  + Regularization compatibility: +0.75 → {score}")
            # No bonus when variables are explicitly not correlated
        # Boost other models that work well with correlated variables
        if variables_correlated == 'yes' and model_name in ['Principal Component Analysis',
                                                          'Factor Analysis', 'Partial Least Squares',
                                                          'Random Forest', 'Gradient Boosting',
                                                          'XGBoost', 'CatBoost', 'LightGBM']:
            score += 1.5
            current_app.logger.debug(f"  + Correlated variable compatibility: +1.5 → {score}")
        # Models that assume independent variables - slight penalty for correlated data
        if variables_correlated == 'yes' and model_name in ['Linear Regression', 'Stepwise Regression']:
            score -= 0.5
            current_app.logger.debug(f"  - Correlated variable penalty: -0.5 → {score}")
        # Boost for Linear Regression in standard prediction scenarios with normal data and linear relationships
        if model_name == 'Linear Regression' and variables_correlated == 'no' and \
           relationship_type == 'linear' and data_distribution == 'normal' and \
           missing_data in ['none', 'little'] and dependent_variable == 'continuous':
            score += 5.0
            current_app.logger.debug(f"  + Linear Regression boost: +5.0 → {score}")
        # Strong boost for Cluster Analysis in exploratory or clustering scenarios
        if model_name == 'Cluster Analysis' and (analysis_goal == 'explore' or analysis_goal == 'cluster'):
            score += 15.0  # Massively increased to ensure Cluster Analysis wins for clustering
            current_app.logger.debug(f"CLUSTER BONUS: {model_name} +15.0 for {analysis_goal} analysis")
        # Penalty for non-clustering models in exploratory or clustering scenarios
        if (analysis_goal == 'explore' or analysis_goal == 'cluster') and model_name not in ['Cluster Analysis', 'Factor Analysis', 'Principal Component Analysis',
                                                           'Multidimensional Scaling', 'UMAP', 'K-Means', 'Hierarchical Clustering', 'DBSCAN', 'Gaussian Mixture Models']:
            score -= 5.0  # Significant penalty for non-exploratory models
            current_app.logger.debug(f"EXPLORE/CLUSTER PENALTY: {model_name} -5.0 for being non-{analysis_goal}")
        # Extra boost for Elastic Net which combines benefits of Lasso and Ridge
        if model_name == 'Elastic Net Regression' and missing_data in ['none', 'little']:
            # Only apply this bonus when variables_correlated is 'yes' or 'unknown'
            if variables_correlated != 'no':
                score += 0.5
                current_app.logger.debug(f"  + Elastic Net bonus: +0.5 → {score}")
        # Check sample size compatibility
        if size_category in model.get('sample_size', []):
            score += 1
            current_app.logger.debug(f"  + Sample size match: +1 → {score}")
        # Check missing data handling
        if missing_data in model.get('missing_data', []):
            score += 1.5
            current_app.logger.debug(f"  + Missing data compatibility: +1.5 → {score}")
        # Check data distribution compatibility
        if data_distribution in model.get('data_distribution', []):
            score += 1.5
            current_app.logger.debug(f"  + Data distribution match: +1.5 → {score}")
        elif data_distribution == 'normal' and 'non_normal' in model.get('data_distribution', []):
            # Models that handle non-normal data can handle normal data too
            score += 1
            current_app.logger.debug(f"  + Non-normal data compatibility: +1 → {score}")
        # Add a bonus for advanced/specialized models
        # This helps counter the bias towards simpler models like linear regression
        if model_name in ['Random Forest', 'XGBoost', 'Gradient Boosting', 'Neural Networks',
                          'Bayesian Linear Regression', 'Generalized Additive Model',
                          'Hierarchical Linear Model', 'Quantile Regression']:
            score += 0.5
            current_app.logger.debug(f"  + Advanced model bonus: +0.5 → {score}")
        # Add extra bonus for Bayesian models - they're often underrepresented
        if 'bayesian' in model_name.lower() or model_name in model_families.get('bayesian', []):
            score += 1.0
            current_app.logger.debug(f"  + Bayesian model bonus: +1.0 → {score}")
        # Add special bonus for models that are particularly suited to hierarchical data
        if relationship_type == 'hierarchical' and model_name in ['Mixed Effects Model', 'Hierarchical Linear Model',
                                                                 'Multilevel Model', 'Bayesian Hierarchical Regression']:
            score += 2.0
            current_app.logger.debug(f"  + Hierarchical model bonus: +2.0 → {score}")
        # Bonus for advanced models that handle complex relationships
        if relationship_type == 'non_linear' and model_name in ['Random Forest', 'XGBoost', 'Neural Networks',
                                                               'Gradient Boosting', 'Support Vector Machines',
                                                               'Bayesian Additive Regression Trees']:
            score += 1.5
            current_app.logger.debug(f"  + Non-linear model bonus: +1.5 → {score}")
        # Add a penalty for overused models to promote diversity
        if model_name == 'Linear Regression':
            score -= 0.25
            current_app.logger.debug(f"  - Linear Regression penalty: -0.25 → {score}")
        # Add a small random component to break ties between similar models
        # Increase the random component to improve diversity in recommendations
        random_component = random.uniform(0, 0.2)
        score += random_component
        current_app.logger.debug(f"  + Random component: +{random_component:.4f} → {score:.4f}")
        # Give a strong bonus to clustering models when the goal is clustering
        if (analysis_goal == 'explore' or analysis_goal == 'cluster') and model_name in clustering_models:
            score += 10.0  # Very significant boost to ensure clustering models win for cluster analysis
            current_app.logger.debug(f"  + Clustering model for cluster goal: +10.0 → {score:.4f}")
        # Special case for Neural Networks - reduce score for clustering tasks
        if (analysis_goal == 'explore' or analysis_goal == 'cluster') and model_name == 'Neural Networks':
            score -= 5.0  # Significant penalty for neural networks in clustering tasks
            current_app.logger.debug(f"  - Neural Networks penalty for clustering: -5.0 → {score:.4f}")
        # Fix for exploratory analysis with continuous dependent variable - ensure clustering models win
        if (analysis_goal == 'explore' or analysis_goal == 'cluster') and dependent_variable == 'continuous' and model_name == 'Cluster Analysis':
            score += 5.0  # Extra boost to ensure Cluster Analysis wins for exploratory analysis
            current_app.logger.debug(f"  + Exploratory/Cluster continuous fix: +5.0 → {score:.4f}")
        model_scores[model_name] = score
    # Get top models
    # First, identify the best matching model
    if model_scores:
        # Sort models by score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        # Log top model scores for debugging
        current_app.logger.debug(f"TOP MODEL SCORES:")
        for model, score in sorted_models[:5]:  # Show top 5 models
            current_app.logger.debug(f"{model}: {score}")
        # Get best model
        best_model = sorted_models[0][0]
        # Generate explanation for best model
        explanation = generate_explanation(best_model, analysis_goal, dependent_variable,
                                        independent_variables, sample_size, missing_data,
                                        data_distribution, relationship_type, variables_correlated)
        # Get alternative models - ensure diversity by selecting from different model families
        best_score = sorted_models[0][1]
        # First, get all candidate alternatives that score at least 70% of the best model
        # Reduced from 75% to 70% to include more diverse alternatives
        candidate_alternatives = [(model, score) for model, score in sorted_models[1:10]  # Look at top 10 candidates
                                if score >= 0.7 * best_score]
        # Determine the family of the best model
        best_model_family = model_to_family.get(best_model, 'unknown')
        # Prioritize models from different families while maintaining decent scores
        alternative_models = []
        families_included = {best_model_family}  # Already included the best model's family
        # First pass: try to include models from different families
        for model, _ in candidate_alternatives:
            family = model_to_family.get(model, 'unknown')
            if family not in families_included and len(alternative_models) < 4:
                alternative_models.append(model)
                families_included.add(family)
        # Second pass: fill remaining slots with highest scoring models
        remaining_slots = 4 - len(alternative_models)
        if remaining_slots > 0:
            for model, _ in candidate_alternatives:
                if model not in alternative_models and len(alternative_models) < 4:
                    alternative_models.append(model)
        return best_model, explanation, alternative_models
    else:
        # Fallback to default model
        default_model = get_default_model(analysis_goal, dependent_variable)
        explanation = f"Based on your analysis goal ({analysis_goal}) and dependent variable type ({dependent_variable}), we recommend using {default_model}."
        # Get some sensible alternatives for the fallback case
        alternative_models = get_default_alternatives(analysis_goal, dependent_variable)
        return default_model, explanation, alternative_models
def get_default_alternatives(analysis_goal, dependent_variable):
    """Get default alternative models based on analysis goal and dependent variable type"""
    alternatives = []
    if analysis_goal == 'predict':
        if dependent_variable == 'continuous':
            alternatives = ['Ridge Regression', 'Random Forest', 'XGBoost', 'Bayesian Linear Regression', 'Gradient Boosting']
        elif dependent_variable == 'binary':
            alternatives = ['Random Forest', 'Support Vector Machine', 'XGBoost', 'Neural Network']
        elif dependent_variable == 'count':
            alternatives = ['Negative Binomial Regression', 'Zero-Inflated Poisson', 'Quantile Regression']
        elif dependent_variable == 'ordinal':
            alternatives = ['Multinomial Logistic Regression', 'Neural Network', 'Ordinal Regression']
        elif dependent_variable == 'time_to_event':
            alternatives = ['Kaplan-Meier', 'Weibull Model', 'Cox Proportional Hazards']
    elif analysis_goal == 'classify':
        if dependent_variable == 'binary':
            alternatives = ['Random Forest', 'Support Vector Machine', 'XGBoost', 'Neural Network', 'Gradient Boosting']
        elif dependent_variable == 'categorical':
            alternatives = ['Random Forest', 'Neural Network', 'Support Vector Machine', 'XGBoost']
    elif analysis_goal == 'explore':
        alternatives = ['Cluster Analysis', 'Factor Analysis', 'Multidimensional Scaling', 'Principal Component Analysis', 'UMAP']
    elif analysis_goal == 'cluster':
        alternatives = ['Cluster Analysis', 'K-Means', 'Hierarchical Clustering', 'DBSCAN', 'Gaussian Mixture Models']
    elif analysis_goal == 'hypothesis_test':
        if dependent_variable == 'continuous':
            alternatives = ['Analysis of Variance (ANOVA)', 'Mann_Whitney_U_Test', 'Wilcoxon Signed-Rank Test', 'T_test']
        elif dependent_variable == 'categorical':
            alternatives = ['Fisher\'s Exact Test', 'G-Test', 'McNemar\'s Test', 'Chi_Square_Test']
    elif analysis_goal == 'non_parametric':
        alternatives = ['Wilcoxon Signed-Rank Test', 'Kruskal_Wallis_Test', 'Spearman Correlation', 'Mann_Whitney_U_Test']
    elif analysis_goal == 'time_series':
        alternatives = ['Exponential Smoothing', 'Prophet', 'ARIMA', 'ARIMAX', 'GARCH']
    # Remove alternatives that might not exist in the database
    MODEL_DATABASE = current_app.config.get('MODEL_DATABASE', {})
    return [alt for alt in alternatives if alt in MODEL_DATABASE][:4]  # Increased from 3 to 4 alternatives
def generate_explanation(model_name, analysis_goal, dependent_variable, independent_variables,
                        sample_size, missing_data, data_distribution, relationship_type,
                        variables_correlated='unknown'):
    """Generate explanation for model recommendation"""
    model_info = get_model_details(model_name) or {}
    explanation = f"\n    Based on your data characteristics, a {model_name} is recommended because:\n    \n"
    reasons = []
    if analysis_goal in model_info.get('analysis_goals', []):
        reasons.append(f"It is suitable for {analysis_goal} analysis with {dependent_variable} dependent variables")
    if independent_variables and all(var in model_info.get('independent_variables', []) for var in independent_variables):
        reasons.append(f"It can handle {', '.join(independent_variables)} independent variables")
    # Convert sample_size to int if needed
    try:
        sample_size_int = int(sample_size)
    except (ValueError, TypeError):
        sample_size_int = 50
    if sample_size_int < 30 and 'small' in model_info.get('sample_size', []):
        reasons.append("It works well with small sample sizes")
    elif sample_size_int >= 30 and sample_size_int < 100 and 'medium' in model_info.get('sample_size', []):
        reasons.append("It works well with medium sample sizes")
    elif sample_size_int >= 100 and 'large' in model_info.get('sample_size', []):
        reasons.append("It is optimized for large datasets")
    if missing_data in model_info.get('missing_data', []):
        reasons.append(f"It can handle {missing_data} missing data patterns")
    if data_distribution in model_info.get('data_distribution', []):
        reasons.append(f"It is appropriate for {data_distribution} data distribution")
    if relationship_type in model_info.get('relationship_type', []):
        reasons.append(f"It can model {relationship_type} relationships")
    # Add reason related to correlated variables if specified
    if variables_correlated == 'yes' and model_name in ['Elastic Net Regression', 'Ridge Regression', 'Lasso Regression',
                                                      'Principal Component Analysis', 'Factor Analysis',
                                                      'Partial Least Squares']:
        reasons.append("It excels at handling correlated predictors")
    # Add numbered reasons
    for i, reason in enumerate(reasons, 1):
        explanation += f"    {i}. {reason}\n"
    # Add implementation notes
    explanation += f"""
    Implementation notes:
    - {model_info.get('description', 'No additional description available.')}
    - Consider preprocessing steps for {', '.join(independent_variables)} variables
    - Check assumptions specific to {model_name}
    """
    return explanation
def get_default_model(analysis_goal, dependent_variable):
    """Get default model based on analysis goal and dependent variable type"""
    # Get model database
    MODEL_DATABASE = current_app.config.get('MODEL_DATABASE', {})
    model_names = MODEL_DATABASE.keys()
    # For clustering analysis, ensure we have a default dependent variable if not provided
    if analysis_goal == 'cluster' and not dependent_variable:
        dependent_variable = 'continuous'  # A sensible default for clustering
    # Define target model patterns based on analysis goal and dependent variable type
    if analysis_goal == 'predict':
        if dependent_variable == 'continuous':
            target_models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression']
        elif dependent_variable == 'binary':
            target_models = ['Logistic Regression', 'Support Vector Machines']
        elif dependent_variable == 'count':
            target_models = ['Poisson Regression', 'Negative Binomial Regression']
        elif dependent_variable == 'ordinal':
            target_models = ['Ordinal Regression', 'Multinomial Regression']
        elif dependent_variable == 'time_to_event':
            target_models = ['Cox Proportional Hazards', 'Kaplan Meier']
    elif analysis_goal == 'classify':
        if dependent_variable == 'binary':
            target_models = ['Logistic Regression', 'Support Vector Machines']
        elif dependent_variable == 'categorical':
            target_models = ['Multinomial Regression', 'Random Forest']
    elif analysis_goal == 'explore':
        target_models = ['Principal Component Analysis', 'Factor Analysis', 'Cluster Analysis']
    elif analysis_goal == 'cluster':
        target_models = ['Cluster Analysis', 'Principal Component Analysis']
    elif analysis_goal == 'hypothesis_test':
        if dependent_variable == 'continuous':
            target_models = ['T_test', 'Analysis of Variance (ANOVA)']
        elif dependent_variable == 'categorical':
            target_models = ['Chi_Square_Test', 'Fisher\'s Exact Test']
    elif analysis_goal == 'non_parametric':
        target_models = ['Mann_Whitney_U_Test', 'Kruskal_Wallis_Test']
    elif analysis_goal == 'time_series':
        target_models = ['ARIMA', 'Exponential Smoothing']
    else:
        target_models = ['Linear Regression', 'Logistic Regression']
    # Find the first matching model that exists in the database
    for model in target_models:
        if model in model_names:
            return model
    # If none of the target models exist, return the first available model
    # as a fallback to prevent errors
    if model_names:
        return list(model_names)[0]
    # If the database is empty (shouldn't happen), return a sensible default
    return "Linear Regression"
@main.route('/')
def home():
    # Create stats for the home page
    stats = {
        'models_count': len(current_app.config.get('MODEL_DATABASE', {})),
        'access_hours': '24/7',
        'verification_rate': '95%'
    }
    return render_template('home.html', stats=stats, now=datetime.now())
@main.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """View and edit user profile"""
    if request.method == 'POST':
        # Update basic profile information
        email = request.form.get('email')
        # Check if email already exists for another user
        if email != current_user.email:
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash('Email already in use.', 'danger')
                return redirect(url_for('main.profile'))
        current_user.email = email
        # If user is an expert, also update expert fields
        if current_user.is_expert:
            current_user.areas_of_expertise = request.form.get('areas_of_expertise', '')
            current_user.institution = request.form.get('institution', '')
            current_user.bio = request.form.get('bio', '')
        db.session.commit()
        flash('Profile updated successfully.', 'success')
        return redirect(url_for('main.profile'))
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).all()
    return render_template('profile.html', user=current_user, analyses=analyses)
@main.route('/results', methods=['GET', 'POST'])
def results():
    """Process form input and return model recommendation"""
    # Get form data
    research_question = request.form.get('research_question', '')
    analysis_goal = request.form.get('analysis_goal', '')
    dependent_variable_type = request.form.get('dependent_variable_type', '')
    # Get independent variables (multiply selected)
    independent_variables = request.form.getlist('independent_variables')
    # Get other attributes
    sample_size = request.form.get('sample_size', '')
    missing_data = request.form.get('missing_data', '')
    data_distribution = request.form.get('data_distribution', '')
    relationship_type = request.form.get('relationship_type', '')
    variables_correlated = request.form.get('variables_correlated', 'unknown')
    # Get model database from app config
    MODEL_DATABASE = current_app.config.get('MODEL_DATABASE', {})
    # For clustering analysis, dependent variable can be empty
    # If it's empty, set it to 'continuous' which works well with clustering models
    if analysis_goal == 'cluster' and not dependent_variable_type:
        dependent_variable_type = 'continuous'
    # Require essential inputs
    if not (research_question and analysis_goal):
        flash('Please provide all required information to get a recommendation.', 'warning')
        return redirect(url_for('main.analysis_form'))
    # For non-clustering analysis, require dependent variable
    if analysis_goal != 'cluster' and not dependent_variable_type:
        flash('Please select what type of outcome you are measuring.', 'warning')
        return redirect(url_for('main.analysis_form'))
    # Get model recommendation
    recommended_model, explanation, alternative_models = get_model_recommendation(
        analysis_goal, dependent_variable_type, independent_variables,
        sample_size, missing_data, data_distribution, relationship_type,
        variables_correlated
    )
    # Verify the recommended model exists in the database
    if recommended_model not in MODEL_DATABASE:
        # Find a replacement model if the recommended one doesn't exist
        recommended_model = get_default_model(analysis_goal, dependent_variable_type)
        explanation = f"Based on your analysis goal ({analysis_goal}) and dependent variable type ({dependent_variable_type}), we recommend using {recommended_model}."
    # Save to history if model recommendation found
    if recommended_model:
        add_to_history(research_question, recommended_model, analysis_goal, dependent_variable_type,
                    independent_variables, sample_size, missing_data, data_distribution, relationship_type,
                    variables_correlated)
    # Save analysis if user is logged in
    if current_user.is_authenticated:
        try:
            save_user_analysis(
                current_user.id, research_question, recommended_model, analysis_goal, dependent_variable_type,
                independent_variables, sample_size, missing_data, data_distribution, relationship_type,
                variables_correlated
            )
            flash('Your analysis has been saved to your profile.', 'info')
        except Exception as e:
            flash(f'Could not save analysis to your profile: {str(e)}', 'danger')
    # Filter for alternative models (don't include the primary recommendation)
    if recommended_model in MODEL_DATABASE:
        # Find similar models based on the same analysis goal and dependent variable type
        # but avoid recommending the same model as the primary recommendation
        similar_models = {
            model_name: model for model_name, model in MODEL_DATABASE.items()
            if (model_name != recommended_model and
                analysis_goal in model.get('analysis_goals', []) and
                (not dependent_variable_type or dependent_variable_type in model.get('dependent_variable', [])))
        }
        # If we have alternative models from the recommendation engine, use those
        # Otherwise, fall back to similar models based on metadata
        if not alternative_models:
            alternative_models = list(similar_models.keys())[:3]
        # Verify all alternative models exist in the database
        alternative_models = [model for model in alternative_models if model in MODEL_DATABASE]
    else:
        alternative_models = []
    return render_template(
        'results.html',
        research_question=research_question,
        analysis_goal=analysis_goal,
        dependent_variable_type=dependent_variable_type,
        independent_variables=independent_variables,
        sample_size=sample_size,
        missing_data=missing_data,
        data_distribution=data_distribution,
        relationship_type=relationship_type,
        variables_correlated=variables_correlated,
        recommended_model=recommended_model,
        explanation=explanation,
        MODEL_DATABASE=MODEL_DATABASE,
        alternative_models=alternative_models
    )
@main.route('/user_analysis/<int:analysis_id>')
@login_required
def user_analysis(analysis_id):
    """View a specific analysis from user history"""
    analysis = Analysis.query.get_or_404(analysis_id)
    # Security check - ensure users can only see their own analyses
    if analysis.user_id != current_user.id:
        return render_template('error.html', error="Unauthorized access")
    # Get model database from app config
    MODEL_DATABASE = current_app.config.get('MODEL_DATABASE', {})
    # Verify the model exists in the database
    recommended_model = analysis.recommended_model
    if recommended_model not in MODEL_DATABASE:
        # Use a fallback model if the original one doesn't exist
        recommended_model = get_default_model(analysis.analysis_goal, analysis.dependent_variable)
    # Create a custom explanation for historical view
    explanation = f"""
    <strong>Historical Analysis from {analysis.created_at.strftime('%Y-%m-%d %H:%M:%S')}</strong><br>
    This is a recommendation previously generated based on your inputs for:
    <ul>
        <li>Analysis Goal: {analysis.analysis_goal}</li>
        <li>Dependent Variable: {analysis.dependent_variable}</li>
        <li>Sample Size: {analysis.sample_size}</li>
    </ul>
    """
    independent_variables = json.loads(analysis.independent_variables)
    variables_correlated = getattr(analysis, 'variables_correlated', 'unknown')
    return render_template('results.html',
                         research_question=analysis.research_question,
                         recommended_model=recommended_model,
                         explanation=explanation,
                         MODEL_DATABASE=MODEL_DATABASE,
                         analysis_goal=analysis.analysis_goal,
                         dependent_variable_type=analysis.dependent_variable,
                         independent_variables=independent_variables,
                         sample_size=analysis.sample_size,
                         missing_data=analysis.missing_data,
                         data_distribution=analysis.data_distribution,
                         relationship_type=analysis.relationship_type,
                         variables_correlated=variables_correlated)
@main.route('/history')
def history():
    """View analysis history"""
    try:
        if current_user.is_authenticated:
            # For logged-in users, redirect to their profile which shows their analyses
            return redirect(url_for('main.profile'))
        else:
            # For guests, redirect to login with a message
            flash('Please log in to view your analysis history.', 'info')
            return redirect(url_for('auth.login', next=url_for('main.history')))
    except Exception as e:
        return render_template('error.html', error=str(e))
@main.route('/models/<group_name>')
def models_in_group(group_name):
    """Display models belonging to a specific group."""
    model_database = current_app.config.get('MODEL_DATABASE', {})
    # Validate group_name
    if group_name not in MODEL_GROUPS:
        flash(f"Invalid model group: {group_name}", "danger")
        return redirect(url_for('main.home')) # Or show a 404 page
    # Get model names for the requested group
    group_model_names = MODEL_GROUPS[group_name]
    # Filter the main database to get details for models in this group
    models_in_group_details = {
        name: details for name, details in model_database.items()
        if name in group_model_names
    }
    # Sort models within the group alphabetically
    sorted_models = sorted(models_in_group_details.items())
    return render_template(
        'models_list.html',
        models=sorted_models,
        group_name=group_name # Pass group name for the title
    )
@main.route('/model/<model_name>')
def model_details(model_name):
    """Display details for a specific model"""
    try:
        model_info = get_model_details(model_name)
        if model_info:
            return render_template('model_details.html',
                                 model_name=model_name,
                                 model_details=model_info)
        else:
            return render_template('error.html', error="Model not found")
    except Exception as e:
        return render_template('error.html', error=str(e))
@main.route('/model/<model_name>/interpretation')
def model_interpretation(model_name):
    """Display interpretation guide for a specific model"""
    try:
        model_info = get_model_details(model_name)
        if not model_info:
            return render_template('error.html', error="Model not found")
        # Import the interpretation utilities
        from utils.interpretation import generate_interpretation_data
        # Generate interpretation data
        interpretation_data = generate_interpretation_data(model_name, model_info)
        return render_template('model_interpretation.html',
                             model_name=model_name,
                             model_details=model_info,
                             interpretation=interpretation_data)
    except Exception as e:
        return render_template('error.html', error=str(e))
@main.route('/model/<model_name>/download-interpretation')
def download_interpretation(model_name):
    """Generate and download interpretation guide as HTML file"""
    try:
        model_info = get_model_details(model_name)
        if not model_info:
            return render_template('error.html', error="Model not found")
        # Import the interpretation utilities
        from utils.interpretation import generate_interpretation_data
        # Generate interpretation data
        interpretation_data = generate_interpretation_data(model_name, model_info)
        # Render the interpretation guide
        html_content = render_template('model_interpretation.html',
                                     model_name=model_name,
                                     model_details=model_info,
                                     interpretation=interpretation_data)
        # Create response with HTML content
        from flask import make_response
        response = make_response(html_content)
        response.headers["Content-Disposition"] = f"attachment; filename={model_name.replace(' ', '_')}_interpretation_guide.html"
        response.headers["Content-Type"] = "text/html"
        return response
    except Exception as e:
        return render_template('error.html', error=str(e))
@main.route('/history/<int:index>')
def view_history_result(index):
    """View a specific result from history"""
    if not current_user.is_authenticated:
        # For guests, redirect to login with a message
        flash('Please log in to view analysis details.', 'info')
        return redirect(url_for('auth.login', next=url_for('main.history')))
    try:
        # For logged-in users, redirect to user's own analyses
        return redirect(url_for('main.profile'))
    except Exception as e:
        return render_template('error.html', error=str(e))
@main.route('/search')
def search():
    """Search models and static pages by keywords across metadata fields."""
    query = request.args.get('q', '').strip()
    model_db = current_app.config.get('MODEL_DATABASE', {})
    results = []
    if query:
        q_lower = query.lower()
        for name, info in model_db.items():
            found = False
            # Check model name
            if q_lower in name.lower():
                found = True
            # Check description
            elif q_lower in info.get('description', '').lower():
                found = True
            else:
                # Check list/string fields
                for field in ['analysis_goals','dependent_variable','relationship_type','missing_data','data_distribution']:
                    vals = info.get(field)
                    if isinstance(vals, list):
                        if any(q_lower in str(v).lower() for v in vals):
                            found = True
                            break
                    elif isinstance(vals, str) and q_lower in vals.lower():
                        found = True
                        break
                # Check independent variables
                if not found and isinstance(info.get('independent_variables'), list):
                    if any(q_lower in str(v).lower() for v in info.get('independent_variables')):
                        found = True
            if found:
                results.append((name, info))
    # Search static pages by name
    pages = [
        {'name': 'Home', 'url': url_for('main.home')},
        {'name': 'Analysis Form', 'url': url_for('main.analysis_form')},
        {'name': 'History', 'url': url_for('main.history')},
        {'name': 'Experts', 'url': url_for('expert.experts_list')},
        {'name': 'Questionnaire Designer', 'url': url_for('questionnaire.index')},
        {'name': 'Contact Us', 'url': url_for('main.contact')}
    ]
    page_results = [p for p in pages if query.lower() in p['name'].lower()]
    return render_template('search_results.html', query=query, results=results, page_results=page_results)
@main.route('/api/search')
def search_api():
    """Return JSON list of model names matching query for autocomplete."""
    q = request.args.get('q', '').strip()
    model_db = current_app.config.get('MODEL_DATABASE', {})
    suggestions = []
    if q:
        for name, info in model_db.items():
            if q.lower() in name.lower():
                suggestions.append({
                    'name': name,
                    'url': url_for('main.model_details', model_name=name)
                })
    return jsonify(suggestions)
@main.route('/contact')
def contact():
    """Render the Contact Us page with support email and social links."""
    return render_template('contact.html')
@main.route('/analysis-form')
def analysis_form():
    return render_template('analysis_form.html')
def save_user_analysis(user_id, research_question, recommended_model, analysis_goal, dependent_variable_type,
                   independent_variables, sample_size, missing_data, data_distribution, relationship_type,
                   variables_correlated='unknown'):
    """Save user analysis to database"""
    try:
        analysis = Analysis(
            user_id=user_id,  # type: ignore
            research_question=research_question,  # type: ignore
            analysis_goal=analysis_goal,  # type: ignore
            dependent_variable=dependent_variable_type,  # type: ignore
            independent_variables=json.dumps(independent_variables),  # type: ignore
            sample_size=sample_size,  # type: ignore
            missing_data=missing_data,  # type: ignore
            data_distribution=data_distribution,  # type: ignore
            relationship_type=relationship_type,  # type: ignore
            variables_correlated=variables_correlated,  # type: ignore
            recommended_model=recommended_model  # type: ignore
        )
        db.session.add(analysis)
        db.session.commit()
        return True
    except Exception as e:
        current_app.logger.error(f"Error saving analysis: {e}")
        db.session.rollback()
        raise