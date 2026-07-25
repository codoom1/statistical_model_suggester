from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from flask_login import login_required, current_user
from models import db, User, Analysis, get_model_details
from datetime import datetime
import hashlib
import json
import logging
from collections import OrderedDict
from urllib.parse import unquote
from sqlalchemy.exc import SQLAlchemyError

from utils.ai_service import OpenAIServiceError, is_ai_enabled
from utils.ai_usage import consume_user_ai_quota
from utils.recommendation_ai import review_recommendation
from utils.validation import is_valid_email, normalize_email


logger = logging.getLogger(__name__)
main = Blueprint('main', __name__)

GOAL_COMPATIBILITY = {
    'predict': {
        'predict',
        'robust_prediction',
        'flexible_prediction',
        'predict survival probabilities',
    },
    'classify': {
        'classify',
        'separate groups',
        'estimate probabilities',
    },
    'explore': {
        'explore',
        'explain',
        'explore relationships',
        'identify latent variables',
        'reduce',
        'reduce dimensions',
        'dimensionality reduction',
        'visualize',
        'visualize relationships',
        'explore similarity structures',
    },
    'cluster': {'cluster'},
    'hypothesis_test': {
        'test',
        'compare',
        'compare means',
        'test group differences',
        'evaluate treatment effects',
        'compare adjusted means',
        'compare multivariate means',
        'test overall group differences',
    },
    'non_parametric': {'test', 'compare', 'rank'},
    'time_series': {'time_series'},
    'risk_assessment': {
        'estimate financial risk',
        'measure tail risk',
        'simulate risk',
    },
    'forensic_analysis': {
        'evaluate forensic evidence',
        'detect forensic anomalies',
        'screen financial records',
    },
}

OUTCOME_COMPATIBILITY = {
    'continuous': {'continuous'},
    'categorical': {'categorical', 'categorical (limited)', 'multiclass'},
    'binary': {'binary', 'binary (with link functions)'},
    'count': {'count'},
    'time_series': {'continuous'},
    'time_to_event': {'time-to-event', 'censored'},
}

MISSING_DATA_COMPATIBILITY = {
    'none': {'none'},
    'little': {'none', 'random', 'random (MAR)'},
    'moderate': {'random', 'random (MAR)', 'imputed'},
    'substantial': {
        'random',
        'random (MAR)',
        'systematic',
        'imputed',
        'handled_via_FIML',
        'handled_via_imputation',
        'handled_automatically',
    },
}


def _supports_goal(selected, supported):
    return bool(GOAL_COMPATIBILITY.get(selected, {selected}) & set(supported))


def _supports_outcome(selected, supported):
    return bool(
        OUTCOME_COMPATIBILITY.get(selected, {selected}) & set(supported)
    )


def _supports_missing_data(selected, supported):
    return bool(
        MISSING_DATA_COMPATIBILITY.get(selected, {selected}) & set(supported)
    )


def _supports_distribution(selected, supported):
    supported = set(supported)
    if 'any' in supported:
        return True
    if selected == 'unknown':
        return False
    if selected == 'normal':
        return bool(
            {'normal', 'gaussian', 'multivariate normal', 'multivariate_normal'}
            & supported
        )
    if selected == 'non_normal':
        return bool(
            {
                'non_normal',
                'nonparametric',
                'heavy_tailed',
                'asymmetric_laplace',
                'non_normal (with robust estimators)',
                'non_normal (with alternative likelihoods)',
            }
            & supported
        )
    return selected in supported


def _supports_relationship(selected, supported):
    supported = set(supported)
    if 'any' in supported:
        return True
    if selected == 'unknown':
        return False
    if selected == 'non_linear':
        return bool({'non_linear', 'non-linear', 'nonlinear'} & supported)
    return selected in supported


def _decode_path_name(name):
    """Decode one path segment left encoded after Flask's URL decoding."""
    return unquote(name)


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
        'Repeated Measures ANOVA',
    ]),
    ('Regression Models', [
        'Linear Regression',
        'Logistic Regression',
        'Multinomial Regression',
        'Ordinal Regression',
        'Poisson Regression',
        'Ridge Regression',
        'Lasso Regression',
        'Elastic Net Regression',
        'Kernel Regression',
        'Negative Binomial Regression',
        'Zero-Inflated Poisson',
        'Quantile Regression',
        'Partial Least Squares',
        'Generalized Additive Model',
        'Bayesian Quantile Regression',
    ]),
    ('Time Series Models', [
        'Autoregressive (AR) Model',
        'Moving Average (MA) Model',
        'Autoregressive Moving Average (ARMA) Model',
        'ARIMA',
        'Seasonal ARIMA (SARIMA)',
        'ARIMAX',
        'Exponential Smoothing',
        'Holt-Winters Model',
        'Vector Autoregression (VAR)',
        'Vector Error Correction Model (VECM)',
        'State Space Model',
        'Bayesian Structural Time Series (BSTS)',
        'Dynamic Linear Model (DLM)',
        'Prophet',
        'GARCH',
        'Stochastic Volatility Model',
        'Threshold Autoregressive (TAR) Model',
        'Markov Switching Model',
        'Recurrent Neural Network for Time Series',
        'Long Short-Term Memory (LSTM)',
        'Gated Recurrent Unit (GRU)',
        'Temporal Convolutional Network (TCN)',
        'Gradient Boosting for Time Series',
        'Random Forest for Time Series',
        'Wavelet Transform Model',
        'Kalman Filter',
    ]),
    ('Multivariate Analysis', [
        'Principal Component Analysis (PCA)',
        'Factor Analysis',
        'K-Means clustering',
        'Discriminant Analysis',
        'Canonical Correlation',
        'Multidimensional Scaling',
        'Partial Least Squares',
        'Multivariate Analysis of Covariance (MANCOVA)',
        'Multivariate Analysis of Variance (MANOVA)',
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
    ]),
    ('Mixed and Hierarchical Models', [
        'Mixed Effects Model',
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
    ('Risk Models', [
        'Value at Risk (VaR)',
        'Conditional Value at Risk (CVaR)',
        'Monte Carlo Risk Simulation',
        'GARCH',
    ]),
    ('Forensic Models', [
        'Likelihood Ratio Evidence Model',
        'Benford Law Analysis',
        'Forensic Anomaly Detection',
    ]),
])
# Make model groups available to all templates
@main.context_processor
def inject_model_groups():
    return dict(model_groups=MODEL_GROUPS)
# Analysis history is stored in PostgreSQL for authenticated users. Guest
# recommendations are intentionally not persisted.
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
        'bayesian': ['Bayesian Hierarchical Regression', 'Bayesian Model Averaging',
                    'Bayesian Quantile Regression', 'Bayesian Linear Regression',
                    'Bayesian Additive Regression Trees (BART)'],
        'hierarchical': ['Mixed Effects Model', 'Bayesian Hierarchical Regression'],
        'neural_network': ['Neural Networks'],
        'nonparametric': ['Support Vector Machines (SVM)', 'K-Nearest Neighbors (KNN)', 'Kernel Regression'],
        'dimensionality_reduction': ['Principal Component Analysis (PCA)', 'Factor Analysis', 'Multidimensional Scaling'],
        'clustering': ['K-Means clustering'],
        'time_series_classical': [
            'Autoregressive (AR) Model', 'Moving Average (MA) Model',
            'Autoregressive Moving Average (ARMA) Model', 'ARIMA',
            'Seasonal ARIMA (SARIMA)', 'ARIMAX', 'Exponential Smoothing',
            'Holt-Winters Model', 'Prophet',
        ],
        'time_series_multivariate': [
            'Vector Autoregression (VAR)',
            'Vector Error Correction Model (VECM)',
        ],
        'time_series_state_space': [
            'State Space Model', 'Bayesian Structural Time Series (BSTS)',
            'Dynamic Linear Model (DLM)', 'Kalman Filter',
        ],
        'time_series_volatility': ['GARCH', 'Stochastic Volatility Model'],
        'time_series_regime': [
            'Threshold Autoregressive (TAR) Model',
            'Markov Switching Model',
        ],
        'time_series_deep': [
            'Recurrent Neural Network for Time Series',
            'Long Short-Term Memory (LSTM)', 'Gated Recurrent Unit (GRU)',
            'Temporal Convolutional Network (TCN)',
        ],
        'time_series_ensemble': [
            'Gradient Boosting for Time Series', 'Random Forest for Time Series',
        ],
        'time_series_signal': ['Wavelet Transform Model'],
        'risk': ['Value at Risk (VaR)', 'Conditional Value at Risk (CVaR)',
                 'Monte Carlo Risk Simulation'],
        'forensic': ['Likelihood Ratio Evidence Model', 'Benford Law Analysis',
                     'Forensic Anomaly Detection'],
        'hypothesis_testing': ['T test', 'Chi-Square Test', 'Mann-Whitney U Test', 'Kruskal-Wallis Test',
                              'Analysis of Variance (ANOVA)', 'Analysis of Covariance (ANCOVA)']
    }
    # Build a reverse lookup of model to family
    model_to_family = {}
    for family, models in model_families.items():
        for model in models:
            model_to_family[model] = family
    time_series_models = {
        model
        for family, models in model_families.items()
        if family.startswith('time_series_')
        for model in models
    }
    # Define clustering models (these don't require a dependent variable)
    clustering_models = ['K-Means clustering']
    # For clustering analysis, ensure we have a default dependent variable if not provided
    if analysis_goal == 'cluster' and not dependent_variable:
        dependent_variable = 'continuous'  # A sensible default for clustering
    # Score models based on compatibility
    model_scores = {}
    for model_name, model in MODEL_DATABASE.items():
        score = 0
        current_app.logger.debug(f"SCORING {model_name}: Starting score = {score}")
        if (
            model_name in time_series_models
            and analysis_goal != 'time_series'
            and not (
                model_name in {'GARCH', 'Stochastic Volatility Model'}
                and analysis_goal == 'risk_assessment'
            )
        ):
            continue
        # Check analysis goal compatibility - heavily weighted
        if _supports_goal(analysis_goal, model.get('analysis_goals', [])):
            score += 3
            current_app.logger.debug(f"  + Analysis goal match: +3 → {score}")
        else:
            # If analysis goal doesn't match, this model is less relevant
            current_app.logger.debug(f"  × Skipping {model_name}: analysis goal mismatch")
            continue  # Skip models that don't match the primary analysis goal
        # Special handling for clustering models when the goal is 'cluster'
        is_clustering_model = model_name in clustering_models
        # Check dependent variable compatibility - heavily weighted
        # Skip this check for clustering models when the goal is 'cluster'
        if is_clustering_model and analysis_goal == 'cluster':
            # Clustering models get a bonus instead of being checked for dependent variable
            score += 3
            current_app.logger.debug(f"  + Clustering model bonus: +3 → {score}")
        elif _supports_outcome(
            dependent_variable,
            model.get('dependent_variable', []),
        ):
            score += 3
            current_app.logger.debug(f"  + Dependent variable match: +3 → {score}")
        else:
            # If dependent variable type doesn't match, this model is less relevant
            current_app.logger.debug(f"  × Skipping {model_name}: dependent variable mismatch")
            continue  # Skip models that don't match the dependent variable type
        # Check relationship type compatibility - important factor
        if _supports_relationship(
            relationship_type,
            model.get('relationship_type', []),
        ):
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
        if variables_correlated == 'yes' and model_name in ['Principal Component Analysis (PCA)',
                                                          'Factor Analysis', 'Partial Least Squares',
                                                          'Random Forest', 'Gradient Boosting',
                                                          'XGBoost', 'CatBoost', 'LightGBM']:
            score += 1.5
            current_app.logger.debug(f"  + Correlated variable compatibility: +1.5 → {score}")
        # Models that assume independent variables - slight penalty for correlated data
        if variables_correlated == 'yes' and model_name == 'Linear Regression':
            score -= 0.5
            current_app.logger.debug(f"  - Correlated variable penalty: -0.5 → {score}")
        # Boost for Linear Regression in standard prediction scenarios with normal data and linear relationships
        if model_name == 'Linear Regression' and variables_correlated == 'no' and \
           relationship_type == 'linear' and data_distribution == 'normal' and \
           missing_data in ['none', 'little'] and dependent_variable == 'continuous':
            score += 5.0
            current_app.logger.debug(f"  + Linear Regression boost: +5.0 → {score}")
        # Strong boost for K-Means in clustering scenarios.
        if model_name == 'K-Means clustering' and analysis_goal == 'cluster':
            score += 5.0
            current_app.logger.debug(
                f"CLUSTER BONUS: {model_name} +5.0 for {analysis_goal} analysis"
            )
        if model_name == 'ARIMA' and analysis_goal == 'time_series':
            score += 3.0
            current_app.logger.debug(
                f"TIME SERIES BASELINE BONUS: {model_name} +3.0"
            )
        # Penalty for non-clustering models in exploratory or clustering scenarios
        if (analysis_goal == 'explore' or analysis_goal == 'cluster') and model_name not in ['K-Means clustering', 'Factor Analysis', 'Principal Component Analysis (PCA)',
                                                           'Multidimensional Scaling']:
            score -= 5.0
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
        if _supports_missing_data(missing_data, model.get('missing_data', [])):
            score += 1.5
            current_app.logger.debug(f"  + Missing data compatibility: +1.5 → {score}")
        # Check data distribution compatibility
        if _supports_distribution(
            data_distribution,
            model.get('data_distribution', []),
        ):
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
                           'Quantile Regression']:
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
                                                               'Gradient Boosting', 'Support Vector Machines (SVM)',
                                                               'Bayesian Additive Regression Trees (BART)']:
            score += 1.5
            current_app.logger.debug(f"  + Non-linear model bonus: +1.5 → {score}")
        # Add a penalty for overused models to promote diversity
        if model_name == 'Linear Regression':
            score -= 0.25
            current_app.logger.debug(f"  - Linear Regression penalty: -0.25 → {score}")
        # Give a strong bonus to clustering models when the goal is clustering
        if (analysis_goal == 'explore' or analysis_goal == 'cluster') and model_name in clustering_models:
            score += 10.0  # Very significant boost to ensure clustering models win for cluster analysis
            current_app.logger.debug(f"  + Clustering model for cluster goal: +10.0 → {score:.4f}")
        # Special case for Neural Networks - reduce score for clustering tasks
        if (analysis_goal == 'explore' or analysis_goal == 'cluster') and model_name == 'Neural Networks':
            score -= 5.0  # Significant penalty for neural networks in clustering tasks
            current_app.logger.debug(f"  - Neural Networks penalty for clustering: -5.0 → {score:.4f}")
        model_scores[model_name] = score
    # Get top models
    # First, identify the best matching model
    if model_scores:
        # Sort models by score
        sorted_models = sorted(
            model_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
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
            alternatives = ['Ridge Regression', 'Random Forest', 'XGBoost', 'Gradient Boosting']
        elif dependent_variable == 'binary':
            alternatives = ['Random Forest', 'Support Vector Machines (SVM)', 'XGBoost', 'Neural Networks']
        elif dependent_variable == 'count':
            alternatives = ['Negative Binomial Regression', 'Zero-Inflated Poisson', 'Quantile Regression']
        elif dependent_variable == 'ordinal':
            alternatives = ['Multinomial Regression', 'Neural Networks', 'Ordinal Regression']
        elif dependent_variable == 'time_to_event':
            alternatives = ['Kaplan-Meier Curve', 'Cox Proportional Hazards Model']
    elif analysis_goal == 'classify':
        if dependent_variable == 'binary':
            alternatives = ['Random Forest', 'Support Vector Machines (SVM)', 'XGBoost', 'Neural Networks', 'Gradient Boosting']
        elif dependent_variable == 'categorical':
            alternatives = ['Random Forest', 'Neural Networks', 'Support Vector Machines (SVM)', 'XGBoost']
    elif analysis_goal == 'explore':
        alternatives = ['Factor Analysis', 'Multidimensional Scaling', 'Principal Component Analysis (PCA)']
    elif analysis_goal == 'cluster':
        alternatives = ['K-Means clustering']
    elif analysis_goal == 'hypothesis_test':
        if dependent_variable == 'continuous':
            alternatives = ['Analysis of Variance (ANOVA)', 'Mann-Whitney U Test', 'T test']
        elif dependent_variable == 'categorical':
            alternatives = ['Chi-Square Test']
    elif analysis_goal == 'non_parametric':
        alternatives = ['Kruskal-Wallis Test', 'Mann-Whitney U Test']
    elif analysis_goal == 'time_series':
        alternatives = [
            'ARIMA',
            'Seasonal ARIMA (SARIMA)',
            'Exponential Smoothing',
            'State Space Model',
        ]
    elif analysis_goal == 'risk_assessment':
        alternatives = ['Value at Risk (VaR)', 'Conditional Value at Risk (CVaR)',
                        'Monte Carlo Risk Simulation', 'GARCH']
    elif analysis_goal == 'forensic_analysis':
        alternatives = ['Likelihood Ratio Evidence Model', 'Benford Law Analysis',
                        'Forensic Anomaly Detection']
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
    if _supports_goal(analysis_goal, model_info.get('analysis_goals', [])):
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
    if _supports_missing_data(missing_data, model_info.get('missing_data', [])):
        reasons.append(f"It can handle {missing_data} missing data patterns")
    if _supports_distribution(
        data_distribution,
        model_info.get('data_distribution', []),
    ):
        reasons.append(f"It is appropriate for {data_distribution} data distribution")
    if _supports_relationship(
        relationship_type,
        model_info.get('relationship_type', []),
    ):
        reasons.append(f"It can model {relationship_type} relationships")
    # Add reason related to correlated variables if specified
    if variables_correlated == 'yes' and model_name in ['Elastic Net Regression', 'Ridge Regression', 'Lasso Regression',
                                                      'Principal Component Analysis (PCA)', 'Factor Analysis']:
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
            target_models = ['Logistic Regression', 'Support Vector Machines (SVM)']
        elif dependent_variable == 'count':
            target_models = ['Poisson Regression', 'Negative Binomial Regression']
        elif dependent_variable == 'ordinal':
            target_models = ['Ordinal Regression', 'Multinomial Regression']
        elif dependent_variable == 'time_to_event':
            target_models = ['Cox Proportional Hazards Model', 'Kaplan-Meier Curve']
    elif analysis_goal == 'classify':
        if dependent_variable == 'binary':
            target_models = ['Logistic Regression', 'Support Vector Machines (SVM)']
        elif dependent_variable == 'categorical':
            target_models = ['Multinomial Regression', 'Random Forest']
    elif analysis_goal == 'explore':
        target_models = ['Principal Component Analysis (PCA)', 'Factor Analysis']
    elif analysis_goal == 'cluster':
        target_models = ['K-Means clustering']
    elif analysis_goal == 'hypothesis_test':
        if dependent_variable == 'continuous':
            target_models = ['T test', 'Analysis of Variance (ANOVA)']
        elif dependent_variable == 'categorical':
            target_models = ['Chi-Square Test']
    elif analysis_goal == 'non_parametric':
        target_models = ['Mann-Whitney U Test', 'Kruskal-Wallis Test']
    elif analysis_goal == 'time_series':
        target_models = ['ARIMA', 'Exponential Smoothing']
    elif analysis_goal == 'risk_assessment':
        target_models = ['Value at Risk (VaR)', 'Conditional Value at Risk (CVaR)',
                         'Monte Carlo Risk Simulation']
    elif analysis_goal == 'forensic_analysis':
        target_models = ['Likelihood Ratio Evidence Model', 'Forensic Anomaly Detection',
                         'Benford Law Analysis']
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
        email = normalize_email(request.form.get('email'))
        if not is_valid_email(email):
            flash('Please provide a valid email address.', 'danger')
            return redirect(url_for('main.profile'))
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
    if independent_variables == ['mixed']:
        independent_variables = ['continuous', 'categorical']
    # Get other attributes
    sample_size = request.form.get('sample_size', '')
    missing_data = request.form.get('missing_data', '')
    data_distribution = request.form.get('data_distribution', '')
    relationship_type = request.form.get('relationship_type', '')
    variables_correlated = request.form.get('variables_correlated', 'unknown')
    use_ai_review = request.form.get('use_ai_review') == 'on'
    # Get model database from app config
    MODEL_DATABASE = current_app.config.get('MODEL_DATABASE', {})
    allowed_goals = set(GOAL_COMPATIBILITY)
    allowed_outcomes = set(OUTCOME_COMPATIBILITY)
    allowed_missing_data = set(MISSING_DATA_COMPATIBILITY)
    allowed_distributions = {'unknown', 'normal', 'non_normal'}
    allowed_relationships = {
        'unknown',
        'linear',
        'non_linear',
        'hierarchical',
    }
    allowed_predictors = {'continuous', 'categorical', 'binary'}
    # For clustering analysis, dependent variable can be empty
    # If it's empty, set it to 'continuous' which works well with clustering models
    if analysis_goal == 'cluster' and not dependent_variable_type:
        dependent_variable_type = 'continuous'
    # Require essential inputs
    if not (research_question and analysis_goal):
        flash('Please provide all required information to get a recommendation.', 'warning')
        return redirect(url_for('main.analysis_form'))
    if len(research_question) > 500 or analysis_goal not in allowed_goals:
        flash('Please provide a valid research question and analysis goal.', 'warning')
        return redirect(url_for('main.analysis_form'))
    # For non-clustering analysis, require dependent variable
    if analysis_goal != 'cluster' and not dependent_variable_type:
        flash('Please select what type of outcome you are measuring.', 'warning')
        return redirect(url_for('main.analysis_form'))
    if (
        dependent_variable_type not in allowed_outcomes
        or missing_data not in allowed_missing_data
        or data_distribution not in allowed_distributions
        or relationship_type not in allowed_relationships
        or not set(independent_variables).issubset(allowed_predictors)
    ):
        flash(
            'Some study-design fields were missing or invalid. Please review the form.',
            'warning',
        )
        return redirect(url_for('main.analysis_form'))
    if sample_size:
        try:
            if int(sample_size) < 1:
                raise ValueError
        except (TypeError, ValueError):
            flash('Sample size must be a positive whole number.', 'warning')
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
    # Filter for alternative models (don't include the primary recommendation)
    if recommended_model in MODEL_DATABASE:
        # Find similar models based on the same analysis goal and dependent variable type
        # but avoid recommending the same model as the primary recommendation
        similar_models = {
            model_name: model for model_name, model in MODEL_DATABASE.items()
            if (model_name != recommended_model and
                _supports_goal(
                    analysis_goal,
                    model.get('analysis_goals', []),
                ) and
                (
                    not dependent_variable_type
                    or _supports_outcome(
                        dependent_variable_type,
                        model.get('dependent_variable', []),
                    )
                ))
        }
        # If we have alternative models from the recommendation engine, use those
        # Otherwise, fall back to similar models based on metadata
        if not alternative_models:
            alternative_models = list(similar_models.keys())[:3]
        # Verify all alternative models exist in the database
        alternative_models = [model for model in alternative_models if model in MODEL_DATABASE]
    else:
        alternative_models = []

    ai_review = None
    ai_review_status = 'not_requested'
    if use_ai_review:
        if not current_user.is_authenticated:
            ai_review_status = 'authentication_required'
        elif not is_ai_enabled():
            ai_review_status = 'unavailable'
        else:
            try:
                allowed, _ = consume_user_ai_quota(current_user.id)
                if not allowed:
                    ai_review_status = 'quota_reached'
                else:
                    candidate_models = [
                        recommended_model,
                        *alternative_models,
                    ]
                    safety_identifier = hashlib.sha256(
                        (
                            f"{current_app.config['SECRET_KEY']}:"
                            f"{current_user.id}"
                        ).encode()
                    ).hexdigest()
                    ai_review = review_recommendation(
                        research_question=research_question,
                        analysis_inputs={
                            'analysis_goal': analysis_goal,
                            'dependent_variable_type': dependent_variable_type,
                            'independent_variable_types': independent_variables,
                            'sample_size': sample_size,
                            'missing_data': missing_data,
                            'data_distribution': data_distribution,
                            'relationship_type': relationship_type,
                            'variables_correlated': variables_correlated,
                        },
                        candidate_models=candidate_models,
                        model_database=MODEL_DATABASE,
                        safety_identifier=safety_identifier,
                    )
                    if ai_review.get('recommended_model') not in candidate_models:
                        raise ValueError(
                            "AI review selected a model outside the shortlist."
                        )
                    rules_engine_model = recommended_model
                    recommended_model = ai_review['recommended_model']
                    if recommended_model != rules_engine_model:
                        alternative_models = [
                            rules_engine_model,
                            *[
                                model for model in alternative_models
                                if model != recommended_model
                            ],
                        ][:4]
                        explanation = generate_explanation(
                            recommended_model,
                            analysis_goal,
                            dependent_variable_type,
                            independent_variables,
                            sample_size,
                            missing_data,
                            data_distribution,
                            relationship_type,
                            variables_correlated,
                        )
                    ai_review_status = 'completed'
            except SQLAlchemyError:
                db.session.rollback()
                logger.exception(
                    "Could not record recommendation AI usage for user %s.",
                    current_user.id,
                )
                ai_review_status = 'unavailable'
            except (OpenAIServiceError, ValueError):
                logger.warning(
                    "AI recommendation review failed for user %s.",
                    current_user.id,
                    exc_info=True,
                )
                ai_review_status = 'unavailable'

    # Save the final, validated recommendation if the user is logged in.
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
        alternative_models=alternative_models,
        ai_review=ai_review,
        ai_review_status=ai_review_status,
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
@main.route('/models')
def models_index():
    """Display every available statistical model."""
    model_database = current_app.config.get('MODEL_DATABASE', {})
    grouped_models = []
    included_models = set()
    for group_name, model_names in MODEL_GROUPS.items():
        group_models = [
            (name, model_database[name])
            for name in model_names
            if name in model_database and name not in included_models
        ]
        if group_models:
            grouped_models.append((group_name, sorted(group_models)))
            included_models.update(name for name, _details in group_models)

    uncategorized_models = sorted(
        (name, details)
        for name, details in model_database.items()
        if name not in included_models
    )
    if uncategorized_models:
        grouped_models.append(('Additional Methods', uncategorized_models))

    return render_template(
        'models_list.html',
        models=sorted(model_database.items()),
        group_name='All Statistical Models',
        grouped_models=grouped_models,
        is_all_models=True,
    )


@main.route('/models/<group_name>')
def models_in_group(group_name):
    """Display models belonging to a specific group."""
    model_database = current_app.config.get('MODEL_DATABASE', {})
    # Some clients and previously rendered links can encode the path segment
    # twice, leaving a literal ``%20`` after Flask's first URL decode.
    group_name = _decode_path_name(group_name)
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
        group_name=group_name,
        is_all_models=False,
    )
@main.route('/model/<model_name>')
def model_details(model_name):
    """Display details for a specific model"""
    try:
        model_name = _decode_path_name(model_name)
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
        model_name = _decode_path_name(model_name)
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
        model_name = _decode_path_name(model_name)
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
