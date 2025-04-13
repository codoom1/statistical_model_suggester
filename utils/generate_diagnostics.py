#!/usr/bin/env python
"""
Generate and save diagnostic plots for all model types.
This script creates PNG image files for diagnostic plots for each model type.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path so we can import our utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.synthetic_data_utils import synthetic_data_to_numpy, synthetic_data_to_dataframe, get_feature_names
from utils.diagnostic_plots.linear_regression import generate_linear_regression_plots
from utils.diagnostic_plots.logistic_regression import generate_logistic_regression_plots
from utils.diagnostic_plots.anova import generate_anova_plots
from utils.diagnostic_plots.random_forest import generate_random_forest_plots
from utils.diagnostic_plots.pca import generate_pca_plots
from utils.diagnostic_plots.poisson_regression import generate_poisson_regression_plots
from utils.diagnostic_plots.ordinal_regression import generate_ordinal_regression_plots
from utils.diagnostic_plots.ttest import generate_ttest_plots
from utils.diagnostic_plots.chi_square import generate_chi_square_plots
from utils.diagnostic_plots.mann_whitney import generate_mann_whitney_plots
from utils.diagnostic_plots.kruskal_wallis import generate_kruskal_wallis_plots
from utils.diagnostic_plots.cluster_analysis import generate_cluster_analysis_plots
# Import all other diagnostic plot modules
from utils.diagnostic_plots.bayesian_model_averaging import generate_bma_plots
from utils.diagnostic_plots.bayesian_quantile_regression import generate_bayesian_quantile_plots
from utils.diagnostic_plots.bayesian_hierarchical_regression import generate_bayesian_hierarchical_plots
from utils.diagnostic_plots.kaplan_meier import generate_kaplan_meier_plots
from utils.diagnostic_plots.cox_proportional_hazards import generate_cox_ph_plots
from utils.diagnostic_plots.repeated_measures_anova import generate_repeated_measures_anova_plots
from utils.diagnostic_plots.elastic_net_regression import generate_elastic_net_plots
from utils.diagnostic_plots.lasso_regression import generate_lasso_plots
from utils.diagnostic_plots.ridge_regression import generate_ridge_plots
from utils.diagnostic_plots.multidimensional_scaling import generate_mds_plots
from utils.diagnostic_plots.bayesian_additive_regression_trees import generate_bart_plots
from utils.diagnostic_plots.k_nearest_neighbors import generate_knn_plots
from utils.diagnostic_plots.decision_trees import generate_decision_tree_plots
from utils.diagnostic_plots.xgboost import generate_xgboost_plots
from utils.diagnostic_plots.catboost import generate_catboost_plots
from utils.diagnostic_plots.lightgbm import generate_lightgbm_plots
from utils.diagnostic_plots.canonical_correlation import generate_canonical_correlation_plots
from utils.diagnostic_plots.mancova import generate_mancova_plots
from utils.diagnostic_plots.manova import generate_manova_plots
from utils.diagnostic_plots.ancova import generate_ancova_plots
from utils.diagnostic_plots.discriminant_analysis import generate_discriminant_analysis_plots
from utils.diagnostic_plots.arima import generate_arima_plots
from utils.diagnostic_plots.path_analysis import generate_path_analysis_plots
from utils.diagnostic_plots.structural_equation_modeling import generate_sem_plots
from utils.diagnostic_plots.naive_bayes import generate_naive_bayes_plots
from utils.diagnostic_plots.neural_network import generate_neural_network_plots
from utils.diagnostic_plots.svm import generate_svm_plots
from utils.diagnostic_plots.gradient_boosting import generate_gradient_boosting_plots
from utils.diagnostic_plots.regularized_regression import generate_regularized_regression_plots
from utils.diagnostic_plots.survival_analysis import generate_survival_analysis_plots
from utils.diagnostic_plots.time_series import generate_time_series_plots
from utils.diagnostic_plots.bayesian_regression import generate_bayesian_regression_plots
from utils.diagnostic_plots.factor_analysis import generate_factor_analysis_plots
from utils.diagnostic_plots.kernel_regression import generate_kernel_regression_plots
from utils.diagnostic_plots.mixed_effects import generate_mixed_effects_plots
from utils.diagnostic_plots.multinomial_regression import generate_multinomial_regression_plots

def save_plot_from_base64(img_data, filename):
    """Convert base64 image to file and save it
    
    Args:
        img_data: Base64 encoded image string
        filename: Path where to save the image
    """
    import base64
    import io
    from PIL import Image
    
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
    """Generate sample data for demonstration
    
    Args:
        model_type: Type of model ('linear', 'logistic', 'anova', 'random_forest', etc.)
        
    Returns:
        Tuple of data appropriate for the model type
    """
    np.random.seed(42)  # For reproducibility
    
    if model_type == 'linear':
        # Linear regression sample data
        X = np.random.normal(0, 1, (100, 3))
        # y = intercept + X1 + 2*X2 + 0.5*X3 + noise
        y = 2 + X[:, 0] + 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 1, 100)
        return X, y
    
    elif model_type == 'logistic':
        # Logistic regression sample data
        X = np.random.normal(0, 1, (100, 2))
        # log-odds = X1 + 2*X2
        logodds = X[:, 0] + 2 * X[:, 1]
        prob = 1 / (1 + np.exp(-logodds))
        y = (np.random.random(100) < prob).astype(int)
        return X, y
    
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
    
    elif model_type == 'random_forest':
        # Random forest sample data
        X = np.random.normal(0, 1, (100, 5))
        # y is a function of features with some non-linearity
        y = (1.5 * X[:, 0] + 0.5 * X[:, 1]**2 + np.exp(0.5 * X[:, 2]) + 
             np.random.normal(0, 0.5, 100))
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
        is_classification = False
        return X, y, feature_names, is_classification
    
    elif model_type == 'pca':
        # PCA sample data - correlated features
        n = 100
        t = np.linspace(0, 2*np.pi, n)
        x1 = np.sin(t) + np.random.normal(0, 0.1, n)
        x2 = np.cos(t) + np.random.normal(0, 0.1, n)
        x3 = x1 + x2 + np.random.normal(0, 0.1, n)
        X = np.column_stack([x1, x2, x3, np.random.normal(0, 1, (n, 2))])
        feature_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
        return X, feature_names
    
    elif model_type == 'poisson':
        # Poisson regression sample data
        n = 100
        X = np.random.normal(0, 1, (n, 3))
        # log(lambda) = intercept + X1 + 0.5*X2 + 0.25*X3
        log_lambda = 0.5 + X[:, 0] + 0.5 * X[:, 1] + 0.25 * X[:, 2]
        lambda_vals = np.exp(log_lambda)
        y = np.random.poisson(lambda_vals)
        return X, y
    
    elif model_type == 'ordinal':
        # Ordinal regression sample data
        n = 100
        np.random.seed(123)  # Ensure reproducibility
        # Generate features without constant term
        X = np.random.normal(0, 1, (n, 3)) 
        
        # Compute latent variable without a constant term
        z = 0.8 * X[:, 0] - 1.2 * X[:, 1] + 0.5 * X[:, 2] + np.random.logistic(0, 0.8, n)
        
        # Convert to ordinal categories (0-4)
        thresholds = [-2, -0.5, 0.5, 2]
        y = np.zeros(n, dtype=int)
        for i in range(len(thresholds)):
            y += (z > thresholds[i]).astype(int)
        
        return X, y
    
    elif model_type == 'ttest':
        # T-test sample data - two groups with different means
        group1 = np.random.normal(10, 2, 30)
        group2 = np.random.normal(12, 2, 30)
        return group1, group2, ['Group A', 'Group B']
    
    elif model_type == 'chi_square':
        # Chi-square sample data - 2x3 contingency table
        # Rows: Treatment/Control, Columns: Outcome (Success/Partial/Failure)
        observed = np.array([
            [45, 30, 25],  # Treatment
            [25, 35, 40]   # Control
        ])
        row_labels = ['Treatment', 'Control']
        col_labels = ['Success', 'Partial', 'Failure']
        return observed, row_labels, col_labels
    
    elif model_type == 'mann_whitney':
        # Mann-Whitney U test data - two groups with different distributions
        # Group 1: normal distribution
        group1 = np.random.normal(50, 10, 30)
        # Group 2: right-skewed distribution
        group2 = np.random.gamma(shape=5, scale=2, size=30) + 40
        return group1, group2, ['Group A', 'Group B']
    
    elif model_type == 'kruskal_wallis':
        # Kruskal-Wallis test data - multiple groups with different distributions
        group1 = np.random.normal(50, 10, 25)
        group2 = np.random.normal(55, 8, 25)
        group3 = np.random.gamma(shape=7, scale=1.5, size=25) + 45
        return [group1, group2, group3], ['Group A', 'Group B', 'Group C']
    
    elif model_type == 'cluster':
        # Cluster analysis sample data - 3 distinct clusters in 4D space
        n_per_cluster = 50
        # Cluster 1
        cluster1 = np.random.multivariate_normal(
            mean=[0, 0, 0, 0], 
            cov=np.eye(4) * 0.5, 
            size=n_per_cluster
        )
        # Cluster 2
        cluster2 = np.random.multivariate_normal(
            mean=[5, 5, 5, 5], 
            cov=np.eye(4) * 0.5, 
            size=n_per_cluster
        )
        # Cluster 3
        cluster3 = np.random.multivariate_normal(
            mean=[-5, 5, -5, 5], 
            cov=np.eye(4) * 0.5, 
            size=n_per_cluster
        )
        # Combine clusters
        X = np.vstack([cluster1, cluster2, cluster3])
        feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
        return X, feature_names
    
    # New sample data generators for additional models
    
    elif model_type in ['ridge', 'lasso', 'elastic_net', 'regularized']:
        # Sample data for regularized regression models (Ridge, Lasso, Elastic Net)
        n = 150
        p = 20  # High-dimensional
        X = np.random.normal(0, 1, (n, p))
        # Only first 5 features have non-zero coefficients
        beta = np.zeros(p)
        beta[:5] = [1.5, -2, 1, -0.8, 1.2]
        # Add multicollinearity
        X[:, 5] = 0.8 * X[:, 0] + 0.2 * np.random.normal(0, 1, n)
        X[:, 6] = 0.8 * X[:, 1] + 0.2 * np.random.normal(0, 1, n)
        y = X @ beta + np.random.normal(0, 1, n)
        feature_names = [f"Feature {i+1}" for i in range(p)]
        return X, y, feature_names
    
    elif model_type in ['decision_trees', 'xgboost', 'catboost', 'lightgbm', 'gradient_boosting']:
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
    
    elif model_type == 'knn':
        # K-Nearest Neighbors sample data - classification with 3 classes in 2D
        n_per_class = 50
        # Class 1: Normal distribution centered at (-2, -2)
        class1 = np.random.multivariate_normal(
            mean=[-2, -2], 
            cov=np.eye(2) * 0.5, 
            size=n_per_class
        )
        # Class 2: Normal distribution centered at (0, 2)
        class2 = np.random.multivariate_normal(
            mean=[0, 2], 
            cov=np.eye(2) * 0.5, 
            size=n_per_class
        )
        # Class 3: Normal distribution centered at (2, -1)
        class3 = np.random.multivariate_normal(
            mean=[2, -1], 
            cov=np.eye(2) * 0.5, 
            size=n_per_class
        )
        # Combine data
        X = np.vstack([class1, class2, class3])
        y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class), np.ones(n_per_class) * 2]).astype(int)
        
        return X, y, ["Feature 1", "Feature 2"], True
    
    elif model_type == 'svm':
        # Support Vector Machine sample data - two classes with margin
        n_per_class = 100
        # Class 0: Points below the line y = x + 1 with some margin
        x1_class0 = np.random.uniform(-3, 3, n_per_class)
        x2_class0 = np.random.uniform(-3, x1_class0, n_per_class) + np.random.normal(0, 0.3, n_per_class)
        # Class 1: Points above the line y = x + 1 with some margin
        x1_class1 = np.random.uniform(-3, 3, n_per_class)
        x2_class1 = np.random.uniform(x1_class1 + 1, 3, n_per_class) + np.random.normal(0, 0.3, n_per_class)
        
        # Combine both classes
        X = np.vstack([
            np.column_stack([x1_class0, x2_class0]),
            np.column_stack([x1_class1, x2_class1])
        ])
        y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)
        
        return X, y, ["Feature 1", "Feature 2"], True
    
    elif model_type == 'naive_bayes':
        # Naive Bayes sample data - text classification-like data (word counts)
        n = 200
        n_features = 10  # Number of "words" in vocabulary
        
        # Generate word counts for two classes (e.g., spam vs. non-spam)
        # Class 0: Low counts for first 5 features, higher for last 5
        class0_counts = np.hstack([
            np.random.poisson(1, (n//2, 5)),
            np.random.poisson(5, (n//2, 5))
        ])
        
        # Class 1: Higher counts for first 5 features, lower for last 5
        class1_counts = np.hstack([
            np.random.poisson(5, (n//2, 5)),
            np.random.poisson(1, (n//2, 5))
        ])
        
        # Combine data
        X = np.vstack([class0_counts, class1_counts])
        y = np.hstack([np.zeros(n//2), np.ones(n//2)]).astype(int)
        feature_names = [f"Word_{i+1}" for i in range(n_features)]
        
        return X, y, feature_names, True
    
    elif model_type == 'neural_network':
        # Neural Network sample data - spiral pattern (non-linear separation)
        n_per_class = 100
        
        # Generate spiral data for 3 classes
        def generate_spiral(n_samples, cls, n_classes=3):
            theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi * 2
            r = 2 * theta + cls * 2.5
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            return np.column_stack([x, y])
        
        # Generate data for each class
        X0 = generate_spiral(n_per_class, 0)
        X1 = generate_spiral(n_per_class, 1)
        X2 = generate_spiral(n_per_class, 2)
        
        # Combine data
        X = np.vstack([X0, X1, X2])
        y = np.hstack([
            np.zeros(n_per_class),
            np.ones(n_per_class),
            np.ones(n_per_class) * 2
        ]).astype(int)
        
        # Add some noise features
        X_noise = np.random.normal(0, 1, (X.shape[0], 3))
        X = np.hstack([X, X_noise])
        
        return X, y, ["X", "Y", "Noise1", "Noise2", "Noise3"], True
    
    elif model_type in ['factor_analysis', 'multidimensional_scaling', 'canonical_correlation']:
        # Latent factor/dimension sample data
        n = 200
        # Generate data with latent factors
        # Two latent factors
        latent_factors = np.random.normal(0, 1, (n, 2))
        
        # Generate 6 observed variables from latent factors
        loadings = np.array([
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.1, 0.9]
        ])
        
        # Observed variables = latent factors * loadings + noise
        X = latent_factors @ loadings.T + np.random.normal(0, 0.3, (n, 6))
        
        # For MDS and factor analysis
        feature_names = [f"Variable_{i+1}" for i in range(6)]
        
        # For canonical correlation
        if model_type == 'canonical_correlation':
            # Split features into two sets
            X1 = X[:, :3]  # First 3 variables (more related to first latent factor)
            X2 = X[:, 3:]  # Last 3 variables (more related to second latent factor)
            return X1, X2, ["Var1", "Var2", "Var3"], ["Var4", "Var5", "Var6"]
        
        return X, feature_names
    
    elif model_type == 'time_series':
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
    
    elif model_type == 'arima':
        # ARIMA sample data
        n = 150
        
        # Generate AR(1) process with coefficient 0.7
        ar = np.zeros(n)
        ar[0] = np.random.normal(0, 1)
        for t in range(1, n):
            ar[t] = 0.7 * ar[t-1] + np.random.normal(0, 1)
        
        # Add trend
        trend = 0.05 * np.arange(n)
        
        # Final series (AR process with trend)
        y = ar + trend
        
        # Add seasonal component
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n) / 12)
        y += seasonal
        
        return np.arange(n), y
    
    elif model_type in ['mixed_effects', 'repeated_measures_anova']:
        # Mixed effects / repeated measures sample data
        # 30 subjects, 4 measurements each, 3 groups
        n_subjects = 30
        n_measurements = 4
        
        # Subject IDs and grouping
        subject_ids = np.repeat(np.arange(n_subjects), n_measurements)
        groups = np.repeat(np.repeat(['A', 'B', 'C'], 10), n_measurements)
        
        # Time points for each subject
        time = np.tile(np.arange(n_measurements), n_subjects)
        
        # Random intercept for each subject
        random_intercepts = np.repeat(np.random.normal(0, 2, n_subjects), n_measurements)
        
        # Fixed effect of group
        group_effects = np.repeat([0, 3, 6], 10 * n_measurements)
        
        # Fixed effect of time
        time_effect = time * 0.5
        
        # Group-time interaction (group C has steeper slope)
        interaction = np.zeros(n_subjects * n_measurements)
        idx_group_c = (groups == 'C')
        interaction[idx_group_c] = time[idx_group_c] * 0.5
        
        # Response variable with noise
        y = 10 + random_intercepts + group_effects + time_effect + interaction + np.random.normal(0, 1, n_subjects * n_measurements)
        
        # Return as DataFrame for easier handling
        data = pd.DataFrame({
            'subject': subject_ids,
            'group': groups,
            'time': time,
            'response': y
        })
        
        return data, 'response', 'group', 'time', 'subject'
    
    elif model_type in ['path_analysis', 'structural_equation_modeling', 'sem']:
        # Path analysis / SEM sample data
        n = 200
        
        # Exogenous variable x1 (independent)
        x1 = np.random.normal(0, 1, n)
        
        # Mediator variables m1 and m2
        m1 = 0.6 * x1 + np.random.normal(0, 0.5, n)
        m2 = 0.4 * x1 + np.random.normal(0, 0.5, n)
        
        # Outcome variable y
        y = 0.3 * m1 + 0.5 * m2 + 0.2 * x1 + np.random.normal(0, 0.5, n)
        
        # Return as DataFrame
        data = pd.DataFrame({
            'x1': x1,
            'm1': m1,
            'm2': m2,
            'y': y
        })
        
        # Define model structure (variables and paths)
        model_structure = {
            'variables': ['x1', 'm1', 'm2', 'y'],
            'paths': [
                ('x1', 'm1'),  # x1 -> m1
                ('x1', 'm2'),  # x1 -> m2
                ('x1', 'y'),   # x1 -> y
                ('m1', 'y'),   # m1 -> y
                ('m2', 'y')    # m2 -> y
            ]
        }
        
        return data, model_structure
    
    elif model_type in ['multinomial', 'multinomial_regression']:
        # Multinomial regression sample data
        n = 200
        p = 3  # Features
        k = 3  # Classes
        
        # Features
        X = np.random.normal(0, 1, (n, p))
        
        # Coefficients for each class (one set per class)
        beta = np.array([
            [0, 0, 0],      # Class 0 (reference level)
            [0.8, 1.2, -0.5],  # Class 1
            [-1.5, 0.5, 1.0]   # Class 2
        ])
        
        # Linear predictors for each class
        eta = X @ beta.T
        
        # Apply softmax to get probabilities
        exp_eta = np.exp(eta)
        probs = exp_eta / exp_eta.sum(axis=1, keepdims=True)
        
        # Sample classes based on probabilities
        y = np.array([np.random.choice(k, p=probs[i]) for i in range(n)])
        
        return X, y, [f"Feature {i+1}" for i in range(p)]
    
    elif model_type in ['survival_analysis', 'cox_proportional_hazards', 'kaplan_meier']:
        # Survival analysis sample data
        n = 200
        
        # Covariates
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.binomial(1, 0.5, n)
        X = np.column_stack([X1, X2])
        
        # Baseline hazard parameters
        baseline_shape = 1.0  # Weibull shape
        baseline_scale = 10.0  # Weibull scale
        
        # True coefficients 
        beta = np.array([0.8, 1.5])
        
        # Generate survival times from Weibull distribution
        # Scale depends on covariates (proportional hazards)
        linear_pred = X @ beta
        scale = baseline_scale * np.exp(-linear_pred / baseline_shape)
        survival_times = np.random.weibull(baseline_shape, n) * scale
        
        # Generate censoring times (uniform distribution)
        censoring_times = np.random.uniform(0, 20, n)
        
        # Observed time is minimum of survival and censoring time
        observed_times = np.minimum(survival_times, censoring_times)
        
        # Event indicator (1 if uncensored, 0 if censored)
        event = (survival_times <= censoring_times).astype(int)
        
        return X, observed_times, event, ["Feature 1", "Feature 2"]
    
    elif model_type in ['bayesian_regression', 'bayesian_hierarchical', 'bayesian_model_averaging', 'bayesian_quantile']:
        # Bayesian regression sample data
        n = 100
        p = 4  # Predictors
        
        # Generate predictors
        X = np.random.normal(0, 1, (n, p))
        
        # True coefficients with different magnitudes
        beta = np.array([0.5, 1.0, 0.0, -0.8])
        
        # Generate outcome with heteroskedastic noise (for quantile regression)
        noise_scale = np.abs(X[:, 0]) + 0.5
        y = X @ beta + np.random.normal(0, noise_scale, n)
        
        # For hierarchical models, add group structure
        if model_type == 'bayesian_hierarchical':
            groups = np.random.choice(5, n)  # 5 different groups
            # Group-specific intercepts
            group_intercepts = np.random.normal(0, 1, 5)[groups]
            y += group_intercepts
            return X, y, groups, [f"X{i+1}" for i in range(p)]
        
        # For model averaging, generate multiple models
        if model_type == 'bayesian_model_averaging':
            # Define model space (different combinations of predictors)
            models = [
                {'predictors': [0], 'name': 'Model 1 (X1)'},
                {'predictors': [0, 1], 'name': 'Model 2 (X1, X2)'},
                {'predictors': [0, 2], 'name': 'Model 3 (X1, X3)'},
                {'predictors': [0, 1, 2], 'name': 'Model 4 (X1, X2, X3)'},
                {'predictors': [0, 1, 2, 3], 'name': 'Model 5 (X1, X2, X3, X4)'}
            ]
            return X, y, models, [f"X{i+1}" for i in range(p)]
            
        return X, y, [f"X{i+1}" for i in range(p)]
    
    elif model_type == 'kernel_regression':
        # Kernel regression sample data - nonlinear pattern
        n = 100
        
        # X values
        X = np.random.uniform(-3, 3, n)
        
        # Y values with non-linear pattern and heteroskedastic noise
        y = np.sin(X) + 0.5 * X**2 + np.random.normal(0, 0.2 * (1 + np.abs(X)), n)
        
        return X.reshape(-1, 1), y

    elif model_type == 'discriminant_analysis':
        # Discriminant analysis sample data
        n_per_class = 50
        n_classes = 3
        p = 2
        
        # Different mean vectors for each class
        means = np.array([
            [-2, -2],  # Class 0
            [0, 2],    # Class 1
            [2, -1]    # Class 2
        ])
        
        # Common covariance matrix for LDA
        cov_lda = np.array([[1.0, 0.6], [0.6, 1.0]])
        
        # Different covariance matrices for QDA
        cov_qda = np.array([
            [[1.0, 0.2], [0.2, 1.0]],    # Class 0
            [[1.5, 0.0], [0.0, 0.8]],    # Class 1
            [[0.7, 0.4], [0.4, 1.2]]     # Class 2
        ])
        
        # Generate data
        X = np.zeros((n_per_class * n_classes, p))
        y = np.repeat(np.arange(n_classes), n_per_class)
        
        for i in range(n_classes):
            start_idx = i * n_per_class
            end_idx = (i + 1) * n_per_class
            
            # For QDA
            if 'qda' in model_type.lower():
                X[start_idx:end_idx] = np.random.multivariate_normal(
                    means[i], cov_qda[i], n_per_class
                )
            # For LDA
            else:
                X[start_idx:end_idx] = np.random.multivariate_normal(
                    means[i], cov_lda, n_per_class
                )
        
        return X, y, ["Feature 1", "Feature 2"]
    
    else:
        # Generic regression data for other model types
        n = 100
        p = 5
        X = np.random.normal(0, 1, (n, p))
        y = 1.5 + 0.5 * X[:, 0] - 0.7 * X[:, 1] + 0.2 * X[:, 2]**2 + np.random.normal(0, 1, n)
        feature_names = [f"Feature {i+1}" for i in range(p)]
        
        return X, y, feature_names, False

def generate_plots_for_model(model_name, model_details, output_dir):
    """Generate and save diagnostic plots for a model
    
    Args:
        model_name: Name of the statistical model
        model_details: Dictionary containing model details
        output_dir: Directory to save the plots
    """
    model_name_lower = model_name.lower()
    model_dir = os.path.join(output_dir, model_name.replace(" ", "_").lower())
    os.makedirs(model_dir, exist_ok=True)
    
    plots = []
    
    try:
        if "linear regression" in model_name_lower:
            # Try to get data from synthetic dataset
            try:
                X, y = synthetic_data_to_numpy(model_details.get("synthetic_data", {}), target_var='y')
                if X is not None and y is not None and len(X) > 0 and len(y) > 0:
                    plots = generate_linear_regression_plots(X, y)
                    print(f"Generated linear regression plots for {model_name} using synthetic data")
                else:
                    # Use sample data
                    X, y = generate_sample_data('linear')
                    plots = generate_linear_regression_plots(X, y)
                    print(f"Generated linear regression plots for {model_name} using sample data")
            except Exception as e:
                print(f"Error with synthetic data: {e}")
                # Use sample data
                X, y = generate_sample_data('linear')
                plots = generate_linear_regression_plots(X, y)
                print(f"Generated linear regression plots for {model_name} using sample data")
                
        elif "logistic regression" in model_name_lower:
            try:
                X, y = synthetic_data_to_numpy(model_details.get("synthetic_data", {}), target_var='y')
                if X is not None and y is not None and len(X) > 0 and len(y) > 0:
                    plots = generate_logistic_regression_plots(X, y)
                    print(f"Generated logistic regression plots for {model_name} using synthetic data")
                else:
                    # Use sample data
                    X, y = generate_sample_data('logistic')
                    plots = generate_logistic_regression_plots(X, y)
                    print(f"Generated logistic regression plots for {model_name} using sample data")
            except Exception as e:
                print(f"Error with synthetic data: {e}")
                # Use sample data
                X, y = generate_sample_data('logistic')
                plots = generate_logistic_regression_plots(X, y)
                print(f"Generated logistic regression plots for {model_name} using sample data")
                
        elif "analysis of variance" in model_name_lower or "anova" in model_name_lower:
            try:
                data = synthetic_data_to_dataframe(model_details.get("synthetic_data", {}))
                if data is not None and not data.empty and 'group' in data.columns and 'value' in data.columns:
                    plots = generate_anova_plots(data, 'value', 'group')
                    print(f"Generated ANOVA plots for {model_name} using synthetic data")
                else:
                    # Use sample data
                    data, value_col, group_col = generate_sample_data('anova')
                    plots = generate_anova_plots(data, value_col, group_col)
                    print(f"Generated ANOVA plots for {model_name} using sample data")
            except Exception as e:
                print(f"Error with synthetic data: {e}")
                # Use sample data
                data, value_col, group_col = generate_sample_data('anova')
                plots = generate_anova_plots(data, value_col, group_col)
                print(f"Generated ANOVA plots for {model_name} using sample data")
                
        elif "random forest" in model_name_lower:
            try:
                X, y = synthetic_data_to_numpy(model_details.get("synthetic_data", {}), target_var='y')
                feature_names = get_feature_names(model_details.get("synthetic_data", {})) or [f"Feature {i+1}" for i in range(X.shape[1])]
                is_classification = 'class' in model_name_lower
                
                if X is not None and y is not None and len(X) > 0 and len(y) > 0:
                    plots = generate_random_forest_plots(X, y, feature_names, is_classification)
                    print(f"Generated random forest plots for {model_name} using synthetic data")
                else:
                    # Use sample data
                    X, y, feature_names, is_classification = generate_sample_data('random_forest')
                    plots = generate_random_forest_plots(X, y, feature_names, is_classification)
                    print(f"Generated random forest plots for {model_name} using sample data")
            except Exception as e:
                print(f"Error with synthetic data: {e}")
                # Use sample data
                X, y, feature_names, is_classification = generate_sample_data('random_forest')
                plots = generate_random_forest_plots(X, y, feature_names, is_classification)
                print(f"Generated random forest plots for {model_name} using sample data")
                
        elif "principal component analysis" in model_name_lower or "pca" in model_name_lower:
            try:
                X = synthetic_data_to_numpy(model_details.get("synthetic_data", {}), target_var=None)
                feature_names = get_feature_names(model_details.get("synthetic_data", {})) or [f"Feature {i+1}" for i in range(X.shape[1])]
                
                if X is not None and len(X) > 0:
                    plots = generate_pca_plots(X, feature_names)
                    print(f"Generated PCA plots for {model_name} using synthetic data")
                else:
                    # Use sample data
                    X, feature_names = generate_sample_data('pca')
                    plots = generate_pca_plots(X, feature_names)
                    print(f"Generated PCA plots for {model_name} using sample data")
            except Exception as e:
                print(f"Error with synthetic data: {e}")
                # Use sample data
                X, feature_names = generate_sample_data('pca')
                plots = generate_pca_plots(X, feature_names)
                print(f"Generated PCA plots for {model_name} using sample data")
                
        elif "poisson regression" in model_name_lower:
            try:
                X, y = synthetic_data_to_numpy(model_details.get("synthetic_data", {}), target_var='y')
                if X is not None and y is not None and len(X) > 0 and len(y) > 0:
                    plots = generate_poisson_regression_plots(X, y)
                    print(f"Generated Poisson regression plots for {model_name} using synthetic data")
                else:
                    # Use sample data
                    X, y = generate_sample_data('poisson')
                    plots = generate_poisson_regression_plots(X, y)
                    print(f"Generated Poisson regression plots for {model_name} using sample data")
            except Exception as e:
                print(f"Error with synthetic data: {e}")
                # Use sample data
                X, y = generate_sample_data('poisson')
                plots = generate_poisson_regression_plots(X, y)
                print(f"Generated Poisson regression plots for {model_name} using sample data")
                
        elif "ordinal regression" in model_name_lower:
            try:
                # Always use sample data for ordinal regression for now
                print("Using generated sample data for Ordinal Regression")
                X, y = generate_sample_data('ordinal')
                plots = generate_ordinal_regression_plots(X, y)
                print(f"Generated Ordinal regression plots for {model_name} using sample data")
            except Exception as e:
                print(f"Error with sample data: {e}")
                # Try again with explicit sample data
                X, y = generate_sample_data('ordinal')
                plots = generate_ordinal_regression_plots(X, y)
                print(f"Generated Ordinal regression plots for {model_name} using sample data")

        elif "t-test" in model_name_lower or "t test" in model_name_lower:
            try:
                # Use sample data for t-test for now
                print("Using generated sample data for T-test")
                group1, group2, group_names = generate_sample_data('ttest')
                plots = generate_ttest_plots(group1, group2, group_names)
                print(f"Generated T-test plots for {model_name} using sample data")
            except Exception as e:
                print(f"Error with sample data: {e}")
                # Try again with explicit sample data
                group1, group2, group_names = generate_sample_data('ttest')
                plots = generate_ttest_plots(group1, group2, group_names)
                print(f"Generated T-test plots for {model_name} using sample data")

        elif "chi-square" in model_name_lower or "chi square" in model_name_lower:
            try:
                # Use sample data for chi-square test for now
                print("Using generated sample data for Chi-Square test")
                observed, row_labels, col_labels = generate_sample_data('chi_square')
                plots = generate_chi_square_plots(observed, row_labels, col_labels)
                print(f"Generated Chi-Square test plots for {model_name} using sample data")
            except Exception as e:
                print(f"Error with sample data: {e}")
                # Try again with explicit sample data
                observed, row_labels, col_labels = generate_sample_data('chi_square')
                plots = generate_chi_square_plots(observed, row_labels, col_labels)
                print(f"Generated Chi-Square test plots for {model_name} using sample data")

        elif "mann-whitney" in model_name_lower or "mann whitney" in model_name_lower:
            try:
                # Use sample data for Mann-Whitney U test for now
                print("Using generated sample data for Mann-Whitney U test")
                group1, group2, group_names = generate_sample_data('mann_whitney')
                plots = generate_mann_whitney_plots(group1, group2, group_names)
                print(f"Generated Mann-Whitney U test plots for {model_name} using sample data")
            except Exception as e:
                print(f"Error with sample data: {e}")
                # Try again with explicit sample data
                group1, group2, group_names = generate_sample_data('mann_whitney')
                plots = generate_mann_whitney_plots(group1, group2, group_names)
                print(f"Generated Mann-Whitney U test plots for {model_name} using sample data")

        elif "kruskal-wallis" in model_name_lower or "kruskal wallis" in model_name_lower:
            try:
                # Use sample data for Kruskal-Wallis test for now
                print("Using generated sample data for Kruskal-Wallis test")
                groups, group_names = generate_sample_data('kruskal_wallis')
                plots = generate_kruskal_wallis_plots(groups, group_names)
                print(f"Generated Kruskal-Wallis test plots for {model_name} using sample data")
            except Exception as e:
                print(f"Error with sample data: {e}")
                # Try again with explicit sample data
                groups, group_names = generate_sample_data('kruskal_wallis')
                plots = generate_kruskal_wallis_plots(groups, group_names)
                print(f"Generated Kruskal-Wallis test plots for {model_name} using sample data")

        elif "cluster" in model_name_lower:
            try:
                # Use sample data for cluster analysis for now
                print("Using generated sample data for Cluster Analysis")
                X, feature_names = generate_sample_data('cluster')
                plots = generate_cluster_analysis_plots(X, feature_names)
                print(f"Generated Cluster Analysis plots for {model_name} using sample data")
            except Exception as e:
                print(f"Error with sample data: {e}")
                # Try again with explicit sample data
                X, feature_names = generate_sample_data('cluster')
                plots = generate_cluster_analysis_plots(X, feature_names)
                print(f"Generated Cluster Analysis plots for {model_name} using sample data")

        elif "bayesian model averaging" in model_name_lower:
            try:
                # Use placeholder data until specific sample data is implemented
                print(f"Generating Bayesian Model Averaging plots for {model_name}")
                plots = generate_bma_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Bayesian Model Averaging plots: {e}")
                
        elif "bayesian quantile regression" in model_name_lower:
            try:
                print(f"Generating Bayesian Quantile Regression plots for {model_name}")
                plots = generate_bayesian_quantile_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Bayesian Quantile Regression plots: {e}")
                
        elif "bayesian hierarchical regression" in model_name_lower:
            try:
                print(f"Generating Bayesian Hierarchical Regression plots for {model_name}")
                plots = generate_bayesian_hierarchical_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Bayesian Hierarchical Regression plots: {e}")
                
        elif "kaplan meier" in model_name_lower:
            try:
                print(f"Generating Kaplan-Meier plots for {model_name}")
                plots = generate_kaplan_meier_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Kaplan-Meier plots: {e}")
                
        elif "cox proportional hazards" in model_name_lower:
            try:
                print(f"Generating Cox Proportional Hazards plots for {model_name}")
                plots = generate_cox_ph_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Cox Proportional Hazards plots: {e}")
                
        elif "repeated measures anova" in model_name_lower:
            try:
                print(f"Generating Repeated Measures ANOVA plots for {model_name}")
                plots = generate_repeated_measures_anova_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Repeated Measures ANOVA plots: {e}")
                
        elif "elastic net regression" in model_name_lower:
            try:
                print(f"Generating Elastic Net Regression plots for {model_name}")
                plots = generate_elastic_net_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Elastic Net Regression plots: {e}")
                
        elif "lasso regression" in model_name_lower:
            try:
                print(f"Generating Lasso Regression plots for {model_name}")
                plots = generate_lasso_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Lasso Regression plots: {e}")
                
        elif "ridge regression" in model_name_lower:
            try:
                print(f"Generating Ridge Regression plots for {model_name}")
                plots = generate_ridge_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Ridge Regression plots: {e}")
                
        elif "multidimensional scaling" in model_name_lower:
            try:
                print(f"Generating Multidimensional Scaling plots for {model_name}")
                plots = generate_mds_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Multidimensional Scaling plots: {e}")
                
        elif "bayesian additive regression trees" in model_name_lower:
            try:
                print(f"Generating Bayesian Additive Regression Trees plots for {model_name}")
                plots = generate_bart_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Bayesian Additive Regression Trees plots: {e}")
                
        elif "k nearest neighbors" in model_name_lower:
            try:
                print(f"Generating K-Nearest Neighbors plots for {model_name}")
                plots = generate_knn_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating K-Nearest Neighbors plots: {e}")
                
        elif "decision trees" in model_name_lower:
            try:
                print(f"Generating Decision Trees plots for {model_name}")
                plots = generate_decision_tree_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Decision Trees plots: {e}")
                
        elif "xgboost" in model_name_lower:
            try:
                print(f"Generating XGBoost plots for {model_name}")
                plots = generate_xgboost_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating XGBoost plots: {e}")
                
        elif "catboost" in model_name_lower:
            try:
                print(f"Generating CatBoost plots for {model_name}")
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
                print(f"Generated CatBoost plots with sample data")
            except Exception as e:
                print(f"Error generating CatBoost plots: {e}")
                
        elif "lightgbm" in model_name_lower:
            try:
                print(f"Generating LightGBM plots for {model_name}")
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
                print(f"Generated LightGBM plots with sample data")
            except Exception as e:
                print(f"Error generating LightGBM plots: {e}")
                
        elif "canonical correlation" in model_name_lower:
            try:
                print(f"Generating Canonical Correlation plots for {model_name}")
                plots = generate_canonical_correlation_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Canonical Correlation plots: {e}")
                
        elif "mancova" in model_name_lower:
            try:
                print(f"Generating MANCOVA plots for {model_name}")
                plots = generate_mancova_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating MANCOVA plots: {e}")
                
        elif "manova" in model_name_lower:
            try:
                print(f"Generating MANOVA plots for {model_name}")
                plots = generate_manova_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating MANOVA plots: {e}")
                
        elif "ancova" in model_name_lower:
            try:
                print(f"Generating ANCOVA plots for {model_name}")
                plots = generate_ancova_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating ANCOVA plots: {e}")
                
        elif "discriminant analysis" in model_name_lower:
            try:
                print(f"Generating Discriminant Analysis plots for {model_name}")
                plots = generate_discriminant_analysis_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Discriminant Analysis plots: {e}")
                
        elif "arima" in model_name_lower:
            try:
                print(f"Generating ARIMA plots for {model_name}")
                plots = generate_arima_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating ARIMA plots: {e}")
                
        elif "path analysis" in model_name_lower:
            try:
                print(f"Generating Path Analysis plots for {model_name}")
                plots = generate_path_analysis_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Path Analysis plots: {e}")
                
        elif "structural equation" in model_name_lower:
            try:
                print(f"Generating Structural Equation Modeling plots for {model_name}")
                plots = generate_sem_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Structural Equation Modeling plots: {e}")
                
        elif "naive bayes" in model_name_lower:
            try:
                print(f"Generating Naive Bayes plots for {model_name}")
                plots = generate_naive_bayes_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Naive Bayes plots: {e}")
                
        elif "neural network" in model_name_lower:
            try:
                print(f"Generating Neural Network plots for {model_name}")
                plots = generate_neural_network_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Neural Network plots: {e}")
                
        elif "svm" in model_name_lower or "support vector" in model_name_lower:
            try:
                print(f"Generating SVM plots for {model_name}")
                plots = generate_svm_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating SVM plots: {e}")
                
        elif "gradient boosting" in model_name_lower:
            try:
                print(f"Generating Gradient Boosting plots for {model_name}")
                plots = generate_gradient_boosting_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Gradient Boosting plots: {e}")
                
        elif "regularized regression" in model_name_lower:
            try:
                print(f"Generating Regularized Regression plots for {model_name}")
                plots = generate_regularized_regression_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Regularized Regression plots: {e}")
                
        elif "survival analysis" in model_name_lower:
            try:
                print(f"Generating Survival Analysis plots for {model_name}")
                plots = generate_survival_analysis_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Survival Analysis plots: {e}")
                
        elif "time series" in model_name_lower:
            try:
                print(f"Generating Time Series plots for {model_name}")
                plots = generate_time_series_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Time Series plots: {e}")
                
        elif "bayesian regression" in model_name_lower:
            try:
                print(f"Generating Bayesian Regression plots for {model_name}")
                plots = generate_bayesian_regression_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Bayesian Regression plots: {e}")
                
        elif "factor analysis" in model_name_lower:
            try:
                print(f"Generating Factor Analysis plots for {model_name}")
                plots = generate_factor_analysis_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Factor Analysis plots: {e}")
                
        elif "kernel regression" in model_name_lower:
            try:
                print(f"Generating Kernel Regression plots for {model_name}")
                plots = generate_kernel_regression_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Kernel Regression plots: {e}")
                
        elif "mixed effects" in model_name_lower:
            try:
                print(f"Generating Mixed Effects plots for {model_name}")
                plots = generate_mixed_effects_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Mixed Effects plots: {e}")
                
        elif "multinomial regression" in model_name_lower:
            try:
                print(f"Generating Multinomial Regression plots for {model_name}")
                plots = generate_multinomial_regression_plots(None)  # Will need appropriate sample data
            except Exception as e:
                print(f"Error generating Multinomial Regression plots: {e}")

    except Exception as e:
        print(f"Error generating plots for {model_name}: {e}")
    
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

def read_model_database(filepath):
    """Read model database from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Generate diagnostic plots for statistical models')
    parser.add_argument('--database', type=str, default='model_database.json',
                        help='Path to the model database JSON file')
    parser.add_argument('--output', type=str, default='static/diagnostic_plots',
                        help='Directory to save the plots')
    parser.add_argument('--model', type=str, 
                        help='Generate plots only for this model name (optional)')
    args = parser.parse_args()
    
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
        # Generate plots for all models
        models_to_process = [
            # Linear Regression models
            "Linear Regression",
            # Logistic Regression models
            "Logistic Regression",
            # ANOVA models
            "Analysis of Variance (ANOVA)",
            # Poisson Regression
            "Poisson Regression",
            # Ordinal Regression
            "Ordinal Regression",
            # Random Forest models
            "Random Forest",
            # PCA models
            "Principal Component Analysis",
            # T-test
            "T_test",
            # Chi-Square Test
            "Chi_Square_Test",
            # Mann-Whitney U Test
            "Mann_Whitney_U_Test",
            # Kruskal-Wallis Test
            "Kruskal_Wallis_Test",
            # Cluster Analysis
            "Cluster Analysis",
            # Bayesian models
            "Bayesian Model Averaging",
            "Bayesian Quantile Regression",
            "Bayesian Hierarchical Regression",
            "Bayesian Additive Regression Trees",
            "Bayesian Regression",
            # Survival Analysis models
            "Kaplan Meier",
            "Cox Proportional Hazards",
            "Survival Analysis",
            # ANOVA variants
            "Repeated Measures ANOVA",
            "MANOVA",
            "MANCOVA",
            "ANCOVA",
            # Regularization models
            "Elastic Net Regression",
            "Lasso Regression",
            "Ridge Regression",
            "Regularized Regression",
            # Dimension reduction
            "Multidimensional Scaling",
            "Factor Analysis",
            "Canonical Correlation",
            # Tree-based models
            "Decision Trees",
            "XGBoost",
            "CatBoost",
            "LightGBM",
            "Gradient Boosting",
            # Other models
            "K Nearest Neighbors",
            "Discriminant Analysis",
            "ARIMA",
            "Path Analysis",
            "Structural Equation Modeling",
            "Naive Bayes",
            "Neural Networks",
            "Support Vector Machines",
            "Time Series",
            "Kernel Regression",
            "Mixed Effects Model",
            "Multinomial Regression"
        ]
        
        for model_name in models_to_process:
            if model_name in model_database:
                print(f"Generating plots for {model_name}")
                generate_plots_for_model(model_name, model_database[model_name], args.output)
            else:
                print(f"Model '{model_name}' not found in database, skipping...")

if __name__ == "__main__":
    main() 