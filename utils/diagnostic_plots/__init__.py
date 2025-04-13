"""
Diagnostic plot utilities for statistical models.
This package provides functions to generate diagnostic plots for different statistical models.
"""

from .linear_regression import generate_linear_regression_plots
from .logistic_regression import generate_logistic_regression_plots
from .anova import generate_anova_plots
from .random_forest import generate_random_forest_plots
from .pca import generate_pca_plots
from .poisson_regression import generate_poisson_regression_plots
from .ordinal_regression import generate_ordinal_regression_plots
from .ttest import generate_ttest_plots
from .chi_square import generate_chi_square_plots
from .mann_whitney import generate_mann_whitney_plots
from .kruskal_wallis import generate_kruskal_wallis_plots
from .cluster_analysis import generate_cluster_analysis_plots
from .multinomial_regression import generate_multinomial_regression_plots
from .mixed_effects import generate_mixed_effects_plots
from .kernel_regression import generate_kernel_regression_plots
from .factor_analysis import generate_factor_analysis_plots
from .bayesian_regression import generate_bayesian_regression_plots
from .time_series import generate_time_series_plots
from .neural_network import generate_neural_network_plots
from .naive_bayes import generate_naive_bayes_plots
from .structural_equation_modeling import generate_sem_plots
from .path_analysis import generate_path_analysis_plots
from .arima import generate_arima_plots
from .discriminant_analysis import generate_discriminant_analysis_plots
from .cox_proportional_hazards import generate_cox_ph_plots
from .kaplan_meier import generate_kaplan_meier_plots
from .bayesian_hierarchical_regression import generate_bayesian_hierarchical_plots
from .bayesian_quantile_regression import generate_bayesian_quantile_plots
from .bayesian_model_averaging import generate_bma_plots
from .catboost import generate_catboost_plots
from .lightgbm import generate_lightgbm_plots

__all__ = [
    'generate_linear_regression_plots',
    'generate_logistic_regression_plots',
    'generate_anova_plots',
    'generate_random_forest_plots', 
    'generate_pca_plots',
    'generate_poisson_regression_plots',
    'generate_ordinal_regression_plots',
    'generate_ttest_plots',
    'generate_chi_square_plots',
    'generate_mann_whitney_plots',
    'generate_kruskal_wallis_plots',
    'generate_cluster_analysis_plots',
    'generate_multinomial_regression_plots',
    'generate_mixed_effects_plots',
    'generate_kernel_regression_plots',
    'generate_factor_analysis_plots',
    'generate_bayesian_regression_plots',
    'generate_time_series_plots',
    'generate_neural_network_plots',
    'generate_naive_bayes_plots',
    'generate_sem_plots',
    'generate_path_analysis_plots',
    'generate_arima_plots',
    'generate_discriminant_analysis_plots',
    'generate_cox_ph_plots',
    'generate_kaplan_meier_plots',
    'generate_bayesian_hierarchical_plots',
    'generate_bayesian_quantile_plots',
    'generate_bma_plots',
    'generate_catboost_plots',
    'generate_lightgbm_plots'
] 