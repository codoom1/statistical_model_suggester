"""Utilities for generating model interpretations."""
import base64
import io
from typing import Dict, Any, List

def generate_interpretation_data(model_name: str, model_details: Dict[str, Any]) -> Dict[str, Any]:
    """Generate interpretation data for a statistical model.
    
    Args:
        model_name: Name of the statistical model
        model_details: Dictionary containing model details
        
    Returns:
        Dictionary containing interpretation data
    """
    # Default interpretation structure
    interpretation = {
        "data_description": "This analysis was performed on a dataset with appropriate characteristics for this model.",
        "data_summary": model_details.get("synthetic_data", {}).get("results", {}).get("text_output", "").split(">")[0] if model_details.get("synthetic_data") else "",
        "model_summary": model_details.get("synthetic_data", {}).get("results", {}).get("text_output", "") if model_details.get("synthetic_data") else "",
        "model_explanation": get_model_explanation(model_name),
        "coefficient_explanation": get_coefficient_explanation(model_name),
        "coefficient_table": get_coefficient_table(model_name, model_details),
        "diagnostic_intro": get_diagnostic_intro(model_name),
        "diagnostic_plots": get_diagnostic_plots(model_name, model_details),
        "diagnostic_warning": get_diagnostic_warning(model_name),
        "assumptions": get_model_assumptions(model_name),
        "assumptions_tip": get_assumptions_tip(model_name),
        "prediction_intro": get_prediction_intro(model_name),
        "prediction_example": get_prediction_example(model_name, model_details),
        "practical_implications": get_practical_implications(model_name),
        "pitfalls": get_model_pitfalls(model_name),
        "further_reading": get_further_reading(model_name)
    }
    
    return interpretation

def get_model_explanation(model_name: str) -> str:
    """Get explanation for interpreting model output."""
    model_name_lower = model_name.lower()
    
    # Linear regression
    if "linear regression" in model_name_lower:
        return """
        <p>The summary table above provides key information about your linear regression model:</p>
        <ul>
            <li><strong>Coefficients</strong>: The estimate column shows the effect of each predictor on the outcome. For every one-unit increase in a predictor, the outcome changes by this amount, holding other variables constant.</li>
            <li><strong>Standard Error</strong>: Measures the precision of the coefficient estimates.</li>
            <li><strong>t value</strong>: The coefficient divided by its standard error, used for hypothesis testing.</li>
            <li><strong>p-value</strong>: The probability of observing the data if the null hypothesis (coefficient = 0) is true. Values below 0.05 are typically considered statistically significant.</li>
            <li><strong>R-squared</strong>: The proportion of variance in the dependent variable explained by the model (ranges from 0 to 1).</li>
            <li><strong>F-statistic</strong>: Tests whether the model as a whole explains significant variation in the outcome.</li>
        </ul>
        """
    
    # Logistic regression
    elif "logistic regression" in model_name_lower:
        return """
        <p>The summary table above provides key information about your logistic regression model:</p>
        <ul>
            <li><strong>Coefficients</strong>: The estimate column shows the effect of each predictor on the log-odds of the outcome. For interpretation, we often exponentiate these values to get odds ratios.</li>
            <li><strong>Standard Error</strong>: Measures the precision of the coefficient estimates.</li>
            <li><strong>z value</strong>: Used for hypothesis testing, similar to t-values in linear regression.</li>
            <li><strong>p-value</strong>: The probability of observing the data if the null hypothesis (coefficient = 0) is true. Values below 0.05 are typically considered statistically significant.</li>
            <li><strong>Null/Residual deviance</strong>: Measures of model fit. Lower residual deviance indicates better fit.</li>
            <li><strong>AIC</strong>: Akaike Information Criterion, used for model comparison. Lower values indicate better models.</li>
        </ul>
        <p>The exponentiated coefficients (odds ratios) tell you how much the odds of the outcome increase multiplicatively with a one-unit increase in the predictor.</p>
        """
    
    # Default explanation
    else:
        return """
        <p>The model output provides essential statistics for understanding your analysis:</p>
        <ul>
            <li><strong>Coefficients/Parameters</strong>: Show the relationship between predictors and the outcome.</li>
            <li><strong>Standard Errors</strong>: Indicate the precision of the estimates.</li>
            <li><strong>Statistical tests</strong>: Help determine which effects are statistically significant.</li>
            <li><strong>Goodness-of-fit measures</strong>: Indicate how well the model explains the data.</li>
        </ul>
        <p>Interpreting these values correctly is key to drawing valid conclusions from your analysis.</p>
        """

def get_coefficient_explanation(model_name: str) -> str:
    """Get explanation for interpreting coefficients."""
    model_name_lower = model_name.lower()
    
    # Linear regression
    if "linear regression" in model_name_lower:
        return """
        <p>In linear regression, coefficients represent the change in the outcome variable associated with a one-unit change in the predictor, holding all other variables constant:</p>
        <ul>
            <li>The <strong>intercept</strong> is the expected value of the outcome when all predictors are zero.</li>
            <li>For <strong>continuous predictors</strong>, the coefficient represents the slope of the relationship between that predictor and the outcome.</li>
            <li>For <strong>binary predictors</strong>, the coefficient represents the difference in the outcome between the two groups.</li>
            <li>For <strong>categorical predictors</strong>, each coefficient represents the difference in the outcome compared to the reference category.</li>
        </ul>
        <p>Statistical significance (p-value) helps determine which coefficients reflect real effects rather than random variation.</p>
        """
    
    # Logistic regression
    elif "logistic regression" in model_name_lower:
        return """
        <p>In logistic regression, we interpret coefficients using odds ratios:</p>
        <ul>
            <li>The <strong>odds ratio</strong> (exp(coefficient)) represents how the odds of the outcome change with a one-unit increase in the predictor.</li>
            <li>Odds ratios > 1 indicate that the predictor is associated with higher odds of the outcome.</li>
            <li>Odds ratios < 1 indicate that the predictor is associated with lower odds of the outcome.</li>
            <li>For example, an odds ratio of 1.5 means the odds increase by 50% for each unit increase in the predictor.</li>
        </ul>
        <p>The 95% confidence interval for the odds ratio helps assess the precision of the estimate and whether the association is statistically significant (if the interval doesn't include 1).</p>
        """
    
    # Default explanation
    else:
        return """
        <p>The coefficients in this model represent the relationship between each predictor and the outcome variable. How you interpret these values depends on the type of model:</p>
        <ul>
            <li>The sign (+ or -) indicates the direction of the relationship.</li>
            <li>The magnitude indicates the strength of the relationship.</li>
            <li>Statistical significance (usually indicated by p-values) helps determine which relationships are likely to be real effects.</li>
        </ul>
        <p>Always interpret coefficients in the context of the specific model type and the scale of your variables.</p>
        """

def get_coefficient_table(model_name: str, model_details: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate a coefficient interpretation table."""
    # Sample coefficient table 
    if "linear regression" in model_name.lower():
        return [
            {"variable": "Intercept", "coefficient": "2.13", "interpretation": "The expected value of the outcome when all predictors are zero."},
            {"variable": "x1 (continuous)", "coefficient": "0.49", "interpretation": "For each one-unit increase in x1, the outcome increases by 0.49 units, holding other variables constant."},
            {"variable": "x2 (binary)", "coefficient": "1.50", "interpretation": "The outcome is 1.50 units higher for the second category of x2 compared to the reference category."}
        ]
    elif "logistic regression" in model_name.lower():
        return [
            {"variable": "Intercept", "coefficient": "-0.99 (OR: 0.37)", "interpretation": "The log odds when all predictors are zero. The odds ratio of 0.37 means the base odds of the event are lower than 1."},
            {"variable": "x1 (continuous)", "coefficient": "0.78 (OR: 2.19)", "interpretation": "For each one-unit increase in x1, the odds of the outcome increase by a factor of 2.19 (119% increase)."},
            {"variable": "x2 (continuous)", "coefficient": "-1.23 (OR: 0.29)", "interpretation": "For each one-unit increase in x2, the odds of the outcome decrease by a factor of 0.29 (71% decrease)."}
        ]
    
    return []

def get_diagnostic_intro(model_name: str) -> str:
    """Get introduction for diagnostic plots."""
    model_name_lower = model_name.lower()
    
    # Linear regression
    if "linear regression" in model_name_lower:
        return "Diagnostic plots help assess the validity of the model's assumptions. These four standard plots provide visual checks for linearity, normality, homoscedasticity, and influential points."
    
    # Logistic regression
    elif "logistic regression" in model_name_lower:
        return "Diagnostic plots for logistic regression help evaluate model fit, check for influential observations, and assess the pattern of residuals."
    
    # Default
    else:
        return "Diagnostic plots are visual tools that help assess whether the model's assumptions are met and identify potential issues with the model fit."

def get_diagnostic_plots(model_name: str, model_details: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get diagnostic plots for a model."""
    # Use any existing plots from synthetic data if available
    plots = []
    existing_plots = model_details.get("synthetic_data", {}).get("results", {}).get("plots", [])
    
    # Linear regression standard plots
    if "linear regression" in model_name.lower():
        plots = [
            {
                "title": "Residuals vs Fitted",
                "img_data": existing_plots[0] if existing_plots and len(existing_plots) > 0 else "",
                "interpretation": "This plot checks the linearity assumption. Look for random scatter around the horizontal line. Patterns or curves indicate non-linearity that should be addressed."
            },
            {
                "title": "Normal Q-Q Plot",
                "img_data": existing_plots[1] if existing_plots and len(existing_plots) > 1 else "",
                "interpretation": "This plot checks if residuals follow a normal distribution. Points should fall along the diagonal line. Deviations suggest non-normality."
            },
            {
                "title": "Scale-Location (Spread-Location)",
                "img_data": existing_plots[2] if existing_plots and len(existing_plots) > 2 else "",
                "interpretation": "This plot checks the homoscedasticity assumption. Look for random scatter with constant spread. A funnel shape indicates heteroscedasticity."
            },
            {
                "title": "Residuals vs Leverage",
                "img_data": existing_plots[3] if existing_plots and len(existing_plots) > 3 else "",
                "interpretation": "This plot helps identify influential observations. Points outside the dashed lines (Cook's distance) have high influence on the model and should be examined."
            }
        ]
    
    # Logistic regression plots
    elif "logistic regression" in model_name.lower():
        plots = [
            {
                "title": "ROC Curve",
                "img_data": existing_plots[0] if existing_plots and len(existing_plots) > 0 else "",
                "interpretation": "The ROC curve shows the trade-off between sensitivity and specificity. The area under the curve (AUC) measures overall model performance, with values closer to 1 indicating better discrimination."
            },
            {
                "title": "Residual Plot",
                "img_data": existing_plots[1] if existing_plots and len(existing_plots) > 1 else "",
                "interpretation": "This plot shows residuals versus fitted values. Look for random scatter around zero. Patterns may indicate non-linearity or other model issues."
            }
        ]
    
    # Default plots if nothing specific
    if not plots:
        plots = [
            {
                "title": "Model Diagnostics",
                "img_data": "",
                "interpretation": "Diagnostic plots for this model type help assess model fit, check assumptions, and identify potential issues."
            }
        ]
    
    return plots

def get_diagnostic_warning(model_name: str) -> str:
    """Get warning about diagnostic interpretation."""
    model_name_lower = model_name.lower()
    
    # Linear regression
    if "linear regression" in model_name_lower:
        return "Violations of model assumptions may lead to incorrect inferences. If the diagnostic plots reveal issues, consider transforming variables, adding interaction terms, or using robust methods."
    
    # Logistic regression
    elif "logistic regression" in model_name_lower:
        return "For logistic regression, also examine the Hosmer-Lemeshow test for goodness-of-fit, classification tables to assess predictive accuracy, and check for multicollinearity using variance inflation factors (VIF)."
    
    # Default
    else:
        return "Always check that your model meets its assumptions before interpreting results. Violation of assumptions can lead to biased estimates, incorrect standard errors, and invalid inferences."

def get_model_assumptions(model_name: str) -> List[Dict[str, str]]:
    """Get model assumptions."""
    model_name_lower = model_name.lower()
    
    # Linear regression
    if "linear regression" in model_name_lower:
        return [
            {"name": "Linearity", "description": "The relationship between predictors and the outcome is linear. Check residuals vs. fitted plot for patterns."},
            {"name": "Independence", "description": "Observations are independent of each other. Critical for time series or clustered data."},
            {"name": "Homoscedasticity", "description": "Variance of residuals is constant across all levels of predictors. Look for funnel shapes in diagnostic plots."},
            {"name": "Normality", "description": "Residuals are normally distributed. Check the Q-Q plot for deviations from the diagonal line."},
            {"name": "No multicollinearity", "description": "Predictors are not highly correlated with each other. Check variance inflation factors (VIF)."}
        ]
    
    # Logistic regression
    elif "logistic regression" in model_name_lower:
        return [
            {"name": "Independence", "description": "Observations are independent of each other."},
            {"name": "Linearity in the logit", "description": "The log odds are linearly related to continuous predictors."},
            {"name": "No multicollinearity", "description": "Predictors are not highly correlated with each other."},
            {"name": "No influential observations", "description": "No single case has undue influence on the model."},
            {"name": "Adequate sample size", "description": "Generally, at least 10 events per variable for stable estimates."}
        ]
    
    # Default assumptions
    else:
        return [
            {"name": "Model-specific assumptions", "description": "Consult literature on this specific model type for detailed assumptions."},
            {"name": "Independence", "description": "In most statistical models, observations should be independent of each other."},
            {"name": "Correct model specification", "description": "The model includes all relevant predictors and the appropriate functional form."}
        ]

def get_assumptions_tip(model_name: str) -> str:
    """Get tip for checking assumptions."""
    model_name_lower = model_name.lower()
    
    # Linear regression
    if "linear regression" in model_name_lower:
        return "If the normality assumption is violated but your sample size is large (n > 30), you can still rely on the Central Limit Theorem. For heteroscedasticity, consider robust standard errors or weighted least squares."
    
    # Logistic regression
    elif "logistic regression" in model_name_lower:
        return "The logistic model doesn't assume normality or homoscedasticity of errors, but it's crucial to check for linearity in the logit for continuous predictors using the Box-Tidwell approach or spline functions."
    
    # Default
    else:
        return "When model assumptions are violated, consider transformation of variables, different link functions, robust methods, or alternative modeling approaches better suited to your data structure."

def get_prediction_intro(model_name: str) -> str:
    """Get introduction for predictions."""
    model_name_lower = model_name.lower()
    
    # Linear regression
    if "linear regression" in model_name_lower:
        return "Linear regression allows you to predict the value of the outcome variable for new observations. The prediction includes both a point estimate and a confidence interval."
    
    # Logistic regression
    elif "logistic regression" in model_name_lower:
        return "Logistic regression allows you to predict the probability of the outcome for new observations, which can be converted to class predictions using a threshold (typically 0.5)."
    
    # Default
    else:
        return "This model can be used to make predictions for new data. When making predictions, be cautious about extrapolating beyond the range of your original data."

def get_prediction_example(model_name: str, model_details: Dict[str, Any]) -> str:
    """Get example code for making predictions."""
    model_name_lower = model_name.lower()
    
    # Extract prediction example from synthetic data if available
    r_code = model_details.get("synthetic_data", {}).get("r_code", "")
    prediction_section = ""
    if r_code:
        # Try to extract prediction section from R code
        prediction_lines = []
        capturing = False
        for line in r_code.split("\n"):
            if "new_data" in line or "predict" in line or capturing:
                capturing = True
                prediction_lines.append(line)
            if capturing and "print" in line and "prediction" in line.lower():
                break
        
        if prediction_lines:
            prediction_section = "\n".join(prediction_lines)
    
    # If no prediction section found in R code, provide a default example
    if not prediction_section:
        if "linear regression" in model_name_lower:
            prediction_section = """# Create new data for prediction
new_data <- data.frame(x1 = c(5, 10, 15), x2 = c(0, 1, 0))

# Generate predictions with confidence intervals
predictions <- predict(model, newdata = new_data, interval = "confidence")

# Display predictions
print(cbind(new_data, predictions))"""
        
        elif "logistic regression" in model_name_lower:
            prediction_section = """# Create new data for prediction
new_data <- data.frame(x1 = c(-1, 0, 1), x2 = c(1, 0, -1))

# Generate probability predictions
pred_prob <- predict(model, newdata = new_data, type = "response")

# Convert to class predictions using 0.5 threshold
pred_class <- ifelse(pred_prob > 0.5, 1, 0)

# Display results
print(cbind(new_data, probability = pred_prob, prediction = pred_class))"""
        
        else:
            prediction_section = """# Create new data for prediction
new_data <- data.frame(
  # Define predictors for new observations
  x1 = c(value1, value2, value3),
  x2 = c(value1, value2, value3)
)

# Generate predictions
predictions <- predict(model, newdata = new_data)

# Display predictions
print(cbind(new_data, predictions))"""
    
    return prediction_section

def get_practical_implications(model_name: str) -> List[str]:
    """Get practical implications of the model."""
    model_name_lower = model_name.lower()
    
    # Linear regression
    if "linear regression" in model_name_lower:
        return [
            "Coefficient estimates quantify the relationships between predictors and the outcome, helping identify key factors.",
            "R-squared indicates how much variation in the outcome is explained by the model, providing an overall assessment of fit.",
            "Predictions can be used for forecasting, decision-making, or establishing baselines.",
            "The model can help identify optimal values of predictors to achieve desired outcomes."
        ]
    
    # Logistic regression
    elif "logistic regression" in model_name_lower:
        return [
            "Odds ratios quantify how each predictor increases or decreases the odds of the outcome.",
            "Predicted probabilities can be used for risk assessment, classification, or decision support.",
            "ROC curves and classification metrics help evaluate the model's discriminative ability.",
            "The model can identify the most important factors affecting the probability of the outcome."
        ]
    
    # Default
    else:
        return [
            "The results help understand the relationships between variables in your data.",
            "The model can be used to make predictions for new observations.",
            "Model diagnostics identify potential issues that might affect the validity of your conclusions.",
            "Understanding the limitations of the model is crucial for appropriate application and interpretation."
        ]

def get_model_pitfalls(model_name: str) -> List[Dict[str, str]]:
    """Get common pitfalls for the model."""
    model_name_lower = model_name.lower()
    
    # Linear regression
    if "linear regression" in model_name_lower:
        return [
            {"name": "Extrapolation", "description": "Predicting beyond the range of your data can lead to unreliable results."},
            {"name": "Confounding", "description": "Unmeasured variables may confound the relationships you observe."},
            {"name": "Overfitting", "description": "Including too many predictors relative to your sample size can lead to unstable estimates and poor generalization."},
            {"name": "Multicollinearity", "description": "Highly correlated predictors can make coefficient estimates unstable and difficult to interpret."},
            {"name": "Ignoring assumptions", "description": "Violating the assumptions of linear regression can lead to biased or inefficient estimates."}
        ]
    
    # Logistic regression
    elif "logistic regression" in model_name_lower:
        return [
            {"name": "Separation", "description": "Complete or quasi-complete separation can occur when a predictor perfectly predicts the outcome, leading to unstable estimates."},
            {"name": "Sample size", "description": "Too few events per variable can lead to biased estimates and wide confidence intervals."},
            {"name": "Probability interpretation", "description": "Misinterpreting odds ratios as risk ratios, especially when events are not rare."},
            {"name": "Threshold selection", "description": "Using 0.5 as a classification threshold may not be optimal when classes are imbalanced."},
            {"name": "Non-linearity", "description": "Assuming linear relationships in the logit scale when they might be non-linear."}
        ]
    
    # Default
    else:
        return [
            {"name": "Overfitting", "description": "Creating a model that fits the training data too closely but performs poorly on new data."},
            {"name": "Assumption violations", "description": "Ignoring the assumptions underlying the statistical model."},
            {"name": "Misinterpretation", "description": "Incorrectly interpreting the meaning of parameters or test statistics."},
            {"name": "Causality claims", "description": "Inferring causation from correlation without proper study design."},
            {"name": "Generalizability", "description": "Applying results beyond the population from which the data were sampled."}
        ]

def get_further_reading(model_name: str) -> List[Dict[str, str]]:
    """Get further reading resources for the model."""
    model_name_lower = model_name.lower()
    
    # Common resources for all models
    common_resources = [
        {
            "title": "UCLA Statistical Methods",
            "url": "https://stats.oarc.ucla.edu/",
            "description": "Comprehensive tutorials and examples for various statistical methods."
        },
        {
            "title": "R for Data Science",
            "url": "https://r4ds.had.co.nz/",
            "description": "Free online book covering data analysis and visualization in R."
        }
    ]
    
    # Model-specific resources
    model_resources = []
    
    # Linear regression
    if "linear regression" in model_name_lower:
        model_resources = [
            {
                "title": "An Introduction to Statistical Learning",
                "url": "https://www.statlearning.com/",
                "description": "Chapter 3 covers linear regression with excellent examples."
            },
            {
                "title": "Regression Diagnostics",
                "url": "https://cran.r-project.org/web/packages/olsrr/vignettes/intro.html",
                "description": "Guide to checking assumptions and diagnosing issues in linear models."
            }
        ]
    
    # Logistic regression
    elif "logistic regression" in model_name_lower:
        model_resources = [
            {
                "title": "Logistic Regression: A Self-Learning Text",
                "url": "https://www.springer.com/gp/book/9781441917416",
                "description": "Comprehensive textbook on logistic regression methods."
            },
            {
                "title": "Applied Logistic Regression",
                "url": "https://onlinelibrary.wiley.com/doi/book/10.1002/9781118548387",
                "description": "Classic reference by Hosmer and Lemeshow."
            }
        ]
    
    return common_resources + model_resources 