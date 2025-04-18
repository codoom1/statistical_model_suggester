#!/usr/bin/env python3
import json
import os

# Load model database
with open('model_database.json', 'r') as f:
    models = json.load(f)

# Define a mapping from model types to example files
model_examples = {
    # Linear models
    "Linear Regression": "linear_regression.R",
    "Multiple Regression": "linear_regression.R",
    "Polynomial Regression": "polynomial_regression.R",
    "Ridge Regression": "linear_regression.R",
    "Lasso Regression": "linear_regression.R",
    "Elastic Net": "linear_regression.R",
    
    # Generalized Linear Models
    "Logistic Regression": "logistic_regression.R",
    "Poisson Regression": "poisson_regression.R",
    "Negative_Binomial_Regression": "poisson_regression.R",
    "Probit_Regression": "logistic_regression.R",
    "Tobit_Regression": "linear_regression.R",
    
    # Categorical outcome models
    "Multinomial Logistic Regression": "multinomial_logistic_regression.R",
    "Ordinal Regression": "ordinal_regression.R",
    
    # Time series models
    "ARIMA": "arima.R",
    "ARIMAX": "arima.R",
    
    # Survival analysis models
    "Cox Regression": "cox_regression.R",
    "Kaplan_Meier": "cox_regression.R",
    
    # Bayesian models
    "Bayesian_Linear_Regression": "bayesian_linear_regression.R",
    "Bayesian_Logistic_Regression": "bayesian_linear_regression.R",
    "Bayesian_Ridge_Regression": "bayesian_linear_regression.R",
    "Bayesian_Hierarchical_Regression": "bayesian_linear_regression.R",
    "Bayesian_Quantile_Regression": "bayesian_linear_regression.R", 
    "Bayesian_Additive_Regression_Trees": "bayesian_linear_regression.R",
    "Bayesian_Model_Averaging": "bayesian_linear_regression.R",
    
    # Machine learning models
    "Random Forest": "random_forest.R",
    "Support Vector Machine": "svm.R",
    "XGBoost": "xgboost.R",
    "Gradient Boosting Machine": "xgboost.R",
    "Neural Network": "neural_network.R",
    
    # Dimensionality reduction
    "Principal Component Analysis": "pca.R",
    "Factor Analysis": "pca.R",
    
    # Clustering models
    "K-Means Clustering": "kmeans.R",
    "Hierarchical_Clustering": "hierarchical_clustering.R",
    "DBSCAN": "kmeans.R",
    "Gaussian_Mixture_Model": "kmeans.R",
    "Spectral_Clustering": "kmeans.R",
    "Agglomerative_Clustering": "hierarchical_clustering.R",
    "Mean_Shift": "kmeans.R",
    "Affinity_Propagation": "kmeans.R",
    "OPTICS": "kmeans.R",
    
    # Hypothesis testing
    "T-Test": "ttest.R",
    "ANOVA": "anova.R",
    "MANOVA": "anova.R",
    "Chi_Square_Test": "chi_square.R",
    "Mann_Whitney_U_Test": "ttest.R",
    "Kruskal_Wallis_H_Test": "anova.R",
    "Wilcoxon Signed Rank Test": "ttest.R",
    "Friedman Test": "anova.R"
}

# Directory for R example files
examples_dir = "synthetic_data_examples"
os.makedirs(examples_dir, exist_ok=True)

# Define default R script templates for different model types
default_templates = {
    "linear_regression.R": """# Generate synthetic data for linear regression
set.seed(123)
n <- 100  # sample size
x1 <- rnorm(n, mean = 10, sd = 2)  # continuous predictor
x2 <- rbinom(n, 1, 0.5)  # binary predictor
x3 <- factor(sample(1:3, n, replace = TRUE))  # categorical predictor with 3 levels
# Create outcome with a linear relationship plus some noise
y <- 2 + 0.5 * x1 + 1.5 * x2 + rnorm(n, mean = 0, sd = 1)
# Combine into a data frame
df <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)

# Descriptive statistics
summary(df)
cor(df[, c("y", "x1", "x2")])
boxplot(y ~ x2, data = df, main = "Y by Binary Predictor", xlab = "X2", ylab = "Y")
plot(x1, y, main = "Scatterplot of Y vs X1", xlab = "X1", ylab = "Y")

# Model fitting
model <- lm(y ~ x1 + x2 + x3, data = df)
summary(model)

# Diagnostic plots
par(mfrow = c(2, 2))
plot(model)

# Predictions
new_data <- data.frame(x1 = c(8, 10, 12), x2 = c(0, 1, 0), x3 = factor(c(1, 2, 3), levels = 1:3))
predictions <- predict(model, newdata = new_data, interval = "confidence")
print(cbind(new_data, predictions))
""",
    
    "logistic_regression.R": """# Generate synthetic data for logistic regression
set.seed(123)
n <- 200  # sample size
x1 <- rnorm(n, mean = 0, sd = 1)  # continuous predictor
x2 <- rnorm(n, mean = 0, sd = 1)  # another continuous predictor
x3 <- factor(sample(1:3, n, replace = TRUE))  # categorical predictor

# Generate binary outcome based on logistic model
logit <- -1 + 0.8 * x1 - 1.2 * x2  # linear predictor
prob <- 1 / (1 + exp(-logit))  # apply logistic function to get probabilities
y <- rbinom(n, 1, prob)  # generate binary outcome

# Combine into a data frame
df <- data.frame(y = factor(y), x1 = x1, x2 = x2, x3 = x3)

# Descriptive statistics
summary(df)
table(df$y)  # frequency of outcome
table(df$y, df$x3)  # contingency table with categorical predictor

# Visualization
boxplot(x1 ~ y, data = df, main = "X1 by Outcome", xlab = "Outcome (Y)", ylab = "X1")
boxplot(x2 ~ y, data = df, main = "X2 by Outcome", xlab = "Outcome (Y)", ylab = "X2")

# Model fitting
model <- glm(y ~ x1 + x2 + x3, family = binomial(link = "logit"), data = df)
summary(model)

# Effects on odds ratios
exp(coef(model))  # exponentiated coefficients give odds ratios
exp(confint(model))  # confidence intervals for odds ratios

# Predictions
new_data <- data.frame(x1 = c(-1, 0, 1), x2 = c(1, 0, -1), x3 = factor(c(1, 2, 3), levels = 1:3))
predicted_probs <- predict(model, newdata = new_data, type = "response")
predicted_class <- ifelse(predicted_probs > 0.5, 1, 0)
print(cbind(new_data, prob = predicted_probs, class = predicted_class))

# ROC curve and AUC
library(pROC)
roc_obj <- roc(df$y, predict(model, type = "response"))
plot(roc_obj, main = "ROC Curve")
auc(roc_obj)  # Area Under the Curve
""",
    
    "poisson_regression.R": """# Generate synthetic data for Poisson regression
set.seed(123)
n <- 200  # sample size
x1 <- runif(n, 0, 3)  # continuous predictor
x2 <- factor(sample(LETTERS[1:3], n, replace = TRUE))  # categorical predictor
offset_var <- runif(n, 0.5, 2)  # exposure variable (e.g., time, area)

# Generate count outcome based on Poisson model
log_mu <- 0.3 + 0.7 * x1 + log(offset_var)  # log(expected count)
mu <- exp(log_mu)  # expected count
y <- rpois(n, mu)  # generate count based on Poisson distribution

# Combine into a data frame
df <- data.frame(y = y, x1 = x1, x2 = x2, offset_var = offset_var)

# Descriptive statistics
summary(df)
table(df$y)  # frequency distribution of counts
aggregate(y ~ x2, data = df, FUN = mean)  # mean counts by group

# Visualization
hist(df$y, breaks = 20, main = "Distribution of Count Outcome", xlab = "Count")
plot(x1, y, main = "Relationship between X1 and Count", xlab = "X1", ylab = "Count")
boxplot(y ~ x2, data = df, main = "Count by Category", xlab = "Category (X2)", ylab = "Count")

# Model fitting
model <- glm(y ~ x1 + x2 + offset(log(offset_var)), family = poisson, data = df)
summary(model)

# Check for overdispersion
dispersion <- sum(residuals(model, type = "pearson")^2) / model$df.residual
cat("Dispersion parameter:", dispersion, "\\n")

# If overdispersion is present (parameter much > 1), consider negative binomial instead
if (dispersion > 1.5) {
  library(MASS)
  nb_model <- glm.nb(y ~ x1 + x2 + offset(log(offset_var)), data = df)
  summary(nb_model)
}

# Predictions
new_data <- data.frame(
  x1 = c(0.5, 1.5, 2.5),
  x2 = factor(c("A", "B", "C"), levels = c("A", "B", "C")),
  offset_var = c(1, 1, 1)
)
predicted_counts <- predict(model, newdata = new_data, type = "response")
print(cbind(new_data, predicted_count = predicted_counts))
"""
}

# Ensure example files exist
for filename, template in default_templates.items():
    filepath = os.path.join(examples_dir, filename)
    if not os.path.exists(filepath):
        print(f"Creating template file {filename}")
        with open(filepath, 'w') as f:
            f.write(template)

# Function to load example R code from file
def load_example_code(filename):
    filepath = os.path.join(examples_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return f.read()
    else:
        # Return a generic template if specific file not found
        return f"""# Generate synthetic data for this model type
set.seed(123)
n <- 100  # sample size

# Generate data
# ...specific code for this model...

# Descriptive statistics
# ...specific code for this model...

# Visualization
# ...specific code for this model...

# Model fitting
# ...specific code for this model...

# Model evaluation
# ...specific code for this model...
"""

# Update models with synthetic data examples
count = 0
for model_name in models:
    if model_name in model_examples:
        example_file = model_examples[model_name]
        r_code = load_example_code(example_file)
        
        # Add synthetic data section
        models[model_name]["synthetic_data"] = {
            "description": f"A dataset suitable for {model_name} analysis",
            "r_code": r_code
        }
        
        count += 1
        print(f"Added synthetic data example to {model_name}")
    else:
        # Try fuzzy matching for models not directly in our mapping
        matched = False
        for example_name, example_file in model_examples.items():
            if example_name.replace("_", " ") in model_name or model_name in example_name.replace("_", " "):
                r_code = load_example_code(example_file)
                
                # Add synthetic data section
                models[model_name]["synthetic_data"] = {
                    "description": f"A dataset suitable for {model_name} analysis",
                    "r_code": r_code
                }
                
                count += 1
                print(f"Added synthetic data example to {model_name} (matched with {example_name})")
                matched = True
                break
        
        if not matched:
            # Use a generic template for unmatched models
            models[model_name]["synthetic_data"] = {
                "description": f"A dataset suitable for {model_name} analysis",
                "r_code": f"""# Generate synthetic data for {model_name}
set.seed(123)
n <- 100  # sample size

# Generate predictors and outcome variables
# (Specific data generation would depend on the model type)

# Descriptive statistics and visualization
# (Specific analysis would depend on the model type)

# Model fitting and evaluation
# (Specific model code would depend on the model type)
"""
            }
            count += 1
            print(f"Added generic synthetic data example to {model_name}")

# Save updated model database
with open('model_database.json', 'w') as f:
    json.dump(models, f, indent=4)

print(f"\nCompleted adding synthetic data examples to {count} models.")
print("Model database has been updated.") 