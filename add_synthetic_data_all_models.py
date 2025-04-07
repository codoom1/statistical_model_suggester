#!/usr/bin/env python3
import json
import os

# Function to load model database
def load_models():
    with open('model_database.json', 'r') as f:
        return json.load(f)

# Function to save model database
def save_models(models):
    with open('model_database.json', 'w') as f:
        json.dump(models, f, indent=4)

# Function to add synthetic data examples to models
def add_synthetic_data(models):
    # Dictionary of synthetic data examples for each model type
    synthetic_data_examples = {}
    
    # Load examples from external files to keep this script manageable
    examples_dir = "synthetic_data_examples"
    os.makedirs(examples_dir, exist_ok=True)
    
    # Linear models
    synthetic_data_examples["Linear Regression"] = {
        "description": "A dataset with a continuous outcome variable and multiple predictor variables",
        "r_code": load_example_code(examples_dir, "linear_regression.R")
    }
    
    synthetic_data_examples["Multiple Regression"] = {
        "description": "A dataset with a continuous outcome variable and multiple predictor variables",
        "r_code": load_example_code(examples_dir, "multiple_regression.R")
    }
    
    synthetic_data_examples["Polynomial Regression"] = {
        "description": "A dataset with non-linear relationships between variables",
        "r_code": load_example_code(examples_dir, "polynomial_regression.R")
    }
    
    # Generalized Linear Models
    synthetic_data_examples["Logistic Regression"] = {
        "description": "A dataset with a binary outcome variable and multiple predictor variables",
        "r_code": load_example_code(examples_dir, "logistic_regression.R")
    }
    
    synthetic_data_examples["Poisson Regression"] = {
        "description": "A dataset with count data as the outcome variable",
        "r_code": load_example_code(examples_dir, "poisson_regression.R")
    }
    
    synthetic_data_examples["Negative_Binomial_Regression"] = {
        "description": "A dataset with overdispersed count data as the outcome variable",
        "r_code": load_example_code(examples_dir, "negative_binomial_regression.R")
    }
    
    synthetic_data_examples["Probit_Regression"] = {
        "description": "A dataset with a binary outcome variable modeled using the probit link function",
        "r_code": load_example_code(examples_dir, "probit_regression.R")
    }
    
    synthetic_data_examples["Tobit_Regression"] = {
        "description": "A dataset with a censored continuous outcome variable",
        "r_code": load_example_code(examples_dir, "tobit_regression.R")
    }
    
    # Categorical outcome models
    synthetic_data_examples["Multinomial Logistic Regression"] = {
        "description": "A dataset with a categorical outcome variable with more than two levels",
        "r_code": load_example_code(examples_dir, "multinomial_logistic_regression.R")
    }
    
    synthetic_data_examples["Ordinal Regression"] = {
        "description": "A dataset with an ordinal outcome variable",
        "r_code": load_example_code(examples_dir, "ordinal_regression.R")
    }
    
    # Time series models
    synthetic_data_examples["ARIMA"] = {
        "description": "A time series dataset with trends and seasonality",
        "r_code": load_example_code(examples_dir, "arima.R")
    }
    
    synthetic_data_examples["ARIMAX"] = {
        "description": "A time series dataset with external regressors",
        "r_code": load_example_code(examples_dir, "arimax.R")
    }
    
    # Survival analysis models
    synthetic_data_examples["Cox Regression"] = {
        "description": "Survival data with time-to-event outcomes and censoring",
        "r_code": load_example_code(examples_dir, "cox_regression.R")
    }
    
    synthetic_data_examples["Kaplan_Meier"] = {
        "description": "Survival data for non-parametric estimation of survival functions",
        "r_code": load_example_code(examples_dir, "kaplan_meier.R")
    }
    
    # Bayesian models
    synthetic_data_examples["Bayesian_Linear_Regression"] = {
        "description": "A dataset for Bayesian inference with prior distributions",
        "r_code": load_example_code(examples_dir, "bayesian_linear_regression.R")
    }
    
    synthetic_data_examples["Bayesian_Logistic_Regression"] = {
        "description": "A dataset with binary outcomes for Bayesian inference",
        "r_code": load_example_code(examples_dir, "bayesian_logistic_regression.R")
    }
    
    # Machine learning models
    synthetic_data_examples["Random Forest"] = {
        "description": "A dataset with multiple features for ensemble learning",
        "r_code": load_example_code(examples_dir, "random_forest.R")
    }
    
    synthetic_data_examples["Support Vector Machine"] = {
        "description": "A dataset for classification or regression with optimal hyperplanes",
        "r_code": load_example_code(examples_dir, "svm.R")
    }
    
    synthetic_data_examples["XGBoost"] = {
        "description": "A dataset for gradient boosting machine learning",
        "r_code": load_example_code(examples_dir, "xgboost.R")
    }
    
    # Dimensionality reduction
    synthetic_data_examples["Principal Component Analysis"] = {
        "description": "A multivariate dataset with correlated variables suitable for dimension reduction",
        "r_code": load_example_code(examples_dir, "pca.R")
    }
    
    # Clustering models
    synthetic_data_examples["K-Means Clustering"] = {
        "description": "A dataset with multiple features to be grouped into clusters",
        "r_code": load_example_code(examples_dir, "kmeans.R")
    }
    
    synthetic_data_examples["Hierarchical_Clustering"] = {
        "description": "A dataset for hierarchical clustering to form a dendrogram",
        "r_code": load_example_code(examples_dir, "hierarchical_clustering.R")
    }
    
    # Hypothesis testing
    synthetic_data_examples["T-Test"] = {
        "description": "A dataset with two groups to compare means",
        "r_code": load_example_code(examples_dir, "ttest.R")
    }
    
    synthetic_data_examples["ANOVA"] = {
        "description": "A dataset with multiple groups to compare means",
        "r_code": load_example_code(examples_dir, "anova.R")
    }
    
    synthetic_data_examples["Chi_Square_Test"] = {
        "description": "Categorical data for testing independence or goodness of fit",
        "r_code": load_example_code(examples_dir, "chi_square.R")
    }
    
    # Add examples to models
    count = 0
    for model_name in models:
        # Try exact match first
        if model_name in synthetic_data_examples:
            models[model_name]["synthetic_data"] = synthetic_data_examples[model_name]
            count += 1
            print(f"Added synthetic data to {model_name}")
        # Try fuzzy matching
        else:
            matched = False
            for example_name in synthetic_data_examples:
                if example_name.replace("_", " ") in model_name or model_name in example_name.replace("_", " "):
                    models[model_name]["synthetic_data"] = synthetic_data_examples[example_name]
                    count += 1
                    print(f"Added synthetic data to {model_name} (matched with {example_name})")
                    matched = True
                    break
            
            if not matched:
                print(f"No synthetic data example found for {model_name}")
    
    print(f"Added synthetic data examples to {count} models")
    return models

# Function to load example R code from file or create it if it doesn't exist
def load_example_code(directory, filename):
    filepath = os.path.join(directory, filename)
    
    # If the file doesn't exist, create a template
    if not os.path.exists(filepath):
        create_example_file(filepath, filename)
    
    # Read the file
    with open(filepath, 'r') as f:
        return f.read()

# Function to create example R code files with templates
def create_example_file(filepath, filename):
    # Extract model type from filename
    model_type = filename.replace(".R", "").replace("_", " ")
    
    # Create a template based on the model type
    template = f"""# Generate synthetic data for {model_type}
set.seed(123)
n <- 100  # sample size

# Generate predictors and outcome
# ... (specific code would go here based on the model type)

# Descriptive statistics
# ... (specific code would go here based on the model type)

# Visualization
# ... (specific code would go here based on the model type)

# Model fitting
# ... (specific code would go here based on the model type)

# Model evaluation
# ... (specific code would go here based on the model type)
"""
    
    # Write template to file
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(template)
    
    print(f"Created template for {filename}")

# Create some example files
def create_initial_examples():
    examples_dir = "synthetic_data_examples"
    os.makedirs(examples_dir, exist_ok=True)
    
    # Create example for Linear Regression
    with open(os.path.join(examples_dir, "linear_regression.R"), 'w') as f:
        f.write("""# Generate synthetic data for linear regression
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
""")
    
    # Create example for Logistic Regression
    with open(os.path.join(examples_dir, "logistic_regression.R"), 'w') as f:
        f.write("""# Generate synthetic data for logistic regression
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
""")
    
    # Create example for K-means clustering
    with open(os.path.join(examples_dir, "kmeans.R"), 'w') as f:
        f.write("""# Generate synthetic data for K-means clustering
set.seed(123)
n_per_cluster <- 50  # observations per cluster

# Create 3 distinct clusters in a 2D space
centers <- matrix(c(
  0, 0,    # center of cluster 1
  5, 5,    # center of cluster 2
  10, 0    # center of cluster 3
), ncol = 2, byrow = TRUE)

# Generate points around these centers
cluster1 <- matrix(rnorm(n_per_cluster * 2, sd = 1), ncol = 2) + rep(centers[1,], each = n_per_cluster)
cluster2 <- matrix(rnorm(n_per_cluster * 2, sd = 1), ncol = 2) + rep(centers[2,], each = n_per_cluster)
cluster3 <- matrix(rnorm(n_per_cluster * 2, sd = 1), ncol = 2) + rep(centers[3,], each = n_per_cluster)

# Combine all points
X <- rbind(cluster1, cluster2, cluster3)
colnames(X) <- c("x", "y")
df <- as.data.frame(X)

# Add a third feature that's related to the clusters
df$z <- df$x + df$y + rnorm(nrow(df), sd = 1)

# True cluster labels (for evaluation)
true_labels <- rep(1:3, each = n_per_cluster)

# Descriptive statistics
summary(df)

# Visualize the data
plot(df$x, df$y, col = true_labels, pch = 19, 
     main = "Actual Clusters", xlab = "X", ylab = "Y")
pairs(df, col = true_labels, pch = 19, 
      main = "Pairwise Scatter Plots with True Clusters")

# Fitting K-means with different values of k
set.seed(123)  # for reproducibility
k_values <- 1:10
within_ss <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  km <- kmeans(df, centers = k_values[i], nstart = 25)
  within_ss[i] <- km$tot.withinss
}

# Elbow plot to determine optimal k
plot(k_values, within_ss, type = "b", pch = 19, 
     xlab = "Number of clusters (k)", ylab = "Within-cluster sum of squares",
     main = "Elbow Method for Optimal k")

# Fit k-means with the chosen k (in this case, k = 3)
k <- 3
km <- kmeans(df, centers = k, nstart = 25)

# Visualize the resulting clusters
plot(df$x, df$y, col = km$cluster, pch = 19, 
     main = "K-means Clustering (k=3)", xlab = "X", ylab = "Y")
points(km$centers[,1], km$centers[,2], col = 1:k, pch = 8, cex = 2)

# Evaluate cluster quality (comparing to true labels)
table(km$cluster, true_labels)  # confusion matrix
""")

# Main execution
if __name__ == "__main__":
    # Create initial examples
    create_initial_examples()
    
    # Load the model database
    print("Loading model database...")
    models = load_models()
    print(f"Loaded {len(models)} models")
    
    # Add synthetic data examples
    print("Adding synthetic data examples...")
    models = add_synthetic_data(models)
    
    # Save the updated model database
    print("Saving updated model database...")
    save_models(models)
    
    print("Done! Model database updated with synthetic data examples.") 