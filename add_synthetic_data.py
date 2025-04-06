#!/usr/bin/env python3
import json

# Load model database
with open('model_database.json', 'r') as f:
    models = json.load(f)

# Define synthetic data examples and analysis code for common model types
synthetic_data_examples = {
    "Linear Regression": {
        "synthetic_data": {
            "description": "A dataset with a continuous outcome variable and multiple predictor variables",
            "r_code": """
# Generate synthetic data for linear regression
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
"""
        }
    },
    "Logistic Regression": {
        "synthetic_data": {
            "description": "A dataset with a binary outcome variable and multiple predictor variables",
            "r_code": """
# Generate synthetic data for logistic regression
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
"""
        }
    },
    "Poisson Regression": {
        "synthetic_data": {
            "description": "A dataset with count data as the outcome variable",
            "r_code": """
# Generate synthetic data for Poisson regression
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
    },
    "Principal Component Analysis": {
        "synthetic_data": {
            "description": "A multivariate dataset with correlated variables suitable for dimension reduction",
            "r_code": """
# Generate synthetic data for PCA
set.seed(123)
n <- 100  # sample size

# Create a correlation matrix to ensure variables are correlated
cor_matrix <- matrix(c(
  1.0, 0.8, 0.6, 0.5, 0.4,
  0.8, 1.0, 0.7, 0.6, 0.5,
  0.6, 0.7, 1.0, 0.7, 0.6,
  0.5, 0.6, 0.7, 1.0, 0.7,
  0.4, 0.5, 0.6, 0.7, 1.0
), nrow = 5)

# Use Cholesky decomposition to generate correlated data
library(MASS)  # for mvrnorm
mu <- c(10, 15, 12, 8, 20)  # means of variables
vars <- c(5, 8, 3, 6, 10)  # variances of variables
sigma <- diag(sqrt(vars)) %*% cor_matrix %*% diag(sqrt(vars))  # covariance matrix
X <- mvrnorm(n, mu, sigma)
colnames(X) <- paste0("V", 1:5)
df <- as.data.frame(X)

# Descriptive statistics
summary(df)
cor(df)  # correlation matrix

# Visualization of correlations
pairs(df, main = "Scatterplot Matrix of Variables")

# Perform PCA
pca_result <- prcomp(df, scale = TRUE)  # standardize variables
summary(pca_result)  # proportion of variance explained by each PC

# Scree plot to visualize eigenvalues
plot(pca_result, type = "l", main = "Scree Plot")

# Biplot to visualize variables and observations in PC space
biplot(pca_result, cex = c(0.8, 1), scale = 0)

# Loadings (correlations between original variables and principal components)
print(pca_result$rotation)

# PC scores (coordinates of observations in PC space)
head(pca_result$x)

# Determine number of components to retain
# Kaiser criterion: eigenvalues > 1
eigenvalues <- pca_result$sdev^2
num_components <- sum(eigenvalues > 1)
cat("Number of components to retain by Kaiser criterion:", num_components, "\\n")

# Cumulative variance explained
cum_var <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
plot(cum_var, type = "b", xlab = "Number of Components", 
     ylab = "Cumulative Proportion of Variance Explained",
     main = "Cumulative Variance Explained")
abline(h = 0.8, col = "red", lty = 2)  # typically aim for 80% explained variance
"""
        }
    },
    "K-Means Clustering": {
        "synthetic_data": {
            "description": "A dataset with multiple features to be grouped into clusters",
            "r_code": """
# Generate synthetic data for K-means clustering
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

# 3D visualization if rgl package is available
if (require(rgl, quietly = TRUE)) {
  plot3d(df, col = km$cluster, size = 3)
  points3d(km$centers, col = 1:k, size = 10)
}

# Evaluate cluster quality (comparing to true labels)
table(km$cluster, true_labels)  # confusion matrix
"""
        }
    },
    "T-Test": {
        "synthetic_data": {
            "description": "A dataset with two groups to compare means",
            "r_code": """
# Generate synthetic data for t-test
set.seed(123)
n1 <- 30  # sample size for group 1
n2 <- 30  # sample size for group 2
group1 <- rnorm(n1, mean = 10, sd = 2)  # data from group 1
group2 <- rnorm(n2, mean = 12, sd = 2)  # data from group 2 with higher mean

# Combine into a data frame
df <- data.frame(
  value = c(group1, group2),
  group = factor(rep(c("Control", "Treatment"), c(n1, n2)))
)

# Descriptive statistics
summary(df)
tapply(df$value, df$group, mean)
tapply(df$value, df$group, sd)

# Visualization
boxplot(value ~ group, data = df, main = "Comparison of Group Means", 
        xlab = "Group", ylab = "Value")
stripchart(value ~ group, data = df, method = "jitter", vertical = TRUE, 
           pch = 19, col = c("blue", "red"), add = TRUE)

# Check assumptions: normality
par(mfrow = c(1, 2))
for (g in levels(df$group)) {
  qqnorm(df$value[df$group == g], main = paste("Q-Q Plot for", g))
  qqline(df$value[df$group == g])
}

# Shapiro-Wilk test for normality
tapply(df$value, df$group, shapiro.test)

# Check assumptions: equal variances
var.test(value ~ group, data = df)

# Perform t-test
t.test(value ~ group, data = df, var.equal = TRUE)  # equal variance assumption
t.test(value ~ group, data = df, var.equal = FALSE)  # Welch's t-test (unequal variances)

# Effect size (Cohen's d)
library(effsize)
cohen.d(df$value, df$group)
"""
        }
    },
    "Cox Regression": {
        "synthetic_data": {
            "description": "Survival data with time-to-event outcomes and censoring",
            "r_code": """
# Generate synthetic data for Cox regression
set.seed(123)
n <- 200  # sample size

# Generate covariates
age <- runif(n, 40, 80)
sex <- factor(rbinom(n, 1, 0.5), labels = c("Male", "Female"))
treatment <- factor(rbinom(n, 1, 0.5), labels = c("Control", "Treatment"))
biomarker <- rnorm(n, mean = 100, sd = 20)

# Generate survival times based on covariates
# Higher age, male sex, and higher biomarker are associated with shorter survival
lambda <- exp(-4 + 0.02 * age + 0.5 * (sex == "Male") - 
              0.8 * (treatment == "Treatment") + 0.01 * biomarker)
event_time <- rexp(n, rate = lambda)
censoring_time <- runif(n, 0, 10)  # administrative censoring

# Determine observed time and censoring indicator
observed_time <- pmin(event_time, censoring_time)
event <- as.numeric(event_time <= censoring_time)  # 1=event observed, 0=censored

# Create data frame
df <- data.frame(
  time = observed_time,
  event = event,
  age = age,
  sex = sex,
  treatment = treatment,
  biomarker = biomarker
)

# Descriptive statistics
summary(df)
table(df$event)  # number of events vs. censored
table(df$event, df$treatment)  # events by treatment group
table(df$event, df$sex)  # events by sex

# Kaplan-Meier curves
library(survival)
km_fit <- survfit(Surv(time, event) ~ treatment, data = df)
plot(km_fit, xlab = "Time", ylab = "Survival Probability", 
     main = "Kaplan-Meier Curves by Treatment", col = c("blue", "red"))
legend("topright", levels(df$treatment), col = c("blue", "red"), lty = 1)

# Log-rank test for equality of survival curves
survdiff(Surv(time, event) ~ treatment, data = df)

# Cox proportional hazards model
cox_model <- coxph(Surv(time, event) ~ age + sex + treatment + biomarker, data = df)
summary(cox_model)

# Hazard ratios and confidence intervals
hazard_ratios <- exp(coef(cox_model))
conf_int <- exp(confint(cox_model))
hr_table <- cbind("Hazard Ratio" = hazard_ratios, conf_int)
print(hr_table)

# Check proportional hazards assumption
ph_test <- cox.zph(cox_model)
print(ph_test)
plot(ph_test)

# Predicted survival curves for specific profiles
new_data <- data.frame(
  age = c(50, 50, 70, 70),
  sex = factor(c("Female", "Female", "Male", "Male"), levels = levels(df$sex)),
  treatment = factor(c("Treatment", "Control", "Treatment", "Control"), levels = levels(df$treatment)),
  biomarker = c(100, 100, 100, 100)
)

pred_surv <- survfit(cox_model, newdata = new_data)
plot(pred_surv, col = 1:4, xlab = "Time", ylab = "Survival Probability",
     main = "Predicted Survival Curves for Different Profiles")
legend("topright", c("Female, 50, Treatment", "Female, 50, Control", 
                     "Male, 70, Treatment", "Male, 70, Control"),
       col = 1:4, lty = 1)
"""
        }
    },
    "Multinomial Logistic Regression": {
        "synthetic_data": {
            "description": "A dataset with a categorical outcome variable with more than two levels",
            "r_code": """
# Generate synthetic data for multinomial logistic regression
set.seed(123)
n <- 300  # sample size

# Generate predictors
x1 <- rnorm(n, mean = 0, sd = 1)
x2 <- rnorm(n, mean = 0, sd = 1)
x3 <- factor(sample(1:3, n, replace = TRUE))

# Generate multinomial outcome probabilities
# Linear predictors for each category relative to the reference
b10 <- 0    # intercept for category 1 (reference)
b11 <- 0    # effect of x1 on category 1 (reference)
b12 <- 0    # effect of x2 on category 1 (reference)

b20 <- 1    # intercept for category 2
b21 <- 1.5  # effect of x1 on category 2
b22 <- -0.8 # effect of x2 on category 2

b30 <- -0.5 # intercept for category 3
b31 <- -1.2 # effect of x1 on category 3
b32 <- 1.0  # effect of x2 on category 3

# Calculate linear predictors for each category
lp1 <- b10 + b11 * x1 + b12 * x2  # reference category (always 0)
lp2 <- b20 + b21 * x1 + b22 * x2
lp3 <- b30 + b31 * x1 + b32 * x2

# Convert to probabilities using softmax function
denom <- exp(lp1) + exp(lp2) + exp(lp3)
p1 <- exp(lp1) / denom
p2 <- exp(lp2) / denom
p3 <- exp(lp3) / denom

# Generate outcome based on these probabilities
probs <- cbind(p1, p2, p3)
y <- apply(probs, 1, function(p) sample(1:3, size = 1, prob = p))
y <- factor(y, levels = 1:3, labels = c("A", "B", "C"))

# Create data frame
df <- data.frame(
  y = y,
  x1 = x1,
  x2 = x2,
  x3 = x3
)

# Descriptive statistics
summary(df)
table(df$y)  # outcome frequencies
prop.table(table(df$y))  # outcome proportions

# Cross-tabulation with categorical predictor
table(df$y, df$x3)
prop.table(table(df$y, df$x3), margin = 2)  # conditional proportions

# Visualization
boxplot(x1 ~ y, data = df, main = "x1 by Outcome Category", xlab = "Outcome", ylab = "x1")
boxplot(x2 ~ y, data = df, main = "x2 by Outcome Category", xlab = "Outcome", ylab = "x2")

# Fit multinomial logistic regression model
library(nnet)
multi_model <- multinom(y ~ x1 + x2 + x3, data = df)
summary(multi_model)

# Convert coefficients to odds ratios
exp(coef(multi_model))

# Calculate and display p-values
z_scores <- summary(multi_model)$coefficients / summary(multi_model)$standard.errors
p_values <- (1 - pnorm(abs(z_scores))) * 2
print(p_values)

# Predictions for new data
new_data <- expand.grid(
  x1 = c(-1, 0, 1),
  x2 = c(-1, 0, 1),
  x3 = factor(c(1, 2, 3), levels = 1:3)
)
new_data <- new_data[c(1, 5, 9), ]  # select a few rows for clarity

# Predict probabilities for each category
pred_probs <- predict(multi_model, newdata = new_data, type = "probs")
pred_class <- predict(multi_model, newdata = new_data, type = "class")

# Display predictions
result <- cbind(new_data, pred_probs, predicted_class = pred_class)
print(result)

# Confusion matrix to evaluate model fit
conf_matrix <- table(Predicted = predict(multi_model, type = "class"), 
                     Actual = df$y)
print(conf_matrix)
print(paste("Accuracy:", sum(diag(conf_matrix)) / sum(conf_matrix)))
"""
        }
    }
}

# Update the models with synthetic data examples
for model_name, example in synthetic_data_examples.items():
    if model_name in models:
        print(f"Adding synthetic data example to {model_name}")
        models[model_name]["synthetic_data"] = example["synthetic_data"]

# Save updated model database
with open('model_database.json', 'w') as f:
    json.dump(models, f, indent=4)

print("Completed adding synthetic data examples to the first batch of models.")
print("Run the full script to add examples to all models.") 