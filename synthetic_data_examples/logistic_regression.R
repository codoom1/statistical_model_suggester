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
