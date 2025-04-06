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
cat("Dispersion parameter:", dispersion, "\n")

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

# Effect sizes (interpreted as rate ratios)
exp(coef(model))
conf_int <- exp(confint(model))
print(cbind("Rate Ratio" = exp(coef(model)), conf_int)) 