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
