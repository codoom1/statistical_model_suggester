# Generate synthetic data for Bayesian Linear Regression
set.seed(123)
n <- 100  # sample size
x1 <- rnorm(n, mean = 0, sd = 1)  # predictor 1
x2 <- rnorm(n, mean = 0, sd = 1)  # predictor 2

# True parameter values
true_beta0 <- 2.5   # intercept
true_beta1 <- 1.8   # effect of x1
true_beta2 <- -0.7  # effect of x2
true_sigma <- 1.3   # residual standard deviation

# Generate outcome with specified effects and noise
y <- true_beta0 + true_beta1 * x1 + true_beta2 * x2 + rnorm(n, 0, true_sigma)

# Create data frame
df <- data.frame(y = y, x1 = x1, x2 = x2)

# Descriptive statistics
summary(df)
cor(df)  # correlation matrix

# Exploratory visualization
par(mfrow = c(1, 2))
plot(x1, y, main = "Y vs X1", xlab = "X1", ylab = "Y")
plot(x2, y, main = "Y vs X2", xlab = "X2", ylab = "Y")
par(mfrow = c(1, 1))

# Fit ordinary least squares for comparison
lm_model <- lm(y ~ x1 + x2, data = df)
summary(lm_model)

# Bayesian linear regression using rstanarm
library(rstanarm)

# Prior specification
# By default, rstanarm uses weakly informative priors:
# - Normal(0, 10) for coefficients
# - half-Cauchy(0, 5) for the residual standard deviation

# Fit Bayesian model
# normal() specifies the likelihood function
# student_t() specifies the prior distribution for the coefficients
stan_model <- stan_glm(
  y ~ x1 + x2, 
  data = df, 
  family = gaussian(),
  prior = normal(0, 2.5),  # adjust the scale for different levels of prior information
  prior_intercept = normal(0, 5),
  prior_aux = exponential(1),
  chains = 4,  # number of Markov chains
  iter = 2000,  # number of iterations per chain
  seed = 123
)

# Summarize posterior distributions
print(stan_model)
summary(stan_model)

# Plot posterior distributions
plot(stan_model, "hist")  # histogram of parameter posterior distributions
plot(stan_model, "trace")  # trace plots to check MCMC convergence
plot(stan_model, "dens")  # kernel density plots of posterior distributions

# Posterior intervals (credible intervals)
posterior_interval(stan_model, prob = 0.95)

# Extract posterior samples for custom analysis
posterior_samples <- as.matrix(stan_model)
head(posterior_samples)

# Calculate posterior probabilities
prob_beta1_positive <- mean(posterior_samples[, "x1"] > 0)
prob_beta2_negative <- mean(posterior_samples[, "x2"] < 0)

cat("Probability that effect of x1 is positive:", prob_beta1_positive, "\n")
cat("Probability that effect of x2 is negative:", prob_beta2_negative, "\n")

# Posterior predictive checks
pp_check(stan_model)  # visual check of model fit

# Predictions for new data
new_data <- data.frame(
  x1 = c(-1, 0, 1),
  x2 = c(1, 0, -1)
)

# Point predictions (using posterior mean)
pred_mean <- posterior_linpred(stan_model, newdata = new_data, transform = TRUE)
pred_mean_vals <- colMeans(pred_mean)

# Posterior predictive distribution (including observation noise)
pred_dist <- posterior_predict(stan_model, newdata = new_data)

# Summarize predictions
pred_interval <- t(apply(pred_dist, 2, quantile, probs = c(0.025, 0.5, 0.975)))
colnames(pred_interval) <- c("2.5%", "50%", "97.5%")

# Display predictions with credible intervals
final_predictions <- cbind(new_data, 
                          "Posterior Mean" = pred_mean_vals,
                          pred_interval)
print(final_predictions) 