# Generate synthetic data for ANOVA
set.seed(123)

# Create a balanced design with 4 groups
n_per_group <- 30  # observations per group
groups <- factor(rep(c("A", "B", "C", "D"), each = n_per_group))

# Set true group means and standard deviation
group_means <- c(10, 12, 15, 13)  # means for groups A, B, C, D
sigma <- 2.5  # within-group standard deviation

# Generate response variable
y <- numeric(length(groups))
for (i in 1:length(y)) {
  group_idx <- as.numeric(groups[i])
  y[i] <- rnorm(1, mean = group_means[group_idx], sd = sigma)
}

# Create data frame
df <- data.frame(y = y, group = groups)

# Add a second factor (to demonstrate two-way ANOVA)
block <- factor(rep(rep(c("Block1", "Block2", "Block3"), each = 10), 4))
df$block <- block

# Descriptive statistics
summary(df)
tapply(df$y, df$group, mean)  # group means
tapply(df$y, df$group, sd)    # group standard deviations

# Visualize group differences
# Box plot
boxplot(y ~ group, data = df, main = "Comparison of Group Means", 
        xlab = "Group", ylab = "Response")

# Add individual observations (jittered)
stripchart(y ~ group, data = df, vertical = TRUE, method = "jitter",
           pch = 20, col = "darkgray", add = TRUE)

# One-way ANOVA
anova_model <- aov(y ~ group, data = df)
summary(anova_model)

# Check ANOVA assumptions
# 1. Normality of residuals
qqnorm(residuals(anova_model), main = "Normal Q-Q Plot of Residuals")
qqline(residuals(anova_model))

# Shapiro-Wilk test for normality of residuals
shapiro.test(residuals(anova_model))

# 2. Homogeneity of variances
plot(anova_model, 1)  # Residuals vs. Fitted values plot

# Bartlett's test for homogeneity of variances
bartlett.test(y ~ group, data = df)

# Levene's test (more robust to non-normality)
library(car)
leveneTest(y ~ group, data = df)

# Multiple comparisons (post-hoc tests)
# Tukey's HSD
tukey_results <- TukeyHSD(anova_model, "group")
print(tukey_results)
plot(tukey_results)

# Two-way ANOVA
twoway_model <- aov(y ~ group * block, data = df)
summary(twoway_model)

# Interaction plot
interaction.plot(df$group, df$block, df$y, 
                 main = "Interaction Plot",
                 xlab = "Group", ylab = "Mean Response",
                 trace.label = "Block")

# Effect size - eta squared
library(effectsize)
eta_squared(anova_model)

# Visualize with more sophisticated plots
library(ggplot2)

# Box plot with points
ggplot(df, aes(x = group, y = y, fill = group)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.5) +
  labs(title = "Response by Group",
       x = "Group",
       y = "Response") +
  theme_minimal()

# Means with confidence intervals
ggplot(df, aes(x = group, y = y, color = group)) +
  stat_summary(fun = mean, geom = "point", size = 3) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2) +
  labs(title = "Group Means with 95% Confidence Intervals",
       x = "Group",
       y = "Response") +
  theme_minimal() 