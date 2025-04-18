# Generate synthetic data for Principal Component Analysis
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
cat("Number of components to retain by Kaiser criterion:", num_components, "\n")

# Cumulative variance explained
cum_var <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
plot(cum_var, type = "b", xlab = "Number of Components", 
     ylab = "Cumulative Proportion of Variance Explained",
     main = "Cumulative Variance Explained")
abline(h = 0.8, col = "red", lty = 2)  # typically aim for 80% explained variance 