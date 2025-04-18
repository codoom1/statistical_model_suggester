# Generate synthetic data for Random Forest
set.seed(123)
n <- 500  # sample size

# Generate predictors
x1 <- rnorm(n)  # continuous predictor
x2 <- rnorm(n)  # continuous predictor
x3 <- factor(sample(letters[1:4], n, replace = TRUE))  # categorical predictor
x4 <- sample(0:1, n, replace = TRUE)  # binary predictor

# Generate outcome based on a non-linear relationship
# We'll make a complex decision boundary that's ideal for random forest
y_reg <- 2*x1^2 + 3*sin(x1*x2) + 0.5*x1*x2 + rnorm(n, 0, 2)  # continuous outcome for regression
y_class <- factor(ifelse(x1^2 + x2^2 + as.numeric(x3) + rnorm(n, 0, 0.7) > 3, "A", "B"))  # binary outcome for classification

# Combine into data frames
df_reg <- data.frame(y = y_reg, x1 = x1, x2 = x2, x3 = x3, x4 = x4)
df_class <- data.frame(y = y_class, x1 = x1, x2 = x2, x3 = x3, x4 = x4)

# Split data into training and testing sets
set.seed(456)
train_idx <- sample(1:n, 0.7*n)
train_reg <- df_reg[train_idx, ]
test_reg <- df_reg[-train_idx, ]
train_class <- df_class[train_idx, ]
test_class <- df_class[-train_idx, ]

# Descriptive statistics and exploratory visualization
summary(df_reg)
summary(df_class)

# Visualize relationships
par(mfrow = c(2, 2))
plot(x1, y_reg, main = "Y vs X1 (Regression)")
plot(x2, y_reg, main = "Y vs X2 (Regression)")
boxplot(x1 ~ y_class, main = "X1 by Class")
boxplot(x2 ~ y_class, main = "X2 by Class")
par(mfrow = c(1, 1))

# Install and load randomForest package if not already installed
if (!require(randomForest)) {
  install.packages("randomForest")
  library(randomForest)
} else {
  library(randomForest)
}

# Random Forest for Regression
rf_reg <- randomForest(
  y ~ x1 + x2 + x3 + x4, 
  data = train_reg,
  ntree = 500,  # number of trees
  mtry = 2,     # number of variables randomly sampled at each split
  importance = TRUE
)

# Model summary
print(rf_reg)

# Variable importance
varImpPlot(rf_reg)
importance(rf_reg)

# Make predictions
pred_reg <- predict(rf_reg, newdata = test_reg)
mse <- mean((test_reg$y - pred_reg)^2)
rmse <- sqrt(mse)
r_squared <- 1 - sum((test_reg$y - pred_reg)^2) / sum((test_reg$y - mean(test_reg$y))^2)

cat("Regression performance metrics:\n")
cat("Mean Squared Error (MSE):", mse, "\n")
cat("Root Mean Squared Error (RMSE):", rmse, "\n")
cat("R-squared:", r_squared, "\n\n")

# Plot predictions vs actual
plot(test_reg$y, pred_reg, main = "Actual vs Predicted Values",
     xlab = "Actual", ylab = "Predicted")
abline(0, 1, col = "red")  # 45-degree line

# Random Forest for Classification
rf_class <- randomForest(
  y ~ x1 + x2 + x3 + x4, 
  data = train_class,
  ntree = 500,
  mtry = 2,
  importance = TRUE
)

# Model summary
print(rf_class)

# Variable importance
varImpPlot(rf_class)
importance(rf_class)

# Make predictions
pred_class <- predict(rf_class, newdata = test_class)
conf_matrix <- table(Predicted = pred_class, Actual = test_class$y)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

cat("Classification performance metrics:\n")
cat("Confusion Matrix:\n")
print(conf_matrix)
cat("Accuracy:", accuracy, "\n")

# ROC curve and AUC (for binary classification)
if (length(levels(test_class$y)) == 2) {
  if (!require(pROC)) {
    install.packages("pROC")
    library(pROC)
  } else {
    library(pROC)
  }
  
  pred_prob <- predict(rf_class, newdata = test_class, type = "prob")
  roc_obj <- roc(test_class$y, pred_prob[, 2])
  auc_value <- auc(roc_obj)
  
  plot(roc_obj, main = paste("ROC Curve (AUC =", round(auc_value, 3), ")"))
  cat("AUC:", auc_value, "\n")
}

# Tuning the model with cross-validation
if (!require(caret)) {
  install.packages("caret")
  library(caret)
} else {
  library(caret)
}

# Define tuning grid
tuneGrid <- expand.grid(
  .mtry = c(1, 2, 3, 4)
)

# Set up cross-validation
ctrl <- trainControl(
  method = "cv",         # k-fold cross-validation
  number = 5,            # number of folds
  verboseIter = FALSE
)

# Train model with cross-validation
set.seed(789)
rf_tuned <- train(
  y ~ x1 + x2 + x3 + x4,
  data = train_class,
  method = "rf",
  trControl = ctrl,
  tuneGrid = tuneGrid,
  importance = TRUE
)

# View results
print(rf_tuned)
plot(rf_tuned)

# Best model results
print(rf_tuned$bestTune)
varImp(rf_tuned)

# Final predictions with tuned model
final_pred <- predict(rf_tuned, newdata = test_class)
final_conf_matrix <- table(Predicted = final_pred, Actual = test_class$y)
final_accuracy <- sum(diag(final_conf_matrix)) / sum(final_conf_matrix)

cat("\nTuned model accuracy:", final_accuracy, "\n") 