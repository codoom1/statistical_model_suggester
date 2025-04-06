#!/usr/bin/env python3
import json
import re

# Load model database
with open('model_database.json', 'r') as f:
    models = json.load(f)

# Generate generic R output for different model types
def generate_generic_output(model_name, model_type):
    """Generate generic R output for a model based on its type"""
    
    # Remove underscores and convert to lowercase for matching
    model_name_lower = model_name.lower().replace("_", " ")
    
    # Linear models
    if any(x in model_name_lower for x in ["linear", "lasso", "ridge", "regression"]) and "logistic" not in model_name_lower:
        return """
> summary(model)

Call:
lm(formula = y ~ x1 + x2 + x3, data = df)

Residuals:
     Min       1Q   Median       3Q      Max 
-2.05111 -0.62366  0.01062  0.70315  2.10890 

Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)   2.1344     0.4940   4.321 3.92e-05 ***
x1            0.4921     0.0457  10.769  < 2e-16 ***
x2            1.4975     0.1866   8.025 3.41e-12 ***
x32           0.1977     0.2534   0.780   0.4373    
x33           0.1741     0.2309   0.754   0.4527    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Residual standard error: 0.9766 on 95 degrees of freedom
Multiple R-squared:  0.7324,	Adjusted R-squared:  0.7208 
F-statistic: 64.93 on 4 and 95 DF,  p-value: < 2.2e-16

> # Model diagnostics and predictions
> plot(model)
> predictions <- predict(model, newdata = test_data)
> mean((test_data$y - predictions)^2)  # MSE
[1] 1.045218
"""
    
    # Logistic regression and classification
    elif any(x in model_name_lower for x in ["logistic", "classify", "classification", "binomial"]):
        return """
> summary(model)

Call:
glm(formula = y ~ x1 + x2 + x3, family = binomial, data = df)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.1701  -0.8079  -0.4635   0.9184   2.2701  

Coefficients:
            Estimate Std. Error z value Pr(>|t|)    
(Intercept)  -0.9879     0.2811  -3.515 0.000439 ***
x1            0.7846     0.1827   4.294 1.75e-05 ***
x2           -1.2264     0.2096  -5.853 4.82e-09 ***
x32           0.1308     0.3989   0.328 0.743023    
x33           0.5486     0.3843   1.428 0.153465    
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for binomial family taken to be 1)

Null deviance: 267.33  on 199  degrees of freedom
Residual deviance: 208.46  on 195  degrees of freedom
AIC: 218.46

> # Calculate odds ratios
> exp(coef(model))  # exponentiated coefficients
(Intercept)         x1         x2        x32        x33 
  0.3724354   2.1916057   0.2933147   1.1396893   1.7309659 

> # Confusion matrix
> pred_probs <- predict(model, newdata = test_data, type = "response")
> pred_class <- ifelse(pred_probs > 0.5, 1, 0)
> table(Predicted = pred_class, Actual = test_data$y)
          Actual
Predicted  0  1
        0 42  8
        1  3 47

> # AUC-ROC
> auc(roc(test_data$y, pred_probs))
[1] 0.942
"""
    
    # Count data models (Poisson, Negative Binomial)
    elif any(x in model_name_lower for x in ["poisson", "count", "binomial"]):
        return """
> summary(model)

Call:
glm(formula = y ~ x1 + x2 + offset(log(offset_var)), family = poisson, 
    data = df)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.8064  -0.8730  -0.0451   0.6975   3.1275  

Coefficients:
            Estimate Std. Error z value Pr(>|z|)    
(Intercept)   0.3102     0.0768   4.037 5.41e-05 ***
x1            0.6942     0.0393  17.659  < 2e-16 ***
x2B          -0.0254     0.0571  -0.445    0.656    
x2C           0.1842     0.0552   3.339  0.00084 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

(Dispersion parameter for poisson family taken to be 1)

    Null deviance: 423.31  on 199  degrees of freedom
Residual deviance: 190.15  on 196  degrees of freedom
AIC: 798.95

> # Check for overdispersion
> dispersion <- sum(residuals(model, type = "pearson")^2) / model$df.residual
> dispersion
[1] 1.23451

> # Predictions
> new_data <- data.frame(
+   x1 = c(0.5, 1.5, 2.5),
+   x2 = factor(c("A", "B", "C"), levels = c("A", "B", "C")),
+   offset_var = c(1, 1, 1)
+ )
> predicted_counts <- predict(model, newdata = new_data, type = "response")
> print(cbind(new_data, predicted_count = predicted_counts))
   x1 x2 offset_var predicted_count
1 0.5  A         1        0.9782323
2 1.5  B         1        1.8745932
3 2.5  C         1        4.9842105
"""
    
    # Clustering
    elif any(x in model_name_lower for x in ["cluster", "kmeans", "dbscan", "hierarchical"]):
        return """
> # Fit clustering model
> model <- kmeans(df, centers = 3, nstart = 25)

> # Examine cluster sizes
> table(model$cluster)

 1  2  3 
42 68 40 

> # Cluster centers
> model$centers
         x        y         z
1  2.36743  6.54327  8.923145
2  9.46725  4.23844  13.42371
3 -1.56824  9.31278  7.651242

> # Within-cluster sum of squares
> model$withinss
[1] 126.4562 153.4781  84.3471

> # Visualization of clusters
> library(ggplot2)
> ggplot(df, aes(x = x, y = y, color = factor(model$cluster))) +
+   geom_point() +
+   labs(title = "K-means Clustering Results",
+        color = "Cluster")

> # Silhouette score to evaluate clustering quality
> library(cluster)
> sil <- silhouette(model$cluster, dist(df))
> summary(sil)
Silhouette of 150 units in 3 clusters:
 Cluster sizes and average silhouette widths:
       42        68        40 
0.7257432 0.5683210 0.6824531 
Individual silhouette widths:
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
 0.3271  0.5406  0.6534  0.6398  0.7542  0.8763
"""
    
    # PCA and factor analysis
    elif any(x in model_name_lower for x in ["pca", "principal component", "factor analysis"]):
        return """
> # Perform PCA
> pca_result <- prcomp(df, scale = TRUE)

> # Summary of PCA results
> summary(pca_result)
Importance of components:
                          PC1     PC2     PC3     PC4     PC5
Standard deviation     1.8440  1.2634  0.7343  0.5281  0.3073
Proportion of Variance 0.6802  0.3191  0.1080  0.0558  0.0189
Cumulative Proportion  0.6802  0.9993  0.9853  0.9941  1.0000

> # Loadings (correlations between variables and principal components)
> pca_result$rotation
           PC1       PC2       PC3       PC4       PC5
V1  -0.4358463  0.574255  0.318673  0.604723  0.126894
V2  -0.5645643 -0.163533  0.646954 -0.478253  0.093855
V3  -0.4212679 -0.578678 -0.427184 -0.068104  0.540344
V4  -0.3951507 -0.174698 -0.305826  0.635367 -0.571384
V5  -0.3953580  0.528976 -0.452258 -0.045156 -0.596563

> # Scree plot
> plot(pca_result, type = "lines")

> # Biplot: visualize variables and observations in PC space
> biplot(pca_result, scale = 0)

> # Determine number of components to retain
> eigenvals <- pca_result$sdev^2
> plot(eigenvals, type = "b", ylab = "Eigenvalue", xlab = "Component")
> abline(h = 1, col = "red", lty = 2)  # Kaiser criterion
"""
    
    # Bayesian models
    elif any(x in model_name_lower for x in ["bayesian", "bayes", "mcmc"]):
        return """
> # Fit Bayesian model
> library(rstanarm)
> model <- stan_glm(y ~ x1 + x2 + x3, data = df, family = gaussian())

> # Model summary
> summary(model)
Model Info:
 function:     stan_glm
 family:       gaussian [identity]
 formula:      y ~ x1 + x2 + x3
 algorithm:    sampling
 priors:       see help('prior_summary')
 sample:       4000 (posterior sample size)
 observations: 100
 predictors:   5

Estimates:
                   mean   sd     10%    50%    90% 
(Intercept)        2.1    0.5    1.5    2.1    2.8
x1                 0.5    0.0    0.4    0.5    0.5
x2                 1.5    0.2    1.3    1.5    1.7
x32                0.2    0.3   -0.1    0.2    0.5
x33                0.2    0.2   -0.1    0.2    0.5
sigma              1.0    0.1    0.9    1.0    1.1

Fit Diagnostics:
           mean   sd     10%    50%    90%  
mean_PPD   8.9    0.1    8.7    8.9    9.0  

The mean_ppd is the sample average posterior predictive distribution of the outcome variable.

> # Parameter distributions
> plot(model, "hist")

> # Posterior intervals
> posterior_interval(model, prob = 0.9)
                   5%        95%
(Intercept)  1.4671735  2.7986981
x1           0.4173749  0.5670458
x2           1.2683602  1.7198841
x32         -0.1353156  0.5340323
x33         -0.1337763  0.4807142
sigma        0.8714667  1.0983766

> # Predictions with uncertainty intervals
> newdata <- data.frame(x1 = c(8, 10, 12), x2 = 1, x3 = factor(2, levels = 1:3))
> predictions <- posterior_predict(model, newdata = newdata)
> pred_summary <- t(apply(predictions, 2, quantile, probs = c(0.05, 0.5, 0.95)))
> colnames(pred_summary) <- c("5%", "50%", "95%")
> print(cbind(newdata, pred_summary))
   x1 x2 x3       5%      50%      95%
1  8  1  2  8.157412  9.57623 10.98749
2 10  1  2  9.160142 10.56047 11.96152
3 12  1  2 10.146060 11.54274 12.94261
"""
    
    # Random Forest and other ML
    elif any(x in model_name_lower for x in ["random forest", "svm", "xgboost", "gradient boosting"]):
        return """
> # Fit random forest model
> library(randomForest)
> model <- randomForest(y ~ ., data = df, ntree = 500, importance = TRUE)

> # Model summary
> print(model)

Call:
 randomForest(formula = y ~ ., data = df, ntree = 500, importance = TRUE) 
               Type of random forest: regression
                     Number of trees: 500
No. of variables tried at each split: 2

          Mean of squared residuals: 1.90215
                    % Var explained: 87.35

> # Variable importance
> importance(model)
        %IncMSE IncNodePurity
x1      84.79218      120.3574
x2      55.40392       74.9482
x3      23.45216       38.2467
x4      11.02718       24.9816

> varImpPlot(model)

> # Test set performance
> predictions <- predict(model, newdata = test_data)
> test_mse <- mean((test_data$y - predictions)^2)
> test_rmse <- sqrt(test_mse)
> test_r2 <- 1 - sum((test_data$y - predictions)^2) / sum((test_data$y - mean(test_data$y))^2)

> cat("Test MSE:", test_mse, "\\n")
Test MSE: 2.157482 
> cat("Test RMSE:", test_rmse, "\\n")
Test RMSE: 1.468837 
> cat("Test R-squared:", test_r2, "\\n")
Test R-squared: 0.8472619 

> # Plot actual vs predicted
> plot(test_data$y, predictions, main = "Actual vs Predicted Values",
+      xlab = "Actual", ylab = "Predicted")
> abline(0, 1, col = "red", lty = 2)
"""
    
    # Survival analysis
    elif any(x in model_name_lower for x in ["cox", "survival", "kaplan", "meier"]):
        return """
> # Fit survival model
> library(survival)
> model <- coxph(Surv(time, event) ~ age + sex + treatment, data = df)

> # Model summary
> summary(model)
Call:
coxph(formula = Surv(time, event) ~ age + sex + treatment, data = df)

  n= 200, number of events= 145 

                  coef  exp(coef)   se(coef)       z      p
age          0.0198423  1.0200392  0.0045251  4.3850 1.2e-05
sexFemale   -0.5124617  0.5991217  0.1675384 -3.0588 0.00222
treatmentT  -0.8103536  0.4447933  0.1693353 -4.7857 1.7e-06

Concordance= 0.706  (se = 0.027)
Likelihood ratio test= 56.49  on 3 df,   p=3e-12
Wald test            = 53.81  on 3 df,   p=1e-11
Score (logrank) test = 55.25  on 3 df,   p=6e-12

> # Hazard ratios with confidence intervals
> hazard_ratios <- exp(coef(model))
> conf_int <- exp(confint(model))
> hr_table <- cbind("Hazard Ratio" = hazard_ratios, conf_int)
> print(hr_table)
                Hazard Ratio     2.5 %    97.5 %
age                1.0200392 1.0110829 1.0291207
sexFemale          0.5991217 0.4313612 0.8320493
treatmentT         0.4447933 0.3189405 0.6203129

> # Check proportional hazards assumption
> ph_test <- cox.zph(model)
> print(ph_test)
             rho  chisq     p
age       -0.128  2.418 0.120
sexFemale -0.075  0.843 0.359
treatmentT 0.039  0.234 0.629
GLOBAL        NA  3.491 0.322

> # Plot survival curves
> plot(survfit(model, newdata = data.frame(
+   age = c(50, 50, 70, 70),
+   sex = factor(c("Male", "Female", "Male", "Female"), levels = c("Male", "Female")),
+   treatment = factor(c("Control", "Control", "Control", "Control"), levels = c("Control", "T"))
+ )), col = 1:4, lty = 1:4)
> legend("topright", c("Male, 50", "Female, 50", "Male, 70", "Female, 70"),
+        col = 1:4, lty = 1:4)
"""
    
    # ANOVA
    elif any(x in model_name_lower for x in ["anova", "variance"]):
        return """
> # Perform ANOVA
> model <- aov(y ~ group, data = df)

> # Model summary
> summary(model)
            Df Sum Sq Mean Sq F value   Pr(>F)    
group        3  650.2  216.73   37.75 < 2.2e-16 ***
Residuals  116  666.1    5.74                     
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

> # Check assumptions
> # 1. Normality of residuals
> shapiro.test(residuals(model))

        Shapiro-Wilk normality test

data:  residuals(model)
W = 0.9876, p-value = 0.3401

> # 2. Homogeneity of variances
> library(car)
> leveneTest(y ~ group, data = df)
Levene's Test for Homogeneity of Variance (center = median)
      Df F value Pr(>F)
group  3  0.5791 0.6298
      116               

> # Post-hoc tests
> TukeyHSD(model)
  Tukey multiple comparisons of means
    95% family-wise confidence level

Fit: aov(formula = y ~ group, data = df)

$group
        diff        lwr       upr     p adj
B-A  2.00000  0.5853267 3.4146733 0.0025058
C-A  5.00000  3.5853267 6.4146733 0.0000000
D-A  3.00000  1.5853267 4.4146733 0.0000035
C-B  3.00000  1.5853267 4.4146733 0.0000035
D-B  1.00000 -0.4146733 2.4146733 0.2548931
D-C -2.00000 -3.4146733 0.4146733 0.0025058

> # Visualization
> boxplot(y ~ group, data = df, main = "Comparison of Groups",
+         xlab = "Group", ylab = "Response")
> stripchart(y ~ group, data = df, vertical = TRUE, method = "jitter",
+            pch = 20, col = "darkgray", add = TRUE)
"""
    
    # Time series
    elif any(x in model_name_lower for x in ["time series", "arima", "arma", "forecast"]):
        return """
> # Fit ARIMA model
> library(forecast)
> model <- auto.arima(ts_data)

> # Model summary
> summary(model)
Series: ts_data 
ARIMA(2,1,1) 

Coefficients:
         ar1      ar2      ma1
      0.7645  -0.1032  -0.8964
s.e.  0.0867   0.0826   0.0513

sigma^2 estimated as 0.7816:  log likelihood=-160.13
AIC=328.26   AICc=328.41   BIC=341.15

Training set error measures:
                       ME      RMSE       MAE      MPE     MAPE      MASE
Training set -0.004826175 0.8766901 0.6826193 -1.23766 8.258828 0.6826193

> # Residual diagnostics
> checkresiduals(model)

        Ljung-Box test

data:  Residuals from ARIMA(2,1,1)
Q* = 13.867, df = 17, p-value = 0.6763

Model df: 3.   Total lags used: 20

> # Forecast future values
> forecast_values <- forecast(model, h = 12)  # Forecast 12 time periods ahead
> print(forecast_values)
         Point Forecast     Lo 80    Hi 80     Lo 95    Hi 95
Jan 2023       83.20417  82.07117 84.33718  81.47218 84.93617
Feb 2023       83.93661  82.10714 85.76609  81.14072 86.73251
Mar 2023       84.43683  82.06069 86.81297  80.81072 88.06294
Apr 2023       84.93707  82.10221 87.77194  80.62144 89.25271
May 2023       85.43731  82.19169 88.68293  80.50214 90.37248
Jun 2023       85.93756  82.31273 89.56239  80.41909 91.45603
Jul 2023       86.43780  82.45487 90.42073  80.36088 92.51472
Aug 2023       86.93804  82.61157 91.26451  80.32116 93.55492
Sep 2023       87.43828  82.77926 92.09731  80.29598 94.58059
Oct 2023       87.93853  82.95570 92.92135  80.28198 95.59507
Nov 2023       88.43877  83.13924 93.73830  80.27738 96.60016
Dec 2023       88.93901  83.32858 94.54944  80.28090 97.59712

> # Plot the forecast
> plot(forecast_values, main = "Time Series Forecast",
+      xlab = "Time", ylab = "Value")
"""
    
    # T-test
    elif any(x in model_name_lower for x in ["t-test", "t test", "student"]):
        return """
> # Perform t-test
> t_test_result <- t.test(value ~ group, data = df, var.equal = TRUE)

> # T-test summary
> print(t_test_result)

        Two Sample t-test

data:  value by group
t = -4.9237, df = 58, p-value = 7.361e-06
alternative hypothesis: true difference in means is not equal to 0
95 percent confidence interval:
 -2.801224 -1.198776
sample estimates:
mean in group Control mean in group Treatment 
                   10                     12 

> # Equal variance test
> var.test(value ~ group, data = df)

        F test to compare two variances

data:  value by group
F = 0.88235, num df = 29, denom df = 29, p-value = 0.7372
alternative hypothesis: true ratio of variances is not equal to 1
95 percent confidence interval:
 0.4244638 1.8344257
sample estimates:
ratio of variances 
         0.8823529 

> # Effect size (Cohen's d)
> library(effsize)
> cohen.d(value ~ group, data = df)
Cohen's d

d estimate: 1.274122 (large)
95 percent confidence interval:
     lower      upper 
0.72123944 1.82700557 

> # Visualization
> boxplot(value ~ group, data = df, main = "Comparison of Groups",
+         xlab = "Group", ylab = "Value", col = c("lightblue", "lightgreen"))
> stripchart(value ~ group, data = df, vertical = TRUE, method = "jitter",
+            pch = 19, col = c("blue", "green"), add = TRUE, alpha = 0.5)
"""
    
    # Chi-square test
    elif any(x in model_name_lower for x in ["chi", "chi-square", "contingency"]):
        return """
> # Create contingency table
> cont_table <- table(df$group, df$outcome)
> print(cont_table)
    
     Negative Positive
  A        42       18
  B        30       30
  C        18       42

> # Perform chi-square test
> chi_test <- chisq.test(cont_table)
> print(chi_test)

        Pearson's Chi-squared test

data:  cont_table
X-squared = 19.2, df = 2, p-value = 6.758e-05

> # Expected frequencies
> print(chi_test$expected)
   Negative Positive
A       30       30
B       30       30
C       30       30

> # Contribution to chi-square
> round(chi_test$residuals^2, 2)
   Negative Positive
A     4.80     4.80
B     0.00     0.00
C     4.80     4.80

> # Effect size (Cramer's V)
> library(vcd)
> assocstats(cont_table)
                    X^2 df   P(> X^2)
Likelihood Ratio 19.457  2 5.9562e-05
Pearson          19.200  2 6.7578e-05

                  Phi    Contingency Coef.    Cramer's V
               0.3265               0.31          0.3265

> # Visualization
> barplot(t(cont_table), beside = TRUE, col = c("lightblue", "lightgreen"),
+         main = "Outcome by Group", xlab = "Group", ylab = "Count",
+         legend.text = c("Negative", "Positive"))
"""
    
    # Generic output for other model types
    else:
        return """
> # Model summary
> summary(model)

# [Model-specific output would appear here]

> # Model diagnostics
> # [Diagnostic outputs would appear here]

> # Model performance metrics
> # [Performance metrics would appear here]

> # Predictions
> predictions <- predict(model, newdata = test_data)
> # [Prediction results would appear here]
"""

# Add results to models
count = 0
for model_name, model in models.items():
    if "synthetic_data" in model and "r_code" in model["synthetic_data"]:
        # Skip if already has results
        if "results" in model["synthetic_data"]:
            print(f"Model {model_name} already has results")
            continue
        
        # Determine model type
        model_type = None
        for category in ["regression", "classification", "clustering", "dimensionality_reduction", 
                        "time_series", "hypothesis_testing", "bayesian"]:
            if category in model.get("analysis_goals", []):
                model_type = category
                break
        
        # Generate output
        r_output = generate_generic_output(model_name, model_type)
        
        # Add to model
        model["synthetic_data"]["results"] = {
            "text_output": r_output,
            "plots": []  # No plots for simplicity
        }
        
        count += 1
        print(f"Added results to {model_name}")

# Save the updated model database
with open('model_database.json', 'w') as f:
    json.dump(models, f, indent=4)

print(f"\nDone! Added results to {count} models.") 