#!/usr/bin/env python3
import json
import os

# Load model database
model_db_path = os.path.join('..', 'data', 'model_database.json')
with open(model_db_path, 'r') as f:
    models = json.load(f)

# Add premade results for Linear Regression
if "Linear Regression" in models and "synthetic_data" in models["Linear Regression"]:
    models["Linear Regression"]["synthetic_data"]["results"] = {
        "text_output": """
> summary(df)
       y                x1             x2             x3    
 Min.   : 4.83   Min.   : 5.40   Min.   :0.00   1:28     
 1st Qu.: 7.68   1st Qu.: 8.68   1st Qu.:0.00   2:31     
 Median : 8.74   Median :10.17   Median :0.00   3:41     
 Mean   : 8.87   Mean   :10.07   Mean   :0.46          
 3rd Qu.:10.20   3rd Qu.:11.49   3rd Qu.:1.00          
 Max.   :12.61   Max.   :14.65   Max.   :1.00          

> cor(df[, c("y", "x1", "x2")])
          y        x1        x2
y  1.000000 0.8046874 0.4486060
x1 0.804687 1.0000000 0.0322727
x2 0.448606 0.0322727 1.0000000

> model <- lm(y ~ x1 + x2 + x3, data = df)
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

> new_data <- data.frame(x1 = c(8, 10, 12), x2 = c(0, 1, 0), x3 = factor(c(1, 2, 3), levels = 1:3))
> predictions <- predict(model, newdata = new_data, interval = "confidence")
> print(cbind(new_data, predictions))
   x1 x2 x3      fit      lwr      upr
1  8  0  1  6.071198  5.67851  6.46389
2 10  1  2  8.576220  8.17784  8.97460
3 12  0  3 10.146060  9.63766 10.65446

> plot(model)
""",
        "plots": []
    }
    print("Added results for Linear Regression")

# Add premade results for Logistic Regression
if "Logistic Regression" in models and "synthetic_data" in models["Logistic Regression"]:
    models["Logistic Regression"]["synthetic_data"]["results"] = {
        "text_output": """
> summary(df)
  y           x1                 x2            x3   
 0:120   Min.   :-2.95305   Min.   :-3.4702   1:74  
 1:80    1st Qu.:-0.63543   1st Qu.:-0.6595   2:61  
         Median : 0.02386   Median : 0.1035   3:65  
         Mean   : 0.02207   Mean   : 0.0222         
         3rd Qu.: 0.70851   3rd Qu.: 0.6863         
         Max.   : 2.81898   Max.   : 3.5814         

> table(df$y)

  0   1 
120  80 

> model <- glm(y ~ x1 + x2 + x3, family = binomial(link = "logit"), data = df)
> summary(model)

Call:
glm(formula = y ~ x1 + x2 + x3, family = binomial(link = "logit"), 
    data = df)

Deviance Residuals: 
    Min       1Q   Median       3Q      Max  
-2.1701  -0.8079  -0.4635   0.9184   2.2701  

Coefficients:
            Estimate Std. Error z value Pr(>|z|)    
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

Number of Fisher Scoring iterations: 4

> exp(coef(model))  # exponentiated coefficients give odds ratios
(Intercept)         x1         x2        x32        x33 
  0.3724354   2.1916057   0.2933147   1.1396893   1.7309659 

> predicted_probs <- predict(model, newdata = new_data, type = "response")
> predicted_class <- ifelse(predicted_probs > 0.5, 1, 0)
> print(cbind(new_data, prob = predicted_probs, class = predicted_class))
   x1 x2 x3        prob class
1 -1  1  1 0.075095095     0
2  0  0  2 0.345347633     0
3  1 -1  3 0.824435337     1
""",
        "plots": []
    }
    print("Added results for Logistic Regression")

# Add premade results for Random Forest
if "Random Forest" in models and "synthetic_data" in models["Random Forest"]:
    models["Random Forest"]["synthetic_data"]["results"] = {
        "text_output": """
> print(rf_reg)

Call:
 randomForest(formula = y ~ x1 + x2 + x3 + x4, data = train_reg,      ntree = 500, mtry = 2, importance = TRUE) 
               Type of random forest: regression
                     Number of trees: 500
No. of variables tried at each split: 2

          Mean of squared residuals: 3.87381
                    % Var explained: 75.55

> importance(rf_reg)
       IncNodePurity
x1          1248.464
x2           695.115
x3           325.693
x4            72.499

> cat("Regression performance metrics:\\n")
Regression performance metrics:
> cat("Mean Squared Error (MSE):", mse, "\\n")
Mean Squared Error (MSE): 3.840835 
> cat("Root Mean Squared Error (RMSE):", rmse, "\\n")
Root Mean Squared Error (RMSE): 1.95982 
> cat("R-squared:", r_squared, "\\n\\n")
R-squared: 0.7587755 

> print(rf_class)

Call:
 randomForest(formula = y ~ x1 + x2 + x3 + x4, data = train_class,      ntree = 500, mtry = 2, importance = TRUE) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 2

        OOB estimate of  error rate: 6.86%
Confusion matrix:
   A   B class.error
A 169  15  0.08152174
B  9 157  0.05421687

> importance(rf_class)
   MeanDecreaseGini
x1        67.209770
x2        66.233992
x3        18.063175
x4         8.204532

> cat("Classification performance metrics:\\n")
Classification performance metrics:
> cat("Confusion Matrix:\\n")
Confusion Matrix:
> print(conf_matrix)
         Actual
Predicted   A   B
        A  73   4
        B   8  65
> cat("Accuracy:", accuracy, "\\n")
Accuracy: 0.92 
""",
        "plots": []
    }
    print("Added results for Random Forest")

# Save the updated model database
with open(model_db_path, 'w') as f:
    json.dump(models, f, indent=4)

print("\nDone! Added premade results to models.") 