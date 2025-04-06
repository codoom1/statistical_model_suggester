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