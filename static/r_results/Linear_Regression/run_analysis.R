
# Load required packages
required_packages <- c('png', 'ggplot2')
for(pkg in required_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        install.packages(pkg, repos = 'https://cloud.r-project.org')
        library(pkg, character.only = TRUE)
    }
}

# Create a file to capture plots
png_files <- c()
plot_counter <- 1

# Original plot function to capture
original_plot <- plot
original_boxplot <- boxplot
original_hist <- hist
original_barplot <- barplot
original_pairs <- pairs

# Override plot functions to capture PNG files
plot <- function(...) {
    filename <- paste0("plot_", plot_counter, ".png")
    png(filename, width = 800, height = 500)
    result <- original_plot(...)
    dev.off()
    png_files <<- c(png_files, filename)
    plot_counter <<- plot_counter + 1
    return(result)
}

boxplot <- function(...) {
    filename <- paste0("plot_", plot_counter, ".png")
    png(filename, width = 800, height = 500)
    result <- original_boxplot(...)
    dev.off()
    png_files <<- c(png_files, filename)
    plot_counter <<- plot_counter + 1
    return(result)
}

hist <- function(...) {
    filename <- paste0("plot_", plot_counter, ".png")
    png(filename, width = 800, height = 500)
    result <- original_hist(...)
    dev.off()
    png_files <<- c(png_files, filename)
    plot_counter <<- plot_counter + 1
    return(result)
}

barplot <- function(...) {
    filename <- paste0("plot_", plot_counter, ".png")
    png(filename, width = 800, height = 500)
    result <- original_barplot(...)
    dev.off()
    png_files <<- c(png_files, filename)
    plot_counter <<- plot_counter + 1
    return(result)
}

pairs <- function(...) {
    filename <- paste0("plot_", plot_counter, ".png")
    png(filename, width = 800, height = 600)
    result <- original_pairs(...)
    dev.off()
    png_files <<- c(png_files, filename)
    plot_counter <<- plot_counter + 1
    return(result)
}

# Sink output to a file
sink("output.txt")

# Run the provided R code
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


# Unsink output
sink()

# Save list of plot files
cat(paste(png_files, collapse=","), file="plots.txt")

# Done
cat("Analysis completed successfully.\n")
