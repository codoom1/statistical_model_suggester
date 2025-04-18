#!/usr/bin/env python3
import json
import subprocess
import base64
from pathlib import Path
import sys

# Specify which model to process
if len(sys.argv) > 1:
    target_model = sys.argv[1]
else:
    target_model = "Linear Regression"  # Default

print(f"Generating results for model: {target_model}")

# Load model database
with open('model_database.json', 'r') as f:
    models = json.load(f)

if target_model not in models:
    print(f"Error: Model '{target_model}' not found in database")
    sys.exit(1)

model = models[target_model]

if "synthetic_data" not in model or "r_code" not in model["synthetic_data"]:
    print(f"Error: No R code found for model '{target_model}'")
    sys.exit(1)

# Create results directory
results_dir = Path("static/r_results")
results_dir.mkdir(parents=True, exist_ok=True)

# Create model-specific directory
model_dir_name = target_model.replace(" ", "_").replace("/", "_")
model_dir = results_dir / model_dir_name
model_dir.mkdir(exist_ok=True)

# Get R code
r_code = model["synthetic_data"]["r_code"]

# Create a simplified R script
r_script = f"""
# Load required packages
required_packages <- c('png', 'ggplot2')
for(pkg in required_packages) {{
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {{
        install.packages(pkg, repos = 'https://cloud.r-project.org')
        library(pkg, character.only = TRUE)
    }}
}}

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
plot <- function(...) {{
    filename <- paste0("plot_", plot_counter, ".png")
    png(filename, width = 800, height = 500)
    result <- original_plot(...)
    dev.off()
    png_files <<- c(png_files, filename)
    plot_counter <<- plot_counter + 1
    return(result)
}}

boxplot <- function(...) {{
    filename <- paste0("plot_", plot_counter, ".png")
    png(filename, width = 800, height = 500)
    result <- original_boxplot(...)
    dev.off()
    png_files <<- c(png_files, filename)
    plot_counter <<- plot_counter + 1
    return(result)
}}

hist <- function(...) {{
    filename <- paste0("plot_", plot_counter, ".png")
    png(filename, width = 800, height = 500)
    result <- original_hist(...)
    dev.off()
    png_files <<- c(png_files, filename)
    plot_counter <<- plot_counter + 1
    return(result)
}}

barplot <- function(...) {{
    filename <- paste0("plot_", plot_counter, ".png")
    png(filename, width = 800, height = 500)
    result <- original_barplot(...)
    dev.off()
    png_files <<- c(png_files, filename)
    plot_counter <<- plot_counter + 1
    return(result)
}}

pairs <- function(...) {{
    filename <- paste0("plot_", plot_counter, ".png")
    png(filename, width = 800, height = 600)
    result <- original_pairs(...)
    dev.off()
    png_files <<- c(png_files, filename)
    plot_counter <<- plot_counter + 1
    return(result)
}}

# Sink output to a file
sink("output.txt")

# Run the provided R code
{r_code}

# Unsink output
sink()

# Save list of plot files
cat(paste(png_files, collapse=","), file="plots.txt")

# Done
cat("Analysis completed successfully.\\n")
"""

# Save R script
r_script_file = model_dir / "run_analysis.R"
with open(r_script_file, "w") as f:
    f.write(r_script)

print(f"Running R script for {target_model}...")

try:
    # Run R script
    process = subprocess.Popen(
        ["Rscript", str(r_script_file)],
        cwd=str(model_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate(timeout=90)  # Timeout after 90 seconds
    
    if process.returncode != 0:
        print(f"Error running R script: {stderr.decode()}")
        sys.exit(1)
    
    print("R script completed successfully")
    
    # Read output text
    try:
        with open(model_dir / "output.txt", "r") as f:
            output_text = f.read()
            print("\nText output preview (first 300 chars):")
            print(output_text[:300], "...")
    except FileNotFoundError:
        output_text = "No text output generated."
        print("No text output file found")
    
    # Get plot files
    try:
        with open(model_dir / "plots.txt", "r") as f:
            plot_files = f.read().split(",")
            plot_files = [p for p in plot_files if p.strip()]
            print(f"\nFound {len(plot_files)} plot files:")
            for p in plot_files:
                print(f"- {p}")
    except FileNotFoundError:
        plot_files = []
        print("No plot files found")
    
    # Process plots
    plot_data = []
    for plot_file in plot_files:
        try:
            plot_path = model_dir / plot_file
            if plot_path.exists():
                with open(plot_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    plot_data.append(img_data)
                print(f"Processed {plot_file} successfully")
        except Exception as e:
            print(f"Error processing plot {plot_file}: {e}")
    
    # Add results to model
    model["synthetic_data"]["results"] = {
        "text_output": output_text,
        "plots": plot_data
    }
    
    # Save the updated model
    models[target_model] = model
    with open('model_database.json', 'w') as f:
        json.dump(models, f, indent=4)
    
    print("\nUpdated model database with results.")
    print(f"Added {len(plot_data)} plots and {len(output_text)} characters of text output.")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1) 