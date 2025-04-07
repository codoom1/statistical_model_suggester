#!/usr/bin/env python3
import json
import os
import subprocess
import base64
from pathlib import Path

# Load model database
def load_models():
    with open('model_database.json', 'r') as f:
        return json.load(f)

# Save model database
def save_models(models):
    with open('model_database.json', 'w') as f:
        json.dump(models, f, indent=4)

# Function to generate R results
def generate_r_results(models):
    # Create results directory if it doesn't exist
    results_dir = Path("static/r_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare R script template
    r_script_template = """
    # Load required packages
    required_packages <- c('knitr', 'png', 'ggplot2')
    for(pkg in required_packages) {
        if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
            install.packages(pkg, repos = 'https://cloud.r-project.org')
            library(pkg, character.only = TRUE)
        }
    }
    
    # Set up to capture plots
    options(width = 120)
    
    # Function to capture output
    capture_all <- function(code_string) {
        # Create temp files for plots
        plot_files <- c()
        plot_counter <- 1
        
        # Set up plot capture
        plot_capture <- function() {
            filename <- paste0("temp_plot_", plot_counter, ".png")
            png(filename, width = 800, height = 500, res = 100)
            plot_counter <<- plot_counter + 1
            plot_files <<- c(plot_files, filename)
        }
        
        # Setup text capture
        output <- capture.output({
            tryCatch({
                # Evaluate the code with special handling for plots
                parsed_code <- parse(text = code_string)
                for (i in seq_along(parsed_code)) {
                    expr <- parsed_code[i]
                    
                    # Capture plots
                    if (any(grepl("plot|boxplot|hist|barplot|pairs", as.character(expr)))) {
                        plot_capture()
                        eval(expr)
                        dev.off()
                    } else {
                        # Normal evaluation
                        print(eval(expr))
                    }
                }
            }, error = function(e) {
                cat("ERROR: ", conditionMessage(e), "\n")
            })
        }, split = TRUE)
        
        # Create result list
        result <- list(
            text_output = paste(output, collapse = "\n"),
            plot_files = plot_files
        )
        
        return(result)
    }
    
    # Run the provided R code and capture results
    code_to_run <- {CODE_PLACEHOLDER}
    
    # Capture all output
    results <- capture_all(code_to_run)
    
    # Save text output
    cat(results$text_output, file = "output.txt")
    
    # Return plot filenames
    cat(paste(results$plot_files, collapse = ","), file = "plots.txt")
    """
    
    count = 0
    for model_name, model in models.items():
        if "synthetic_data" in model and "r_code" in model["synthetic_data"]:
            print(f"Generating results for {model_name}...")
            
            # Create model-specific directory
            model_dir_name = model_name.replace(" ", "_").replace("/", "_")
            model_dir = results_dir / model_dir_name
            model_dir.mkdir(exist_ok=True)
            
            # Get R code
            r_code = model["synthetic_data"]["r_code"]
            
            # Create R script file with code embedded
            r_script = r_script_template.replace("{CODE_PLACEHOLDER}", repr(r_code))
            r_script_file = model_dir / "run_analysis.R"
            with open(r_script_file, "w") as f:
                f.write(r_script)
            
            try:
                # Run R script
                process = subprocess.Popen(
                    ["Rscript", r_script_file],
                    cwd=str(model_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate(timeout=60)  # Timeout after 60 seconds
                
                if process.returncode != 0:
                    print(f"Error running R script for {model_name}: {stderr.decode()}")
                    continue
                
                # Read output text
                try:
                    with open(model_dir / "output.txt", "r") as f:
                        output_text = f.read()
                except FileNotFoundError:
                    output_text = "No text output generated."
                
                # Get plot files
                try:
                    with open(model_dir / "plots.txt", "r") as f:
                        plot_files = f.read().split(",")
                        plot_files = [p for p in plot_files if p.strip()]
                except FileNotFoundError:
                    plot_files = []
                
                # Process plots
                plot_data = []
                for plot_file in plot_files:
                    try:
                        plot_path = model_dir / plot_file
                        if plot_path.exists():
                            with open(plot_path, "rb") as img_file:
                                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                                plot_data.append(img_data)
                    except Exception as e:
                        print(f"Error processing plot {plot_file}: {e}")
                
                # Add results to model
                model["synthetic_data"]["results"] = {
                    "text_output": output_text,
                    "plots": plot_data
                }
                
                count += 1
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
    
    print(f"Generated results for {count} models")
    return models

if __name__ == "__main__":
    # Load the model database
    print("Loading model database...")
    models = load_models()
    print(f"Loaded {len(models)} models")
    
    # Generate R results
    print("Generating R results...")
    models = generate_r_results(models)
    
    # Save the updated model database
    print("Saving updated model database...")
    save_models(models)
    
    print("Done! Model database updated with R analysis results.") 