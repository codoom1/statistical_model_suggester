#!/usr/bin/env python
"""
Create placeholder diagnostic plot files for PCA and ANOVA models.
This is a workaround for the circular import issue.
"""
import os
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Models to create placeholder plots for
MODELS = {
    "Principal Component Analysis": [
        {"title": "Scree Plot", "interpretation": "Shows the proportion of variance explained by each principal component. The cumulative line helps determine how many components to retain."},
        {"title": "PCA Biplot", "interpretation": "Visualizes both samples (points) and variables (arrows) in the PCA space. Variables pointing in similar directions are positively correlated."},
        {"title": "PC1 Loadings", "interpretation": "Shows how strongly each original feature influences the first principal component."},
        {"title": "Correlation Circle", "interpretation": "Shows correlations between variables and principal components."}
    ],
    "ANOVA": [
        {"title": "Box Plot", "interpretation": "Compares the distribution of the dependent variable across groups."},
        {"title": "Residuals vs Fitted", "interpretation": "Checks for homogeneity of variance and linearity."},
        {"title": "Q-Q Plot", "interpretation": "Assesses normality of residuals."},
        {"title": "F-test Results", "interpretation": "Shows the F-statistic, degrees of freedom, and p-value for testing differences between group means."}
    ],
    "Repeated Measures ANOVA": [
        {"title": "Profile Plot", "interpretation": "Shows the mean response for each condition and subject."},
        {"title": "Residuals vs Fitted", "interpretation": "Checks for homogeneity of variance and linearity."},
        {"title": "Q-Q Plot", "interpretation": "Assesses normality of residuals."},
        {"title": "Sphericity Test", "interpretation": "Assesses if the variances of the differences between all combinations of related groups are equal."}
    ]
}

def create_placeholder_image(title, size=(800, 600), color=(240, 240, 240)):
    """Create a placeholder image with title text"""
    # Create a new image with a light gray background
    image = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(image)
    
    # Try to use a standard font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial", 36)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw the title text centered in the image
    text_width, text_height = draw.textsize(title, font=font) if hasattr(draw, 'textsize') else (200, 36)
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    
    # Draw with a dark gray text color
    draw.text(position, title, fill=(80, 80, 80), font=font)
    
    # Add some fake chart elements
    if "Scree Plot" in title or "Loadings" in title:
        # Bar chart
        num_bars = 5
        bar_width = size[0] // (num_bars * 2)
        height_values = np.random.rand(num_bars) * size[1] // 3
        
        for i in range(num_bars):
            x = size[0] // 4 + i * bar_width * 2
            y = size[1] * 2 // 3
            bar_height = int(height_values[i])
            draw.rectangle([x, y - bar_height, x + bar_width, y], fill=(70, 130, 180))
    
    elif "Box Plot" in title or "Means" in title:
        # Box plots
        num_boxes = 3
        box_width = size[0] // (num_boxes * 3)
        
        for i in range(num_boxes):
            x = size[0] // 4 + i * box_width * 3
            y = size[1] // 3
            draw.rectangle([x, y, x + box_width, y + size[1] // 3], outline=(70, 130, 180), width=2)
            
            # Median line
            median_y = y + np.random.randint(0, size[1] // 3)
            draw.line([x, median_y, x + box_width, median_y], fill=(255, 0, 0), width=2)
    
    elif "Biplot" in title or "Interaction" in title or "Circle" in title:
        # Scatter plot with arrows/lines
        # Draw axes
        center_x, center_y = size[0] // 2, size[1] // 2
        draw.line([center_x, size[1] // 4, center_x, size[1] * 3 // 4], fill=(0, 0, 0), width=1)
        draw.line([size[0] // 4, center_y, size[0] * 3 // 4, center_y], fill=(0, 0, 0), width=1)
        
        # Draw some random points
        num_points = 20
        points = np.random.randint(0, 100, (num_points, 2))
        for x, y in points:
            point_x = size[0] // 4 + x * size[0] // 200
            point_y = size[1] // 4 + y * size[1] // 200
            r = 3  # point radius
            draw.ellipse([point_x-r, point_y-r, point_x+r, point_y+r], fill=(220, 20, 60))
            
        # Draw some arrows for biplots
        if "Biplot" in title or "Circle" in title:
            num_arrows = 4
            for i in range(num_arrows):
                angle = np.pi * 2 * i / num_arrows
                end_x = center_x + int(np.cos(angle) * size[0] // 6)
                end_y = center_y + int(np.sin(angle) * size[1] // 6)
                draw.line([center_x, center_y, end_x, end_y], fill=(0, 100, 0), width=2)
    
    elif "F-test" in title:
        # Create a table-like view for F-test results
        header_y = size[1] // 3
        draw.line([size[0] // 4, header_y, size[0] * 3 // 4, header_y], fill=(0, 0, 0), width=2)
        
        # Column dividers
        col1_x = size[0] // 4 + size[0] // 6
        col2_x = size[0] // 4 + size[0] * 2 // 6
        draw.line([col1_x, header_y - 50, col1_x, header_y + 150], fill=(0, 0, 0), width=1)
        draw.line([col2_x, header_y - 50, col2_x, header_y + 150], fill=(0, 0, 0), width=1)
        
        # Add headers
        if hasattr(draw, 'textsize'):
            draw.text((size[0] // 4 + 20, header_y - 40), "Source", fill=(0, 0, 0), font=font)
            draw.text((col1_x + 20, header_y - 40), "F-value", fill=(0, 0, 0), font=font)
            draw.text((col2_x + 20, header_y - 40), "p-value", fill=(0, 0, 0), font=font)
            
            # Add data row
            draw.text((size[0] // 4 + 20, header_y + 20), "Between Groups", fill=(0, 0, 0), font=font)
            draw.text((col1_x + 20, header_y + 20), "12.43", fill=(0, 0, 0), font=font)
            draw.text((col2_x + 20, header_y + 20), "0.001", fill=(0, 0, 0), font=font)
                
    elif "Q-Q Plot" in title:
        # Diagonal line for Q-Q plot
        draw.line([size[0] // 4, size[1] // 4, size[0] * 3 // 4, size[1] * 3 // 4], fill=(0, 0, 0), width=2)
        
        # Add points close to the line
        num_points = 15
        x_points = np.linspace(size[0] // 4, size[0] * 3 // 4, num_points)
        
        for i, x in enumerate(x_points):
            ideal_y = size[1] // 4 + (i / (num_points-1)) * (size[1] * 2 // 4)
            # Add some noise
            noise = np.random.normal(0, size[1] // 30)
            y = int(ideal_y + noise)
            r = 3  # point radius
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(70, 130, 180))
    
    elif "Residuals" in title:
        # Create random scatter of residuals
        num_points = 30
        x_points = np.linspace(size[0] // 4, size[0] * 3 // 4, num_points)
        
        # Horizontal zero line
        zero_y = size[1] // 2
        draw.line([size[0] // 4, zero_y, size[0] * 3 // 4, zero_y], fill=(255, 0, 0), width=2)
        
        for x in x_points:
            y = zero_y + np.random.normal(0, size[1] // 10)
            r = 3  # point radius
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(70, 130, 180))
            
    return image

def create_plots():
    """Create placeholder plot files for all models"""
    base_dir = "static/diagnostic_plots"
    os.makedirs(base_dir, exist_ok=True)
    
    for model, plots in MODELS.items():
        # Create directory for model
        model_dir = os.path.join(base_dir, model.replace(" ", "_").lower())
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"Creating plots for {model}...")
        
        # Create each plot
        for i, plot_data in enumerate(plots):
            title = plot_data["title"]
            interpretation = plot_data["interpretation"]
            filename_base = f"{i+1}_{title.replace(' ', '_').lower()}"
            
            # Create and save the image file
            img_path = os.path.join(model_dir, f"{filename_base}.png")
            img = create_placeholder_image(title)
            img.save(img_path)
            
            # Create the JSON metadata file
            json_path = os.path.join(model_dir, f"{filename_base}.json")
            with open(json_path, 'w') as f:
                json.dump({
                    "title": title,
                    "interpretation": interpretation
                }, f, indent=2)
                
            print(f"  Created {filename_base}.png and {filename_base}.json")
            
        print(f"Completed {model} plots\n")

if __name__ == "__main__":
    create_plots() 