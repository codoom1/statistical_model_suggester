#!/usr/bin/env python
"""
Debug utility for diagnostic plot issue.
This script creates a patch for the interpretation.py file to add support for 
static PNG files while maintaining backward compatibility with base64 encoding.
"""
import os
import sys
import re
from pathlib import Path

def patch_interpretation_py(file_path='utils/interpretation.py'):
    """
    Patch the interpretation.py file to support both static PNG files and base64 encoding.
    
    Args:
        file_path: Path to the interpretation.py file
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add a parameter to check for static files
    get_diagnostic_plots_pattern = r'def get_diagnostic_plots\(model_name: str, model_details: Dict\[str, Any\], check_static_files: bool = True\) -> List\[Dict\[str, Any\]\]:'
    get_diagnostic_plots_replacement = 'def get_diagnostic_plots(model_name: str, model_details: Dict[str, Any], check_static_files: bool = True) -> List[Dict[str, Any]]:'
    
    # Add code to check for static files
    model_name_lower_pattern = r'    model_name_lower = model_name\.lower\(\)'
    model_name_lower_replacement = '''    model_name_lower = model_name.lower()
    
    # Check if static files exist
    if check_static_files:
        static_dir = os.path.join('static', 'diagnostic_plots', model_name.replace(" ", "_").lower())
        has_static_files = os.path.exists(static_dir) and len(os.listdir(static_dir)) > 0
        
        if has_static_files:
            print(f"Found static diagnostic plot files for {model_name} in {static_dir}")
            
            # Get titles from JSON files
            plot_files = [f for f in os.listdir(static_dir) if f.endswith('.json')]
            plots = []
            
            for plot_file in sorted(plot_files):
                # Get the plot file path and title
                plot_path = os.path.join(static_dir, plot_file)
                
                try:
                    with open(plot_path, 'r') as f:
                        import json
                        plot_data = json.load(f)
                        plots.append({
                            "title": plot_data.get("title", "Diagnostic Plot"),
                            "img_data": "static_file",  # Special marker to indicate static file
                            "interpretation": plot_data.get("interpretation", "No interpretation available.")
                        })
                except Exception as e:
                    print(f"Error reading plot data from {plot_path}: {e}")
            
            if plots:
                print(f"Using {len(plots)} static diagnostic plot files for {model_name}")
                return plots'''
    
    # Update the imports
    imports_pattern = r'import base64\nimport io\nfrom typing import Dict, Any, List'
    imports_replacement = 'import base64\nimport io\nimport os\nfrom typing import Dict, Any, List'
    
    # Replace the patterns
    content = re.sub(get_diagnostic_plots_pattern, get_diagnostic_plots_replacement, content)
    content = re.sub(model_name_lower_pattern, model_name_lower_replacement, content)
    content = re.sub(imports_pattern, imports_replacement, content)
    
    # Save the updated file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {file_path} to support static diagnostic plot files")

def update_template(template_path='templates/model_interpretation.html'):
    """
    Update the template to support both static PNG files and base64 encoding.
    
    Args:
        template_path: Path to the template file
    """
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Find the img tag in the template
    img_tag_pattern = r'<img src="\{\{ url_for\(\'static\', filename=\'diagnostic_plots/\' \+ model_name\.replace\(\' \', \'_\'\)\.lower\(\) \+ \'/\' \+ \(loop\.index\|string\) \+ \'_\' \+ plot\.title\.replace\(\' \', \'_\'\)\.lower\(\) \+ \'\.png\'\) \}\}" alt="\{\{ plot\.title \}\}">'
    
    # Updated img tag that checks for static_file marker
    img_tag_replacement = '''<img src="{% if plot.img_data == 'static_file' %}{{ url_for('static', filename='diagnostic_plots/' + model_name.replace(' ', '_').lower() + '/' + (loop.index|string) + '_' + plot.title.replace(' ', '_').lower() + '.png') }}{% else %}data:image/png;base64,{{ plot.img_data }}{% endif %}" alt="{{ plot.title }}">'''
    
    # Replace the img tag
    updated_content = re.sub(img_tag_pattern, img_tag_replacement, content)
    
    # Write the updated template
    with open(template_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated {template_path} to support both static files and base64 encoding")

def main():
    if len(sys.argv) > 1:
        interpretation_path = sys.argv[1]
    else:
        interpretation_path = 'utils/interpretation.py'
    
    if len(sys.argv) > 2:
        template_path = sys.argv[2]
    else:
        template_path = 'templates/model_interpretation.html'
    
    # Patch the interpretation.py file
    patch_interpretation_py(interpretation_path)
    
    # Update the template
    update_template(template_path)
    
    print("\nDone! The changes should allow using both static PNG files and base64 encoding.")
    print("Try refreshing the interpretation page in your browser to see if the plots appear.")

if __name__ == "__main__":
    main() 