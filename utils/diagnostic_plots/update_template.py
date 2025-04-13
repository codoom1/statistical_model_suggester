#!/usr/bin/env python
"""
Update model_interpretation.html template to use saved PNG files instead of base64 strings.
This script modifies the template to look for PNG files in the static/diagnostic_plots directory.
"""
import os
import sys
import re
from pathlib import Path

def update_template(template_path, output_path=None):
    """
    Update the model_interpretation.html template to use saved PNG files.
    
    Args:
        template_path: Path to the original template file
        output_path: Path to save the updated template (if None, overwrites original)
    """
    if output_path is None:
        output_path = template_path
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Pattern to find the diagnostic plots section
    diagnostic_section_pattern = r'({% for plot in interpretation\.diagnostic_plots %}.*?{% endfor %})'
    
    # New diagnostic plots section that uses static files
    new_section = """{% for plot in interpretation.diagnostic_plots %}
                    <div class="figure-container">
                        {% if plot.img_data %}
                        <img src="{{ url_for('static', filename='diagnostic_plots/' + model_name.replace(' ', '_').lower() + '/' + (loop.index|string) + '_' + plot.title.replace(' ', '_').lower() + '.png') }}" alt="{{ plot.title }}">
                        {% else %}
                        <div class="border p-4 bg-light rounded">
                            <i class="bi bi-image text-secondary" style="font-size: 3rem;"></i>
                            <p class="text-center text-muted mb-0 mt-2">Plot: {{ plot.title }}</p>
                        </div>
                        {% endif %}
                        <div class="figure-caption">Figure: {{ plot.title }}</div>
                        <div class="mt-3 text-start">
                            <i class="bi bi-eye-fill me-2 text-primary"></i>
                            <strong>How to interpret:</strong> {{ plot.interpretation }}
                        </div>
                    </div>
                    {% endfor %}"""
    
    # Replace the diagnostic plots section with the new version
    updated_content = re.sub(diagnostic_section_pattern, new_section, content, flags=re.DOTALL)
    
    with open(output_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated template saved to {output_path}")

def main():
    if len(sys.argv) > 1:
        template_path = sys.argv[1]
    else:
        template_path = 'templates/model_interpretation.html'
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = None
    
    update_template(template_path, output_path)

if __name__ == "__main__":
    main() 