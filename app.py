from flask import Flask, render_template, request, redirect, url_for
import json
import os
from datetime import datetime
import random

app = Flask(__name__)

# Load model database
with open('model_database.json', 'r') as f:
    MODEL_DATABASE = json.load(f)

# History file path
HISTORY_FILE = 'history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def add_to_history(research_question, recommended_model, analysis_goal, dependent_variable_type, 
                  independent_variables, sample_size, missing_data, data_distribution, relationship_type):
    """Add an analysis to the history file"""
    history = load_history()
    history.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'research_question': research_question,
        'analysis_goal': analysis_goal,
        'dependent_variable': dependent_variable_type,  # Maintain backward compatibility with history.html
        'independent_variables': independent_variables,
        'sample_size': sample_size,
        'missing_data': missing_data,
        'data_distribution': data_distribution,
        'relationship_type': relationship_type,
        'recommended_model': recommended_model
    })
    save_history(history)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    try:
        # Get form data
        research_question = request.form.get('research_question', '')
        analysis_goal = request.form.get('analysis_goal', '')
        dependent_variable_type = request.form.get('dependent_variable_type', '')
        independent_variables = request.form.getlist('independent_variables')
        sample_size = request.form.get('sample_size', '')
        missing_data = request.form.get('missing_data', '')
        data_distribution = request.form.get('data_distribution', '')
        relationship_type = request.form.get('relationship_type', '')
        
        # Get model recommendation
        recommended_model, explanation = get_model_recommendation(
            analysis_goal, dependent_variable_type, independent_variables,
            sample_size, missing_data, data_distribution, relationship_type
        )
        
        # Save to history
        add_to_history(research_question, recommended_model, analysis_goal, dependent_variable_type, 
                     independent_variables, sample_size, missing_data, data_distribution, relationship_type)
        
        return render_template('results.html', 
                             research_question=research_question,
                             recommended_model=recommended_model,
                             explanation=explanation,
                             MODEL_DATABASE=MODEL_DATABASE,
                             analysis_goal=analysis_goal,
                             dependent_variable_type=dependent_variable_type,
                             independent_variables=independent_variables,
                             sample_size=sample_size,
                             missing_data=missing_data,
                             data_distribution=data_distribution,
                             relationship_type=relationship_type)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/results/<int:index>')
def view_result(index):
    try:
        history = load_history()
        if 0 <= index < len(history):
            entry = history[index]
            # Get model information from database for the model explanation
            model_name = entry['recommended_model']
            model_info = MODEL_DATABASE.get(model_name, {})
            
            # Create a custom explanation for historical view
            explanation = f"""
            <strong>Historical Analysis from {entry.get('timestamp', 'unknown date')}</strong><br>
            This is a recommendation previously generated based on your inputs for:
            <ul>
                <li>Analysis Goal: {entry.get('analysis_goal', 'Not specified')}</li>
                <li>Dependent Variable: {entry.get('dependent_variable', 'Not specified')}</li>
                <li>Sample Size: {entry.get('sample_size', 'Not specified')}</li>
            </ul>
            """
            
            return render_template('results.html',
                                 research_question=entry['research_question'],
                                 recommended_model=entry['recommended_model'],
                                 explanation=explanation,
                                 MODEL_DATABASE=MODEL_DATABASE,
                                 analysis_goal=entry.get('analysis_goal', ''),
                                 dependent_variable_type=entry.get('dependent_variable', ''),
                                 independent_variables=entry.get('independent_variables', []),
                                 sample_size=entry.get('sample_size', ''),
                                 missing_data=entry.get('missing_data', ''),
                                 data_distribution=entry.get('data_distribution', ''),
                                 relationship_type=entry.get('relationship_type', ''))
        else:
            return render_template('error.html', error="Invalid history index")
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/history')
def history():
    try:
        history = load_history()
        return render_template('history.html', history=history)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/model/<model_name>')
def model_details(model_name):
    try:
        if model_name in MODEL_DATABASE:
            return render_template('model_details.html',
                                model_name=model_name,
                                model_details=MODEL_DATABASE[model_name])
        else:
            return render_template('error.html', error="Model not found")
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/models')
def models_list():
    try:
        return render_template('models_list.html', models=MODEL_DATABASE)
    except Exception as e:
        return render_template('error.html', error=str(e))

def get_model_recommendation(analysis_goal, dependent_variable, independent_variables,
                           sample_size, missing_data, data_distribution, relationship_type):
    # Convert sample_size to integer if it's a string
    try:
        sample_size = int(sample_size)
    except (ValueError, TypeError):
        sample_size = 50  # Default to medium sample size if not provided

    # Categorize sample size
    if sample_size < 30:
        size_category = 'small'
    elif sample_size < 100:
        size_category = 'medium'
    else:
        size_category = 'large'

    # Score models based on compatibility
    model_scores = {}
    for model_name, model_info in MODEL_DATABASE.items():
        score = 0

        # Check analysis goal compatibility
        if analysis_goal in model_info.get('analysis_goals', []):
            score += 2

        # Check dependent variable compatibility
        if dependent_variable in model_info.get('dependent_variable', []):
            score += 2

        # Check sample size compatibility
        if size_category in model_info.get('sample_size', []):
            score += 1

        # Check missing data handling
        if missing_data in model_info.get('missing_data', []):
            score += 1

        # Check independent variable compatibility
        independent_var_score = 0
        for var in independent_variables:
            if var in model_info.get('independent_variables', []):
                independent_var_score += 1
        if independent_variables and independent_var_score == len(independent_variables):
            score += 2
        elif independent_variables and independent_var_score > 0:
            score += 1

        # Check data distribution compatibility
        if data_distribution in model_info.get('data_distribution', []):
            score += 1

        # Check relationship type compatibility
        if relationship_type in model_info.get('relationship_type', []):
            score += 1

        model_scores[model_name] = score

    # Get best matching model
    if model_scores:
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        explanation = generate_explanation(best_model, analysis_goal, dependent_variable,
                                        independent_variables, sample_size, missing_data,
                                        data_distribution, relationship_type)
        return best_model, explanation
    else:
        # Fallback to default model
        default_model = get_default_model(analysis_goal, dependent_variable)
        explanation = f"Based on your analysis goal ({analysis_goal}) and dependent variable type ({dependent_variable}), we recommend using {default_model}."
        return default_model, explanation

def generate_explanation(model_name, analysis_goal, dependent_variable, independent_variables,
                        sample_size, missing_data, data_distribution, relationship_type):
    model_info = MODEL_DATABASE.get(model_name, {})
    explanation = f"\n    Based on your data characteristics, a {model_name} is recommended because:\n    \n"
    
    reasons = []
    if analysis_goal in model_info.get('analysis_goals', []):
        reasons.append(f"It is suitable for {analysis_goal} analysis with {dependent_variable} dependent variables")
    
    if all(var in model_info.get('independent_variables', []) for var in independent_variables):
        reasons.append(f"It can handle {', '.join(independent_variables)} independent variables")
    
    # Convert sample_size to int if needed
    try:
        sample_size_int = int(sample_size)
    except (ValueError, TypeError):
        sample_size_int = 50
        
    if sample_size_int < 30 and 'small' in model_info.get('sample_size', []):
        reasons.append("It works well with small sample sizes")
    elif sample_size_int >= 30 and sample_size_int < 100 and 'medium' in model_info.get('sample_size', []):
        reasons.append("It works well with medium sample sizes")
    elif sample_size_int >= 100 and 'large' in model_info.get('sample_size', []):
        reasons.append("It is optimized for large datasets")
    
    if missing_data in model_info.get('missing_data', []):
        reasons.append(f"It can handle {missing_data} missing data patterns")
    
    if data_distribution in model_info.get('data_distribution', []):
        reasons.append(f"It is appropriate for {data_distribution} data distribution")
    
    if relationship_type in model_info.get('relationship_type', []):
        reasons.append(f"It can model {relationship_type} relationships")
    
    # Add numbered reasons
    for i, reason in enumerate(reasons, 1):
        explanation += f"    {i}. {reason}\n"
    
    # Add implementation notes
    explanation += f"""    
    Implementation notes:
    - {model_info.get('description', 'No additional description available.')}
    - Consider preprocessing steps for {', '.join(independent_variables)} variables
    - Check assumptions specific to {model_name}
    """
    
    return explanation

def get_default_model(analysis_goal, dependent_variable):
    # Default models based on analysis goal and dependent variable type
    if analysis_goal == 'predict':
        if dependent_variable == 'continuous':
            return 'Linear Regression'
        elif dependent_variable == 'binary':
            return 'Logistic Regression'
        elif dependent_variable == 'count':
            return 'Poisson Regression'
        elif dependent_variable == 'ordinal':
            return 'Ordinal Regression'
        elif dependent_variable == 'time_to_event':
            return 'Cox Regression'
    elif analysis_goal == 'classify':
        if dependent_variable == 'binary':
            return 'Logistic Regression'
        elif dependent_variable == 'categorical':
            return 'Multinomial Logistic Regression'
    elif analysis_goal == 'explore':
        return 'Principal Component Analysis'
    elif analysis_goal == 'hypothesis_test':
        if dependent_variable == 'continuous':
            return 'T-Test'
        elif dependent_variable == 'categorical':
            return 'Chi-Square Test'
    elif analysis_goal == 'non_parametric':
        return 'Mann-Whitney U Test'
    elif analysis_goal == 'time_series':
        return 'ARIMA'
    
    # Fallback to Linear Regression if no specific match
    return 'Linear Regression'

if __name__ == '__main__':
    app.run(debug=True, port=8080)


