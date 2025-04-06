from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from models import db, User, Analysis, get_model_details
from datetime import datetime
import json
import os

main = Blueprint('main', __name__)

# Path for history file (legacy support)
HISTORY_FILE = 'history.json'

def load_history():
    """Load analysis history from JSON file (legacy support)"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(history):
    """Save analysis history to JSON file (legacy support)"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def add_to_history(research_question, recommended_model, analysis_goal, dependent_variable_type, 
                   independent_variables, sample_size, missing_data, data_distribution, relationship_type):
    """Add analysis to history file (legacy support)"""
    history = load_history()
    history.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'research_question': research_question,
        'analysis_goal': analysis_goal,
        'dependent_variable': dependent_variable_type,
        'independent_variables': independent_variables,
        'sample_size': sample_size,
        'missing_data': missing_data,
        'data_distribution': data_distribution,
        'relationship_type': relationship_type,
        'recommended_model': recommended_model
    })
    save_history(history)

def get_model_recommendation(analysis_goal, dependent_variable, independent_variables,
                            sample_size, missing_data, data_distribution, relationship_type):
    """Get model recommendation based on input parameters"""
    # Get the model database from app config
    MODEL_DATABASE = current_app.config.get('MODEL_DATABASE', {})
    
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
    for model_name, model in MODEL_DATABASE.items():
        score = 0

        # Check analysis goal compatibility
        if analysis_goal in model.get('analysis_goals', []):
            score += 2

        # Check dependent variable compatibility
        if dependent_variable in model.get('dependent_variable', []):
            score += 2

        # Check sample size compatibility
        if size_category in model.get('sample_size', []):
            score += 1

        # Check missing data handling
        if missing_data in model.get('missing_data', []):
            score += 1

        # Check independent variable compatibility
        independent_var_score = 0
        for var in independent_variables:
            if var in model.get('independent_variables', []):
                independent_var_score += 1
        if independent_variables and independent_var_score == len(independent_variables):
            score += 2
        elif independent_variables and independent_var_score > 0:
            score += 1

        # Check data distribution compatibility
        if data_distribution in model.get('data_distribution', []):
            score += 1

        # Check relationship type compatibility
        if relationship_type in model.get('relationship_type', []):
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
    """Generate explanation for model recommendation"""
    model_info = get_model_details(model_name) or {}
    
    explanation = f"\n    Based on your data characteristics, a {model_name} is recommended because:\n    \n"
    
    reasons = []
    if analysis_goal in model_info.get('analysis_goals', []):
        reasons.append(f"It is suitable for {analysis_goal} analysis with {dependent_variable} dependent variables")
    
    if independent_variables and all(var in model_info.get('independent_variables', []) for var in independent_variables):
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
    """Get default model based on analysis goal and dependent variable type"""
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
    
    # Default fallback
    return 'Linear Regression'

@main.route('/')
def home():
    # Create stats for the home page
    stats = {
        'models_count': len(current_app.config.get('MODEL_DATABASE', {})),
        'access_hours': '24/7',
        'verification_rate': '95%'
    }
    return render_template('home.html', stats=stats, now=datetime.now())

@main.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """View and edit user profile"""
    if request.method == 'POST':
        # Update basic profile information
        email = request.form.get('email')
        
        # Check if email already exists for another user
        if email != current_user.email:
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash('Email already in use.', 'danger')
                return redirect(url_for('main.profile'))
        
        current_user.email = email
        
        # If user is an expert, also update expert fields
        if current_user.is_expert:
            current_user.areas_of_expertise = request.form.get('areas_of_expertise', '')
            current_user.institution = request.form.get('institution', '')
            current_user.bio = request.form.get('bio', '')
        
        db.session.commit()
        flash('Profile updated successfully.', 'success')
        return redirect(url_for('main.profile'))
    
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).all()
    return render_template('profile.html', user=current_user, analyses=analyses)

@main.route('/results', methods=['POST'])
def results():
    """Process form submission and show results"""
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
        
        # Save to history (legacy support)
        add_to_history(research_question, recommended_model, analysis_goal, dependent_variable_type, 
                       independent_variables, sample_size, missing_data, data_distribution, relationship_type)
        
        # If user is logged in, save to their history in the database
        if current_user.is_authenticated:
            analysis = Analysis(
                user_id=current_user.id,
                research_question=research_question,
                analysis_goal=analysis_goal,
                dependent_variable=dependent_variable_type, 
                independent_variables=json.dumps(independent_variables),
                sample_size=sample_size,
                missing_data=missing_data,
                data_distribution=data_distribution,
                relationship_type=relationship_type,
                recommended_model=recommended_model
            )
            db.session.add(analysis)
            db.session.commit()
        
        return render_template('results.html', 
                             research_question=research_question,
                             recommended_model=recommended_model,
                             explanation=explanation,
                             MODEL_DATABASE=current_app.config.get('MODEL_DATABASE', {}),
                             analysis_goal=analysis_goal,
                             dependent_variable_type=dependent_variable_type,
                             independent_variables=independent_variables,
                             sample_size=sample_size,
                             missing_data=missing_data,
                             data_distribution=data_distribution,
                             relationship_type=relationship_type)
    except Exception as e:
        return render_template('error.html', error=str(e))

@main.route('/user_analysis/<int:analysis_id>')
@login_required
def user_analysis(analysis_id):
    """View a specific analysis from user history"""
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Security check - ensure users can only see their own analyses
    if analysis.user_id != current_user.id:
        return render_template('error.html', error="Unauthorized access")
    
    # Create a custom explanation for historical view
    explanation = f"""
    <strong>Historical Analysis from {analysis.created_at.strftime('%Y-%m-%d %H:%M:%S')}</strong><br>
    This is a recommendation previously generated based on your inputs for:
    <ul>
        <li>Analysis Goal: {analysis.analysis_goal}</li>
        <li>Dependent Variable: {analysis.dependent_variable}</li>
        <li>Sample Size: {analysis.sample_size}</li>
    </ul>
    """
    
    independent_variables = json.loads(analysis.independent_variables)
    
    return render_template('results.html',
                         research_question=analysis.research_question,
                         recommended_model=analysis.recommended_model,
                         explanation=explanation,
                         MODEL_DATABASE=current_app.config.get('MODEL_DATABASE', {}),
                         analysis_goal=analysis.analysis_goal,
                         dependent_variable_type=analysis.dependent_variable,
                         independent_variables=independent_variables,
                         sample_size=analysis.sample_size,
                         missing_data=analysis.missing_data,
                         data_distribution=analysis.data_distribution,
                         relationship_type=analysis.relationship_type)

@main.route('/history')
def history():
    """View analysis history"""
    try:
        if current_user.is_authenticated:
            # For logged-in users, redirect to their profile which shows their analyses
            return redirect(url_for('main.profile'))
        else:
            # For anonymous users, use the legacy JSON file-based history
            history_data = load_history()
            return render_template('history.html', history=history_data)
    except Exception as e:
        return render_template('error.html', error=str(e))

@main.route('/models')
def models_list():
    """Display list of all available models"""
    try:
        return render_template('models_list.html', models=current_app.config.get('MODEL_DATABASE', []))
    except Exception as e:
        return render_template('error.html', error=str(e))

@main.route('/model/<model_name>')
def model_details(model_name):
    """Display details for a specific model"""
    try:
        model_info = get_model_details(model_name)
        if model_info:
            return render_template('model_details.html',
                                 model_name=model_name,
                                 model_details=model_info)
        else:
            return render_template('error.html', error="Model not found")
    except Exception as e:
        return render_template('error.html', error=str(e))

@main.route('/history/<int:index>')
def view_history_result(index):
    """View a specific result from history"""
    try:
        history = load_history()
        if 0 <= index < len(history):
            entry = history[index]
            return render_template('results.html',
                                research_question=entry['research_question'],
                                recommended_model=entry['recommended_model'],
                                explanation=f"Historical analysis from {entry['timestamp']}",
                                MODEL_DATABASE=current_app.config.get('MODEL_DATABASE', {}),
                                analysis_goal=entry['analysis_goal'],
                                dependent_variable_type=entry['dependent_variable'],
                                independent_variables=entry['independent_variables'],
                                sample_size=entry['sample_size'],
                                missing_data=entry['missing_data'],
                                data_distribution=entry['data_distribution'],
                                relationship_type=entry['relationship_type'])
        else:
            return render_template('error.html', error="History entry not found")
    except Exception as e:
        return render_template('error.html', error=str(e))

@main.route('/analysis-form')
@login_required
def analysis_form():
    return render_template('analysis_form.html') 