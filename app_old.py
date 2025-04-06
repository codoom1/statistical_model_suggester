from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import json
import os
from datetime import datetime
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration from environment variables
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', 'True').lower() == 'true'
app.config['REMEMBER_COOKIE_SECURE'] = os.environ.get('REMEMBER_COOKIE_SECURE', 'True').lower() == 'true'
app.config['SESSION_COOKIE_HTTPONLY'] = os.environ.get('SESSION_COOKIE_HTTPONLY', 'True').lower() == 'true'
app.config['REMEMBER_COOKIE_HTTPONLY'] = os.environ.get('REMEMBER_COOKIE_HTTPONLY', 'True').lower() == 'true'

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Load model database
try:
    with open('model_database.json', 'r') as f:
        MODEL_DATABASE = json.load(f)
except FileNotFoundError:
    print("Error: model_database.json not found")
    MODEL_DATABASE = {}

# History file path
HISTORY_FILE = 'history.json'

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    role = db.Column(db.String(20), default='user')  # 'user', 'expert', 'admin'
    is_approved_expert = db.Column(db.Boolean, default=False)
    expertise = db.Column(db.String(500), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    institution = db.Column(db.String(200), nullable=True)
    analyses = db.relationship('Analysis', backref='user', lazy=True)
    consultations_requested = db.relationship('Consultation', backref='requester', lazy=True, foreign_keys='Consultation.requester_id')
    consultations_provided = db.relationship('Consultation', backref='expert', lazy=True, foreign_keys='Consultation.expert_id')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def is_expert(self):
        return self.role == 'expert' and self.is_approved_expert

# Analysis model to store user analyses
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    research_question = db.Column(db.String(500), nullable=False)
    analysis_goal = db.Column(db.String(50))
    dependent_variable = db.Column(db.String(50))
    independent_variables = db.Column(db.String(200))  # Store as JSON string
    sample_size = db.Column(db.String(20))
    missing_data = db.Column(db.String(50))
    data_distribution = db.Column(db.String(50))
    relationship_type = db.Column(db.String(50))
    recommended_model = db.Column(db.String(100))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_independent_variables(self):
        return json.loads(self.independent_variables)

# Consultation model for expert advice requests
class Consultation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    requester_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    expert_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis.id'), nullable=True)
    title = db.Column(db.String(200), nullable=False)
    question = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='pending')  # 'pending', 'assigned', 'completed', 'declined'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    public = db.Column(db.Boolean, default=False)  # Whether this consultation is visible to all users

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create the database and tables
with app.app_context():
    db.create_all()

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

# Save analysis to database for logged in user
def save_user_analysis(user_id, research_question, recommended_model, analysis_goal, dependent_variable_type, 
                      independent_variables, sample_size, missing_data, data_distribution, relationship_type):
    """Save analysis to user's history in database"""
    analysis = Analysis(
        user_id=user_id,
        research_question=research_question,
        analysis_goal=analysis_goal,
        dependent_variable=dependent_variable_type, 
        independent_variables=json.dumps(independent_variables),
        sample_size=sample_size,
        missing_data=missing_data,
        data_distribution=data_distribution,
        relationship_type=relationship_type,
        recommended_model=recommended_model,
        timestamp=datetime.utcnow()
    )
    db.session.add(analysis)
    db.session.commit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=request.form.get('remember', False))
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page or url_for('home'))
        else:
            flash('Login failed. Please check your username and password.', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('register.html')
            
        user_exists = User.query.filter_by(username=username).first()
        email_exists = User.query.filter_by(email=email).first()
        
        if user_exists:
            flash('Username already exists!', 'danger')
        elif email_exists:
            flash('Email already registered!', 'danger')
        else:
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/profile')
@login_required
def profile():
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.timestamp.desc()).all()
    return render_template('profile.html', user=current_user, analyses=analyses)

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
        
        # Save to history (legacy support)
        add_to_history(research_question, recommended_model, analysis_goal, dependent_variable_type, 
                     independent_variables, sample_size, missing_data, data_distribution, relationship_type)
        
        # If user is logged in, save to their history in the database
        if current_user.is_authenticated:
            save_user_analysis(
                current_user.id, research_question, recommended_model, analysis_goal, dependent_variable_type,
                independent_variables, sample_size, missing_data, data_distribution, relationship_type
            )
        
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

@app.route('/user_analysis/<int:analysis_id>')
@login_required
def user_analysis(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Security check - ensure users can only see their own analyses
    if analysis.user_id != current_user.id:
        return render_template('error.html', error="Unauthorized access")
    
    # Create a custom explanation for historical view
    explanation = f"""
    <strong>Historical Analysis from {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</strong><br>
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
                         MODEL_DATABASE=MODEL_DATABASE,
                         analysis_goal=analysis.analysis_goal,
                         dependent_variable_type=analysis.dependent_variable,
                         independent_variables=independent_variables,
                         sample_size=analysis.sample_size,
                         missing_data=analysis.missing_data,
                         data_distribution=analysis.data_distribution,
                         relationship_type=analysis.relationship_type)

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
        if current_user.is_authenticated:
            # For logged-in users, redirect to their profile which shows their analyses
            return redirect(url_for('profile'))
        else:
            # For anonymous users, use the legacy JSON file-based history
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
    
    # Default fallback
    return 'Linear Regression'

# Expert-related routes
@app.route('/experts')
def experts_list():
    try:
        experts = User.query.filter_by(role='expert', is_approved_expert=True).all()
        return render_template('experts_list.html', experts=experts)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/expert/<int:expert_id>')
def expert_profile(expert_id):
    try:
        expert = User.query.filter_by(id=expert_id, role='expert', is_approved_expert=True).first_or_404()
        # Get public consultations this expert has answered
        consultations = Consultation.query.filter_by(expert_id=expert_id, public=True, status='completed').all()
        return render_template('expert_profile.html', expert=expert, consultations=consultations)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/apply-expert', methods=['GET', 'POST'])
@login_required
def apply_expert():
    try:
        if request.method == 'POST':
            current_user.role = 'expert'
            current_user.expertise = request.form.get('expertise', '')
            current_user.bio = request.form.get('bio', '')
            current_user.institution = request.form.get('institution', '')
            # Expert needs approval before they can offer consultations
            current_user.is_approved_expert = False
            db.session.commit()
            flash('Your expert application has been submitted for review!', 'success')
            return redirect(url_for('profile'))
        
        return render_template('apply_expert.html')
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/request-consultation', methods=['GET', 'POST'])
@login_required
def request_consultation():
    try:
        analysis_id = request.args.get('analysis_id')
        analysis = None
        if analysis_id:
            analysis = Analysis.query.get(analysis_id)
            if analysis and analysis.user_id != current_user.id:
                analysis = None
        
        if request.method == 'POST':
            title = request.form.get('title', '')
            question = request.form.get('question', '')
            analysis_id = request.form.get('analysis_id')
            expert_id = request.form.get('expert_id')
            
            if not title or not question:
                flash('Please provide both title and question', 'danger')
                return render_template('request_consultation.html', analysis=analysis)
            
            consultation = Consultation(
                requester_id=current_user.id,
                title=title,
                question=question,
                status='pending'
            )
            
            if analysis_id:
                consultation.analysis_id = analysis_id
            
            if expert_id:
                expert = User.query.filter_by(id=expert_id, role='expert', is_approved_expert=True).first()
                if expert:
                    consultation.expert_id = expert.id
                    consultation.status = 'assigned'
            
            db.session.add(consultation)
            db.session.commit()
            
            flash('Your consultation request has been submitted!', 'success')
            return redirect(url_for('my_consultations'))
        
        # Get all approved experts for the select list
        experts = User.query.filter_by(role='expert', is_approved_expert=True).all()
        return render_template('request_consultation.html', analysis=analysis, experts=experts)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/my-consultations')
@login_required
def my_consultations():
    try:
        # Get consultations requested by the user
        requested = Consultation.query.filter_by(requester_id=current_user.id).order_by(Consultation.created_at.desc()).all()
        
        # If user is an expert, also show consultations assigned to them
        provided = []
        if current_user.is_expert():
            provided = Consultation.query.filter_by(expert_id=current_user.id).order_by(Consultation.created_at.desc()).all()
        
        return render_template('my_consultations.html', requested=requested, provided=provided)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/consultation/<int:consultation_id>', methods=['GET', 'POST'])
@login_required
def view_consultation(consultation_id):
    try:
        consultation = Consultation.query.get_or_404(consultation_id)
        
        # Security check - only the requester, assigned expert, or admin can view
        if (consultation.requester_id != current_user.id and 
            consultation.expert_id != current_user.id and 
            not (consultation.public and consultation.status == 'completed') and
            current_user.role != 'admin'):
            return render_template('error.html', error="Unauthorized access")
        
        # If the current user is the assigned expert and it's a POST request (responding)
        if request.method == 'POST' and consultation.expert_id == current_user.id:
            response = request.form.get('response', '')
            make_public = 'make_public' in request.form
            
            if response:
                consultation.response = response
                consultation.status = 'completed'
                consultation.public = make_public
                consultation.updated_at = datetime.utcnow()
                db.session.commit()
                flash('Your response has been submitted!', 'success')
                return redirect(url_for('my_consultations'))
        
        # Get related analysis if exists
        analysis = None
        if consultation.analysis_id:
            analysis = Analysis.query.get(consultation.analysis_id)
        
        return render_template('view_consultation.html', 
                             consultation=consultation, 
                             analysis=analysis)
    except Exception as e:
        return render_template('error.html', error=str(e))

# Admin routes
@app.route('/admin/expert-applications')
@login_required
def admin_expert_applications():
    # Check if user is admin
    if current_user.role != 'admin':
        return render_template('error.html', error="Unauthorized access")
    
    experts_pending = User.query.filter_by(role='expert', is_approved_expert=False).all()
    return render_template('admin_expert_applications.html', experts_pending=experts_pending)

@app.route('/admin/approve-expert/<int:user_id>', methods=['POST'])
@login_required
def admin_approve_expert(user_id):
    # Check if user is admin
    if current_user.role != 'admin':
        return render_template('error.html', error="Unauthorized access")
    
    expert = User.query.get_or_404(user_id)
    if expert.role == 'expert':
        expert.is_approved_expert = True
        db.session.commit()
        flash(f'Expert {expert.username} has been approved!', 'success')
    
    return redirect(url_for('admin_expert_applications'))

if __name__ == '__main__':
    app.run(debug=True, port=8083)


