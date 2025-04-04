from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Collect form data
    data = {
        "dependent_variable": request.form['dependent_variable'],
        "independent_variables": request.form.getlist('independent_variables'),
        "size": int(request.form['size']),
        "missing_data": request.form['missing_data'],
        "research_question": request.form['research_question'],
        "analysis_goal": request.form['analysis_goal']
    }

    # Model recommendation logic
    recommended_model = ""
    explanation = ""
    competing_models = []

    if data["analysis_goal"] == "predict":
        if data["dependent_variable"] == "numerical":
            if "categorical" in data["independent_variables"]:
                recommended_model = "Multiple Linear Regression (with Categorical Variables Encoded)"
                explanation = "Multiple Linear Regression is used when predicting a continuous dependent variable with both numerical and categorical independent variables. You may need to encode categorical variables using dummy variables or one-hot encoding."
                competing_models = [
                    ("Ridge Regression", "Ridge Regression can also handle multicollinearity and overfitting by adding a penalty term."),
                    ("Lasso Regression", "Lasso Regression adds L1 regularization, which can also perform feature selection.")
                ]
            else:
                recommended_model = "Linear Regression"
                explanation = "Linear Regression is effective when predicting continuous outcomes using numerical independent variables."
                competing_models = [
                    ("Polynomial Regression", "Polynomial Regression can model more complex relationships between the independent and dependent variables."),
                    ("Ridge Regression", "Ridge Regression helps when there's multicollinearity in the data.")
                ]
        else:
            recommended_model = "Logistic Regression"
            explanation = "Logistic Regression is used when the dependent variable is categorical and the independent variables are numerical or mixed."
            competing_models = [
                ("Random Forest", "Random Forest is a powerful classification model that handles non-linear relationships and interactions well."),
                ("Support Vector Machine", "Support Vector Machine (SVM) works well for high-dimensional spaces and is robust to overfitting.")
            ]
    elif data["analysis_goal"] == "classify":
        recommended_model = "Random Forest or Support Vector Machine (SVM)"
        explanation = "For classification tasks, models like Random Forest or SVM perform well with numerical and categorical variables."
        competing_models = [
            ("Logistic Regression", "Logistic Regression is a simpler classification model that can work well with linearly separable data."),
            ("K-Nearest Neighbors", "KNN is a non-parametric method that works well for classification when you have fewer features.")
        ]
    elif data["analysis_goal"] == "explore":
        recommended_model = "Principal Component Analysis (PCA) or Clustering (K-Means)"
        explanation = "For exploratory analysis, PCA is useful for dimensionality reduction, while K-Means helps identify clusters in the data."
        competing_models = [
            ("t-SNE", "t-SNE is another dimensionality reduction technique, especially for visualizing high-dimensional data."),
            ("DBSCAN", "DBSCAN can be used for clustering, and it handles noise well.")
        ]
    else:
        recommended_model = "Generalized Linear Models"
        explanation = "Consider a more generalized approach based on your data structure."
        competing_models = [
            ("Naive Bayes", "Naive Bayes is often used for classification tasks with categorical data, particularly when the features are conditionally independent."),
            ("Decision Trees", "Decision Trees are simple, interpretable models that can be used for both classification and regression.")
        ]

    # Pass the results to the results page
    return redirect(url_for('results', recommended_model=recommended_model, explanation=explanation, competing_models=competing_models))

@app.route('/results')
def results():
    # Extract the data passed via the redirect
    recommended_model = request.args.get('recommended_model')
    explanation = request.args.get('explanation')
    competing_models = request.args.getlist('competing_models')

    # If no data was passed, redirect to the home page
    if not recommended_model:
        return redirect(url_for('home'))

    return render_template('results.html', recommended_model=recommended_model, explanation=explanation, competing_models=competing_models)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


