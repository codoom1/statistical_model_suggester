"""
Questionnaire Designer Service Routes

This module provides routes for the questionnaire design service,
allowing users to create, preview, and edit questionnaires.
"""

from flask import Blueprint, render_template, request, redirect, url_for, session, flash, Response, send_file
from utils.questionnaire_generator import generate_questionnaire
from utils.export_utils import export_to_word, export_to_pdf
import json
import os
from datetime import datetime

questionnaire_bp = Blueprint('questionnaire', __name__, url_prefix='/questionnaire')

@questionnaire_bp.route('/')
def index():
    """Landing page for the questionnaire design service."""
    return render_template('questionnaire/index.html')

@questionnaire_bp.route('/design', methods=['GET', 'POST'])
def design():
    """
    GET: Show form to enter research description
    POST: Process form input and generate questionnaire
    """
    if request.method == 'POST':
        research_topic = request.form.get('research_topic', '')
        research_description = request.form.get('research_description', '')
        target_audience = request.form.get('target_audience', '')
        questionnaire_purpose = request.form.get('questionnaire_purpose', '')
        
        # Generate questionnaire based on research description
        questionnaire = generate_questionnaire(
            research_description,
            research_topic,
            target_audience,
            questionnaire_purpose
        )
        
        # Store questionnaire data in session
        session['questionnaire'] = questionnaire
        session['research_topic'] = research_topic
        session['research_description'] = research_description
        session['target_audience'] = target_audience
        session['questionnaire_purpose'] = questionnaire_purpose
        
        return redirect(url_for('questionnaire.preview'))
    
    return render_template('questionnaire/design.html')

@questionnaire_bp.route('/preview')
def preview():
    """Preview the generated questionnaire."""
    # Check if questionnaire data exists in session
    if 'questionnaire' not in session:
        flash('Please design a questionnaire first.', 'error')
        return redirect(url_for('questionnaire.design'))
    
    return render_template(
        'questionnaire/preview.html',
        questionnaire=session['questionnaire'],
        research_topic=session.get('research_topic', ''),
        research_description=session.get('research_description', ''),
        target_audience=session.get('target_audience', ''),
        questionnaire_purpose=session.get('questionnaire_purpose', '')
    )

@questionnaire_bp.route('/edit', methods=['GET', 'POST'])
def edit():
    """
    GET: Show form to edit questionnaire
    POST: Process edits and update the questionnaire
    """
    # Check if questionnaire data exists in session
    if 'questionnaire' not in session:
        flash('Please design a questionnaire first.', 'error')
        return redirect(url_for('questionnaire.design'))
    
    if request.method == 'POST':
        # Process the form data
        research_topic = request.form.get('research_topic', '')
        target_audience = request.form.get('target_audience', '')
        questionnaire_purpose = request.form.get('questionnaire_purpose', '')
        research_description = request.form.get('research_description', '')
        
        # Process sections data from the form
        sections_data = []
        form_data = request.form.to_dict(flat=False)
        
        # Get all unique section indices from form data
        section_indices = set()
        for key in form_data:
            if key.startswith('sections[') and '][title]' in key:
                section_index = key.split('[')[1].split(']')[0]
                section_indices.add(section_index)
        
        # Sort section indices to maintain order
        section_indices = sorted(section_indices, key=int)
        
        # Process each section
        for section_index in section_indices:
            section_title = request.form.get(f'sections[{section_index}][title]', '')
            section_description = request.form.get(f'sections[{section_index}][description]', '')
            
            # Get questions for this section
            questions = []
            question_indices = set()
            
            # Find all question indices for this section
            for key in form_data:
                if key.startswith(f'sections[{section_index}][questions][') and '][text]' in key:
                    question_index = key.split('[')[3].split(']')[0]
                    question_indices.add(question_index)
            
            # Sort question indices to maintain order
            question_indices = sorted(question_indices, key=int)
            
            # Process each question
            for question_index in question_indices:
                question_text = request.form.get(f'sections[{section_index}][questions][{question_index}][text]', '')
                question_type = request.form.get(f'sections[{section_index}][questions][{question_index}][type]', '')
                
                # Get options if applicable
                options = []
                if question_type in ['Multiple Choice', 'Checkbox']:
                    option_indices = set()
                    
                    # Find all option indices for this question
                    for key in form_data:
                        if key.startswith(f'sections[{section_index}][questions][{question_index}][options]['):
                            option_index = key.split('[')[5].split(']')[0]
                            option_indices.add(option_index)
                    
                    # Sort option indices to maintain order
                    option_indices = sorted(option_indices, key=int)
                    
                    # Process each option
                    for option_index in option_indices:
                        option_text = request.form.get(f'sections[{section_index}][questions][{question_index}][options][{option_index}]', '')
                        if option_text:
                            options.append(option_text)
                
                question_data = {
                    'text': question_text,
                    'type': question_type
                }
                
                if options:
                    question_data['options'] = options
                
                if question_text:  # Only add non-empty questions
                    questions.append(question_data)
            
            if section_title:  # Only add sections with a title
                sections_data.append({
                    'title': section_title,
                    'description': section_description,
                    'questions': questions
                })
        
        # Update session with edited data
        session['questionnaire'] = sections_data
        session['research_topic'] = research_topic
        session['target_audience'] = target_audience
        session['questionnaire_purpose'] = questionnaire_purpose
        session['research_description'] = research_description
        
        flash('Questionnaire updated successfully.', 'success')
        return redirect(url_for('questionnaire.preview'))
    
    return render_template(
        'questionnaire/edit.html',
        questionnaire=session['questionnaire'],
        research_topic=session.get('research_topic', ''),
        research_description=session.get('research_description', ''),
        target_audience=session.get('target_audience', ''),
        questionnaire_purpose=session.get('questionnaire_purpose', '')
    )

@questionnaire_bp.route('/export/word')
def export_word():
    """Export questionnaire to Word document."""
    # Check if questionnaire data exists in session
    if 'questionnaire' not in session:
        flash('Please design a questionnaire first.', 'error')
        return redirect(url_for('questionnaire.design'))
    
    # Get questionnaire data from session
    questionnaire = session['questionnaire']
    research_topic = session.get('research_topic', 'Questionnaire')
    research_description = session.get('research_description', '')
    target_audience = session.get('target_audience', '')
    questionnaire_purpose = session.get('questionnaire_purpose', '')
    
    # Generate a filename
    filename = f"{research_topic.replace(' ', '_')}_questionnaire.docx"
    
    # Create the Word document
    file_path = export_to_word(
        questionnaire, 
        research_topic, 
        research_description,
        target_audience,
        questionnaire_purpose
    )
    
    # Send the file to the user
    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )

@questionnaire_bp.route('/export/pdf')
def export_pdf():
    """Export questionnaire to PDF document."""
    # Check if questionnaire data exists in session
    if 'questionnaire' not in session:
        flash('Please design a questionnaire first.', 'error')
        return redirect(url_for('questionnaire.design'))
    
    # Get questionnaire data from session
    questionnaire = session['questionnaire']
    research_topic = session.get('research_topic', 'Questionnaire')
    research_description = session.get('research_description', '')
    target_audience = session.get('target_audience', '')
    questionnaire_purpose = session.get('questionnaire_purpose', '')
    
    # Generate a filename
    filename = f"{research_topic.replace(' ', '_')}_questionnaire.pdf"
    
    # Create the PDF document
    file_path = export_to_pdf(
        questionnaire, 
        research_topic, 
        research_description,
        target_audience,
        questionnaire_purpose
    )
    
    # Send the file to the user
    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    ) 