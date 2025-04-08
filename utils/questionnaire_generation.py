"""
Utility functions for generating questionnaires.
"""

def generate_questionnaire(research_description):
    """
    Generate a questionnaire structure based on a research description.
    
    Args:
        research_description (str): Description of the research
        
    Returns:
        list: A list of questionnaire sections with questions
    """
    # Default questionnaire structure with common sections
    questionnaire = []
    
    # Add demographics section
    demographics_section = {
        'title': 'Demographics',
        'description': 'Please provide the following demographic information to help us analyze the results.',
        'questions': [
            {
                'text': 'What is your age range?',
                'type': 'Multiple Choice',
                'options': ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65 or older']
            },
            {
                'text': 'What is your gender?',
                'type': 'Multiple Choice',
                'options': ['Male', 'Female', 'Non-binary', 'Prefer not to say', 'Other']
            },
            {
                'text': 'What is your highest level of education?',
                'type': 'Multiple Choice',
                'options': ['High School', 'Some College', 'Associate Degree', 'Bachelor\'s Degree', 'Master\'s Degree', 'Doctoral Degree', 'Professional Degree']
            }
        ]
    }
    questionnaire.append(demographics_section)
    
    # Generate additional sections based on research description
    keywords = research_description.lower()
    
    # Experience section (if relevant)
    if any(word in keywords for word in ['experience', 'background', 'history', 'skill', 'expertise', 'professional']):
        experience_section = {
            'title': 'Experience & Background',
            'description': 'Please tell us about your relevant experience and background.',
            'questions': [
                {
                    'text': 'How many years of experience do you have in this field?',
                    'type': 'Multiple Choice',
                    'options': ['Less than 1 year', '1-3 years', '4-6 years', '7-10 years', 'More than 10 years']
                },
                {
                    'text': 'How would you rate your expertise level in this area?',
                    'type': 'Rating',
                    'options': []
                },
                {
                    'text': 'Please describe your relevant experience in this field.',
                    'type': 'Open-Ended',
                    'options': []
                }
            ]
        }
        questionnaire.append(experience_section)
    
    # Preferences section (if relevant)
    if any(word in keywords for word in ['preference', 'like', 'dislike', 'favorite', 'opinion', 'attitude', 'choice']):
        preferences_section = {
            'title': 'Preferences & Opinions',
            'description': 'Please share your preferences and opinions on the following items.',
            'questions': [
                {
                    'text': 'What factors influence your preferences in this area the most?',
                    'type': 'Checkbox',
                    'options': ['Cost', 'Quality', 'Convenience', 'Brand reputation', 'Recommendations', 'Previous experience', 'Other']
                },
                {
                    'text': 'How strongly do you agree with the statement: "I am well-informed about this topic"?',
                    'type': 'Likert Scale',
                    'options': []
                },
                {
                    'text': 'Please describe what you like most about this topic or area.',
                    'type': 'Open-Ended',
                    'options': []
                }
            ]
        }
        questionnaire.append(preferences_section)
    
    # Behaviors section (if relevant)
    if any(word in keywords for word in ['behavior', 'habit', 'use', 'practice', 'routine', 'activity', 'frequency']):
        behaviors_section = {
            'title': 'Behaviors & Practices',
            'description': 'Please tell us about your behaviors and practices related to this topic.',
            'questions': [
                {
                    'text': 'How frequently do you engage with this topic/activity?',
                    'type': 'Multiple Choice',
                    'options': ['Daily', 'Several times a week', 'Once a week', 'A few times a month', 'Once a month', 'Less than once a month', 'Never']
                },
                {
                    'text': 'Which of the following activities do you regularly engage in? (Select all that apply)',
                    'type': 'Checkbox',
                    'options': ['Research on the topic', 'Discussing with others', 'Reading related content', 'Watching related videos', 'Taking classes/courses', 'Practice/implementation', 'Other']
                },
                {
                    'text': 'Describe your typical process or routine related to this topic.',
                    'type': 'Open-Ended',
                    'options': []
                }
            ]
        }
        questionnaire.append(behaviors_section)
    
    # Feedback section (always include)
    feedback_section = {
        'title': 'Feedback & Suggestions',
        'description': 'Please provide your feedback and suggestions.',
        'questions': [
            {
                'text': 'How satisfied are you with your current understanding/experience in this area?',
                'type': 'Rating',
                'options': []
            },
            {
                'text': 'What challenges or difficulties have you encountered in this area?',
                'type': 'Open-Ended',
                'options': []
            },
            {
                'text': 'Do you have any suggestions for improvements or changes in this area?',
                'type': 'Open-Ended',
                'options': []
            }
        ]
    }
    questionnaire.append(feedback_section)
    
    return questionnaire 