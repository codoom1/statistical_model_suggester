"""
Questionnaire Generator Module

This module analyzes research descriptions and generates appropriate
questionnaire sections and questions based on the content.
"""

import re
from collections import defaultdict

# Research domains and their associated intent keywords
RESEARCH_DOMAINS = {
    'education': ['learning', 'teaching', 'education', 'student', 'school', 'academic', 'course', 'training', 'classroom', 'curriculum'],
    'health': ['health', 'medical', 'wellness', 'patient', 'disease', 'treatment', 'doctor', 'hospital', 'symptom', 'diagnosis', 'therapy'],
    'business': ['business', 'market', 'customer', 'product', 'service', 'company', 'employee', 'management', 'strategy', 'sales', 'revenue'],
    'technology': ['technology', 'software', 'application', 'device', 'user', 'interface', 'system', 'digital', 'tech', 'computer', 'program'],
    'social': ['social', 'community', 'relationship', 'communication', 'network', 'interaction', 'people', 'group', 'society', 'culture'],
    'psychology': ['psychology', 'behavior', 'cognitive', 'emotion', 'mental', 'perception', 'attitude', 'personality', 'motivation', 'stress'],
    'environment': ['environment', 'sustainability', 'climate', 'green', 'ecology', 'conservation', 'pollution', 'recycling', 'waste', 'energy']
}

# Research intent categories
RESEARCH_INTENTS = {
    'exploratory': ['explore', 'discover', 'understand', 'identify', 'investigate', 'examine', 'assess', 'learn about', 'insight', 'initial'],
    'descriptive': ['describe', 'document', 'profile', 'detail', 'characterize', 'outline', 'illustrate', 'portray'],
    'explanatory': ['explain', 'cause', 'effect', 'reason', 'why', 'how', 'influence', 'impact', 'correlation', 'relationship'],
    'evaluative': ['evaluate', 'assess', 'test', 'compare', 'measure', 'rate', 'review', 'analyze', 'effectiveness', 'quality', 'performance'],
    'predictive': ['predict', 'forecast', 'future', 'anticipate', 'project', 'estimate', 'likelihood', 'probable', 'potential'],
    'prescriptive': ['improve', 'enhance', 'optimize', 'solve', 'recommendation', 'strategy', 'solution', 'guideline', 'best practice']
}

# Target audience segments
TARGET_AUDIENCES = {
    'general': ['general public', 'everyone', 'all', 'people', 'individuals', 'adults'],
    'professionals': ['professional', 'worker', 'employee', 'expert', 'practitioner', 'specialist', 'staff'],
    'students': ['student', 'learner', 'pupil', 'scholar', 'undergraduate', 'graduate', 'academic'],
    'consumers': ['consumer', 'customer', 'buyer', 'shopper', 'user', 'client', 'purchaser'],
    'patients': ['patient', 'healthcare', 'medical', 'clinic', 'hospital', 'treatment', 'care'],
    'parents': ['parent', 'caregiver', 'guardian', 'family', 'mother', 'father', 'child']
}

# Keywords for categorizing questions (expanded from original)
DEMOGRAPHICS_KEYWORDS = [
    'demographics', 'age', 'gender', 'education', 'income', 'occupation', 
    'location', 'marital', 'ethnicity', 'race', 'background', 'language',
    'nationality', 'residence', 'household', 'family', 'children'
]

EXPERIENCE_KEYWORDS = [
    'experience', 'history', 'usage', 'frequency', 'habits', 'past', 
    'previous', 'training', 'expertise', 'skill', 'knowledge', 'familiarity',
    'proficiency', 'competence', 'qualification', 'capability', 'background'
]

PREFERENCE_KEYWORDS = [
    'preference', 'like', 'dislike', 'favorite', 'opinion', 'view', 
    'perception', 'attitude', 'satisfaction', 'interest', 'importance', 'value',
    'priority', 'choice', 'selection', 'taste', 'desire', 'wish', 'want'
]

BEHAVIOR_KEYWORDS = [
    'behavior', 'behaviour', 'habits', 'routine', 'activity', 'practice', 
    'use', 'consumption', 'purchase', 'adoption', 'interaction', 'engagement',
    'pattern', 'trend', 'frequency', 'regularity', 'method', 'approach', 'style'
]

FEEDBACK_KEYWORDS = [
    'feedback', 'suggestion', 'recommendation', 'improvement', 'issue', 
    'challenge', 'problem', 'difficulty', 'concern', 'barrier', 'obstacle', 'comment',
    'review', 'opinion', 'evaluation', 'assessment', 'critique', 'insight', 'idea'
]

MOTIVATION_KEYWORDS = [
    'motivation', 'reason', 'driver', 'goal', 'aim', 'purpose', 'incentive', 'desire',
    'aspiration', 'ambition', 'intention', 'objective', 'target', 'why', 'what for'
]

SATISFACTION_KEYWORDS = [
    'satisfaction', 'happy', 'unhappy', 'content', 'discontent', 'pleased', 'displeased',
    'fulfilled', 'unfulfilled', 'enjoy', 'like', 'dislike', 'appreciate', 'value'
]

PAIN_POINT_KEYWORDS = [
    'pain point', 'frustration', 'annoyance', 'problem', 'issue', 'challenge', 'difficulty',
    'concern', 'dissatisfaction', 'disappointment', 'complaint', 'struggle', 'obstacle'
]

# Question type templates
MULTIPLE_CHOICE_TEMPLATE = {
    'type': 'Multiple Choice',
    'options': []
}

CHECKBOX_TEMPLATE = {
    'type': 'Checkbox',
    'options': []
}

LIKERT_TEMPLATE = {
    'type': 'Likert Scale'
}

OPEN_ENDED_TEMPLATE = {
    'type': 'Open-Ended'
}

RATING_TEMPLATE = {
    'type': 'Rating'
}

# Common question templates for each category
DEMOGRAPHICS_QUESTIONS = [
    {
        'text': 'What is your age group?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['18-24', '25-34', '35-44', '45-54', '55-64', '65 or older']
    },
    {
        'text': 'What is your gender?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Male', 'Female', 'Non-binary', 'Prefer not to say', 'Other']
    },
    {
        'text': 'What is your highest level of education?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['High School or less', 'Some College', 'Associate Degree', 'Bachelor\'s Degree', 'Master\'s Degree', 'Doctoral or Professional Degree']
    },
    {
        'text': 'What is your current employment status?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Employed full-time', 'Employed part-time', 'Self-employed', 'Student', 'Retired', 'Unemployed']
    }
]

EXPERIENCE_QUESTIONS = [
    {
        'text': 'How would you rate your level of experience with [TOPIC]?',
        **RATING_TEMPLATE
    },
    {
        'text': 'How long have you been using [TOPIC]?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Less than 6 months', '6-12 months', '1-3 years', '3-5 years', 'More than 5 years']
    },
    {
        'text': 'How frequently do you engage with [TOPIC]?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Daily', 'Several times a week', 'Once a week', 'A few times a month', 'Monthly or less frequently']
    },
    {
        'text': 'What specific aspects of [TOPIC] are you most familiar with?',
        **OPEN_ENDED_TEMPLATE
    }
]

PREFERENCE_QUESTIONS = [
    {
        'text': 'How satisfied are you with [TOPIC]?',
        **LIKERT_TEMPLATE
    },
    {
        'text': 'What aspects of [TOPIC] do you find most valuable?',
        **CHECKBOX_TEMPLATE,
        'options': ['Ease of use', 'Quality', 'Cost', 'Availability', 'Features', 'Other']
    },
    {
        'text': 'What is your preferred method of [TOPIC]?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Option A', 'Option B', 'Option C', 'Other']
    },
    {
        'text': 'How important is [TOPIC] to you?',
        **LIKERT_TEMPLATE
    }
]

BEHAVIOR_QUESTIONS = [
    {
        'text': 'Which of the following best describes your typical usage of [TOPIC]?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Option A', 'Option B', 'Option C', 'Other']
    },
    {
        'text': 'What factors influence your decision to use [TOPIC]?',
        **CHECKBOX_TEMPLATE,
        'options': ['Convenience', 'Necessity', 'Cost', 'Quality', 'Recommendations', 'Other']
    },
    {
        'text': 'How has your usage of [TOPIC] changed over time?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'In what context do you typically engage with [TOPIC]?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Home', 'Work', 'School', 'Social settings', 'Other']
    }
]

FEEDBACK_QUESTIONS = [
    {
        'text': 'What improvements would you suggest for [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'What challenges or difficulties have you experienced with [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'How likely are you to recommend [TOPIC] to others?',
        **RATING_TEMPLATE
    },
    {
        'text': 'What additional features or aspects would you like to see in [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    }
]

MOTIVATION_QUESTIONS = [
    {
        'text': 'What motivates you to use [TOPIC]?',
        **CHECKBOX_TEMPLATE,
        'options': ['Personal interest', 'Professional requirements', 'Recommendation from others', 'Necessity', 'Curiosity', 'Other']
    },
    {
        'text': 'What goals are you trying to achieve with [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'What would encourage you to engage more with [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    }
]

SATISFACTION_QUESTIONS = [
    {
        'text': 'How satisfied are you with your experience of [TOPIC]?',
        **RATING_TEMPLATE
    },
    {
        'text': 'Which aspects of [TOPIC] give you the most satisfaction?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'What would make your experience with [TOPIC] more satisfying?',
        **OPEN_ENDED_TEMPLATE
    }
]

PAIN_POINT_QUESTIONS = [
    {
        'text': 'What frustrations or challenges do you experience with [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'Which aspects of [TOPIC] do you find most difficult to deal with?',
        **CHECKBOX_TEMPLATE,
        'options': ['Complexity', 'Time required', 'Cost', 'Technical issues', 'Lack of support', 'Other']
    },
    {
        'text': 'How do you currently overcome challenges related to [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    }
]

# Domain-specific question templates
EDUCATION_QUESTIONS = [
    {
        'text': 'How effective do you find the current methods of teaching [TOPIC]?',
        **RATING_TEMPLATE
    },
    {
        'text': 'What learning resources for [TOPIC] do you find most helpful?',
        **CHECKBOX_TEMPLATE,
        'options': ['Textbooks', 'Online courses', 'Video tutorials', 'In-person classes', 'Discussion groups', 'Practical exercises', 'Other']
    },
    {
        'text': 'What challenges do you face when learning about [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    }
]

HEALTH_QUESTIONS = [
    {
        'text': 'How has [TOPIC] affected your overall health and wellbeing?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'What health-related concerns do you have regarding [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'How often do you discuss [TOPIC] with healthcare professionals?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Never', 'Rarely', 'Occasionally', 'Frequently', 'At every visit']
    }
]

BUSINESS_QUESTIONS = [
    {
        'text': 'How has [TOPIC] impacted your business operations?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'What business value do you get from [TOPIC]?',
        **CHECKBOX_TEMPLATE,
        'options': ['Cost savings', 'Increased revenue', 'Improved efficiency', 'Better customer satisfaction', 'Competitive advantage', 'Other']
    },
    {
        'text': 'What ROI have you observed from implementing [TOPIC]?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Negative ROI', 'Break-even', 'Moderate positive ROI', 'Significant positive ROI', 'Not measured']
    }
]

TECHNOLOGY_QUESTIONS = [
    {
        'text': 'How user-friendly do you find [TOPIC]?',
        **RATING_TEMPLATE
    },
    {
        'text': 'What technical challenges have you encountered with [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'How well does [TOPIC] integrate with your existing technologies?',
        **RATING_TEMPLATE
    }
]

DOMAIN_QUESTIONS = {
    'education': EDUCATION_QUESTIONS,
    'health': HEALTH_QUESTIONS,
    'business': BUSINESS_QUESTIONS,
    'technology': TECHNOLOGY_QUESTIONS
}

# Intent-specific question templates
EXPLORATORY_QUESTIONS = [
    {
        'text': 'What aspects of [TOPIC] are you most curious about?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'What would you like to learn more about regarding [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    }
]

EVALUATIVE_QUESTIONS = [
    {
        'text': 'How would you rate the overall quality of [TOPIC]?',
        **RATING_TEMPLATE
    },
    {
        'text': 'What criteria do you use to evaluate [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    }
]

PRESCRIPTIVE_QUESTIONS = [
    {
        'text': 'What specific improvements would make [TOPIC] more effective?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'What recommendations would you give to someone new to [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    }
]

INTENT_QUESTIONS = {
    'exploratory': EXPLORATORY_QUESTIONS,
    'evaluative': EVALUATIVE_QUESTIONS,
    'prescriptive': PRESCRIPTIVE_QUESTIONS
}

def analyze_research_description(description, topic):
    """
    Analyze the research description to identify relevant categories
    and generate appropriate questions.
    
    Args:
        description (str): The research description provided by the user
        topic (str): The main research topic
        
    Returns:
        list: Structured sections and questions for the questionnaire
    """
    # Convert to lowercase for better matching
    description_lower = description.lower()
    topic_lower = topic.lower()
    
    # Count keyword matches for each category
    category_scores = defaultdict(int)
    
    # Check for demographics keywords
    for keyword in DEMOGRAPHICS_KEYWORDS:
        if keyword in description_lower:
            category_scores['demographics'] += 1
    
    # Check for experience keywords
    for keyword in EXPERIENCE_KEYWORDS:
        if keyword in description_lower:
            category_scores['experience'] += 1
    
    # Check for preference keywords
    for keyword in PREFERENCE_KEYWORDS:
        if keyword in description_lower:
            category_scores['preferences'] += 1
    
    # Check for behavior keywords
    for keyword in BEHAVIOR_KEYWORDS:
        if keyword in description_lower:
            category_scores['behaviors'] += 1
    
    # Check for feedback keywords
    for keyword in FEEDBACK_KEYWORDS:
        if keyword in description_lower:
            category_scores['feedback'] += 1
    
    # Check for motivation keywords
    for keyword in MOTIVATION_KEYWORDS:
        if keyword in description_lower:
            category_scores['motivation'] += 1
    
    # Check for satisfaction keywords
    for keyword in SATISFACTION_KEYWORDS:
        if keyword in description_lower:
            category_scores['satisfaction'] += 1
    
    # Check for pain point keywords
    for keyword in PAIN_POINT_KEYWORDS:
        if keyword in description_lower:
            category_scores['pain_points'] += 1
    
    # Always include demographics unless explicitly not needed
    if 'demographics' not in category_scores and not any(word in description_lower for word in ['no demographics', 'without demographics', 'skip demographics']):
        category_scores['demographics'] = 1
    
    # Detect research domain
    domain_scores = {}
    for domain, keywords in RESEARCH_DOMAINS.items():
        score = sum(1 for keyword in keywords if keyword in description_lower)
        if score > 0:
            domain_scores[domain] = score
    
    # Detect research intent
    intent_scores = {}
    for intent, keywords in RESEARCH_INTENTS.items():
        score = sum(1 for keyword in keywords if keyword in description_lower)
        if score > 0:
            intent_scores[intent] = score
    
    # Get primary domain and intent (if any)
    primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else None
    primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else None
    
    # Generate questionnaire sections based on the analysis
    sections = []
    
    # Sort categories by their score (highest first)
    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Generate sections for each category with at least one keyword match
    for category, score in sorted_categories:
        if score > 0:
            section = generate_section_for_category(category, topic, description, primary_domain, primary_intent)
            sections.append(section)
    
    # If no categories matched, create a general section
    if not sections:
        sections.append({
            'title': 'General Questions',
            'description': f'Questions about {topic}',
            'questions': [
                {
                    'text': f'What is your overall experience with {topic}?',
                    **OPEN_ENDED_TEMPLATE
                },
                {
                    'text': f'How satisfied are you with {topic}?',
                    **LIKERT_TEMPLATE
                },
                {
                    'text': f'What aspects of {topic} are most important to you?',
                    **OPEN_ENDED_TEMPLATE
                },
                {
                    'text': f'How would you improve {topic}?',
                    **OPEN_ENDED_TEMPLATE
                }
            ]
        })
    
    return sections

def generate_section_for_category(category, topic, description, domain=None, intent=None):
    """
    Generate a questionnaire section for a specific category.
    
    Args:
        category (str): The category name ('demographics', 'experience', etc.)
        topic (str): The main research topic
        description (str): The research description
        domain (str, optional): The primary research domain
        intent (str, optional): The primary research intent
        
    Returns:
        dict: Section with title, description and questions
    """
    section = {
        'title': '',
        'description': '',
        'questions': []
    }
    
    if category == 'demographics':
        section['title'] = 'Demographics'
        section['description'] = 'Please provide some information about yourself.'
        section['questions'] = DEMOGRAPHICS_QUESTIONS.copy()
    
    elif category == 'experience':
        section['title'] = 'Experience & Background'
        section['description'] = f'Please share your experience with {topic}.'
        section['questions'] = customize_questions(EXPERIENCE_QUESTIONS.copy(), topic)
    
    elif category == 'preferences':
        section['title'] = 'Preferences & Opinions'
        section['description'] = f'Please share your preferences regarding {topic}.'
        section['questions'] = customize_questions(PREFERENCE_QUESTIONS.copy(), topic)
    
    elif category == 'behaviors':
        section['title'] = 'Behaviors & Practices'
        section['description'] = f'Please share information about your usage and behaviors related to {topic}.'
        section['questions'] = customize_questions(BEHAVIOR_QUESTIONS.copy(), topic)
    
    elif category == 'feedback':
        section['title'] = 'Feedback & Suggestions'
        section['description'] = f'Please provide feedback and suggestions for improving {topic}.'
        section['questions'] = customize_questions(FEEDBACK_QUESTIONS.copy(), topic)
    
    elif category == 'motivation':
        section['title'] = 'Motivation & Goals'
        section['description'] = f'Please share what motivates you regarding {topic}.'
        section['questions'] = customize_questions(MOTIVATION_QUESTIONS.copy(), topic)
    
    elif category == 'satisfaction':
        section['title'] = 'Satisfaction & Experience'
        section['description'] = f'Please share your level of satisfaction with {topic}.'
        section['questions'] = customize_questions(SATISFACTION_QUESTIONS.copy(), topic)
    
    elif category == 'pain_points':
        section['title'] = 'Challenges & Pain Points'
        section['description'] = f'Please share any challenges you face with {topic}.'
        section['questions'] = customize_questions(PAIN_POINT_QUESTIONS.copy(), topic)
    
    # Add domain-specific questions if applicable
    if domain and domain in DOMAIN_QUESTIONS:
        domain_specific = customize_questions(DOMAIN_QUESTIONS[domain].copy(), topic)
        section['questions'].extend(domain_specific)
    
    # Add intent-specific questions if applicable
    if intent and intent in INTENT_QUESTIONS:
        intent_specific = customize_questions(INTENT_QUESTIONS[intent].copy(), topic)
        section['questions'].extend(intent_specific)
    
    return section

def customize_questions(questions, topic):
    """
    Customize standard questions by replacing placeholders with the specific topic.
    
    Args:
        questions (list): List of question templates
        topic (str): The main research topic
        
    Returns:
        list: Customized questions
    """
    customized = []
    
    for question in questions:
        new_question = question.copy()
        if '[TOPIC]' in new_question['text']:
            new_question['text'] = new_question['text'].replace('[TOPIC]', topic)
        
        # For multiple choice/checkbox questions that have generic options (Option A, B, C)
        # we could customize these based on the topic in a more advanced implementation
        
        customized.append(new_question)
    
    return customized

def analyze_intent(description):
    """
    Analyze the research description to identify the primary intent,
    domain, and target audience.
    
    Args:
        description (str): The research description
        
    Returns:
        dict: Detected intent information
    """
    description_lower = description.lower()
    
    # Detect research domain
    domain_scores = {}
    for domain, keywords in RESEARCH_DOMAINS.items():
        score = sum(1 for keyword in keywords if keyword in description_lower)
        if score > 0:
            domain_scores[domain] = score
    
    # Detect research intent
    intent_scores = {}
    for intent, keywords in RESEARCH_INTENTS.items():
        score = sum(1 for keyword in keywords if keyword in description_lower)
        if score > 0:
            intent_scores[intent] = score
    
    # Detect target audience
    audience_scores = {}
    for audience, keywords in TARGET_AUDIENCES.items():
        score = sum(1 for keyword in keywords if keyword in description_lower)
        if score > 0:
            audience_scores[audience] = score
    
    # Get primary domain, intent, and audience (if any)
    primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else None
    primary_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else None
    primary_audience = max(audience_scores.items(), key=lambda x: x[1])[0] if audience_scores else None
    
    return {
        'domain': primary_domain,
        'intent': primary_intent,
        'audience': primary_audience,
        'domain_score': domain_scores.get(primary_domain, 0) if primary_domain else 0,
        'intent_score': intent_scores.get(primary_intent, 0) if primary_intent else 0,
        'audience_score': audience_scores.get(primary_audience, 0) if primary_audience else 0
    }

def generate_questionnaire(research_description, research_topic=None, target_audience=None, questionnaire_purpose=None):
    """
    Generate a questionnaire structure based on a research description.
    
    Args:
        research_description (str): Description of the research
        research_topic (str, optional): Title of the research
        target_audience (str, optional): Target audience for the questionnaire
        questionnaire_purpose (str, optional): Purpose of the questionnaire
        
    Returns:
        list: A list of questionnaire sections with questions
    """
    if not research_topic:
        research_topic = "this topic"
    
    # Analyze the intent, domain, and target audience
    intent_analysis = analyze_intent(research_description)
    
    # Default questionnaire structure with common sections
    questionnaire = []
    
    # Add demographics section
    demographics_section = {
        'title': 'Demographics',
        'description': 'Please provide the following demographic information.',
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
    
    # Customize demographics questions based on domain and audience
    if intent_analysis['domain'] == 'education':
        demographics_section['questions'].append({
            'text': 'What is your current role in education?',
            'type': 'Multiple Choice',
            'options': ['Student', 'Teacher/Professor', 'Administrator', 'Parent', 'Researcher', 'Other']
        })
    elif intent_analysis['domain'] == 'health':
        demographics_section['questions'].append({
            'text': 'How would you describe your overall health?',
            'type': 'Multiple Choice',
            'options': ['Excellent', 'Very good', 'Good', 'Fair', 'Poor']
        })
    elif intent_analysis['domain'] == 'business':
        demographics_section['questions'].append({
            'text': 'What is the size of your organization?',
            'type': 'Multiple Choice',
            'options': ['Self-employed', '1-10 employees', '11-50 employees', '51-200 employees', '201-1000 employees', 'More than 1000 employees']
        })
    
    questionnaire.append(demographics_section)
    
    # Generate additional sections based on research description
    # (This is a simple implementation - would be enhanced with NLP in a production system)
    keywords = research_description.lower()
    
    # Experience section (if relevant)
    if any(word in keywords for word in ['experience', 'background', 'history', 'skill', 'expertise', 'professional']):
        experience_section = {
            'title': 'Experience & Background',
            'description': 'Please tell us about your experiences related to this topic.',
            'questions': [
                {
                    'text': 'How many years of experience do you have in this field?',
                    'type': 'Multiple Choice',
                    'options': ['None', 'Less than 1 year', '1-3 years', '4-6 years', '7-10 years', 'More than 10 years']
                },
                {
                    'text': 'How would you rate your expertise in this area?',
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
        
        # Add domain-specific experience questions
        if intent_analysis['domain'] == 'technology':
            experience_section['questions'].append({
                'text': f'Which technologies related to {research_topic} have you used?',
                'type': 'Checkbox',
                'options': ['Software applications', 'Mobile apps', 'Web platforms', 'Hardware devices', 'APIs or integrations', 'Other']
            })
        elif intent_analysis['domain'] == 'education':
            experience_section['questions'].append({
                'text': f'In what educational contexts have you encountered {research_topic}?',
                'type': 'Checkbox',
                'options': ['K-12 education', 'Higher education', 'Professional training', 'Self-directed learning', 'Online courses', 'Other']
            })
        
        questionnaire.append(experience_section)
    
    # Preferences section (if relevant)
    if any(word in keywords for word in ['preference', 'like', 'dislike', 'favorite', 'opinion', 'interest']):
        preferences_section = {
            'title': 'Preferences & Opinions',
            'description': 'Please share your preferences and opinions.',
            'questions': [
                {
                    'text': 'What aspects of this topic interest you the most? (Select all that apply)',
                    'type': 'Checkbox',
                    'options': ['Learning new skills', 'Social connections', 'Career advancement', 'Personal fulfillment', 'Financial benefits', 'Other']
                },
                {
                    'text': 'How strongly do you agree with the statement: "I am passionate about this topic"?',
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
        
        # Add intent-specific preference questions
        if intent_analysis['intent'] == 'evaluative':
            preferences_section['questions'].append({
                'text': f'What criteria do you use to evaluate {research_topic}?',
                'type': 'Open-Ended',
                'options': []
            })
        elif intent_analysis['intent'] == 'prescriptive':
            preferences_section['questions'].append({
                'text': f'What improvements would make {research_topic} more aligned with your preferences?',
                'type': 'Open-Ended',
                'options': []
            })
        
        questionnaire.append(preferences_section)
    
    # Usage/Behavior section (if relevant)
    if any(word in keywords for word in ['use', 'utilize', 'behavior', 'habit', 'frequency', 'practice', 'consume']):
        usage_section = {
            'title': 'Usage & Behaviors',
            'description': 'Please tell us about how you engage with this topic.',
            'questions': [
                {
                    'text': 'How frequently do you engage with this topic?',
                    'type': 'Multiple Choice',
                    'options': ['Daily', 'Several times a week', 'Once a week', 'A few times a month', 'Once a month', 'Less than once a month', 'Never']
                },
                {
                    'text': 'When do you typically engage with this topic? (Select all that apply)',
                    'type': 'Checkbox',
                    'options': ['Mornings', 'Afternoons', 'Evenings', 'Weekdays', 'Weekends', 'No specific time']
                },
                {
                    'text': 'Describe your typical approach or process related to this topic.',
                    'type': 'Open-Ended',
                    'options': []
                }
            ]
        }
        questionnaire.append(usage_section)
    
    # Feedback section (always include)
    feedback_section = {
        'title': 'Feedback & Suggestions',
        'description': 'Please provide your feedback and suggestions.',
        'questions': [
            {
                'text': 'How satisfied are you with your current experience in this area?',
                'type': 'Rating',
                'options': []
            },
            {
                'text': 'What challenges or difficulties have you encountered?',
                'type': 'Open-Ended',
                'options': []
            },
            {
                'text': 'Do you have any suggestions for improvements?',
                'type': 'Open-Ended',
                'options': []
            }
        ]
    }
    
    # Add domain-specific feedback questions
    if intent_analysis['domain']:
        if intent_analysis['domain'] == 'technology':
            feedback_section['questions'].append({
                'text': f'What technical improvements would make {research_topic} more effective?',
                'type': 'Open-Ended',
                'options': []
            })
        elif intent_analysis['domain'] == 'health':
            feedback_section['questions'].append({
                'text': f'How has {research_topic} impacted your health and wellbeing?',
                'type': 'Open-Ended',
                'options': []
            })
        elif intent_analysis['domain'] == 'education':
            feedback_section['questions'].append({
                'text': f'How could {research_topic} be better integrated into educational settings?',
                'type': 'Open-Ended',
                'options': []
            })
        elif intent_analysis['domain'] == 'business':
            feedback_section['questions'].append({
                'text': f'What business value do you get from {research_topic}?',
                'type': 'Checkbox',
                'options': ['Cost savings', 'Increased revenue', 'Improved efficiency', 'Better customer satisfaction', 'Competitive advantage', 'Other']
            })
    
    questionnaire.append(feedback_section)
    
    return questionnaire 