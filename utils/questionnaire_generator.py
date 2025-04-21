"""
Questionnaire Generator Module

This module analyzes research descriptions and generates appropriate
questionnaire sections and questions based on the content.
"""

import re
from collections import defaultdict
import os
import logging
import requests
import json

# Import the new AI service and error class
from utils.ai_service import call_huggingface_api, is_ai_enabled, HuggingFaceError, get_huggingface_config

# Configure logging
logger = logging.getLogger(__name__)

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
    },
    {
        'text': 'What is your household income range?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Less than $25,000', '$25,000-$49,999', '$50,000-$74,999', '$75,000-$99,999', '$100,000-$149,999', '$150,000 or more', 'Prefer not to say']
    },
    {
        'text': 'What is your marital status?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Single', 'Married', 'Domestic partnership', 'Divorced', 'Widowed', 'Separated', 'Prefer not to say']
    },
    {
        'text': 'In what type of area do you live?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Urban', 'Suburban', 'Rural', 'Small town']
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
    },
    {
        'text': 'How did you initially learn about [TOPIC]?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Formal education', 'Self-taught', 'Professional training', 'From colleagues', 'Online resources', 'Other']
    },
    {
        'text': 'In what contexts do you typically use or encounter [TOPIC]?',
        **CHECKBOX_TEMPLATE,
        'options': ['Work', 'School', 'Personal projects', 'Hobbies', 'Daily life', 'Other']
    },
    {
        'text': 'What resources have been most helpful in developing your knowledge of [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'How has your approach to [TOPIC] evolved over time?',
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
    },
    {
        'text': 'What alternatives to [TOPIC] have you tried, and how do they compare?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'Which specific features of [TOPIC] do you find most useful?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'If you could change one thing about [TOPIC], what would it be?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'How likely are you to recommend [TOPIC] to others?',
        **RATING_TEMPLATE
    },
    {
        'text': 'What factors influence your preference for [TOPIC]?',
        **CHECKBOX_TEMPLATE,
        'options': ['Price', 'Quality', 'Convenience', 'Reliability', 'Brand reputation', 'Recommendations', 'Other']
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
    },
    {
        'text': 'What time of day do you most frequently use [TOPIC]?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Morning', 'Afternoon', 'Evening', 'Late night', 'Throughout the day']
    },
    {
        'text': 'What other activities do you typically combine with [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'How do you prepare before engaging with [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'What triggers your decision to use [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'How do you adjust your use of [TOPIC] based on different situations?',
        **OPEN_ENDED_TEMPLATE
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
    },
    {
        'text': 'What specific aspects of [TOPIC] need the most improvement?',
        **CHECKBOX_TEMPLATE,
        'options': ['Usability', 'Performance', 'Features', 'Cost', 'Support', 'Documentation', 'Other']
    },
    {
        'text': 'How well does [TOPIC] meet your expectations?',
        **LIKERT_TEMPLATE
    },
    {
        'text': 'What is your most significant frustration with [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'What do you appreciate most about [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'How responsive has the team been to your previous feedback about [TOPIC]?',
        **LIKERT_TEMPLATE
    },
    {
        'text': 'If you were in charge of [TOPIC], what would be your top three priorities for improvement?',
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
    },
    {
        'text': 'What initially sparked your interest in [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'What keeps you coming back to [TOPIC]?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'What benefits do you hope to gain from using [TOPIC]?',
        **CHECKBOX_TEMPLATE,
        'options': ['Increased knowledge', 'Improved skills', 'Better outcomes', 'Time savings', 'Cost savings', 'Enjoyment', 'Other']
    },
    {
        'text': 'How does [TOPIC] align with your personal or professional values?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'What specific outcome are you hoping to achieve through [TOPIC]?',
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
    },
    {
        'text': 'How has [TOPIC] met or exceeded your expectations?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'Which aspects of [TOPIC] do you find most disappointing?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'How does your satisfaction with [TOPIC] compare to similar alternatives?',
        **LIKERT_TEMPLATE
    },
    {
        'text': 'What specific experiences with [TOPIC] have influenced your overall satisfaction?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'How has your satisfaction with [TOPIC] changed over time?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['Significantly improved', 'Somewhat improved', 'Stayed the same', 'Somewhat decreased', 'Significantly decreased']
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
    },
    {
        'text': 'What specific tasks related to [TOPIC] cause you the most frustration?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'What barriers prevent you from getting the most out of [TOPIC]?',
        **CHECKBOX_TEMPLATE,
        'options': ['Knowledge gaps', 'Time constraints', 'Resource limitations', 'Technical limitations', 'Organizational policies', 'Other']
    },
    {
        'text': 'What problems does [TOPIC] solve for you, and what new problems does it create?',
        **OPEN_ENDED_TEMPLATE
    },
    {
        'text': 'How much time do you spend troubleshooting issues with [TOPIC]?',
        **MULTIPLE_CHOICE_TEMPLATE,
        'options': ['None', 'A few minutes occasionally', 'Several hours monthly', 'Several hours weekly', 'Daily']
    },
    {
        'text': 'What would eliminate your biggest pain point with [TOPIC]?',
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

def analyze_research_description(description, topic, use_ai=False, num_ai_questions=3):
    """
    Analyze the research description to identify relevant categories
    and generate appropriate questions.
    
    Args:
        description (str): The research description provided by the user
        topic (str): The main research topic
        use_ai (bool): Whether to use AI enhancement for questions
        num_ai_questions (int): Number of AI questions to generate per type/section
        
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
    
    # Get intent analysis
    intent_analysis = analyze_intent(description)
    primary_domain = intent_analysis.get('domain')
    primary_intent = intent_analysis.get('intent')
    
    # Sort categories by their score (highest first)
    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Generate sections for each category with at least one keyword match
    sections = []
    for category, score in sorted_categories:
        if score > 0:
            section = generate_section_for_category(
                category, 
                topic, 
                description, 
                primary_domain, 
                primary_intent,
                use_ai=use_ai,
                num_ai_questions=num_ai_questions
            )
            sections.append(section)
    
    # If no categories matched, create a general section
    if not sections:
        general_questions = [
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
        
        # If AI is enabled, generate an entirely new section
        if use_ai:
            # First enhance existing template questions
            enhanced_questions = enhance_questions_with_ai(
                general_questions,
                topic,
                description,
                primary_domain,
                primary_intent,
                use_ai=True
            )
            
            # Then add AI-generated questions for the general section
            ai_questions = generate_ai_questions(
                topic,
                description,
                primary_domain,
                primary_intent,
                category=None,  # General category
                num_questions=num_ai_questions # Use the parameter
            )
            
            # Combine enhanced template questions with AI-generated ones
            all_questions = enhanced_questions + ai_questions
            
            sections.append({
                'title': 'General Questions',
                'description': f'Questions about {topic}',
                'questions': all_questions
            })
        else:
            sections.append({
                'title': 'General Questions',
                'description': f'Questions about {topic}',
                'questions': general_questions
            })
    
    # Add a special AI-only section with completely fresh questions if AI is enabled
    if use_ai:
        # Create a comprehensive set of AI-generated questions
        comprehensive_ai_questions = generate_ai_questions(
            topic,
            description,
            primary_domain,
            primary_intent,
            category=None,
            num_questions=num_ai_questions + 1 # Generate slightly more for the dedicated section
        )
        
        if comprehensive_ai_questions:
            # Only add the AI section if we successfully generated questions
            sections.append({
                'title': 'Additional Insights',
                'description': f'Additional questions to gain deeper insights about {topic}',
                'questions': comprehensive_ai_questions
            })
    
    return sections

def generate_section_for_category(category, topic, description, domain=None, intent=None, use_ai=False, num_ai_questions=3):
    """
    Generate a questionnaire section for a specific category.
    
    Args:
        category (str): The category name ('demographics', 'experience', etc.)
        topic (str): The main research topic
        description (str): The research description
        domain (str, optional): The primary research domain
        intent (str, optional): The primary research intent
        use_ai (bool): Whether to use AI enhancement
        num_ai_questions (int): Number of AI questions to generate for this section
        
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
    
    # Generate entirely new AI questions if enabled
    if use_ai:
        # First enhance existing template questions
        section['questions'] = enhance_questions_with_ai(
            section['questions'], 
            topic, 
            description, 
            domain, 
            intent, 
            use_ai=True
        )
        
        # Then add completely AI-generated questions (2 per category)
        ai_questions = generate_ai_questions(
            topic,
            description,
            domain,
            intent,
            category,
            num_questions=num_ai_questions
        )
        
        if ai_questions:
            # Add the AI-created questions to the section
            section['questions'].extend(ai_questions)
            logger.info(f"Added {len(ai_questions)} AI-created questions to {category} section")
    
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

def enhance_questions_with_ai(questions, research_topic, research_description, domain=None, intent=None, use_ai=False):
    """
    Enhance questions using AI to make them more relevant to the specific research context.
    
    Args:
        questions (list): List of question dictionaries to enhance
        research_topic (str): The main research topic
        research_description (str): Detailed description of the research
        domain (str, optional): The identified research domain
        intent (str, optional): The identified research intent
        use_ai (bool): Whether to use AI enhancement (can be toggled off for testing/fallback)
        
    Returns:
        list: Enhanced questions with more relevant content
    """
    # If AI is disabled or not available, return original questions
    if not use_ai or not is_ai_enabled():
        return questions
    
    # Get the default model from config in case API call fails early
    _, configured_model = get_huggingface_config()

    enhanced_questions = []
    api_error_occurred = False # Track if any API call failed

    # Loop through each question
    for question in questions:
        enhanced_question = question.copy()
        enhanced_question['ai_enhanced'] = False # Initialize as not enhanced

        # Only enhance open-ended questions, which benefit most from AI processing
        if question.get('type') == 'Open-Ended':
            # Prepare AI prompt
            prompt = f"""<s>[INST] You are an expert questionnaire designer. Improve the following open-ended question to make it more specific, relevant, and insightful for the research context described below. Keep the question concise and clear.

Research Topic: {research_topic}
Research Description: {research_description}
Domain: {domain or 'Not specified'}
Intent: {intent or 'Not specified'}

Original Question: {question.get('text', '')}

Provide only the text of the enhanced question with no explanations or additional comments. [/INST]"""
            
            try:
                # Call the centralized AI service
                enhanced_text = call_huggingface_api(prompt)
                
                # Use AI-enhanced text if it's valid, otherwise keep original
                if enhanced_text and len(enhanced_text) > 10:  # Basic validation
                    enhanced_question['text'] = enhanced_text
                    enhanced_question['ai_enhanced'] = True
                    logger.info(f"AI enhanced question: Original: '{question.get('text')}' -> Enhanced: '{enhanced_text}'")
                else:
                    logger.warning(f"AI returned invalid or short response for question: '{question.get('text')}', Response: '{enhanced_text}'")

            except (HuggingFaceError, ValueError) as e:
                logger.error(f"Error enhancing question '{question.get('text')}' with AI: {e}")
                api_error_occurred = True # Mark that an error happened
                # Keep original question on API error

        enhanced_questions.append(enhanced_question)

    # If any API error occurred during the process, fallback for all questions might be safer
    if api_error_occurred:
        logger.warning("Fallback to rule-based enhancement due to API errors during enhancement.")
        return fallback_question_enhancement(questions, research_topic, research_description, domain, intent)

    return enhanced_questions

def fallback_question_enhancement(questions, research_topic, research_description, domain=None, intent=None):
    """
    Fallback method for enhancing questions when AI API is not available.
    Uses simple rule-based enhancements instead.
    """
    enhanced_questions = []
    
    for question in questions:
        enhanced_question = question.copy()
        
        if question.get('type') == 'Open-Ended':
            current_text = question.get('text', '')
            if '[TOPIC]' in current_text:
                # Already using the standard template with [TOPIC] placeholder
                enhanced_text = current_text
            else:
                # Add more specificity based on domain and intent
                domain_context = f" in the context of {domain}" if domain else ""
                intent_qualifier = ""
                if intent == "exploratory":
                    intent_qualifier = " from your perspective"
                elif intent == "evaluative":
                    intent_qualifier = " based on your experience"
                elif intent == "prescriptive":
                    intent_qualifier = " that could lead to improvements"
                
                enhanced_text = f"{current_text}{domain_context}{intent_qualifier}"
            
            enhanced_question['text'] = enhanced_text
            enhanced_question['ai_enhanced'] = True
        
        enhanced_questions.append(enhanced_question)
    
    return enhanced_questions

def get_dummy_enhanced_questions(questions, research_topic, research_description):
    """
    Demonstrates how AI-enhanced questions would look without actually calling an API.
    This is for demonstration purposes only.
    
    Args:
        questions (list): List of question dictionaries to enhance
        research_topic (str): The main research topic
        research_description (str): The research description
        
    Returns:
        list: Questions with simulated AI enhancements
    """
    enhanced_questions = []
    
    # Example mappings for common question patterns
    enhancement_map = {
        "How satisfied are you with": f"On a scale from 1-10, how would you rate your overall satisfaction with the features and functionality of the {research_topic}?",
        "What aspects of": f"Which specific features or components of the {research_topic} do you find most useful for your daily needs?",
        "What improvements would you suggest": f"If you could change three things about the {research_topic} to improve your experience, what would they be and why?",
        "What challenges or difficulties": f"What are the most frustrating obstacles or limitations you've encountered while using the {research_topic}?",
        "How has your usage": f"How has your pattern of engagement with the {research_topic} evolved since you first started using it?",
        "What would make your experience": f"What missing features or improvements would significantly enhance your satisfaction with the {research_topic}?",
        "What frustrations": f"When using the {research_topic}, what specific aspects cause you the most frustration or decrease your productivity?",
        "How do you currently overcome challenges": f"What workarounds or alternative methods have you developed to address limitations in the {research_topic}?"
    }
    
    for question in questions:
        enhanced_question = question.copy()
        
        if question.get('type') == 'Open-Ended':
            current_text = question.get('text', '')
            
            # Check if the current text matches any of our enhancement patterns
            enhanced_text = None
            for pattern, enhanced_version in enhancement_map.items():
                if pattern in current_text:
                    enhanced_text = enhanced_version
                    break
            
            # If no specific enhancement found, make a generic improvement
            if not enhanced_text:
                # Replace generic topic references with the specific topic
                if research_topic in current_text:
                    enhanced_text = current_text
                else:
                    enhanced_text = f"{current_text} specifically regarding {research_topic}"
            
            enhanced_question['text'] = enhanced_text
            enhanced_question['ai_enhanced'] = True
        
        enhanced_questions.append(enhanced_question)
    
    return enhanced_questions

def generate_ai_questions(research_topic, research_description, domain=None, intent=None, category=None, num_questions=3):
    """
    Generate entirely new questions using AI based on research context.
    
    Args:
        research_topic (str): The main research topic
        research_description (str): The research description
        domain (str, optional): The identified research domain
        intent (str, optional): The identified research intent
        category (str, optional): The question category (e.g., 'experience', 'feedback')
        num_questions (int): Number of questions to generate
        
    Returns:
        list: AI-generated questions
    """
    if not is_ai_enabled():
        logger.warning("AI question generation skipped: AI features are disabled.")
        return []

    ai_questions = []
    
    # Define question types to generate
    question_types = [
        {"type": "Open-Ended", "name": "open-ended"},
        {"type": "Multiple Choice", "name": "multiple choice", "options_count": 4},
        {"type": "Likert Scale", "name": "rating scale"}
    ]
    
    # Create a descriptive category name if provided
    category_desc = ""
    if category:
        if category == 'demographics':
            category_desc = "demographic information about respondents"
        elif category == 'experience':
            category_desc = "experience with and exposure to the topic"
        elif category == 'preferences':
            category_desc = "preferences and opinions"
        elif category == 'behaviors':
            category_desc = "behaviors and usage patterns"
        elif category == 'feedback':
            category_desc = "feedback and suggestions for improvement"
        elif category == 'motivation':
            category_desc = "motivation and reasons for usage"
        elif category == 'satisfaction':
            category_desc = "satisfaction levels and expectations"
        elif category == 'pain_points':
            category_desc = "challenges, pain points, and difficulties"
    
    try:
        # Generate different question types
        for q_type in question_types:
            # Prepare AI prompt for generating a question
            prompt = f"""<s>[INST] You are an expert questionnaire designer. Based on the research context below, generate {num_questions} distinct {q_type['name']} questions relevant to the topic{category_desc}.

Research Topic: {research_topic}
Research Description: {research_description}
Domain: {domain or 'Not specified'}
Intent: {intent or 'Not specified'}

For each question:
1. The question should be specific, clear, and directly relevant to the research topic
2. Avoid leading or biased questions
3. Make sure questions will generate meaningful data for the research purpose
4. Focus on aspects that would be most insightful for this kind of research

{f"For multiple choice questions, suggest 4-6 answer options separated by | symbols" if q_type['name'] == 'multiple choice' else ""}

Return ONLY the questions with no explanations or additional text, one question per line.
{f"For multiple choice questions, include the options on the next line after each question, separated by | symbols" if q_type['name'] == 'multiple choice' else ""}
[/INST]"""
            
            try:
                # Call the centralized AI service
                generated_text = call_huggingface_api(prompt)
                
                if generated_text and len(generated_text) > 10:
                    # Process the generated questions
                    lines = [line.strip() for line in generated_text.strip().split('\n') if line.strip()]
                    
                    # Handle different question types
                    if q_type['name'] == 'multiple choice':
                        for i in range(0, len(lines), 2):
                            if i+1 < len(lines):  # Make sure we have options line
                                question_text = lines[i]
                                options = [opt.strip() for opt in lines[i+1].split('|')]
                                
                                ai_questions.append({
                                    'text': question_text,
                                    'type': q_type['type'],
                                    'options': options,
                                    'ai_created': True
                                })
                    else:
                        for line in lines:
                            if '?' in line or line.strip().endswith('.'):  # Basic check that it's a question
                                ai_questions.append({
                                    'text': line,
                                    'type': q_type['type'],
                                    'ai_created': True
                                })
                
                logger.info(f"Generated {len(ai_questions)} AI questions of type: {q_type['name']}")
                
            except (HuggingFaceError, ValueError) as e:
                logger.error(f"Error generating {q_type['name']} questions: {e}")
    
    except Exception as e:
        logger.error(f"Error in AI question generation: {e}")
    
    return ai_questions

def generate_questionnaire(research_description, research_topic=None, target_audience=None, questionnaire_purpose=None, use_ai_enhancement=False, num_ai_questions=3):
    """
    Generate a questionnaire structure based on a research description.
    
    Args:
        research_description (str): Description of the research
        research_topic (str, optional): Title of the research
        target_audience (str, optional): Target audience for the questionnaire
        questionnaire_purpose (str, optional): Purpose of the questionnaire
        use_ai_enhancement (bool, optional): Whether to use AI to enhance question relevance
        num_ai_questions (int): Number of AI questions to generate per type/section
        
    Returns:
        list: A list of questionnaire sections with questions
    """
    if not research_topic:
        research_topic = "this topic"
    
    # Analyze the intent, domain, and target audience
    intent_analysis = analyze_intent(research_description)
    
    # Generate sections and questions based on the research description
    sections = analyze_research_description(research_description, research_topic, use_ai=use_ai_enhancement, num_ai_questions=num_ai_questions)
    
    # If AI enhancement is requested but API fails, use our dummy enhancement for demo purposes
    if use_ai_enhancement:
        for section in sections:
            try:
                # First try the actual AI enhancement
                section['questions'] = enhance_questions_with_ai(
                    section['questions'], 
                    research_topic, 
                    research_description, 
                    intent_analysis.get('domain'), 
                    intent_analysis.get('intent'), 
                    use_ai=True
                )
                
                # If no questions got AI-enhanced, fall back to dummy enhancement
                if not any(q.get('ai_enhanced', False) for q in section['questions']):
                    section['questions'] = get_dummy_enhanced_questions(
                        section['questions'], 
                        research_topic, 
                        research_description
                    )
            except Exception as e:
                logger.error(f"Error in AI enhancement, using dummy enhancement: {e}")
                section['questions'] = get_dummy_enhanced_questions(
                    section['questions'], 
                    research_topic, 
                    research_description
                )
    
    return sections 