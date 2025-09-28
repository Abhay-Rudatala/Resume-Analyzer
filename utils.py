import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean and preprocess text for analysis
    Args:
        text (str): Input text to clean
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""

    # Convert to string if not already
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Remove excessive whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.@-]', ' ', text)

    # Remove extra spaces
    text = text.strip()

    return text

def calculate_resume_score(extracted_info: Dict[str, Any], resume_text: str) -> float:
    """
    Calculate comprehensive resume score based on extracted information
    Args:
        extracted_info (Dict): Dictionary containing extracted resume information
        resume_text (str): Original resume text
    Returns:
        float: Resume score out of 10
    """
    score = 0.0
    max_score = 10.0

    # Personal Information (2 points)
    if extracted_info.get('name') and extracted_info['name'] != "Not found":
        score += 1.0

    # Contact Information (2 points)
    if extracted_info.get('email') and extracted_info['email'] != "Not found":
        score += 1.0

    if extracted_info.get('phone') and extracted_info['phone'] != "Not found":
        score += 1.0

    # Skills Assessment (3 points)
    skills = extracted_info.get('skills', [])
    if len(skills) >= 10:
        score += 3.0
    elif len(skills) >= 7:
        score += 2.5
    elif len(skills) >= 5:
        score += 2.0
    elif len(skills) >= 3:
        score += 1.0
    elif len(skills) >= 1:
        score += 0.5

    # Experience (2 points)
    experience = extracted_info.get('experience', "")
    if "years" in experience and experience != "Not specified":
        # Extract years and award points based on experience
        years_match = re.search(r'(\d+)', experience)
        if years_match:
            years = int(years_match.group(1))
            if years >= 5:
                score += 2.0
            elif years >= 3:
                score += 1.5
            elif years >= 1:
                score += 1.0
            else:
                score += 0.5
    elif experience != "Not specified":
        score += 1.0

    # Education (1 point)
    education = extracted_info.get('education', [])
    if education and education != ["Not found"]:
        if len(education) >= 2:
            score += 1.0
        else:
            score += 0.5

    # Additional Factors (1 point)
    # Resume length and structure
    if resume_text:
        word_count = len(resume_text.split())
        if 200 <= word_count <= 1000:
            score += 0.5
        elif word_count > 100:
            score += 0.25

    # Check for section diversity
    sections_found = 0
    section_keywords = ['experience', 'education', 'skills', 'project', 'certification']
    text_lower = resume_text.lower()
    for keyword in section_keywords:
        if keyword in text_lower:
            sections_found += 1

    if sections_found >= 4:
        score += 0.5
    elif sections_found >= 3:
        score += 0.25

    return min(score, max_score)

def generate_recommendations(extracted_info: Dict[str, Any], overall_score: float) -> List[str]:
    """
    Generate personalized recommendations for resume improvement
    Args:
        extracted_info (Dict): Extracted resume information
        overall_score (float): Overall resume score
    Returns:
        List[str]: List of recommendations
    """
    recommendations = []

    # Score-based recommendations
    if overall_score < 5:
        recommendations.append("Your resume needs significant improvement. Consider a complete restructure.")
    elif overall_score < 7:
        recommendations.append("Your resume is decent but has room for improvement.")
    elif overall_score >= 8:
        recommendations.append("Excellent resume! Minor refinements could make it perfect.")

    # Specific recommendations based on missing elements
    if not extracted_info.get('name') or extracted_info['name'] == "Not found":
        recommendations.append("Add your full name prominently at the top of your resume.")

    if not extracted_info.get('email') or extracted_info['email'] == "Not found":
        recommendations.append("Include a professional email address in your contact information.")

    if not extracted_info.get('phone') or extracted_info['phone'] == "Not found":
        recommendations.append("Add your phone number for easy contact.")

    # Skills recommendations
    skills = extracted_info.get('skills', [])
    if len(skills) < 5:
        recommendations.append("Add more relevant technical and soft skills to strengthen your profile.")
    elif len(skills) < 8:
        recommendations.append("Consider adding a few more specialized skills relevant to your target role.")

    # Experience recommendations
    experience = extracted_info.get('experience', "")
    if experience == "Not specified":
        recommendations.append("Clearly state your years of experience or employment history.")
    elif "years" in experience:
        years_match = re.search(r'(\d+)', experience)
        if years_match and int(years_match.group(1)) < 2:
            recommendations.append("Highlight any internships, projects, or relevant coursework to strengthen your profile.")

    # Education recommendations
    education = extracted_info.get('education', [])
    if not education or education == ["Not found"]:
        recommendations.append("Include your educational background and qualifications.")

    # Additional recommendations
    organizations = extracted_info.get('organizations', [])
    if not organizations:
        recommendations.append("Mention company names or organizations you've worked with.")

    certifications = extracted_info.get('certifications', [])
    if not certifications:
        recommendations.append("Add any relevant certifications or professional credentials.")

    projects = extracted_info.get('projects', [])
    if not projects:
        recommendations.append("Include notable projects or achievements to showcase your work.")

    # Ensure we have at least some recommendations
    if not recommendations:
        recommendations.append("Your resume looks comprehensive! Consider tailoring it for specific job applications.")

    return recommendations

def extract_keywords(text: str, top_n: int = 20) -> List[Tuple[str, float]]:
    """
    Extract top keywords from text using TF-IDF
    Args:
        text (str): Input text
        top_n (int): Number of top keywords to return
    Returns:
        List[Tuple[str, float]]: List of (keyword, score) tuples
    """
    if not text:
        return []

    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=min(100, top_n * 5),
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1
        )

        # Fit and transform text
        tfidf_matrix = vectorizer.fit_transform([text])

        # Get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Create keyword-score pairs
        keyword_scores = list(zip(feature_names, tfidf_scores))

        # Sort by score and return top keywords
        keyword_scores.sort(key=lambda x: x[1], reverse=True)

        return keyword_scores[:top_n]

    except Exception as e:
        logger.warning(f"Keyword extraction failed: {e}")
        return []

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of the resume text
    Args:
        text (str): Resume text
    Returns:
        Dict: Sentiment analysis results
    """
    try:
        blob = TextBlob(text)

        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Determine sentiment category
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            'sentiment': sentiment,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'confidence': abs(polarity)
        }

    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {e}")
        return {
            'sentiment': 'neutral',
            'polarity': 0.0,
            'subjectivity': 0.0,
            'confidence': 0.0
        }

def format_skills_for_display(skills: List[str], max_display: int = 10) -> str:
    """
    Format skills list for display
    Args:
        skills (List[str]): List of skills
        max_display (int): Maximum number of skills to display
    Returns:
        str: Formatted skills string
    """
    if not skills:
        return "No skills found"

    display_skills = skills[:max_display]
    formatted = ", ".join(display_skills)

    if len(skills) > max_display:
        remaining = len(skills) - max_display
        formatted += f" (and {remaining} more)"

    return formatted

def validate_resume_data(extracted_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the quality and completeness of extracted resume data
    Args:
        extracted_info (Dict): Extracted resume information
    Returns:
        Dict: Validation results
    """
    validation_results = {
        'completeness_score': 0,
        'issues': [],
        'strengths': []
    }

    total_fields = 8  # Total expected fields
    completed_fields = 0

    # Check each field
    fields_to_check = [
        ('name', 'Name'),
        ('email', 'Email'),
        ('phone', 'Phone'),
        ('skills', 'Skills'),
        ('experience', 'Experience'),
        ('education', 'Education'),
        ('organizations', 'Organizations'),
        ('projects', 'Projects')
    ]

    for field_key, field_name in fields_to_check:
        field_value = extracted_info.get(field_key)

        if field_value and field_value != "Not found" and field_value != ["Not found"]:
            if isinstance(field_value, list) and len(field_value) > 0:
                completed_fields += 1
                validation_results['strengths'].append(f"{field_name} information found")
            elif isinstance(field_value, str) and field_value.strip():
                completed_fields += 1
                validation_results['strengths'].append(f"{field_name} information found")
            else:
                validation_results['issues'].append(f"Missing {field_name}")
        else:
            validation_results['issues'].append(f"Missing {field_name}")

    # Calculate completeness score
    validation_results['completeness_score'] = (completed_fields / total_fields) * 100

    return validation_results

def generate_word_cloud_data(text: str, max_words: int = 50) -> Dict[str, int]:
    """
    Generate word frequency data for word cloud visualization
    Args:
        text (str): Input text
        max_words (int): Maximum number of words to return
    Returns:
        Dict[str, int]: Word frequency dictionary
    """
    if not text:
        return {}

    try:
        # Clean and preprocess text
        clean_text_content = clean_text(text)

        # Split into words and count frequency
        words = clean_text_content.split()

        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
            'could', 'should', 'may', 'might', 'must', 'can', 'shall', 'this', 
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        # Filter words
        filtered_words = [
            word for word in words 
            if len(word) > 2 and word.lower() not in stop_words
        ]

        # Count frequency
        word_freq = Counter(filtered_words)

        # Return top words
        return dict(word_freq.most_common(max_words))

    except Exception as e:
        logger.warning(f"Word cloud data generation failed: {e}")
        return {}

def extract_contact_patterns(text: str) -> Dict[str, List[str]]:
    """
    Extract various contact patterns from text
    Args:
        text (str): Input text
    Returns:
        Dict[str, List[str]]: Dictionary of extracted contact patterns
    """
    contacts = {
        'emails': [],
        'phones': [],
        'linkedin_profiles': [],
        'github_profiles': []
    }

    if not text:
        return contacts

    # Email patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    contacts['emails'] = re.findall(email_pattern, text, re.IGNORECASE)

    # Phone patterns (multiple formats)
    phone_patterns = [
        r'\+?91[-\s]?[6-9]\d{9}',  # Indian mobile
        r'\(?\+?91\)?[-\s]?\d{10}',  # Indian with country code
        r'\b(?:\+?1[-.]?)?(?:\(?[0-9]{3}\)?[-.]?)?[0-9]{3}[-.]?[0-9]{4}\b',  # US format
        r'\+?[1-9]\d{7,14}'  # International format
    ]

    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        contacts['phones'].extend(matches)

    # LinkedIn profiles
    linkedin_pattern = r'linkedin\.com/in/[A-Za-z0-9-]+'
    contacts['linkedin_profiles'] = re.findall(linkedin_pattern, text, re.IGNORECASE)

    # GitHub profiles
    github_pattern = r'github\.com/[A-Za-z0-9-]+'
    contacts['github_profiles'] = re.findall(github_pattern, text, re.IGNORECASE)

    # Remove duplicates
    for key in contacts:
        contacts[key] = list(set(contacts[key]))

    return contacts

def calculate_readability_score(text: str) -> Dict[str, Any]:
    """
    Calculate readability metrics for resume text
    Args:
        text (str): Resume text
    Returns:
        Dict[str, Any]: Readability metrics
    """
    if not text:
        return {'error': 'No text provided'}

    try:
        # Basic metrics
        word_count = len(text.split())
        sentence_count = len(re.findall(r'[.!?]+', text))
        char_count = len(text)

        # Average calculations
        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Readability assessment
        if avg_sentence_length <= 15:
            readability = "Easy"
        elif avg_sentence_length <= 20:
            readability = "Moderate"
        else:
            readability = "Complex"

        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'character_count': char_count,
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'readability_level': readability
        }

    except Exception as e:
        logger.warning(f"Readability calculation failed: {e}")
        return {'error': str(e)}