import os
import logging
from typing import Dict, List, Any, Optional
import re

# Import Groq client
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("groq package not installed. Install with: pip install groq")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaResumeRecommender:
    """
    Generate intelligent resume recommendations using Groq's Llama models
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        """
        Initialize Groq Resume Recommender
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY environment variable)
            model: Groq model to use (default: llama-3.1-8b-instant)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        
        if not GROQ_AVAILABLE:
            raise ImportError(
                "groq package not found. "
                "Install it with: pip install groq"
            )
        
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. "
                "Set GROQ_API_KEY environment variable or pass api_key parameter"
            )
        
        try:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise
    
    def generate_recommendations(
        self,
        extracted_data: Dict[str, Any],
        resume_text: str,
        overall_score: float,
        predicted_category: str = None,
        max_recommendations: int = 10
    ) -> Dict[str, Any]:
        """
        Generate AI-powered resume recommendations using Groq
        
        Args:
            extracted_data: Dictionary containing extracted resume information
            resume_text: Full resume text
            overall_score: Resume quality score (0-10)
            predicted_category: Predicted job category
            max_recommendations: Maximum number of recommendations
        
        Returns:
            Dictionary containing recommendations and analysis
        """
        try:
            # Build comprehensive context for Llama
            context = self._build_context(
                extracted_data, 
                resume_text, 
                overall_score, 
                predicted_category
            )
            
            # Create prompt for Llama
            prompt = self._create_prompt(context, max_recommendations)
            
            # Call Groq API
            logger.info("Calling Groq API for recommendations...")
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert career counselor and resume reviewer with years of experience helping professionals improve their resumes. Provide specific, actionable, and personalized recommendations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=2000
            )
            
            # Extract recommendations from response
            recommendations_text = response.choices[0].message.content
            
            # Parse the response
            parsed_recommendations = self._parse_recommendations(recommendations_text)
            
            logger.info(f"Generated {len(parsed_recommendations['recommendations'])} recommendations")
            
            return {
                'success': True,
                'recommendations': parsed_recommendations['recommendations'],
                'summary': parsed_recommendations.get('summary', ''),
                'priority_areas': parsed_recommendations.get('priority_areas', []),
                'strengths': parsed_recommendations.get('strengths', []),
                'overall_assessment': parsed_recommendations.get('overall_assessment', ''),
                'model_used': self.model,
                'raw_response': recommendations_text
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {
                'success': False,
                'error': str(e),
                'recommendations': self._get_fallback_recommendations(overall_score)
            }
    
    def _build_context(
        self,
        extracted_data: Dict[str, Any],
        resume_text: str,
        overall_score: float,
        predicted_category: str = None
    ) -> str:
        """Build comprehensive context for Llama"""
        
        entities = extracted_data.get('entities', {})
        
        context_parts = [
            f"Resume Quality Score: {overall_score:.1f}/10",
            f"Word Count: {len(resume_text.split())} words"
        ]
        
        if predicted_category:
            context_parts.append(f"Predicted Job Category: {predicted_category}")
        
        # Add personal information
        if entities.get('name') and entities['name'] != 'Not found':
            context_parts.append(f"Name: {entities['name']}")
        
        if entities.get('email') and entities['email'] != 'Not found':
            context_parts.append(f"Email: {entities['email']}")
        
        if entities.get('phone') and entities['phone'] != 'Not found':
            context_parts.append(f"Phone: {entities['phone']}")
        
        # Add experience
        experience = entities.get('experience', 'Not specified')
        context_parts.append(f"Experience: {experience}")
        
        # Add skills
        skills = entities.get('skills', [])
        if skills:
            context_parts.append(f"Skills Found: {len(skills)}")
            context_parts.append(f"Top Skills: {', '.join(skills[:10])}")
        else:
            context_parts.append("Skills Found: 0")
        
        # Add education
        education = entities.get('education', [])
        if education:
            context_parts.append(f"Education: {education[0] if education else 'Not found'}")
        
        # Add organizations
        organizations = entities.get('organizations', [])
        if organizations:
            context_parts.append(f"Organizations: {', '.join(organizations[:3])}")
        
        # Add certifications
        certifications = entities.get('certifications', [])
        if certifications:
            context_parts.append(f"Certifications: {len(certifications)} found")
        
        # Add projects
        projects = entities.get('projects', [])
        if projects:
            context_parts.append(f"Projects: {len(projects)} found")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, context: str, max_recommendations: int) -> str:
        """Create a structured prompt for Llama"""
        
        prompt = f"""Analyze this resume and provide detailed, actionable recommendations for improvement.

RESUME INFORMATION:
{context}

INSTRUCTIONS:
1. Provide {max_recommendations} specific, actionable recommendations to improve this resume
2. Prioritize recommendations based on impact (highest impact first)
3. For each recommendation, explain WHY it matters and HOW to implement it
4. Consider the job category and tailor recommendations accordingly
5. Identify 3-5 key strengths to maintain
6. Identify 3-5 priority improvement areas
7. Provide an overall assessment summary

FORMAT YOUR RESPONSE AS FOLLOWS:

OVERALL ASSESSMENT:
[2-3 sentences about the resume's current state]

STRENGTHS:
- [Strength 1]
- [Strength 2]
- [Strength 3]

PRIORITY AREAS:
- [Area 1]
- [Area 2]
- [Area 3]

RECOMMENDATIONS:
1. [Title]: [Detailed recommendation with specific actions]
2. [Title]: [Detailed recommendation with specific actions]
3. [Title]: [Detailed recommendation with specific actions]
...

SUMMARY:
[1-2 sentences with final advice]"""

        return prompt
    
    def _parse_recommendations(self, response_text: str) -> Dict[str, Any]:
        """Parse structured recommendations from Llama response"""
        
        result = {
            'recommendations': [],
            'strengths': [],
            'priority_areas': [],
            'overall_assessment': '',
            'summary': ''
        }
        
        try:
            # Split response into sections
            sections = {
                'overall_assessment': '',
                'strengths': [],
                'priority_areas': [],
                'recommendations': [],
                'summary': ''
            }
            
            current_section = None
            lines = response_text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Detect section headers
                if 'OVERALL ASSESSMENT' in line.upper():
                    current_section = 'overall_assessment'
                    continue
                elif 'STRENGTHS' in line.upper() and 'PRIORITY' not in line.upper():
                    current_section = 'strengths'
                    continue
                elif 'PRIORITY AREAS' in line.upper() or 'PRIORITY IMPROVEMENT' in line.upper():
                    current_section = 'priority_areas'
                    continue
                elif 'RECOMMENDATIONS' in line.upper():
                    current_section = 'recommendations'
                    continue
                elif 'SUMMARY' in line.upper():
                    current_section = 'summary'
                    continue
                
                # Parse content based on current section
                if not line:
                    continue
                
                if current_section == 'overall_assessment':
                    sections['overall_assessment'] += line + ' '
                elif current_section == 'strengths':
                    if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                        sections['strengths'].append(line[1:].strip())
                elif current_section == 'priority_areas':
                    if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                        sections['priority_areas'].append(line[1:].strip())
                elif current_section == 'recommendations':
                    # Parse numbered recommendations
                    if line[0].isdigit() or line.startswith('-') or line.startswith('•'):
                        sections['recommendations'].append(line.lstrip('0123456789.-•* '))
                elif current_section == 'summary':
                    sections['summary'] += line + ' '
            
            # Clean up and assign
            result['overall_assessment'] = sections['overall_assessment'].strip()
            result['strengths'] = sections['strengths'][:5]
            result['priority_areas'] = sections['priority_areas'][:5]
            result['recommendations'] = sections['recommendations']
            result['summary'] = sections['summary'].strip()
            
            # If recommendations are empty, extract all numbered/bulleted items
            if not result['recommendations']:
                result['recommendations'] = self._extract_bullet_points(response_text)
            
        except Exception as e:
            logger.warning(f"Error parsing recommendations: {e}")
            # Fallback: extract all bullet points
            result['recommendations'] = self._extract_bullet_points(response_text)
        
        return result
    
    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points or numbered items from text"""
        
        # Pattern to match numbered items or bullet points
        patterns = [
            r'^\d+[\.\)]\s+(.+)$',  # 1. Item or 1) Item
            r'^[-•*]\s+(.+)$',   # - Item or • Item or * Item
        ]
        
        items = []
        for line in text.split('\n'):
            line = line.strip()
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    items.append(match.group(1))
                    break
        
        return items
    
    def _get_fallback_recommendations(self, overall_score: float) -> List[str]:
        """Provide fallback recommendations if API fails"""
        
        recommendations = [
            "Add quantifiable achievements with specific metrics and numbers",
            "Include relevant technical skills for your target role",
            "Ensure consistent formatting throughout the document",
            "Add action verbs at the beginning of bullet points",
            "Include a professional summary at the top",
            "Tailor your resume to match the job description",
            "Remove outdated or irrelevant information",
            "Add relevant certifications and courses"
        ]
        
        if overall_score < 5:
            recommendations.insert(0, "Consider a complete resume restructure for better impact")
        elif overall_score < 7:
            recommendations.insert(0, "Focus on highlighting your key achievements more prominently")
        else:
            recommendations.insert(0, "Your resume is strong - focus on fine-tuning for specific roles")
        
        return recommendations


# Fallback function for when Groq API is not available
def generate_basic_recommendations(
    extracted_data: Dict[str, Any],
    overall_score: float
) -> Dict[str, Any]:
    """
    Generate basic recommendations without AI
    (Fallback when Groq API is not configured)
    """
    
    entities = extracted_data.get('entities', {})
    recommendations = []
    strengths = []
    priority_areas = []
    
    # Check what's missing and make recommendations
    if not entities.get('name') or entities['name'] == 'Not found':
        recommendations.append("Add your full name prominently at the top of your resume")
        priority_areas.append("Missing contact information")
    else:
        strengths.append("Clear contact information present")
    
    if not entities.get('email') or entities['email'] == 'Not found':
        recommendations.append("Include a professional email address")
        priority_areas.append("Missing email contact")
    
    skills = entities.get('skills', [])
    if len(skills) < 5:
        recommendations.append("Add more relevant technical and soft skills (aim for 8-12 key skills)")
        priority_areas.append("Limited skills listed")
    elif len(skills) >= 10:
        strengths.append(f"Good range of skills ({len(skills)} skills identified)")
    
    experience = entities.get('experience', 'Not specified')
    if experience == 'Not specified':
        recommendations.append("Clearly state your years of experience")
        priority_areas.append("Experience not clearly stated")
    else:
        strengths.append("Experience clearly documented")
    
    education = entities.get('education', [])
    if not education or education == ['Not found']:
        recommendations.append("Include your educational background and qualifications")
        priority_areas.append("Education information missing")
    else:
        strengths.append("Educational background included")
    
    certifications = entities.get('certifications', [])
    if not certifications:
        recommendations.append("Add relevant certifications to strengthen your credentials")
    else:
        strengths.append(f"Professional certifications present ({len(certifications)} found)")
    
    projects = entities.get('projects', [])
    if not projects:
        recommendations.append("Include notable projects to showcase your practical experience")
    else:
        strengths.append(f"Projects documented ({len(projects)} found)")
    
    # Score-based recommendations
    if overall_score >= 8:
        assessment = "Excellent resume with strong fundamentals. Focus on minor refinements."
    elif overall_score >= 6:
        assessment = "Good resume foundation with room for strategic improvements."
    else:
        assessment = "Resume needs significant enhancements to meet professional standards."
        priority_areas.append("Overall structure and content depth")
    
    if len(recommendations) == 0:
        recommendations.append("Your resume looks comprehensive - consider tailoring it for specific roles")
        recommendations.append("Use strong action verbs to describe your achievements")
        recommendations.append("Quantify your accomplishments with specific metrics where possible")
    
    return {
        'success': True,
        'recommendations': recommendations,
        'strengths': strengths[:5],
        'priority_areas': priority_areas[:5],
        'overall_assessment': assessment,
        'summary': "Continue refining your resume based on target job requirements.",
        'model_used': 'Basic Rule-Based System',
        'is_fallback': True
    }