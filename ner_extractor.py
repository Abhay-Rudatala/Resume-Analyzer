import spacy
import re
import logging
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NamedEntityExtractor:
    """Named Entity Extraction using SpaCy Fallback"""
    def __init__(self):
        # Try to load SpaCy model
        self.nlp = self._load_spacy_model()

        # Core skills for extraction
        self.technical_skills = [
            'python', 'java', 'javascript', 'html', 'css', 'react', 'angular', 'vue',
            'node.js', 'express.js', 'django', 'flask', 'spring', 'laravel', 'rails',
            'mysql', 'postgresql', 'mongodb', 'redis', 'sql', 'git', 'github',
            'aws', 'azure', 'docker', 'kubernetes', 'tensorflow', 'pytorch',
            'machine learning', 'deep learning', 'nlp', 'computer vision',
            'pandas', 'numpy', 'scikit-learn', 'api', 'rest', 'matplotlib', 'seaborn'
        ]

        self.soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving',
            'project management', 'time management', 'adaptability'
        ]

        logger.info("NamedEntityExtractor initialized successfully")

    def _load_spacy_model(self):
        """Load SpaCy model"""
        try:
            nlp = spacy.load('en_core_web_sm')
            logger.info("Loaded SpaCy model: en_core_web_sm")
            return nlp
        except OSError:
            logger.warning("SpaCy model not found. Using regex-based extraction.")
            return None

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Main extraction method"""
        if not text:
            return {}

        entities = {
            'name': self._extract_name_simple(text),
            'email': self._extract_email(text),
            'phone': self._extract_phone(text),
            'skills': self._extract_skills(text),
            'experience': self._extract_experience_simple(text),
            'education': self._extract_education_simple(text),
            'organizations': self._extract_organizations_simple(text),
            'locations': self._extract_locations_final(text),
            'certifications': self._extract_certifications_simple(text),
            'projects': self._extract_projects_simple(text)
        }

        return entities

    def _extract_name_simple(self, text: str) -> str:
        """Simple name extraction"""
        lines = text.split('\n')[:10]

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line or len(line) < 3:
                continue

            # Skip lines with obvious non-name content
            if any(word in line.lower() for word in [
                'email', 'phone', 'contact', '@', '+91', 'resume', 'cv',
                'objective', 'summary', 'experience', 'education', 'skills'
            ]):
                continue

            # Check for spaced name format "A B H A Y  R U D A T A L A"
            if re.match(r'^[A-Z](?:\s+[A-Z]){3,}\s*$', line):
                letters = line.split()
                if 4 <= len(letters) <= 12:
                    # Convert to normal name format
                    if len(letters) >= 6:
                        mid = len(letters) // 2
                        first_name = ''.join(letters[:mid])
                        last_name = ''.join(letters[mid:])
                        return f"{first_name} {last_name}"
                    else:
                        return ''.join(letters)

            # Check for normal name patterns
            words = line.split()
            if 2 <= len(words) <= 3:
                # Check if all words start with capital letter and are alphabetic
                if all(word[0].isupper() and word.isalpha() for word in words):
                    # Make sure it's not too long (likely not a name)
                    if len(line) <= 30:
                        return line

        return "Not found"

    def _extract_experience_simple(self, text: str) -> str:
        """Simple experience extraction"""
        text_lower = text.lower()

        # Look for direct experience statements
        patterns = [
            r'(\d+)\+?\s*years?\s*of\s*experience',
            r'(\d+)\+?\s*years?\s*experience',
            r'experience.*?(\d+)\+?\s*years?'
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                years = int(match.group(1))
                if 1 <= years <= 40:
                    return f"{years} years"

        # Check for fresher indicators
        fresher_words = ['fresher', 'entry level', 'recent graduate', 'student', 'seeking entry']
        if any(word in text_lower for word in fresher_words):
            return "Fresher/Entry-level"

        # Check for internship
        if 'intern' in text_lower or 'training' in text_lower:
            return "Internship/Training experience"

        return "Not specified"

    def _extract_education_simple(self, text: str) -> List[str]:
        """Simple education extraction"""
        education = []

        # Common degree patterns
        patterns = [
            r'(Master of Computer Applications?)',
            r'(Bachelor of Computer Applications?)',
            r'(M\.?C\.?A\.?)',
            r'(B\.?C\.?A\.?)',
            r'(Master of Science)',
            r'(Bachelor of Science)',
            r'(M\.?Tech|B\.?Tech)',
            r'(MBA|MCA|BCA)',
            r'([A-Z][a-z]+\s+University)',
            r'([A-Z][a-z]+\s+College)'
        ]

        text_lines = text.split('\n')
        for line in text_lines:
            for pattern in patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    clean_match = match.strip()
                    if len(clean_match) > 3 and clean_match not in education:
                        education.append(clean_match.title())

        return education[:3] if education else ["Not found"]

    def _extract_organizations_simple(self, text: str) -> List[str]:
        """Simple organization extraction"""
        organizations = []

        # Look for company patterns
        patterns = [
            r'(?:at|with)\s+([A-Z][a-zA-Z\s]+(?:Company|Corp|Inc|Ltd|Limited|Technologies|Systems|Solutions))',
            r'([A-Z][a-zA-Z\s]+(?:Company|Corp|Inc|Ltd|Limited|Technologies|Systems|Solutions))',
            r'([A-Z][a-zA-Z\s]+University)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                org = match.strip()
                if len(org) > 5 and org not in organizations:
                    organizations.append(org)

        return organizations[:3]

    def _extract_locations_final(self, text: str) -> List[str]:
        """
        FINAL SUPER STRICT: Only extract genuine geographical locations
        """
        locations = []

        # Comprehensive blacklist of ALL technical and non-location terms
        tech_blacklist = {
            # Programming languages & frameworks
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node', 'express',
            'django', 'flask', 'spring', 'laravel', 'php', 'ruby', 'go', 'kotlin', 'swift',
            'typescript', 'dart', 'scala', 'rust', 'perl', 'matlab', 'r',

            # Data science & ML libraries
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'bokeh', 'tensorflow', 
            'pytorch', 'keras', 'scikit', 'learn', 'opencv', 'jupyter', 'notebook', 
            'anaconda', 'spyder', 'colab',

            # Databases & tools
            'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle', 'cassandra',
            'elasticsearch', 'neo4j', 'dynamodb', 'firebase', 'git', 'github', 'gitlab',
            'bitbucket', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible',
            'aws', 'azure', 'gcp', 'heroku', 'netlify', 'vercel',

            # Web technologies
            'html', 'css', 'sass', 'scss', 'less', 'bootstrap', 'tailwind', 'jquery',
            'ajax', 'json', 'xml', 'api', 'rest', 'graphql', 'websocket', 'http', 'https',

            # Generic resume terms
            'company', 'current', 'present', 'experience', 'project', 'projects', 'skills',
            'education', 'work', 'job', 'role', 'position', 'internship', 'training',
            'certification', 'certifications', 'development', 'software', 'application',
            'system', 'platform', 'framework', 'library', 'database', 'server', 'client',
            'frontend', 'backend', 'fullstack', 'mobile', 'web', 'android', 'ios',
            'linux', 'windows', 'macos', 'ubuntu', 'centos', 'debian',

            # Resume sections and common words
            'summary', 'objective', 'profile', 'contact', 'phone', 'email', 'address',
            'languages', 'tools', 'technologies', 'achievements', 'awards', 'honors',
            'references', 'portfolio', 'github', 'linkedin', 'website', 'blog',

            # Academic and professional terms
            'university', 'college', 'institute', 'school', 'degree', 'bachelor', 'master',
            'phd', 'doctorate', 'diploma', 'certificate', 'course', 'program', 'major',
            'minor', 'gpa', 'cgpa', 'semester', 'year', 'completed', 'ongoing',
            'pursuing', 'graduated', 'alumni',

            # Time and status terms
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
            'september', 'october', 'november', 'december', 'jan', 'feb', 'mar', 'apr',
            'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'monday', 'tuesday',
            'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'today', 'yesterday',
            'tomorrow', 'week', 'month', 'year', 'daily', 'weekly', 'monthly', 'yearly'
        }

        # Approved Indian geographical locations only
        indian_cities = {
            'ahmedabad', 'mumbai', 'delhi', 'bangalore', 'chennai', 'hyderabad', 'pune', 
            'kolkata', 'jaipur', 'surat', 'kanpur', 'nagpur', 'indore', 'thane', 'bhopal',
            'visakhapatnam', 'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik',
            'faridabad', 'meerut', 'rajkot', 'kalyan', 'vasai', 'varanasi', 'srinagar',
            'aurangabad', 'dhanbad', 'amritsar', 'howrah', 'gwalior', 'jabalpur', 
            'coimbatore', 'vijayawada', 'jodhpur', 'madurai', 'raipur', 'kota', 
            'chandigarh', 'guwahati', 'solapur', 'hubli', 'mysore', 'tiruchirappalli',
            'bareilly', 'aligarh', 'tiruppur', 'gurgaon', 'moradabad', 'jalandhar',
            'bhubaneswar', 'salem', 'warangal', 'gandhinagar', 'bhavnagar', 'jamnagar',
            'rajkot', 'bhuj', 'anand', 'nadiad', 'mehsana', 'palanpur'
        }

        indian_states = {
            'gujarat', 'maharashtra', 'karnataka', 'tamil nadu', 'rajasthan', 'uttar pradesh',
            'bihar', 'west bengal', 'madhya pradesh', 'andhra pradesh', 'odisha', 'punjab',
            'haryana', 'assam', 'jharkhand', 'kerala', 'telangana', 'chhattisgarh',
            'uttarakhand', 'himachal pradesh', 'tripura', 'manipur', 'meghalaya', 'nagaland',
            'goa', 'arunachal pradesh', 'mizoram', 'sikkim'
        }

        # Priority 1: Look for "City, State" patterns (highest priority)
        city_state_patterns = [
            r'\b(Ahmedabad,\s*Gujarat)\b',
            r'\b(Mumbai,\s*Maharashtra)\b',
            r'\b(Bangalore,\s*Karnataka)\b',
            r'\b(Chennai,\s*Tamil Nadu)\b',
            r'\b(Pune,\s*Maharashtra)\b',
            r'\b(Jaipur,\s*Rajasthan)\b',
            r'\b(Jamnagar,\s*Gujarat)\b',
            r'\b(Rajkot,\s*Gujarat)\b',
            r'\b(Surat,\s*Gujarat)\b',
            r'\b([A-Z][a-z]+,\s*Gujarat)\b',
            r'\b([A-Z][a-z]+,\s*Maharashtra)\b'
        ]

        for pattern in city_state_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_match = match.strip()
                # Triple-check it's not any technical term
                words_in_location = [word.strip(',').lower() for word in clean_match.split()]
                if not any(word in tech_blacklist for word in words_in_location):
                    if clean_match not in locations:
                        locations.append(clean_match)

        # Priority 2: Individual approved cities (only if no city,state found)
        if not locations:
            # Find all words in text
            words_in_text = re.findall(r'\b[A-Za-z]+\b', text.lower())

            for word in words_in_text:
                # Check if it's an approved Indian city
                if word in indian_cities and word not in tech_blacklist:
                    proper_name = word.title()
                    if proper_name not in locations:
                        locations.append(proper_name)
                        if len(locations) >= 2:  # Limit to 2 individual cities
                            break

        # Remove any remaining technical terms that might have slipped through
        final_locations = []
        for loc in locations:
            words_in_loc = [word.strip(',').lower() for word in loc.split()]
            if not any(word in tech_blacklist for word in words_in_loc):
                final_locations.append(loc)

        # Remove duplicates while preserving order
        unique_locations = []
        for loc in final_locations:
            if loc not in unique_locations:
                unique_locations.append(loc)

        return unique_locations[:2]  # Return max 2 locations

    def _extract_email(self, text: str) -> str:
        """Extract email address"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(pattern, text)
        return matches[0] if matches else "Not found"

    def _extract_phone(self, text: str) -> str:
        """Extract phone number"""
        patterns = [
            r'\+91[-\s]?([6-9]\d{9})',
            r'\b([6-9]\d{9})\b'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return f"+91 {matches[0]}"
        return "Not found"

    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills"""
        text_lower = text.lower()
        found_skills = []

        all_skills = self.technical_skills + self.soft_skills

        for skill in all_skills:
            if skill.lower() in text_lower:
                # Use word boundaries for better matching
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.append(skill.title())

        return sorted(list(set(found_skills)))

    def _extract_certifications_simple(self, text: str) -> List[str]:
        """Simple certification extraction"""
        certifications = []
        cert_keywords = ['certified', 'certification', 'certificate']

        lines = text.split('\n')
        for line in lines:
            for keyword in cert_keywords:
                if keyword.lower() in line.lower() and len(line.strip()) > 10:
                    certifications.append(line.strip())
                    break

        return certifications[:3]

    def _extract_projects_simple(self, text: str) -> List[str]:
        """Simple project extraction"""
        projects = []
        project_keywords = ['project', 'built', 'developed', 'created']

        lines = text.split('\n')
        for line in lines:
            if (any(keyword in line.lower() for keyword in project_keywords) and
                len(line.strip()) > 20):
                projects.append(line.strip())
                if len(projects) >= 3:
                    break

        return projects

# ANONYMIZED TEST FUNCTION (NO PERSONAL DATA)
def test_system():
    """Test the extractor with completely anonymized data"""
    extractor = NamedEntityExtractor()

    # Test with anonymized sample text
    test_text = """
    John Smith
    john.smith@email.com
    +91 9876543210
    Mumbai, Maharashtra

    Skills: Python, JavaScript, React, Machine Learning
    Experience with modern web technologies
    """

    print("SYSTEM TEST:")
    entities = extractor.extract_entities(test_text)
    print(f"Locations: {entities['locations']}")
    print(f"Skills: {entities['skills']}")
    # Should show: ['Mumbai, Maharashtra'] and proper skills

if __name__ == "__main__":
    test_system()