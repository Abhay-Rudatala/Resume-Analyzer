import os
import sys
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test all required imports"""
    print("🔧 Testing imports...")
    try:
        # Test core libraries
        import pandas as pd
        import numpy as np
        import sklearn
        import nltk
        import re
        print("✅ Core libraries imported successfully")

        # Test optional libraries
        try:
            import spacy
            print("✅ SpaCy imported successfully")
        except ImportError:
            print("⚠️ SpaCy not found - NER will use fallback methods")

        try:
            import streamlit as st
            print("✅ Streamlit imported successfully")
        except ImportError:
            print("❌ Streamlit not found - please install: pip install streamlit")
            return False

        # Test custom modules
        try:
            from resume_parser import ResumeParser
            from ml_models import ResumeClassifier
            from ner_extractor import NamedEntityExtractor
            from utils import calculate_resume_score
            print("✅ Custom modules imported successfully")
        except ImportError as e:
            print(f"❌ Custom module import failed: {e}")
            return False

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_dataset():
    """Test dataset availability"""
    print("\n📊 Testing dataset...")

    if not os.path.exists('resume_dataset.csv'):
        print("❌ Dataset not found!")
        return False

    try:
        df = pd.read_csv('resume_dataset.csv')
        print(f"✅ Dataset loaded: {len(df)} samples, {df['Category'].nunique()} categories")

        # Check required columns
        if 'Category' not in df.columns or 'Text' not in df.columns:
            print("❌ Dataset missing required columns")
            return False

        print("✅ Dataset structure is correct")
        return True

    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return False

def test_models():
    """Test trained models"""
    print("\n🤖 Testing trained models...")

    model_files = [
        'models/naive_bayes_model.pkl',
        'models/svm_model.pkl',
        'models/label_encoder.pkl',
        'models/categories.pkl'
    ]

    for model_file in model_files:
        if not os.path.exists(model_file):
            print(f"❌ Model file missing: {model_file}")
            return False
        print(f"✅ Found: {model_file}")

    # Test loading models
    try:
        from ml_models import ResumeClassifier
        classifier = ResumeClassifier()
        success = classifier.load_models()

        if success:
            print("✅ Models loaded successfully")
        else:
            print("❌ Model loading failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_resume_parser():
    """Test resume parser"""
    print("\n📄 Testing resume parser...")

    try:
        from resume_parser import ResumeParser
        parser = ResumeParser()

        # Test with sample text (anonymized)
        sample_text = """
        John Doe
        Software Engineer
        Email: john.doe@email.com
        Phone: (555) 123-4567

        EXPERIENCE
        5+ years of software development experience

        SKILLS
        Python, JavaScript, React, AWS

        EDUCATION
        Bachelor of Computer Science
        """

        # Test contact extraction
        contact_info = parser.extract_contact_info(sample_text)
        print(f"✅ Contact extraction: {len(contact_info)} fields found")

        # Test validation
        validation = parser.validate_resume(sample_text)
        print(f"✅ Resume validation: {validation['confidence']:.1%} confidence")

        return True

    except Exception as e:
        print(f"❌ Resume parser error: {e}")
        return False

def test_ner_extractor():
    """Test NER extractor"""
    print("\n🔍 Testing NER extractor...")

    try:
        from ner_extractor import NamedEntityExtractor
        extractor = NamedEntityExtractor()

        # Test with sample text (anonymized)
        sample_text = """
        Jane Smith
        Data Scientist
        jane.smith@email.com
        (555) 987-6543

        5+ years experience in machine learning and Python programming.
        Master's degree in Computer Science.
        Worked at Technology Company and Software Corp.

        Skills: Python, TensorFlow, SQL, AWS
        """

        entities = extractor.extract_entities(sample_text)
        print(f"✅ NER extraction: {len(entities)} entity types detected")
        print(f"   - Name: {entities.get('name', 'Not found')}")
        print(f"   - Skills: {len(entities.get('skills', []))} found")

        return True

    except Exception as e:
        print(f"❌ NER extractor error: {e}")
        return False

def test_classification():
    """Test ML classification"""
    print("\n🎯 Testing ML classification...")

    try:
        from ml_models import ResumeClassifier
        classifier = ResumeClassifier()

        # Test with sample text
        sample_text = """
        Software Engineer with 5+ years experience in Python programming and web development.
        Expertise in React, Node.js, and AWS cloud services. Built scalable applications and
        led development teams. Strong background in algorithms and database design.
        """

        results = classifier.classify_resume(sample_text)

        if results and not all('error' in result for result in results.values()):
            print("✅ Classification successful")
            for model_name, result in results.items():
                if 'error' not in result:
                    print(f"   - {model_name}: {result['category']} ({result['confidence']:.1%})")
            return True
        else:
            print("❌ Classification failed")
            return False

    except Exception as e:
        print(f"❌ Classification error: {e}")
        return False

def test_utilities():
    """Test utility functions"""
    print("\n🛠️ Testing utilities...")

    try:
        from utils import calculate_resume_score, generate_recommendations

        # Test with sample entities
        sample_entities = {
            'name': 'John Smith',
            'email': 'john@email.com',
            'phone': '555-123-4567',
            'skills': ['Python', 'JavaScript', 'React'],
            'experience': '5+ years',
            'education': ['Bachelor of Computer Science']
        }

        sample_text = "Software engineer with Python and JavaScript experience"

        score = calculate_resume_score(sample_entities, sample_text)
        recommendations = generate_recommendations(sample_entities, score)

        print(f"✅ Resume scoring: {score:.1f}/10")
        print(f"✅ Recommendations: {len(recommendations)} generated")

        return True

    except Exception as e:
        print(f"❌ Utilities error: {e}")
        return False

def test_full_pipeline():
    """Test the complete pipeline"""
    print("\n🔄 Testing full pipeline...")

    try:
        from resume_parser import ResumeParser
        from ml_models import ResumeClassifier
        from ner_extractor import NamedEntityExtractor
        from utils import calculate_resume_score, generate_recommendations

        # Sample resume text (completely anonymized)
        sample_resume = """
        Sarah Johnson
        Data Scientist
        sarah.johnson@email.com
        (555) 789-0123
        San Francisco, CA

        PROFESSIONAL SUMMARY
        Experienced Data Scientist with 6+ years in machine learning and statistical analysis.

        EXPERIENCE
        Senior Data Scientist at Tech Company (2020-2023)
        - Developed recommendation algorithms using Python and TensorFlow
        - Led a team of 4 data scientists
        - Improved user engagement by 25% through ML models

        Data Analyst at Music Streaming Company (2018-2020)
        - Analyzed user behavior data using SQL and Python
        - Created dashboards using visualization tools

        EDUCATION
        Master of Science in Data Science
        Technology University (2018)

        Bachelor of Science in Statistics
        State University (2016)

        SKILLS
        Programming: Python, R, SQL, Java
        ML/AI: TensorFlow, PyTorch, Scikit-learn
        Tools: Visualization Software, Docker, AWS, Git
        Statistics: Hypothesis Testing, A/B Testing, Regression Analysis

        CERTIFICATIONS
        AWS Certified Machine Learning Specialist
        Cloud Platform Professional Data Engineer
        """

        # Step 1: Parse resume
        parser = ResumeParser()
        print("   Step 1: Parsing resume...")

        # Step 2: Extract entities
        ner_extractor = NamedEntityExtractor()
        entities = ner_extractor.extract_entities(sample_resume)
        print(f"   Step 2: Extracted {len(entities)} entity types")

        # Step 3: Classify resume
        classifier = ResumeClassifier()
        classification = classifier.classify_resume(sample_resume)
        print("   Step 3: Classification completed")

        # Step 4: Calculate score and recommendations
        score = calculate_resume_score(entities, sample_resume)
        recommendations = generate_recommendations(entities, score)
        print(f"   Step 4: Score calculated ({score:.1f}/10)")

        # Display sample results
        print("\n📋 Sample Results:")
        print(f"   Name: {entities.get('name', 'Not found')}")
        print(f"   Skills: {len(entities.get('skills', []))} found")
        print(f"   Overall Score: {score:.1f}/10")

        if classification:
            for model_name, result in list(classification.items())[:1]:  # Show first model
                if 'error' not in result:
                    print(f"   Predicted Category: {result['category']}")

        print("✅ Full pipeline test successful!")
        return True

    except Exception as e:
        print(f"❌ Full pipeline error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 NLP RESUME ANALYZER - COMPREHENSIVE TEST SUITE")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Dataset", test_dataset),
        ("Trained Models", test_models),
        ("Resume Parser", test_resume_parser),
        ("NER Extractor", test_ner_extractor),
        ("ML Classification", test_classification),
        ("Utilities", test_utilities),
        ("Full Pipeline", test_full_pipeline)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")

    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Your system is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Run: streamlit run app.py")
        print("   2. Open your browser to the displayed URL")
        print("   3. Upload a resume file and start analyzing!")
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the errors above.")
        print("\n💡 Common fixes:")
        print("   - Install missing packages: pip install -r requirements.txt")
        print("   - Train models: python ml_models.py")
        print("   - Download SpaCy model: python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    main()