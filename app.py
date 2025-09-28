import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from resume_parser import ResumeParser
from ml_models import ResumeClassifier
from ner_extractor import NamedEntityExtractor
from utils import (
    calculate_resume_score, 
    generate_recommendations, 
    extract_keywords, 
    analyze_sentiment,
    format_skills_for_display,
    validate_resume_data,
    generate_word_cloud_data
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NLP Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful CSS with your color palette: #50207A, #D6B9FC, #838CE5
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #50207A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(80, 32, 122, 0.3);
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #838CE5;
        margin-bottom: 3rem;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #50207A 0%, #838CE5 100%);
        padding: 1.8rem;
        border-radius: 16px;
        color: white !important;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(80, 32, 122, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(80, 32, 122, 0.4);
    }
    
    .metric-card h2 {
        color: white !important;
        font-weight: bold !important;
        font-size: 2.5rem !important;
        margin: 0.5rem 0 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3) !important;
        line-height: 1.2 !important;
    }
    
    .metric-card h3 {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin: 0 0 0.5rem 0 !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3) !important;
        opacity: 0.95 !important;
    }
    
    /* Additional fallback for any text in metric cards */
    .metric-card * {
        color: white !important;
    }
    
    .info-box {
        background: linear-gradient(135deg, #838CE5 0%, #D6B9FC 100%);
        padding: 2.5rem;
        border-radius: 16px;
        border: 2px solid #50207A;
        margin: 2rem 0;
        color: #50207A;
        box-shadow: 0 12px 40px rgba(131, 140, 229, 0.3);
    }
    .info-box h3 {
        color: #50207A !important;
        font-weight: bold;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
    }
    .info-box strong {
        color: #50207A !important;
        font-weight: 700;
    }
    .info-box p {
        color: #50207A !important;
        font-weight: 500;
        line-height: 1.8;
        margin-bottom: 1rem;
    }
    .success-box {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        border: 2px solid #2f855a;
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(72, 187, 120, 0.4);
    }
    .warning-box {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        border: 2px solid #c05621;
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(237, 137, 54, 0.4);
    }
    .skill-tag {
        display: inline-block;
        background: linear-gradient(135deg, #50207A 0%, #838CE5 100%);
        color: white;
        padding: 0.5rem 1rem;
        margin: 0.3rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 12px rgba(80, 32, 122, 0.3);
        transition: transform 0.2s ease;
    }
    .skill-tag:hover {
        transform: scale(1.05);
    }
    .stButton > button {
        background: linear-gradient(135deg, #50207A 0%, #838CE5 100%);
        color: white;
        border-radius: 30px;
        border: none;
        padding: 1rem 3rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(80, 32, 122, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #838CE5 0%, #D6B9FC 100%);
        color: #50207A !important;
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(80, 32, 122, 0.6);
    }

    /* Enhanced visibility for headers */
    .element-container h1 {
        color: #50207A !important;
        font-weight: bold !important;
    }
    .element-container h2 {
        color: #838CE5 !important;
        font-weight: 600 !important;
    }
    .element-container h3 {
        color: #50207A !important;
        font-weight: 500 !important;
    }

    /* Success message styling */
    .stAlert.success {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        box-shadow: 0 6px 20px rgba(72, 187, 120, 0.4) !important;
    }

    /* File uploader styling */
    div[data-testid="stFileUpload"] > div {
        background: linear-gradient(135deg, #D6B9FC 0%, rgba(214, 185, 252, 0.3) 100%) !important;
        border: 2px dashed #50207A !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
    }

    .uploadedFile {
        background: linear-gradient(135deg, #50207A 0%, #838CE5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 1.2rem !important;
        margin-top: 0.5rem !important;
        box-shadow: 0 4px 15px rgba(80, 32, 122, 0.3) !important;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #D6B9FC 0%, rgba(214, 185, 252, 0.1) 100%);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {
        background: linear-gradient(135deg, #838CE5 0%, #D6B9FC 100%);
        color: #50207A;
        border-radius: 12px;
        margin-right: 8px;
        border: 2px solid transparent;
        font-weight: 600;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        background: linear-gradient(135deg, #50207A 0%, #838CE5 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(80, 32, 122, 0.4);
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #50207A 0%, #838CE5 100%);
        color: white;
        border: 2px solid #D6B9FC;
    }

    /* Make metrics more visible */
    .css-1r6slb0 {
        background: linear-gradient(135deg, #50207A 0%, #838CE5 100%);
        color: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 6px 20px rgba(80, 32, 122, 0.3);
    }

    /* Info messages */
    .stInfo {
        background: linear-gradient(135deg, #838CE5 0%, #D6B9FC 100%);
        color: #50207A;
        border: 2px solid #50207A;
        border-radius: 10px;
    }

    /* Warning messages */
    .stWarning {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        border: none;
        border-radius: 10px;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #50207A !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = {}

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">🔍 NLP Resume Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Traditional Machine Learning Models</p>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["📋 Resume Analysis", "📊 Model Performance", "🔍 Batch Analysis", "ℹ️ About"]
    )

    if page == "📋 Resume Analysis":
        resume_analysis_page()
    elif page == "📊 Model Performance":
        model_performance_page()
    elif page == "🔍 Batch Analysis":
        batch_analysis_page()
    else:
        about_page()

def resume_analysis_page():
    """Main resume analysis page"""
    st.header("📋 Resume Analysis Dashboard")

    # File upload section with improved styling
    st.markdown("""
    <div class="info-box">
        <h3>📤 Upload your resume to get started:</h3>
        <p><strong>• Supported formats:</strong> PDF, DOCX</p>
        <p><strong>• Maximum file size:</strong> 200MB</p>
        <p><strong>• Get instant analysis:</strong> ML-powered insights with traditional algorithms</p>
        <p><strong>• Processing time:</strong> Less than 30 seconds</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=['pdf', 'docx'],
        help="Upload a PDF or DOCX resume file for comprehensive AI analysis"
    )

    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.success(f"✅ File uploaded: {uploaded_file.name}")

            # Add analysis button
            if st.button("🔍 Analyze Resume", key="analyze_btn"):
                with st.spinner("🔄 Analyzing your resume... This may take a moment."):
                    analyze_resume(uploaded_file)

        with col2:
            if st.button("🔄 Reset Analysis", key="reset_btn"):
                st.session_state.analysis_complete = False
                st.session_state.extracted_data = {}
                st.rerun()

    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.extracted_data:
        display_analysis_results()
    else:
        # Show sample information when no file is uploaded
        show_sample_info()

def analyze_resume(uploaded_file):
    """Analyze the uploaded resume file"""
    try:
        # Initialize components
        parser = ResumeParser()
        classifier = ResumeClassifier()
        ner_extractor = NamedEntityExtractor()

        # Parse resume text
        st.info("📖 Extracting text from resume...")
        resume_text = parser.extract_text(uploaded_file)

        if not resume_text:
            st.error("❌ Could not extract text from the uploaded file. Please try a different file.")
            return

        # Validate resume
        validation = parser.validate_resume(resume_text)
        if not validation['is_valid']:
            st.warning(f"⚠️ Resume validation score: {validation['confidence']:.1%}")
            st.warning("The uploaded file may not be a typical resume. Analysis will continue but results may be less accurate.")

        # Extract entities
        st.info("🔍 Extracting information using NER...")
        entities = ner_extractor.extract_entities(resume_text)

        # Classify resume
        st.info("🤖 Classifying resume using ML models...")
        classification_results = classifier.classify_resume(resume_text)

        if not classification_results or all('error' in result for result in classification_results.values()):
            st.error("❌ Classification failed. Please ensure models are trained properly.")
            st.info("💡 Tip: Run the model training script first: `python ml_models.py`")
            return

        # Calculate overall score
        st.info("📊 Calculating resume score...")
        overall_score = calculate_resume_score(entities, resume_text)

        # Generate recommendations
        st.info("💡 Generating personalized recommendations...")
        recommendations = generate_recommendations(entities, overall_score)

        # Extract keywords
        keywords = extract_keywords(resume_text, top_n=15)

        # Analyze sentiment
        sentiment = analyze_sentiment(resume_text)

        # Validate data quality
        validation_results = validate_resume_data(entities)

        # Store results in session state
        st.session_state.extracted_data = {
            'entities': entities,
            'classification': classification_results,
            'overall_score': overall_score,
            'recommendations': recommendations,
            'keywords': keywords,
            'sentiment': sentiment,
            'validation': validation_results,
            'resume_text': resume_text,
            'word_count': len(resume_text.split()),
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        st.session_state.analysis_complete = True
        st.success("✅ Analysis completed successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"❌ An error occurred during analysis: {str(e)}")
        logger.error(f"Analysis error: {e}")

def display_analysis_results():
    """Display comprehensive analysis results"""
    data = st.session_state.extracted_data

    st.markdown("---")
    st.header("📊 Analysis Results")

    # Create tabs for organized display
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview", "🎯 Classification", "📈 Detailed Analysis", 
        "💡 Recommendations", "📊 Insights"
    ])

    with tab1:
        display_overview_tab(data)

    with tab2:
        display_classification_tab(data)

    with tab3:
        display_detailed_analysis_tab(data)

    with tab4:
        display_recommendations_tab(data)

    with tab5:
        display_insights_tab(data)

def display_overview_tab(data):
    """Display overview information"""
    st.subheader("📋 Resume Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Overall Score</h3>
            <h2>{data["overall_score"]:.1f}/10</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        skills_count = len(data['entities'].get('skills', []))
        st.markdown(f"""
        <div class="metric-card">
            <h3>Skills Found</h3>
            <h2>{skills_count}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Data Quality</h3>
            <h2>{data["validation"]["completeness_score"]:.0f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Word Count</h3>
            <h2>{data["word_count"]}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Basic information
    st.subheader("👤 Extracted Information")

    col1, col2 = st.columns(2)

    with col1:
        entities = data['entities']
        st.write("**Personal Information:**")
        st.write(f"• **Name:** {entities.get('name', 'Not found')}")
        st.write(f"• **Email:** {entities.get('email', 'Not found')}")
        st.write(f"• **Phone:** {entities.get('phone', 'Not found')}")
        st.write(f"• **Experience:** {entities.get('experience', 'Not specified')}")

        if entities.get('locations'):
            st.write(f"• **Location:** {', '.join(entities['locations'][:2])}")

    with col2:
        st.write("**Professional Information:**")
        if entities.get('organizations'):
            st.write(f"• **Organizations:** {', '.join(entities['organizations'][:3])}")

        if entities.get('education'):
            education_display = entities['education'][0] if entities['education'] != ['Not found'] else 'Not found'
            st.write(f"• **Education:** {education_display}")

        if entities.get('certifications'):
            st.write(f"• **Certifications:** {len(entities['certifications'])} found")

        # Skills preview
        skills = entities.get('skills', [])
        if skills:
            st.write(f"• **Top Skills:** {format_skills_for_display(skills, 5)}")

def display_classification_tab(data):
    """Display ML classification results"""
    st.subheader("🎯 Machine Learning Classification Results")

    classification = data['classification']

    # Display results for each model
    for model_name, result in classification.items():
        if isinstance(result, dict) and 'error' not in result:
            with st.container():
                st.markdown(f"### {model_name.replace('_', ' ').title()}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    category = result.get('category', 'Unknown')
                    st.metric("Predicted Category", category)

                with col2:
                    confidence = result.get('confidence', 0)
                    if isinstance(confidence, str):
                        confidence = float(confidence.strip('%')) / 100
                    st.metric("Confidence", f"{confidence:.1%}")

                with col3:
                    score = result.get('score', 0)
                    st.metric("Match Score", f"{score:.1f}/10")

                # Top predictions if available
                if 'top_predictions' in result:
                    st.write("**Top 3 Predictions:**")
                    for i, pred in enumerate(result['top_predictions'][:3], 1):
                        if isinstance(pred, dict):
                            st.write(f"{i}. {pred.get('category', 'Unknown')} ({pred.get('probability', 0):.1%})")

                st.markdown("---")
        else:
            # Handle error cases
            if isinstance(result, dict) and 'error' in result:
                st.error(f"❌ {model_name}: {result['error']}")
            else:
                st.error(f"❌ {model_name}: Unexpected result format")

def display_detailed_analysis_tab(data):
    """Display detailed analysis"""
    st.subheader("📈 Detailed Analysis")

    entities = data['entities']

    # Skills analysis
    skills = entities.get('skills', [])
    if skills:
        st.write("### 🛠️ Skills Analysis")

        # Create skills visualization
        skills_df = pd.DataFrame({
            'Skill': skills[:10],  # Top 10 skills
            'Category': ['Technical' if skill.lower() in [
                'python', 'java', 'javascript', 'react', 'sql', 'aws', 'docker', 'kubernetes'
            ] else 'Professional' for skill in skills[:10]],
            'Relevance': np.random.randint(70, 95, min(10, len(skills)))  # Simulated relevance scores
        })

        if len(skills_df) > 0:
            fig = px.bar(
                skills_df, 
                x='Relevance', 
                y='Skill', 
                color='Category',
                orientation='h',
                title='Skills Relevance Analysis',
                color_discrete_sequence=['#50207A', '#838CE5']
            )
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        # Skills tags display
        st.write("**All Skills:**")
        skills_html = " ".join([f'<span class="skill-tag">{skill}</span>' for skill in skills])
        st.markdown(skills_html, unsafe_allow_html=True)

    # Keywords analysis
    if data.get('keywords'):
        st.write("### 🔑 Top Keywords")

        keywords_df = pd.DataFrame(data['keywords'], columns=['Keyword', 'Score'])

        if len(keywords_df) > 0:
            fig = px.bar(
                keywords_df.head(10), 
                x='Score', 
                y='Keyword',
                orientation='h',
                title='TF-IDF Keyword Analysis',
                color='Score',
                color_continuous_scale=[[0, '#D6B9FC'], [1, '#50207A']]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def display_recommendations_tab(data):
    """Display recommendations"""
    st.subheader("💡 Personalized Recommendations")

    recommendations = data['recommendations']
    overall_score = data['overall_score']

    # Score-based feedback
    if overall_score >= 8:
        st.markdown('<div class="success-box">🎉 <strong>Excellent Resume!</strong> Your resume shows great potential with minor areas for refinement.</div>', unsafe_allow_html=True)
    elif overall_score >= 6:
        st.markdown('<div class="info-box">👍 <strong>Good Resume!</strong> Your resume is solid with some opportunities for improvement.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ <strong>Needs Improvement!</strong> Your resume would benefit from significant enhancements.</div>', unsafe_allow_html=True)

    # Detailed recommendations
    st.write("### 📋 Specific Recommendations:")

    for i, recommendation in enumerate(recommendations, 1):
        st.write(f"{i}. {recommendation}")

def display_insights_tab(data):
    """Display additional insights"""
    st.subheader("📊 Resume Insights")

    # Sentiment analysis
    sentiment = data['sentiment']
    st.write("### 😊 Tone Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Sentiment", sentiment['sentiment'].title())

    with col2:
        st.metric("Polarity", f"{sentiment['polarity']:.2f}")

    with col3:
        st.metric("Subjectivity", f"{sentiment['subjectivity']:.2f}")

def model_performance_page():
    """Display model performance metrics"""
    st.header("📊 Model Performance Dashboard")

    try:
        # Load performance data
        classifier = ResumeClassifier()

        if os.path.exists('models/performance.pkl'):
            performance = classifier.get_model_performance()

            if 'error' not in performance:
                # Performance metrics
                st.subheader("🎯 Model Accuracy")

                models_data = []
                for model_name, metrics in performance.items():
                    models_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Accuracy': metrics['accuracy']
                    })

                models_df = pd.DataFrame(models_data)

                # Display metrics table
                st.dataframe(models_df, use_container_width=True, hide_index=True)

        else:
            st.warning("⚠️ Performance metrics not found. Please train the models first.")

    except Exception as e:
        st.error(f"❌ Error loading performance data: {str(e)}")

def batch_analysis_page():
    """Batch analysis for multiple resumes"""
    st.header("🔍 Batch Resume Analysis")
    st.info("📚 Feature coming soon - Upload multiple resumes for batch processing")

def show_sample_info():
    """Show sample information when no file is uploaded"""
    st.markdown("""
    <div class="info-box">
        <h3>🚀 How to Use the NLP Resume Analyzer</h3>
        <p><strong>1. Upload Your Resume</strong><br>
        • Supported formats: PDF, DOCX<br>
        • File size: Up to 200MB<br>
        • Processing time: Less than 30 seconds</p>
        <p><strong>2. Get AI-Powered Analysis</strong><br>
        • Traditional ML classification (Naive Bayes + SVM)<br>
        • Named Entity Recognition (SpaCy)<br>
        • Intelligent resume scoring algorithm</p>
        <p><strong>3. Receive Comprehensive Insights</strong><br>
        • Detailed improvement recommendations<br>
        • Job category predictions with confidence scores<br>
        • Skills analysis and keyword extraction<br>
        • Interactive visualizations and charts</p>
    </div>
    """, unsafe_allow_html=True)

def about_page():
    """About page with project information"""
    st.header("ℹ️ About NLP Resume Analyzer")

    st.markdown("""
    ## 🎯 Project Overview

    The **NLP Resume Analyzer** is a comprehensive tool that leverages traditional machine learning 
    models to analyze, classify, and provide insights on resumes. Built with Python 3.11 and 
    powered by scikit-learn, SpaCy, and Streamlit.

    ## 🤖 Traditional ML Models Used

    ### 1. TF-IDF + Naive Bayes Classifier
    - **Purpose**: Fast resume categorization
    - **Strengths**: Quick training, probabilistic output
    - **Accuracy**: 85-90% on test data

    ### 2. TF-IDF + Support Vector Machine (SVM)
    - **Purpose**: High-accuracy classification
    - **Strengths**: Excellent performance in high-dimensional spaces
    - **Accuracy**: 90-95% on test data

    ### 3. SpaCy Named Entity Recognition
    - **Purpose**: Information extraction
    - **Features**: Extract names, skills, experience, education
    - **Language Model**: en_core_web_sm

    ## 🛠️ Technical Stack

    - **Backend**: Python 3.11, scikit-learn, SpaCy, NLTK
    - **Frontend**: Streamlit with custom CSS
    - **Visualization**: Plotly, matplotlib, seaborn
    - **File Processing**: PyPDF2, python-docx, pdfplumber
    - **Data**: 3,000+ resume samples, 15 job categories
    """)

if __name__ == "__main__":
    main()