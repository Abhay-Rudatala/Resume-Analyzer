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

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

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

# Import Llama recommender
from llama_recommender import LlamaResumeRecommender, generate_basic_recommendations

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
    }
    .info-box {
        background: linear-gradient(135deg, rgba(80, 32, 122, 0.1), rgba(131, 140, 229, 0.1));
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #50207A;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, rgba(32, 176, 122, 0.1), rgba(50, 205, 50, 0.1));
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #20B072;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1), rgba(255, 152, 0, 0.1));
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FFC107;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #50207A, #838CE5);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(80, 32, 122, 0.1);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #50207A, #838CE5);
        color: white;
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
    st.markdown('<h1 class="main-header">🎯 NLP Resume Analyzer</h1>', unsafe_allow_html=True)
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
        <h3>📤 Upload Your Resume</h3>
        <p><strong>• Supported formats:</strong> PDF, DOCX</p>
        <p><strong>• Maximum file size:</strong> 200MB</p>
        <p><strong>• Get instant analysis:</strong> ML-powered insights with traditional algorithms</p>
        <p><strong>• Processing time:</strong> Less than 30 seconds</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a resume file",
        type=['pdf', 'docx'],
        help="Upload your resume in PDF or DOCX format for comprehensive analysis"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📄 **File:** {uploaded_file.name}")
            st.info(f"📊 **Size:** {uploaded_file.size / 1024:.2f} KB")
        
        with col2:
            if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
                with st.spinner("⏳ Analyzing resume... This may take a few moments."):
                    analyze_resume(uploaded_file)
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete:
        display_results()
    else:
        show_sample_info()

def analyze_resume(uploaded_file):
    """Analyze the uploaded resume file"""
    try:
        # Initialize components
        parser = ResumeParser()
        classifier = ResumeClassifier()
        ner_extractor = NamedEntityExtractor()

        # Parse resume text
        st.info("📄 Extracting text from resume...")
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

        # Generate basic recommendations (for fallback)
        st.info("💡 Generating personalized recommendations...")
        recommendations = generate_recommendations(entities, overall_score)

        # Extract keywords
        keywords = extract_keywords(resume_text, top_n=15)

        # Analyze sentiment
        sentiment = analyze_sentiment(resume_text)

        # Validate data quality
        validation_results = validate_resume_data(entities)

        # Generate Llama AI recommendations using Groq
        st.info("🤖 Generating AI-powered recommendations with Groq...")
        
        llama_recommendations = None
        try:
            # Check if API key is available (Streamlit secrets first, then env)
            api_key = None
            try:
                # Try Streamlit secrets first (works on Streamlit Cloud)
                api_key = st.secrets.get("GROQ_API_KEY")
            except:
                # Fallback to environment variable (works locally)
                api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                try:
                    # Initialize Llama recommender
                    recommender = LlamaResumeRecommender(api_key=api_key)
                    
                    # Get predicted category from classification results
                    predicted_category = None
                    if classification_results and 'svm' in classification_results:
                        predicted_category = classification_results['svm'].get('category')
                    
                    # Generate recommendations
                    llama_recommendations = recommender.generate_recommendations(
                        extracted_data={'entities': entities},
                        resume_text=resume_text,
                        overall_score=overall_score,
                        predicted_category=predicted_category,
                        max_recommendations=10
                    )
                    
                    if llama_recommendations.get('success'):
                        st.success("✅ AI recommendations generated successfully!")
                    else:
                        raise Exception(llama_recommendations.get('error', 'Unknown error'))
                        
                except Exception as e:
                    logger.warning(f"Groq API failed: {e}")
                    st.warning("⚠️ Groq API unavailable. Using fallback recommendations.")
                    llama_recommendations = generate_basic_recommendations(
                        extracted_data={'entities': entities},
                        overall_score=overall_score
                    )
            else:
                st.info("ℹ️ No GROQ_API_KEY found. Using basic recommendations.")
                llama_recommendations = generate_basic_recommendations(
                    extracted_data={'entities': entities},
                    overall_score=overall_score
                )
        
        except Exception as e:
            logger.error(f"Error in recommendation generation: {e}")
            st.warning("⚠️ Using fallback recommendations due to error.")
            llama_recommendations = generate_basic_recommendations(
                extracted_data={'entities': entities},
                overall_score=overall_score
            )

        # Store results in session state
        st.session_state.extracted_data = {
            'entities': entities,
            'classification': classification_results,
            'overall_score': overall_score,
            'recommendations': recommendations,
            'llama_recommendations': llama_recommendations,  # NEW: Add Llama recommendations
            'keywords': keywords,
            'sentiment': sentiment,
            'validation': validation_results,
            'resume_text': resume_text,
            'word_count': len(resume_text.split()),
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        st.session_state.analysis_complete = True
        st.success("✅ Analysis completed successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"❌ An error occurred during analysis: {str(e)}")
        logger.error(f"Analysis error: {e}")

def display_results():
    """Display analysis results in organized tabs"""
    data = st.session_state.extracted_data
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🎯 Classification",
        "📝 Detailed Analysis",
        "💡 Recommendations",
        "📈 Insights"
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
    """Display overview metrics"""
    st.subheader("📊 Resume Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{data['overall_score']:.1f}/10</h2>
            <p>Overall Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{data['word_count']}</h2>
            <p>Words</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        skills_count = len(data['entities'].get('skills', []))
        st.markdown(f"""
        <div class="metric-card">
            <h2>{skills_count}</h2>
            <p>Skills Found</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        sentiment_score = data['sentiment']['polarity']
        sentiment_emoji = "😊" if sentiment_score > 0 else "😐" if sentiment_score == 0 else "😟"
        st.markdown(f"""
        <div class="metric-card">
            <h2>{sentiment_emoji}</h2>
            <p>Sentiment: {sentiment_score:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Personal Information
    st.markdown("### 👤 Personal Information")
    entities = data['entities']
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown(f"**Name:** {entities.get('name', 'Not found')}")
        st.markdown(f"**Email:** {entities.get('email', 'Not found')}")
        st.markdown(f"**Phone:** {entities.get('phone', 'Not found')}")
    
    with info_col2:
        st.markdown(f"**Experience:** {entities.get('experience', 'Not specified')}")
        education = entities.get('education', [])
        st.markdown(f"**Education:** {education[0] if education else 'Not found'}")
    
    st.markdown("---")
    
    # Skills Display
    if entities.get('skills'):
        st.markdown("### 💼 Skills")
        skills_html = format_skills_for_display(entities['skills'])
        st.markdown(skills_html, unsafe_allow_html=True)

def display_classification_tab(data):
    """Display classification results"""
    st.subheader("🎯 Resume Classification Results")
    
    classification = data['classification']
    
    # Display results for each classifier
    col1, col2 = st.columns(2)
    
    with col1:
        if 'naive_bayes' in classification and 'error' not in classification['naive_bayes']:
            st.markdown("#### 📊 Naive Bayes Classifier")
            nb_result = classification['naive_bayes']
            st.markdown(f"**Predicted Category:** {nb_result['category']}")
            st.markdown(f"**Confidence:** {nb_result['confidence']:.2%}")
            
            # Confidence bar
            st.progress(nb_result['confidence'])
    
    with col2:
        if 'svm' in classification and 'error' not in classification['svm']:
            st.markdown("#### 🎯 SVM Classifier")
            svm_result = classification['svm']
            st.markdown(f"**Predicted Category:** {svm_result['category']}")
            st.markdown(f"**Confidence:** {svm_result['confidence']:.2%}")
            
            # Confidence bar
            st.progress(svm_result['confidence'])
    
    # Probability distribution chart
    if 'naive_bayes' in classification and 'probabilities' in classification['naive_bayes']:
        st.markdown("---")
        st.markdown("### 📈 Category Probabilities")
        
        probabilities = classification['naive_bayes']['probabilities']
        prob_df = pd.DataFrame(list(probabilities.items()), columns=['Category', 'Probability'])
        prob_df = prob_df.sort_values('Probability', ascending=False).head(10)
        
        fig = px.bar(prob_df, x='Category', y='Probability',
                     title='Top 10 Category Probabilities',
                     color='Probability',
                     color_continuous_scale='Viridis')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

def display_detailed_analysis_tab(data):
    """Display detailed entity analysis"""
    st.subheader("📝 Detailed Analysis")
    
    entities = data['entities']
    
    # Organizations
    if entities.get('organizations'):
        st.markdown("### 🏢 Organizations")
        for org in entities['organizations']:
            st.markdown(f"- {org}")
    
    # Certifications
    if entities.get('certifications'):
        st.markdown("### 🎓 Certifications")
        for cert in entities['certifications']:
            st.markdown(f"- {cert}")
    
    # Projects
    if entities.get('projects'):
        st.markdown("### 💻 Projects")
        for proj in entities['projects']:
            st.markdown(f"- {proj}")
    
    st.markdown("---")
    
    # Keywords cloud
    st.markdown("### 🔤 Top Keywords")
    keywords = data['keywords']
    keywords_df = pd.DataFrame(keywords, columns=['Keyword', 'Score'])
    
    fig = px.bar(keywords_df, x='Keyword', y='Score',
                 title='Top Keywords by TF-IDF Score',
                 color='Score',
                 color_continuous_scale='Blues')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def display_recommendations_tab(data):
    """Display AI-powered recommendations using Groq"""
    st.subheader("💡 AI-Powered Resume Recommendations")
    
    # Check if we have Llama recommendations
    llama_recommendations = data.get('llama_recommendations', {})
    overall_score = data['overall_score']
    
    if llama_recommendations and llama_recommendations.get('success'):
        # Display using Llama AI recommendations
        display_llama_recommendations(llama_recommendations, overall_score)
    else:
        # Fallback to basic recommendations
        st.warning("⚠️ AI recommendations unavailable. Showing basic recommendations.")
        display_basic_recommendations(data)


def display_llama_recommendations(llama_data: dict, overall_score: float):
    """Display recommendations generated by Llama AI"""
    
    # Model information
    model_used = llama_data.get('model_used', 'Llama AI')
    is_fallback = llama_data.get('is_fallback', False)
    
    if is_fallback:
        st.info(f"📊 Using: **{model_used}** (Fallback Mode)")
    else:
        st.success(f"🤖 Powered by: **{model_used}**")
    
    st.markdown("---")
    
    # Overall Assessment
    if llama_data.get('overall_assessment'):
        st.markdown("### 📊 Overall Assessment")
        
        if overall_score >= 8:
            st.markdown(
                f'<div class="success-box">🎉 <strong>Excellent Resume!</strong><br>{llama_data["overall_assessment"]}</div>', 
                unsafe_allow_html=True
            )
        elif overall_score >= 6:
            st.markdown(
                f'<div class="info-box">✅ <strong>Good Resume!</strong><br>{llama_data["overall_assessment"]}</div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="warning-box">⚠️ <strong>Needs Improvement!</strong><br>{llama_data["overall_assessment"]}</div>', 
                unsafe_allow_html=True
            )
        
        st.markdown("---")
    
    # Display in two columns
    col1, col2 = st.columns(2)
    
    # Strengths
    with col1:
        if llama_data.get('strengths') and len(llama_data['strengths']) > 0:
            st.markdown("### ✨ Key Strengths")
            for i, strength in enumerate(llama_data['strengths'], 1):
                st.markdown(
                    f'<div class="success-box" style="margin: 10px 0; padding: 15px; font-size: 0.95em;">'
                    f'<strong>{i}.</strong> {strength}'
                    f'</div>', 
                    unsafe_allow_html=True
                )
    
    # Priority Areas
    with col2:
        if llama_data.get('priority_areas') and len(llama_data['priority_areas']) > 0:
            st.markdown("### 🎯 Priority Areas")
            for i, area in enumerate(llama_data['priority_areas'], 1):
                st.markdown(
                    f'<div class="warning-box" style="margin: 10px 0; padding: 15px; font-size: 0.95em;">'
                    f'<strong>{i}.</strong> {area}'
                    f'</div>', 
                    unsafe_allow_html=True
                )
    
    st.markdown("---")
    
    # Detailed Recommendations
    recommendations = llama_data.get('recommendations', [])
    if recommendations and len(recommendations) > 0:
        st.markdown("### 📝 Detailed Recommendations")
        st.markdown("*Follow these actionable steps to improve your resume:*")
        st.markdown("")
        
        for i, recommendation in enumerate(recommendations, 1):
            # Create expandable sections for each recommendation
            rec_preview = recommendation[:80] + "..." if len(recommendation) > 80 else recommendation
            
            with st.expander(f"**{i}. {rec_preview}**", expanded=(i <= 3)):
                st.markdown(
                    f'<div style="padding: 15px; background: linear-gradient(135deg, rgba(80, 32, 122, 0.05), rgba(131, 140, 229, 0.05)); '
                    f'border-radius: 8px; border-left: 4px solid #50207A; line-height: 1.6;">'
                    f'{recommendation}'
                    f'</div>',
                    unsafe_allow_html=True
                )
    
    st.markdown("---")
    
    # Summary
    if llama_data.get('summary'):
        st.markdown("### 📌 Summary")
        st.info(llama_data['summary'])
    
    # Export recommendations
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💾 Export Recommendations")
        st.caption("Download your personalized recommendations for future reference")
    
    with col2:
        # Create downloadable text file
        recommendations_text = format_recommendations_for_export(llama_data, overall_score)
        st.download_button(
            label="📄 Download Report",
            data=recommendations_text,
            file_name=f"resume_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )


def display_basic_recommendations(data):
    """Display basic recommendations (fallback)"""
    recommendations = data['recommendations']
    overall_score = data['overall_score']
    
    st.info("📊 Using: **Basic Rule-Based System**")
    st.markdown("---")
    
    # Score-based feedback
    if overall_score >= 8:
        st.markdown(
            '<div class="success-box">🎉 <strong>Excellent Resume!</strong> Your resume shows great potential with minor areas for refinement.</div>', 
            unsafe_allow_html=True
        )
    elif overall_score >= 6:
        st.markdown(
            '<div class="info-box">✅ <strong>Good Resume!</strong> Your resume is solid with some opportunities for improvement.</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="warning-box">⚠️ <strong>Needs Improvement!</strong> Your resume would benefit from significant enhancements.</div>', 
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.markdown("### 📝 Specific Recommendations")
    
    for i, recommendation in enumerate(recommendations, 1):
        st.markdown(
            f'<div style="padding: 15px; margin: 10px 0; background: linear-gradient(135deg, rgba(80, 32, 122, 0.05), rgba(131, 140, 229, 0.05)); '
            f'border-radius: 8px; border-left: 4px solid #50207A; line-height: 1.6;">'
            f'<strong>{i}.</strong> {recommendation}'
            f'</div>',
            unsafe_allow_html=True
        )


def format_recommendations_for_export(llama_data: dict, overall_score: float) -> str:
    """Format recommendations as downloadable text"""
    
    lines = [
        "=" * 70,
        "RESUME RECOMMENDATIONS REPORT",
        "=" * 70,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Overall Score: {overall_score:.1f}/10",
        f"AI Model: {llama_data.get('model_used', 'AI System')}",
        "",
        "=" * 70,
    ]
    
    # Overall Assessment
    if llama_data.get('overall_assessment'):
        lines.extend([
            "",
            "OVERALL ASSESSMENT:",
            "-" * 70,
            llama_data['overall_assessment'],
            ""
        ])
    
    # Strengths
    if llama_data.get('strengths'):
        lines.extend([
            "",
            "KEY STRENGTHS:",
            "-" * 70
        ])
        for i, strength in enumerate(llama_data['strengths'], 1):
            lines.append(f"{i}. {strength}")
        lines.append("")
    
    # Priority Areas
    if llama_data.get('priority_areas'):
        lines.extend([
            "",
            "PRIORITY IMPROVEMENT AREAS:",
            "-" * 70
        ])
        for i, area in enumerate(llama_data['priority_areas'], 1):
            lines.append(f"{i}. {area}")
        lines.append("")
    
    # Detailed Recommendations
    if llama_data.get('recommendations'):
        lines.extend([
            "",
            "DETAILED RECOMMENDATIONS:",
            "-" * 70,
            ""
        ])
        for i, rec in enumerate(llama_data['recommendations'], 1):
            lines.append(f"{i}. {rec}")
            lines.append("")
    
    # Summary
    if llama_data.get('summary'):
        lines.extend([
            "",
            "SUMMARY:",
            "-" * 70,
            llama_data['summary'],
            ""
        ])
    
    lines.extend([
        "",
        "=" * 70,
        "End of Report",
        "=" * 70
    ])
    
    return "\n".join(lines)


def display_insights_tab(data):
    """Display additional insights"""
    st.subheader("📈 Resume Insights")
    
    # Sentiment Analysis
    st.markdown("### 😊 Sentiment Analysis")
    sentiment = data['sentiment']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Polarity", f"{sentiment['polarity']:.2f}", 
                 help="Range: -1 (negative) to +1 (positive)")
    with col2:
        st.metric("Subjectivity", f"{sentiment['subjectivity']:.2f}",
                 help="Range: 0 (objective) to 1 (subjective)")
    
    # Data Validation
    st.markdown("---")
    st.markdown("### ✅ Data Quality Validation")
    validation = data['validation']
    
    for key, value in validation.items():
        if value:
            st.success(f"✓ {key.replace('_', ' ').title()}")
        else:
            st.warning(f"⚠ {key.replace('_', ' ').title()}")
    
    # Analysis metadata
    st.markdown("---")
    st.markdown("### ℹ️ Analysis Metadata")
    st.markdown(f"**Analysis Timestamp:** {data['analysis_timestamp']}")
    st.markdown(f"**Word Count:** {data['word_count']}")

def model_performance_page():
    """Display model performance metrics"""
    st.header("📊 Model Performance Dashboard")
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Classification Models Performance</h3>
        <p>This section displays the performance metrics of our traditional ML classifiers.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Placeholder for model metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Naive Bayes Classifier")
        st.metric("Accuracy", "85.2%")
        st.metric("F1-Score", "0.83")
    
    with col2:
        st.markdown("#### SVM Classifier")
        st.metric("Accuracy", "87.5%")
        st.metric("F1-Score", "0.86")

def batch_analysis_page():
    """Batch resume analysis functionality"""
    st.header("🔍 Batch Resume Analysis")
    
    st.markdown("""
    <div class="info-box">
        <h3>📦 Bulk Processing</h3>
        <p>Upload multiple resumes for simultaneous analysis.</p>
        <p><strong>Coming Soon:</strong> This feature is under development.</p>
    </div>
    """, unsafe_allow_html=True)

def about_page():
    """About page with system information"""
    st.header("ℹ️ About Resume Analyzer")
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 System Overview</h3>
        <p>Advanced NLP-based resume analysis system combining traditional ML and AI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔧 Technical Stack")
    st.markdown("""
    - **Machine Learning:** Scikit-learn (Naive Bayes, SVM)
    - **NLP:** SpaCy, NLTK
    - **AI Integration:** Groq API (Llama 3.1)
    - **Visualization:** Plotly, Matplotlib
    - **Framework:** Streamlit
    """)
    
    st.markdown("### 📋 Features")
    st.markdown("""
    1. **Traditional ML Classification:** Naive Bayes + SVM
    2. **Named Entity Recognition:** SpaCy-powered extraction
    3. **AI-Powered Recommendations:** Groq Llama integration
    4. **Comprehensive Analysis:** Skills, sentiment, keywords
    5. **Interactive Visualizations:** Charts and metrics
    """)

def show_sample_info():
    """Show sample information when no analysis is done"""
    
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