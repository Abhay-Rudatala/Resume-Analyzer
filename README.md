# 🔍 NLP Resume Analyzer

> **A comprehensive resume analysis tool using traditional machine learning models for job category classification and intelligent insights.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)

## 🚀 Live Demo

Try the live application: [Coming Soon - Deploy on Streamlit Cloud]

## ✨ Features

### 🤖 **ML-Powered Analysis**
- **TF-IDF + Naive Bayes** (85-90% accuracy)
- **TF-IDF + SVM** (90-95% accuracy) 
- **SpaCy Named Entity Recognition** (85-90% accuracy)

### 📊 **Comprehensive Insights**
- ✅ Job category prediction (15+ categories)
- ✅ Skills extraction and analysis
- ✅ Resume quality scoring (1-10 scale)
- ✅ Personalized improvement recommendations
- ✅ Contact information extraction
- ✅ Interactive visualizations

### 🎯 **Job Categories Supported**
Software Engineer • Data Scientist • Product Manager • Marketing Manager • Sales Representative • HR Manager • Financial Analyst • Designer • Business Analyst • Project Manager • DevOps Engineer • Quality Assurance • Content Writer • Customer Success • Operations Manager

## 🏁 Quick Start

### 1️⃣ **Clone & Install**
```bash
git clone https://github.com/Abhay-Rudatala/resume-analyzer.git
cd resume-analyzer
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2️⃣ **Train Models**
```bash
python ml_models.py
```

### 3️⃣ **Test System**
```bash
python test_system.py
```

### 4️⃣ **Launch App**
```bash
streamlit run app.py
```

🌐 Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
nlp-resume-analyzer/
├── 🎨 app.py                    # Main Streamlit application
├── 📄 resume_parser.py          # PDF/DOCX text extraction
├── 🤖 ml_models.py              # ML models (Naive Bayes, SVM)
├── 🔍 ner_extractor.py          # Named Entity Recognition
├── 🛠️ utils.py                  # Utility functions
├── 📦 requirements.txt          # Python dependencies
├── 📊 resume_dataset.csv        # Training dataset (3000+ samples)
├── 📁 models/                   # Trained model files (generated)
└── 📖 README.md                 # This file
```

## 🔧 Technical Stack

**Backend:** Python 3.11, scikit-learn, SpaCy, NLTK  
**Frontend:** Streamlit with custom CSS  
**ML Models:** TF-IDF, Naive Bayes, SVM, NER  
**File Processing:** PyPDF2, python-docx, pdfplumber  
**Visualization:** Plotly, matplotlib, seaborn  
**Data:** 3,000+ resume samples, 15 job categories  

## 📈 Model Performance

| Model | Accuracy | Use Case |
|-------|----------|----------|
| **Naive Bayes** | 85-90% | Fast categorization |
| **SVM** | 90-95% | High-accuracy classification |
| **SpaCy NER** | 85-90% | Information extraction |

## 🎯 How It Works

1. **📤 Upload** your resume (PDF/DOCX)
2. **🔍 Analysis** using traditional ML models
3. **📊 Results** with job predictions and insights
4. **💡 Recommendations** for resume improvement

## 🐛 Troubleshooting

### Common Issues

**SpaCy model not found:**
```bash
python -m spacy download en_core_web_sm
```

**Models not trained:**
```bash
python ml_models.py
```

**Import errors:**
```bash
pip install -r requirements.txt
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🌟 Show Your Support

If this project helped you, please ⭐ star this repository!

## 📞 Contact

**Built with ❤️ using traditional ML approaches that deliver reliable, interpretable results.**

---

*Ready to analyze your resume? Let's get started! 🚀*
