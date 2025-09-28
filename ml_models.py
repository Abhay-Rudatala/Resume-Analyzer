import pandas as pd
import numpy as np
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional, Any

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeClassifier:
    """
    Complete Resume Classification System using Traditional ML Models
    """

    def __init__(self):
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.categories = []
        self.model_performance = {}

        # Initialize text preprocessing components
        self._init_text_preprocessing()
        logger.info("ResumeClassifier initialized successfully")

    def _init_text_preprocessing(self):
        """Initialize NLTK components for text preprocessing"""
        try:
            # Download required NLTK data
            nltk_downloads = ['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger']
            for download in nltk_downloads:
                try:
                    nltk.data.find(f'tokenizers/{download}')
                except LookupError:
                    nltk.download(download, quiet=True)

            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            logger.info("Text preprocessing components initialized")

        except Exception as e:
            logger.warning(f"NLTK initialization warning: {e}")
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            self.lemmatizer = None

    def preprocess_text(self, text: str) -> str:
        """
        Comprehensive text preprocessing
        """
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        # Remove phone numbers
        text = re.sub(r'[\+]?[1-9]?[0-9]{7,14}', '', text)

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenization and lemmatization
        if self.lemmatizer:
            try:
                tokens = word_tokenize(text)
                # Remove stopwords and lemmatize
                tokens = [
                    self.lemmatizer.lemmatize(token)
                    for token in tokens
                    if token not in self.stop_words and len(token) > 2
                ]
                text = ' '.join(tokens)
            except Exception as e:
                logger.warning(f"Advanced preprocessing failed: {e}")
                # Fallback to basic preprocessing
                words = text.split()
                text = ' '.join([word for word in words if word not in self.stop_words and len(word) > 2])

        return text

    def load_and_preprocess_data(self, data_path: str = 'resume_dataset.csv') -> Tuple[List[str], List[str]]:
        """
        Load and preprocess the resume dataset
        """
        try:
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            logger.info(f"Dataset loaded: {df.shape[0]} samples, {df['Category'].nunique()} categories")

            # Preprocess text data
            logger.info("Preprocessing text data...")
            preprocessed_texts = []
            for i, text in enumerate(df['Text'].values):
                if i % 500 == 0:
                    logger.info(f"Processed {i}/{len(df)} texts")
                processed_text = self.preprocess_text(str(text))
                preprocessed_texts.append(processed_text)

            # Store categories
            self.categories = list(df['Category'].unique())
            labels = df['Category'].values

            logger.info(f"Preprocessing completed. Categories: {len(self.categories)}")
            return preprocessed_texts, labels

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def train_models(self, data_path: str = 'resume_dataset.csv', test_size: float = 0.2):
        """
        Train both Naive Bayes and SVM models
        """
        logger.info("Starting model training...")

        # Load and preprocess data
        texts, labels = self.load_and_preprocess_data(data_path)

        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels_encoded, test_size=test_size, random_state=42,
            stratify=labels_encoded
        )

        logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        # Train TF-IDF + Naive Bayes
        self._train_naive_bayes(X_train, X_test, y_train, y_test)

        # Train TF-IDF + SVM
        self._train_svm(X_train, X_test, y_train, y_test)

        # Save models
        self._save_models()

        logger.info("Model training completed successfully!")

    def _train_naive_bayes(self, X_train, X_test, y_train, y_test):
        """
        Train Naive Bayes model with hyperparameter tuning
        """
        logger.info("Training Naive Bayes model...")

        # Create pipeline
        nb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                sublinear_tf=True
            )),
            ('nb', MultinomialNB())
        ])

        # Hyperparameter tuning
        param_grid = {
            'nb__alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
            'tfidf__max_features': [3000, 5000, 8000],
            'tfidf__ngram_range': [(1, 1), (1, 2)]
        }

        grid_search = GridSearchCV(
            nb_pipeline, param_grid,
            cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Best model
        self.models['naive_bayes'] = grid_search.best_estimator_

        # Evaluate
        nb_pred = self.models['naive_bayes'].predict(X_test)
        nb_accuracy = accuracy_score(y_test, nb_pred)

        self.model_performance['naive_bayes'] = {
            'accuracy': nb_accuracy,
            'best_params': grid_search.best_params_,
            'classification_report': classification_report(y_test, nb_pred,
                                                         target_names=self.categories,
                                                         output_dict=True)
        }

        logger.info(f"Naive Bayes - Accuracy: {nb_accuracy:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")

    def _train_svm(self, X_train, X_test, y_train, y_test):
        """
        Train SVM model with hyperparameter tuning
        """
        logger.info("Training SVM model...")

        # Create pipeline
        svm_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                sublinear_tf=True
            )),
            ('svm', SVC(probability=True, random_state=42))
        ])

        # Hyperparameter tuning (limited for faster training)
        param_grid = {
            'svm__C': [0.1, 1.0, 10.0],
            'svm__kernel': ['linear', 'rbf'],
            'tfidf__max_features': [3000, 5000]
        }

        grid_search = GridSearchCV(
            svm_pipeline, param_grid,
            cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Best model
        self.models['svm'] = grid_search.best_estimator_

        # Evaluate
        svm_pred = self.models['svm'].predict(X_test)
        svm_accuracy = accuracy_score(y_test, svm_pred)

        self.model_performance['svm'] = {
            'accuracy': svm_accuracy,
            'best_params': grid_search.best_params_,
            'classification_report': classification_report(y_test, svm_pred,
                                                         target_names=self.categories,
                                                         output_dict=True)
        }

        logger.info(f"SVM - Accuracy: {svm_accuracy:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")

    def _save_models(self):
        """
        Save trained models and components using pickle
        """
        os.makedirs('models', exist_ok=True)

        try:
            # Save models
            for model_name, model in self.models.items():
                model_path = f'models/{model_name}_model.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {model_name} model to {model_path}")

            # Save label encoder
            with open('models/label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)

            # Save categories and performance
            with open('models/categories.pkl', 'wb') as f:
                pickle.dump(self.categories, f)

            with open('models/performance.pkl', 'wb') as f:
                pickle.dump(self.model_performance, f)

            logger.info("All models and components saved successfully")

        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise

    def load_models(self):
        """
        Load previously trained models
        """
        try:
            # Load models
            model_files = ['naive_bayes_model.pkl', 'svm_model.pkl']
            for model_file in model_files:
                model_path = f'models/{model_file}'
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model_name = model_file.replace('_model.pkl', '')
                        self.models[model_name] = pickle.load(f)
                        logger.info(f"Loaded {model_name} model")

            # Load other components
            component_files = {
                'label_encoder.pkl': 'label_encoder',
                'categories.pkl': 'categories',
                'performance.pkl': 'model_performance'
            }

            for file_name, attr_name in component_files.items():
                file_path = f'models/{file_name}'
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        setattr(self, attr_name, pickle.load(f))
                        logger.info(f"Loaded {attr_name}")

            logger.info("All models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def classify_resume(self, resume_text: str) -> Dict[str, Any]:
        """
        Classify a resume using trained models
        """
        if not self.models:
            if not self.load_models():
                raise ValueError("No trained models available. Please train models first.")

        # Preprocess text
        processed_text = self.preprocess_text(resume_text)

        if not processed_text:
            return {
                'error': 'No valid text found after preprocessing'
            }

        results = {}

        # Classify with each model
        for model_name, model in self.models.items():
            try:
                # Predict
                prediction = model.predict([processed_text])[0]
                probabilities = model.predict_proba([processed_text])[0]

                # Get category name
                category = self.label_encoder.inverse_transform([prediction])[0]
                confidence = max(probabilities)

                # Get top 3 predictions
                top_indices = np.argsort(probabilities)[-3:][::-1]
                top_predictions = []
                for idx in top_indices:
                    cat = self.label_encoder.inverse_transform([idx])[0]
                    prob = probabilities[idx]
                    top_predictions.append({'category': cat, 'probability': float(prob)})

                results[model_name] = {
                    'category': category,
                    'confidence': float(confidence),
                    'score': min(float(confidence * 10), 10.0),
                    'top_predictions': top_predictions
                }

            except Exception as e:
                logger.error(f"Error classifying with {model_name}: {e}")
                results[model_name] = {
                    'error': str(e)
                }

        return results

    def get_model_performance(self) -> Dict[str, Any]:
        """
        Get detailed model performance metrics
        """
        if not self.model_performance:
            if os.path.exists('models/performance.pkl'):
                with open('models/performance.pkl', 'rb') as f:
                    self.model_performance = pickle.load(f)
            else:
                return {"error": "No performance metrics available"}

        return self.model_performance

    def predict_batch(self, texts: List[str], model_name: str = 'svm') -> List[Dict[str, Any]]:
        """
        Classify multiple resumes at once
        """
        if model_name not in self.models:
            if not self.load_models():
                raise ValueError("Models not available")

        results = []
        for text in texts:
            result = self.classify_resume(text)
            if model_name in result:
                results.append(result[model_name])
            else:
                results.append({'error': 'Classification failed'})

        return results

def train_models_main():
    """
    Main function to train and save models
    """
    classifier = ResumeClassifier()

    # Check if dataset exists
    if not os.path.exists('resume_dataset.csv'):
        logger.error("Dataset not found! Please ensure resume_dataset.csv exists.")
        return

    # Train models
    classifier.train_models()

    # Display performance
    performance = classifier.get_model_performance()
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETED!")
    print("="*50)

    for model_name, metrics in performance.items():
        print(f"\n{model_name.upper()}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        if 'best_params' in metrics:
            print(f"Best Parameters: {metrics['best_params']}")

    print("\n✅ Models saved in the 'models/' directory")
    print("✅ Ready to use for classification!")

if __name__ == "__main__":
    train_models_main()