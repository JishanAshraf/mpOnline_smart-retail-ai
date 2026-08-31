"""NLP Service for text preprocessing and sentiment analysis inference.

Implements NLTK-based text cleaning (lowercasing, punctuation removal,
stopword removal, lemmatization) and sentiment model pickle loading/inference.
"""

import os
import re
import string
import pickle
from typing import Tuple, Optional, Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Base path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")


class NLPService:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._init_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.load_model()

    def _init_nltk_resources(self) -> None:
        """Initialize optional NLTK resources without blocking startup.

        Attempting to download corpora during app bootstrap can hang in offline or
        restricted environments. We prefer a safe built-in fallback set and only use
        NLTK features when the data is already available locally.
        """
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = {"a", "an", "the", "in", "on", "at", "and", "or", "is", "it", "to"}

        # Explicitly avoid network access during startup. The tokenizer and lemmatizer
        # already fall back to simple string splitting and best-effort lemmatization when
        # additional NLTK resources are unavailable.

    def load_model(self) -> None:
        """Load trained sentiment model & vectorizer from pickle if available."""
        if os.path.exists(SENTIMENT_MODEL_PATH):
            try:
                with open(SENTIMENT_MODEL_PATH, "rb") as f:
                    saved_dict = pickle.load(f)
                    if isinstance(saved_dict, dict):
                        self.model = saved_dict.get("model")
                        self.vectorizer = saved_dict.get("vectorizer")
                    else:
                        self.model = saved_dict
                print(f"[NLPService] Loaded sentiment model from {SENTIMENT_MODEL_PATH}")
            except Exception as e:
                print(f"[NLPService] Warning: Could not load sentiment model ({e}). Running in stub mode.")
                self.model = None
                self.vectorizer = None
        else:
            print(f"[NLPService] Notice: {SENTIMENT_MODEL_PATH} not found. Running sentiment analysis in stub mode.")
            self.model = None
            self.vectorizer = None

    def clean_text(self, text: str) -> str:
        """Clean raw text: lowercase, remove punctuation/numbers, remove stopwords, lemmatize."""
        if not text:
            return ""

        # 1. Lowercase
        text = text.lower()

        # 2. Remove punctuation and numbers
        text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub(r'\d+', '', text)

        # 3. Tokenize
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()

        # 4. Remove stopwords and lemmatize
        cleaned_tokens = []
        for word in tokens:
            if word not in self.stop_words and len(word) > 1:
                try:
                    lemmatized = self.lemmatizer.lemmatize(word)
                except Exception:
                    lemmatized = word
                cleaned_tokens.append(lemmatized)

        return " ".join(cleaned_tokens)

    def analyze_sentiment(self, text: str) -> Tuple[str, str, float]:
        """Analyze sentiment for input text string.

        Returns:
            Tuple[original_text, sentiment_label ("positive"|"negative"|"neutral"), confidence]
        """
        cleaned = self.clean_text(text)

        # Use trained ML model if loaded
        if self.model is not None and self.vectorizer is not None:
            try:
                vec = self.vectorizer.transform([cleaned])
                pred_label = self.model.predict(vec)[0]
                probs = self.model.predict_proba(vec)[0]
                confidence = float(max(probs))
                return text, str(pred_label), round(confidence, 2)
            except Exception as e:
                print(f"[NLPService] Sentiment prediction error: {e}")

        # Rule-based fallback heuristic for stubs before model training
        positive_keywords = {"love", "great", "excellent", "fast", "good", "amazing", "happy", "best", "satisfied"}
        negative_keywords = {"bad", "terrible", "slow", "broken", "poor", "horrible", "delay", "disappointed", "worst"}

        words = set(cleaned.split())
        pos_score = len(words.intersection(positive_keywords))
        neg_score = len(words.intersection(negative_keywords))

        if pos_score > neg_score:
            return text, "positive", 0.88
        elif neg_score > pos_score:
            return text, "negative", 0.85
        else:
            return text, "neutral", 0.75
