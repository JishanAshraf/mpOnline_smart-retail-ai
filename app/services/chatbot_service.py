"""Chatbot Service for customer intelligence FAQ and support inquiries.

Implements hybrid approach: rule-based intent matching first (from data/intents.json),
falling back to ML intent classifier (app/models/chatbot_model.pkl) if no rule matches.
"""

import os
import json
import re
import pickle
import random
from typing import Tuple, List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

INTENTS_FILE_PATH = os.path.join(DATA_DIR, "intents.json")
CHATBOT_MODEL_PATH = os.path.join(MODEL_DIR, "chatbot_model.pkl")


class ChatbotService:
    def __init__(self):
        self.intents: List[Dict[str, Any]] = []
        self.ml_model = None
        self.ml_vectorizer = None
        self.load_intents()
        self.load_ml_model()

    def load_intents(self) -> None:
        """Load retail FAQ intents from data/intents.json."""
        if os.path.exists(INTENTS_FILE_PATH):
            try:
                with open(INTENTS_FILE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.intents = data.get("intents", [])
                print(f"[ChatbotService] Loaded {len(self.intents)} intents from {INTENTS_FILE_PATH}")
            except Exception as e:
                print(f"[ChatbotService] Warning loading intents.json: {e}")
                self.intents = []
        else:
            print(f"[ChatbotService] Notice: {INTENTS_FILE_PATH} not found.")
            self.intents = []

    def load_ml_model(self) -> None:
        """Load ML intent classification model from pickle if available."""
        if os.path.exists(CHATBOT_MODEL_PATH):
            try:
                with open(CHATBOT_MODEL_PATH, "rb") as f:
                    saved_dict = pickle.load(f)
                    if isinstance(saved_dict, dict):
                        self.ml_model = saved_dict.get("model")
                        self.ml_vectorizer = saved_dict.get("vectorizer")
                    else:
                        self.ml_model = saved_dict
                print(f"[ChatbotService] Loaded ML chatbot model from {CHATBOT_MODEL_PATH}")
            except Exception as e:
                print(f"[ChatbotService] Warning loading chatbot model: {e}")
                self.ml_model = None
        else:
            print(f"[ChatbotService] Notice: {CHATBOT_MODEL_PATH} not found. Using rule-based matching.")
            self.ml_model = None

    def match_rule_based(self, message: str) -> Optional[Tuple[str, str]]:
        """Match user message against rule patterns in intents.json.

        Returns:
            Optional[Tuple[response_text, intent_tag]]
        """
        if not message or not self.intents:
            return None

        clean_msg = message.lower().strip()

        for intent_item in self.intents:
            tag = intent_item.get("intent", intent_item.get("tag", "unknown"))
            patterns = intent_item.get("patterns", [])
            responses = intent_item.get("responses", [])

            for pattern in patterns:
                # Regex word boundary search or simple substring match
                pattern_regex = r'\b' + re.escape(pattern.lower()) + r'\b'
                if re.search(pattern_regex, clean_msg) or pattern.lower() in clean_msg:
                    response_text = random.choice(responses) if responses else "How can I help you with that?"
                    return response_text, tag

        return None

    def predict_ml_intent(self, message: str) -> Optional[Tuple[str, str]]:
        """Predict intent using ML model fallback if rule matching misses."""
        if self.ml_model is not None and self.ml_vectorizer is not None:
            try:
                vec = self.ml_vectorizer.transform([message])
                predicted_tag = str(self.ml_model.predict(vec)[0])

                # Find response corresponding to predicted tag
                for intent_item in self.intents:
                    tag = intent_item.get("intent", intent_item.get("tag"))
                    if tag == predicted_tag:
                        responses = intent_item.get("responses", [])
                        resp = random.choice(responses) if responses else "Thank you for contacting customer service."
                        return resp, tag
            except Exception as e:
                print(f"[ChatbotService] ML prediction error: {e}")

        return None

    def get_response(self, message: str) -> Tuple[str, str, str]:
        """Main method: Try rule-based first, fall back to ML model, then default fallback.

        Returns:
            Tuple[reply_message, intent_tag, match_type ("rule_based"|"ml_classifier"|"fallback")]
        """
        # 1. Rule-based matching
        rule_result = self.match_rule_based(message)
        if rule_result:
            reply, intent = rule_result
            return reply, intent, "rule_based"

        # 2. ML Classifier fallback
        ml_result = self.predict_ml_intent(message)
        if ml_result:
            reply, intent = ml_result
            return reply, intent, "ml_classifier"

        # 3. Default fallback
        fallback_reply = (
            "I'm sorry, I didn't quite understand that. "
            "You can ask me about order status, return policies, store hours, or shipping!"
        )
        return fallback_reply, "general_fallback", "fallback"
