# Project Proposal: AI-Powered Smart Retail & Customer Intelligence Platform

**Author:** Jishan Ashraf  
**Registration No.:** 23MEI10015  
**Enrollment No.:** IN26013868  
**Institution:** VIT Bhopal University  
**Course:** Machine Learning & Artificial Intelligence  

---

## 2. Project Overview & Problem Statement
Traditional physical retail stores struggle to capture real-time customer analytics, provide automated personalized VIP check-ins, assess customer feedback sentiment dynamically, and offer 24/7 instant support. 

This project delivers an enterprise-grade, microservice-ready platform that integrates **Computer Vision**, **Natural Language Processing (NLP)**, and **Machine Learning Chatbot Pipelines** into a unified **FastAPI** backend with an interactive web intelligence dashboard.

---

## 3. Key Features & API Specification

| Module | Endpoint | Method | Technology Stack | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Facial Recognition** | `/recognize-face` | `POST` | OpenCV, dlib, `face_recognition` | Detects facial features and matches customer encodings against registered member database (`face_db.pkl`). |
| **Product Classifier** | `/classify-product` | `POST` | MobileNetV2 (TensorFlow/Keras) | Preprocesses item images (224x224) and classifies retail product categories (`product_classifier.h5`). |
| **Sentiment Analysis** | `/analyze-sentiment` | `POST` | NLTK, TF-IDF, Scikit-Learn | Cleans review text (lowercasing, stopword removal, lemmatization) and classifies sentiment (`positive`/`negative`/`neutral`). |
| **Hybrid Chatbot** | `/chatbot` | `POST` | JSON Intents, Scikit-Learn ML | Rule-based intent matching from `intents.json` with fallback to ML intent classifier (`chatbot_model.pkl`). |
| **Executive Dashboard** | `/dashboard/stats` | `GET` | FastAPI, Pydantic | Returns aggregate store visit counts, review sentiment breakdown, and top inquiry topics for store analytics. |

---

## 4. Proposed Technical Architecture

```
                               ┌──────────────────────────────────────────────┐
                               │       Client Web Dashboard & Swagger UI       │
                               └──────────────────────┬───────────────────────┘
                                                      │ HTTP Requests
                                                      ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                       FastAPI Core Backend Application                                 │
 ├──────────────────────────────┬──────────────────────────────┬──────────────────────────────────────────┤
 │      Vision Router           │          NLP Router          │              Chatbot Router              │
 │  • /recognize-face           │     • /analyze-sentiment     │      • /chatbot                          │
 │  • /classify-product         │                              │      • /dashboard/stats                  │
 ├──────────────────────────────┼──────────────────────────────┼──────────────────────────────────────────┤
 │    CVService (OpenCV, dlib)  │   NLPService (NLTK, TF-IDF) │ ChatbotService (Hybrid Intents & ML)     │
 └──────────────┬───────────────┴──────────────┬───────────────┴────────────────────┬─────────────────────┘
                │                              │                                    │
                ▼                              ▼                                    ▼
 ┌──────────────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────────────────────────┐
 │     Model Artifacts          │ │     Model Artifacts          │ │     Model Artifacts                  │
 │  • face_db.pkl               │ │  • sentiment_model.pkl       │ │  • chatbot_model.pkl                 │
 │  • product_classifier.h5     │ │  • reviews.csv               │ │  • intents.json                      │
 └──────────────────────────────┘ └──────────────────────────────┘ └──────────────────────────────────────┘
```

---

## 5. Software & Technology Stack

- **Core Framework**: Python 3.10, FastAPI, Uvicorn, Pydantic V2.
- **Computer Vision**: OpenCV (preprocessing, edge detection, resizing), `face_recognition` (dlib 128-d facial embeddings), TensorFlow / Keras (MobileNetV2 transfer learning).
- **Natural Language Processing**: NLTK (WordNet lemmatizer, tokenization, stopword filtering), Scikit-Learn (TF-IDF vectorizer, Logistic Regression / Naive Bayes).
- **Frontend Dashboard**: Single-Page Web Dashboard (HTML5, CSS3 Glassmorphism, Vanilla JS Fetch API), Swagger UI OpenAPI at `/docs`.
- **Testing & Quality Assurance**: Pytest (`TestClient`), Flake8 code linting.
- **Containerization & CI/CD**: Docker (`python:3.10-slim` base image), GitHub Actions pipeline (Lint → Test → Docker Build).

---

## 6. Project Directory Structure

```
smart-retail-ai/
├── app/
│   ├── main.py                          # FastAPI entrypoint (loads model pipelines ONCE at startup)
│   ├── schemas.py                       # Pydantic request & response models
│   ├── routers/
│   │   ├── vision.py                    # /recognize-face, /classify-product
│   │   ├── nlp.py                       # /analyze-sentiment
│   │   └── chatbot.py                   # /chatbot, /dashboard/stats
│   ├── models/                          # Model artifacts directory (.gitkeep)
│   ├── services/
│   │   ├── cv_service.py                # OpenCV + dlib + MobileNetV2 service
│   │   ├── nlp_service.py               # NLTK text processing + sentiment inference
│   │   └── chatbot_service.py           # Rule-based intents + ML classifier fallback
│   └── static/                          # Web Dashboard UI (index.html, style.css, app.js)
├── notebooks/
│   ├── 01_image_classifier_training.ipynb   # MobileNetV2 model training pipeline
│   ├── 02_face_recognition_setup.ipynb      # Face encodings & face_db setup
│   └── 03_sentiment_model_training.ipynb    # Sentiment model & vectorizer training
├── data/
│   ├── reviews.csv                      # Starter retail reviews dataset (20 rows)
│   └── intents.json                     # 25 Retail FAQ intents with patterns & responses
├── tests/
│   └── test_endpoints.py                # Pytest TestClient test suite (6/6 passing)
├── Dockerfile                           # Production Docker image configuration
├── requirements.txt                     # Pinned project dependencies
├── README.md                            # Complete setup & API documentation
└── .github/workflows/deploy.yml         # GitHub Actions CI/CD workflow
```

---

## 7. Model Training & Implementation Roadmap

1. **Phase 1: Environment & API Scaffolding (Completed)**
   - FastAPI server initialization with models loaded ONCE at startup via lifespan context manager.
   - Implementation of Pydantic schemas, routes, and robust fallback service stubs.
   - Comprehensive test suite (100% pass rate) and interactive Web UI Dashboard.

2. **Phase 2: Model Training via Notebooks**
   - **`01_image_classifier_training.ipynb`**: Train MobileNetV2 model on retail product image dataset and export `product_classifier.h5`.
   - **`02_face_recognition_setup.ipynb`**: Process registered customer photos, extract 128-d encodings, and export `face_db.pkl`.
   - **`03_sentiment_model_training.ipynb`**: Clean `reviews.csv` with NLTK, fit TF-IDF + Logistic Regression classifier, and export `sentiment_model.pkl`.

3. **Phase 3: Deployment & Evaluation**
   - Docker container build and local testing.
   - GitHub Actions automated CI/CD pipeline integration.
