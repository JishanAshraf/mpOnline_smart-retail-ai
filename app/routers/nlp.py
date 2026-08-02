"""NLP Router: Customer review sentiment analysis endpoint.

Includes input text validation, payload length safeguards, and secure error handling.
"""

from fastapi import APIRouter, Depends, Request, HTTPException

try:
    from app.schemas import SentimentRequest, SentimentResponse
    from app.services.nlp_service import NLPService
except ImportError:
    from ..schemas import SentimentRequest, SentimentResponse
    from ..services.nlp_service import NLPService

router = APIRouter(tags=["NLP & Sentiment Services"])

MAX_TEXT_LENGTH = 5000  # 5,000 character limit per review


def get_nlp_service(request: Request) -> NLPService:
    """Dependency helper to access shared NLPService instance from app state."""
    nlp_service = getattr(request.app.state, "nlp_service", None)
    if nlp_service is None:
        nlp_service = NLPService()
    return nlp_service


@router.post(
    "/analyze-sentiment",
    response_model=SentimentResponse,
    summary="Analyze text sentiment for customer reviews"
)
async def analyze_sentiment(
    payload: SentimentRequest,
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """Accepts review text (max 5,000 chars), performs NLTK cleaning (lowercasing, stopword removal,
    lemmatization), runs sentiment model inference, and returns sentiment classification.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text string cannot be empty.")

    if len(payload.text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Text length exceeds maximum limit of {MAX_TEXT_LENGTH} characters.")

    try:
        text, sentiment, confidence = nlp_service.analyze_sentiment(payload.text)
        return SentimentResponse(
            text=text,
            sentiment=sentiment,
            confidence=confidence
        )
    except Exception as e:
        print(f"[NLPRouter] Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error analyzing review sentiment.")
