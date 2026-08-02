"""NLP Router: Customer review sentiment analysis endpoint.
"""

from fastapi import APIRouter, Depends, Request, HTTPException

try:
    from app.schemas import SentimentRequest, SentimentResponse
    from app.services.nlp_service import NLPService
except ImportError:
    from ..schemas import SentimentRequest, SentimentResponse
    from ..services.nlp_service import NLPService

router = APIRouter(tags=["NLP & Sentiment Services"])


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
    """Accepts review text, performs NLTK cleaning (lowercasing, stopword removal, lemmatization),
    runs sentiment model inference, and returns sentiment classification and confidence score.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        text, sentiment, confidence = nlp_service.analyze_sentiment(payload.text)
        return SentimentResponse(
            text=text,
            sentiment=sentiment,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")
