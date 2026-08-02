"""Vision Router: Face recognition and Product classification endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request

try:
    from app.schemas import FaceRecognitionResponse, ProductClassificationResponse
    from app.services.cv_service import CVService
except ImportError:
    from ..schemas import FaceRecognitionResponse, ProductClassificationResponse
    from ..services.cv_service import CVService

router = APIRouter(tags=["Vision Services"])


def get_cv_service(request: Request) -> CVService:
    """Dependency helper to access shared CVService instance from app state."""
    cv_service = getattr(request.app.state, "cv_service", None)
    if cv_service is None:
        cv_service = CVService()
    return cv_service


@router.post(
    "/recognize-face",
    response_model=FaceRecognitionResponse,
    summary="Recognize customer face from uploaded image"
)
async def recognize_face(
    file: UploadFile = File(...),
    cv_service: CVService = Depends(get_cv_service)
):
    """Accepts an uploaded image file, detects faces, compares encodings against face database,
    and returns matched customer ID, recognition status, and confidence score.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        customer_id, status, confidence = cv_service.recognize_face(contents)
        return FaceRecognitionResponse(
            customer_id=customer_id,
            status=status,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image for face recognition: {str(e)}")


@router.post(
    "/classify-product",
    response_model=ProductClassificationResponse,
    summary="Classify retail product category from uploaded image"
)
async def classify_product(
    file: UploadFile = File(...),
    cv_service: CVService = Depends(get_cv_service)
):
    """Accepts an uploaded product image file, performs MobileNetV2 feature extraction & classification,
    and returns predicted product category and model confidence.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        category, confidence = cv_service.classify_product(contents)
        return ProductClassificationResponse(
            category=category,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying product image: {str(e)}")
