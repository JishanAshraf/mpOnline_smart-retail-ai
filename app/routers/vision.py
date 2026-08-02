"""Vision Router: Face recognition and Product classification endpoints.

Includes input validation, file size safeguards (max 10MB), and secure error handling.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request

try:
    from app.schemas import FaceRecognitionResponse, ProductClassificationResponse
    from app.services.cv_service import CVService
except ImportError:
    from ..schemas import FaceRecognitionResponse, ProductClassificationResponse
    from ..services.cv_service import CVService

router = APIRouter(tags=["Vision Services"])

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB limit


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
    """Accepts an uploaded image file (max 10MB), detects faces, compares encodings
    against customer database, and returns matched customer ID, status, and confidence.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image (JPEG/PNG).")

    contents = await file.read()

    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image file size exceeds maximum 10MB limit.")

    try:
        customer_id, status, confidence = cv_service.recognize_face(contents)
        return FaceRecognitionResponse(
            customer_id=customer_id,
            status=status,
            confidence=confidence
        )
    except Exception as e:
        print(f"[VisionRouter] Face recognition error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error processing face recognition.")


@router.post(
    "/classify-product",
    response_model=ProductClassificationResponse,
    summary="Classify retail product category from uploaded image"
)
async def classify_product(
    file: UploadFile = File(...),
    cv_service: CVService = Depends(get_cv_service)
):
    """Accepts an uploaded product image file (max 10MB), performs MobileNetV2 feature extraction,
    and returns predicted product category and model confidence score.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image (JPEG/PNG).")

    contents = await file.read()

    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image file size exceeds maximum 10MB limit.")

    try:
        category, confidence = cv_service.classify_product(contents)
        return ProductClassificationResponse(
            category=category,
            confidence=confidence
        )
    except Exception as e:
        print(f"[VisionRouter] Product classification error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error classifying product image.")
