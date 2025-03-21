from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.prediction_service import get_prediction
from app.schemas.prediction import PredictionResponse
import io
from PIL import Image

router = APIRouter(
    prefix="/api",
    tags=["prediction"]
)

@router.post("/predict", response_model=PredictionResponse)
async def predict_skin_type(file: UploadFile = File(...)):
    """
    Upload an image to classify the skin type.
    Returns the predicted skin type and confidence scores.
    """
    # ファイルが画像かチェック
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # 画像ファイルの読み込み
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # 予測を取得
    prediction_result = get_prediction(image)
    
    return prediction_result
