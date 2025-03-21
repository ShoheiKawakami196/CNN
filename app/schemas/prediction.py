from pydantic import BaseModel
from typing import Dict

class PredictionResponse(BaseModel):
    """予測結果のレスポンスモデル"""
    predicted_class: str
    confidence_scores: Dict[str, float]
