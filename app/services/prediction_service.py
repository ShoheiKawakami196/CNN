import torch
from torchvision import transforms
from app.models.model import SkinClassifierCNN
from PIL import Image
import os
from app.schemas.prediction import PredictionResponse

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルのロード
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                         "model_weights", "skin_classifier.pth")
model = SkinClassifierCNN(num_classes=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# クラス名（データセットに合わせて調整）
class_names = ["dry", "normal", "oily"]

# 画像変換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_prediction(image: Image.Image) -> PredictionResponse:
    """
    画像を受け取り、肌タイプの予測を返す
    
    Args:
        image: PIL Image オブジェクト
        
    Returns:
        PredictionResponse: 予測結果とスコア
    """
    # 画像の前処理
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 予測
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # 予測結果の取得
    predicted_class_idx = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_class_idx]
    
    # 各クラスの確率
    confidence_scores = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    return PredictionResponse(
        predicted_class=predicted_class,
        confidence_scores=confidence_scores
    )
