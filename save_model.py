import torch
import os
from app.models.model import SkinClassifierCNN

# 学習済みモデルを保存するためのスクリプト
def save_trained_model(model, output_dir="model_weights", filename="skin_classifier.pth"):
    """
    トレーニング済みモデルを保存
    
    Args:
        model: 学習済みのモデル
        output_dir: 保存先ディレクトリ
        filename: 保存するファイル名
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # トレーニング済みモデルの読み込み
    model = SkinClassifierCNN(num_classes=3).to(device)
    
    # ここでモデルをロードして保存
    # model.load_state_dict(torch.load('path_to_trained_model'))
    
    save_trained_model(model)
