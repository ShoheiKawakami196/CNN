from PIL import Image
import io

def read_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    バイト形式の画像データをPIL Imageに変換
    
    Args:
        image_bytes: 画像データのバイト列
        
    Returns:
        PIL Image オブジェクト
    """
    return Image.open(io.BytesIO(image_bytes))
