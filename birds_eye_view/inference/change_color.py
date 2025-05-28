import numpy as np
from PIL import Image

def replace_black_with_green(image, threshold=30):
    """
    画像内の黒い部分を緑色に置き換える関数です。
    
    Parameters:
        image (PIL.Image.Image または numpy.ndarray):
            入力画像。PIL Image または (H, W, 3) の NumPy 配列。
        threshold (int):
            黒と判定するための閾値です。各 RGB チャンネルがこの値以下なら黒とみなします。
            デフォルトは 30 です。
            
    Returns:
        PIL.Image.Image:
            黒い部分が緑色に置き換えられた画像。
    """
    # PIL Image なら NumPy 配列に変換
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # もし画像が RGBA なら RGB の部分のみ使用
    if image.shape[-1] == 4:
        image = image[..., :3]
    
    # 黒の部分を判定するマスクを作成
    # 各チャンネルが threshold 以下であれば黒と判定
    black_mask = (image[..., 0] <= threshold) & (image[..., 1] <= threshold) & (image[..., 2] <= threshold)
    
    # 入力画像のコピーを作成
    result = image.copy()
    
    # 黒い部分を緑 (0, 255, 0) に置き換え
    result[black_mask] = [0, 255/2, 0]
    
    # NumPy 配列を PIL Image に戻して返す
    return Image.fromarray(result)

# 使用例
if __name__ == "__main__":
    # 画像の読み込み
    img = Image.open("black.jpg")
    
    # 黒い部分を緑に置き換えた画像を作成
    new_img = replace_black_with_green(img, threshold=30)
    
    # 結果を保存または表示
    new_img.save("green.jpg")
    
