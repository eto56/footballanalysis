import json
from typing import Dict, List, Tuple

class AnnotationManager:
    """
    各フレームごとの検出結果を (x, y, color) の配列で保存し、
    まとめて JSON に出力するためのクラス。
    """

    def __init__(self, image_x, image_y):
        # 例: frames[0] = [(100, 200, [255,0,0]), (120,220,[0,255,0]), ...]
        #     frames[1] = [(...] ... ]
        print ("init annotation manager")
        self.image_size = Tuple[float, float]
        self.image_size = (image_x, image_y)
        
        print (self.image_size)
        self.frames: Dict[int, List[Tuple[float, float, List[int]]]] = {}

    def add_detection(self, frame: int, x: float, y: float, color: Tuple[int, int, int], isplayer:bool) -> None:
        """
        1つの検出結果を「どのフレームか」をキーにして登録する。
        
        Parameters
        ----------
        frame : int
            フレーム番号
        x : float
            検出点の x 座標
        y : float
            検出点の y 座標
        color : (int, int, int)
            検出物体の色情報 (R, G, B)

        isball : bool
        """
        if frame not in self.frames:
            self.frames[frame] = []

        # color は JSON 化するとき配列になり、(x, y, [R, G, B]) の構造を持ちます
        # x,yは0-1の値に正規化する
        x = x / self.image_size[0]
        y = y / self.image_size[1]

        player = 0  
        if isplayer:
            player = 1

        detection_tuple = (float(x), float(y), [int(color[0]), int(color[1]), int(color[2])], player)
        print (detection_tuple)
        self.frames[frame].append(detection_tuple)

    def save_json(self, file_path: str) -> None:
        """
        フレームごとに蓄えた (x, y, color) のリストを JSON ファイルに保存する。
        
        Parameters
        ----------
        file_path : str
            出力先の JSON ファイルパス
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            # Python の辞書は {フレーム番号: [(x, y, [R,G,B]), ...], ...} の形
            # これをそのまま dump すると、下記のような JSON になる:
            # {
            #   "0": [
            #     [123.0, 456.0, [255, 0, 0],True],
            #     [200.0, 300.0, [0, 255, 0],False]
            #   ],
            #   "1": [
            #     ...
            #   ]
            # }
            json.dump(self.frames, f, indent=4, ensure_ascii=False)
        print(f"JSON file has been saved to {file_path}")
