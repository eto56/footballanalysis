from elements.yolo import YOLO
from elements.deep_sort import DEEPSORT
from elements.perspective_transform import Perspective_Transform
from elements.assets import transform_matrix, detect_color, detect_color_remake
from elements.assets import  player_manager,player_color_manager
from elements.assets import player_color_manager
#from elements.assets import detect_color, detect_color_remake
#from elements.detect_color import classify_team_colors
from arguments import Arguments
from yolov5.utils.plots import plot_one_box

from annotation import AnnotationManager

import torch
import os
import cv2
import numpy as np
import sys


 


max_frames = 100

def main(opt):
    # Load models
    detector = YOLO(opt.yolov5_model, opt.conf_thresh, opt.iou_thresh)
    deep_sort = DEEPSORT(opt.deepsort_config)
    perspective_transform = Perspective_Transform()

    # Video capture
    cap = cv2.VideoCapture(opt.source)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    

    # # Save output
    # if opt.save:
    #     output_name = opt.source.split('/')[-1]
    #     output_name = output_name.split('.')[0] + '_output.' + output_name.split('.')[-1]
        
    #     output_path = os.path.join(os.getcwd(), 'inference/output')
    #     os.makedirs(output_path, exist_ok=True)
    #     output_name = os.path.join(os.getcwd(), 'inference/output', output_name)

    #     w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #     h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    #     out = cv2.VideoWriter(output_name,  
    #                             cv2.VideoWriter_fourcc(*'mp4v'), 
    #                             opt.outputfps, (int(w), int(h)))
 

    frame_num = 0

    # Black Image (Soccer Field)
    bg_ratio = int(np.ceil(w/(3*115)))
    gt_img = cv2.imread('./inference/green.jpg')
    gt_img = cv2.resize(gt_img,(115*bg_ratio, 74*bg_ratio))
    gt_h, gt_w, _ = gt_img.shape

    AM = AnnotationManager(115*bg_ratio, 74*bg_ratio)
    pcm = player_color_manager()


    ##first loop

    while(cap.isOpened()):
        
        ret, frame = cap.read()
        bg_img = gt_img.copy()

        if ret:
            main_frame = frame.copy()
            yoloOutput = detector.detect(frame)
            print (len(yoloOutput))

            # Output: Homography Matrix and Warped image 
            if frame_num % 5 ==0: # Calculate the homography matrix every 5 frames
                M, warped_image = perspective_transform.homography_matrix(main_frame)

            
            

           

            if yoloOutput:

                # Tracking
                deep_sort_output = deep_sort.detection_to_deepsort(yoloOutput, frame)
                # 返り値が None ではなく、少なくとも 2 次元配列にする
                if deep_sort_output is None:
                    print("deep_sort_output is None! Setting to empty array.")
                    deep_sort_output = np.empty((0, 5), dtype=int)
                else:
                    deep_sort_output = np.array(deep_sort_output)
                    # deep_sort_output が 0-d（スカラー）の場合、最低でも 2-d に変換
                    if deep_sort_output.ndim == 0:
                        deep_sort_output = np.atleast_2d(deep_sort_output)
                    # 1-d で要素数が 5 個の場合、1 件の bbox として扱う（(1,5) に変換）
                    elif deep_sort_output.ndim == 1 and deep_sort_output.shape[0] == 5:
                        deep_sort_output = np.expand_dims(deep_sort_output, axis=0)
                

 
                # print (deep_sort_output)
                # print (deep_sort_output.shape)
                pm = player_manager()
                # The homography matrix is applied to the center of the lower side of the bbox.
                for i, obj in enumerate(yoloOutput):
                    xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
                    x_center = (xyxy[0] + xyxy[2])/2 
                    y_center = xyxy[3]
                    id = 0
                    min_dist = 1e8
                    print ("deep",deep_sort_output)
                     
                    for box in deep_sort_output:

                        dist = (box[0] - x_center)**2 + (box[1] - y_center)**2 + (box[2] - xyxy[2])**2 + (box[3] - xyxy[3])**2
                        if dist < min_dist:
                            min_dist = dist
                            id = box[4]
                             
                    print("!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(id)
                    

                    
                    if obj['label'] == 'player':
                        coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                         
                        #  color = detect_color_remake(main_frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])

                        #cv2.circle(bg_img, coords, bg_ratio + 1, color, -1)
                        x= float(coords[0])
                        y= float(coords[1])
                        color = (0, 0, 255)
                        detection_frame = main_frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                        pm.add_player(id,detection_frame,coords, color, True, frame_num)
                            
                        

                        #  circles_list.append(circle_info(coords, color, True))
                         #   AM.add_detection(frame_num, x, y, color,True)
                        

                         
                    elif obj['label'] == 'ball':
                        coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                        #cv2.circle(bg_img, coords, bg_ratio + 1, (102, 0, 102), -1)
                        #AM.add_detection(frame_num, coords[0], coords[1], (102, 0, 102), False)

                        detection_frame = main_frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                        pm.add_player(id,detection_frame,coords, (102, 0, 102), False, frame_num)

                        #plot_one_box(xyxy, frame, (102, 0, 102), label="ball")
                        x= float(coords[0])
                        y= float(coords[1])
                         
                    
               
                pm.classify_colors()
                get_players = pm.get_players()
                for player in get_players:
                    print ("pcm id :",player.id)
                    pcm.add_color(player.id, player.color)
            
        else:
                break


    frame_num =0
    cap.release()
    cv2.destroyAllWindows()

    # Video capture
    cap = cv2.VideoCapture(opt.source)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    

    # Save output
    if opt.save:
        output_name = opt.source.split('/')[-1]
        output_name = output_name.split('.')[0] + '_output.' + output_name.split('.')[-1]
        
        output_path = os.path.join(os.getcwd(), 'inference/output')
        os.makedirs(output_path, exist_ok=True)
        output_name = os.path.join(os.getcwd(), 'inference/output', output_name)

        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        out = cv2.VideoWriter(output_name,  
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                opt.outputfps, (int(w), int(h)))

                    
               




    print("first loop done")

    pcm.calculate_final_colors()
    deep_sort = DEEPSORT(opt.deepsort_config)
     
    ##second loop

    while(cap.isOpened()):
        
        ret, frame = cap.read()
        bg_img = gt_img.copy()

        if ret:
            main_frame = frame.copy()
            yoloOutput = detector.detect(frame)
 
            # Output: Homography Matrix and Warped image 
            if frame_num % 5 ==0: # Calculate the homography matrix every 5 frames
                M, warped_image = perspective_transform.homography_matrix(main_frame)

            
            

           

            if yoloOutput:

                # Tracking
                deep_sort_output = deep_sort.detection_to_deepsort(yoloOutput, frame)
                # 返り値が None ではなく、少なくとも 2 次元配列にする
                if deep_sort_output is None:
                    print("deep_sort_output is None! Setting to empty array.")
                    deep_sort_output = np.empty((0, 5), dtype=int)
                else:
                    deep_sort_output = np.array(deep_sort_output)
                    # deep_sort_output が 0-d（スカラー）の場合、最低でも 2-d に変換
                    if deep_sort_output.ndim == 0:
                        deep_sort_output = np.atleast_2d(deep_sort_output)
                    # 1-d で要素数が 5 個の場合、1 件の bbox として扱う（(1,5) に変換）
                    elif deep_sort_output.ndim == 1 and deep_sort_output.shape[0] == 5:
                        deep_sort_output = np.expand_dims(deep_sort_output, axis=0)
                

 
                # print (deep_sort_output)
                # print (deep_sort_output.shape)
                print ("max:" ,pcm.id_max)
                pm = player_manager()
                cnt=0
                # The homography matrix is applied to the center of the lower side of the bbox.
                for i, obj in enumerate(yoloOutput):
                    xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
                    x_center = (xyxy[0] + xyxy[2])/2 
                    y_center = xyxy[3]
                    id = 0
                    min_dist = 1e8
                    print ("deep",deep_sort_output)

                    cnt+=1
                    print ("cnt:",cnt)
                    for box in deep_sort_output:

                        dist = (box[0] - x_center)**2 + (box[1] - y_center)**2 + (box[2] - xyxy[2])**2 + (box[3] - xyxy[3])**2
                        if dist < min_dist:
                            min_dist = dist
                            id = box[4]
                             
                    print("!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("id:",id)
                    # if id ==0 :
                    #     continue
                    

                    
                    if obj['label'] == 'player':
                        coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                        # try:
                          #  color = detect_color_remake(main_frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])

                        #cv2.circle(bg_img, coords, bg_ratio + 1, color, -1)
                        x= float(coords[0])
                        y= float(coords[1])
                        color = pcm.get_colors(id)
                        detection_frame = main_frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                        pm.add_player(id,detection_frame,coords, color, True, frame_num)
                          


                          #  circles_list.append(circle_info(coords, color, True))
                         #   AM.add_detection(frame_num, x, y, color,True)
                        

                        # except:
                        #   print ("error")
                        #   pass
                    elif obj['label'] == 'ball':
                        coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                        #cv2.circle(bg_img, coords, bg_ratio + 1, (102, 0, 102), -1)
                        #AM.add_detection(frame_num, coords[0], coords[1], (102, 0, 102), False)

                        detection_frame = main_frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                        pm.add_player(id,detection_frame,coords, (102, 0, 102), False, frame_num)

                        plot_one_box(xyxy, frame, (102, 0, 102), label="ball")
                        x= float(coords[0])
                        y= float(coords[1])
                         
                    
               
                
                pm.draw_circles(bg_img, bg_ratio)
                players = pm.get_players()

                for player in players:
                    x = float(player.coords[0])
                    y = float(player.coords[1])

                    AM.add_detection(player.frame_num, x, y, player.color, player.is_player)
                       
                    

                
                #save bg_img
                #cv2.imwrite('bg_img_{}.jpg'.format(frame_num), bg_img)

                
            else:
                deep_sort.deepsort.increment_ages()

            frame[frame.shape[0]-bg_img.shape[0]:, frame.shape[1]-bg_img.shape[1]:] = bg_img  
            
            if opt.view:
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & ord('q') == 0xFF:
                    break
            

            # Saving the output
            if opt.save:
                out.write(frame)
                print(f'\n\nOutput video saved at {output_name}')


            frame_num += 1

            if frame_num == max_frames:
                break
             
        else:
            break

        sys.stdout.write(
            "\r[Input Video : %s] [%d/%d Frames Processed]"
            % (
                opt.source,
                frame_num,
                frame_count,
            )
        )

    if opt.save:
        
        

        output_path = "../soccerAR/data/annotation.json"
        print(f'\n\nOutput json file saved at {output_path}')
        AM.save_json(output_path)

        out_video_path = "../soccerAR/data/output.mp4"
 
     

    

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    opt = Arguments().parse()
    with torch.no_grad():
        main(opt)

#  import json
# from typing import Dict, List, Tuple

# class AnnotationManager:
#     """
#     各フレームごとの検出結果を (x, y, color) の配列で保存し、
#     まとめて JSON に出力するためのクラス。
#     """

#     def __init__(self, image_size: Tuple[int, int]):
#         # 例: frames[0] = [(100, 200, [255,0,0]), (120,220,[0,255,0]), ...]
#         #     frames[1] = [(...] ... ]
#         print (image_size)
#         self.frames: Dict[int, List[Tuple[float, float, List[int]]]] = {}

#     def add_detection(self, frame: int, x: float, y: float, color: Tuple[int, int, int]) -> None:
#         """
#         1つの検出結果を「どのフレームか」をキーにして登録する。
        
#         Parameters
#         ----------
#         frame : int
#             フレーム番号
#         x : float
#             検出点の x 座標
#         y : float
#             検出点の y 座標
#         color : (int, int, int)
#             検出物体の色情報 (R, G, B)
#         """
#         if frame not in self.frames:
#             self.frames[frame] = []

#         # color は JSON 化するとき配列になり、(x, y, [R, G, B]) の構造を持ちます
#         # x,yは0-1の値に正規化する
#         x = x / self.image_size[0]
#         y = y / self.image_size[1]

#         detection_tuple = (float(x), float(y), [int(color[0]), int(color[1]), int(color[2])])
#         self.frames[frame].append(detection_tuple)

#     def save_json(self, file_path: str) -> None:
#         """
#         フレームごとに蓄えた (x, y, color) のリストを JSON ファイルに保存する。
        
#         Parameters
#         ----------
#         file_path : str
#             出力先の JSON ファイルパス
#         """
#         with open(file_path, 'w', encoding='utf-8') as f:
#             # Python の辞書は {フレーム番号: [(x, y, [R,G,B]), ...], ...} の形
#             # これをそのまま dump すると、下記のような JSON になる:
#             # {
#             #   "0": [
#             #     [123.0, 456.0, [255, 0, 0]],
#             #     [200.0, 300.0, [0, 255, 0]]
#             #   ],
#             #   "1": [
#             #     ...
#             #   ]
#             # }
#             json.dump(self.frames, f, indent=4, ensure_ascii=False)
#         print(f"JSON file has been saved to {file_path}")
