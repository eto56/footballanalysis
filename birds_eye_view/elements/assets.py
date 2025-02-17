import cv2
from sklearn.cluster import KMeans
import numpy as np


pallete = {'b': (0, 0, 128),
        'g': (0, 200, 0),
        'r': (255, 0, 0),
        'c': (0, 192, 192),
        'm': (192, 0, 192),
        'y': (192, 192, 0),
        'k': (0, 0, 0),
        'w': (255, 255, 255)}

# white , black ,orange ,blue ,green only

# pallete = {'b': (0, 0, 128),
#         'o': (0, 165, 255),
#         'k': (0, 0, 0),
#         'w': (255, 255, 255),
#         'g': (0, 128, 0)}





color_for_labels = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in color_for_labels]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    print ("bbox", len(bbox))
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        print ("id", id)
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def transform_matrix(matrix, p, vid_shape, gt_shape):
    p = (p[0]*1280/vid_shape[1], p[1]*720/vid_shape[0])
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))

    p_after = (int(px*gt_shape[1]/115) , int(py*gt_shape[0]/74))

    return p_after


# Color Detection with K-means
def detect_color(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[1]*img.shape[0],3))

    kmeans = KMeans(n_clusters=2)
    s = kmeans.fit(img)

    labels = kmeans.labels_ 
    centroid = kmeans.cluster_centers_  # list of RGB values of the centroids of the clusters
    labels = list(labels)
    percent=[]
    
    for i in range(len(centroid)):
        j=labels.count(i)
        j=j/(len(labels))
        percent.append(j)

    detected_color = centroid[np.argmin(percent)]
    
    list_of_colors = list(pallete.values())
    assigned_color = closest_color(list_of_colors, detected_color)[0]
    assigned_color = (int(assigned_color[2]), int(assigned_color[1]), int(assigned_color[0]))

    if assigned_color == (0, 0, 0):
        assigned_color = (128, 128, 128)

    return assigned_color


def detect_color_remake(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # put premiums on pixels in the center of the image

    # 1/{1+abs(x-y)} is a function that gives a premium to the center of the image

    h, w, _ = img.shape
    
  

    center = img [int(h/2)-10:int(h/2)+10, int(w/2)-10:int(w/2)+10]

    center = center.reshape((center.shape[1]*center.shape[0],3))

    center = np.array(center)
    center = center.astype('float32')
    centeral_color= np.mean(center, axis=0)
   

    closest_color = cos_color(centeral_color[0], centeral_color[1], centeral_color[2])

    return closest_color


 

class object:
    def __init__(self,id,frame,coords, color, is_player, frame_num):
        self.id = id
        self.frame = frame
        self.frame_num = frame_num
        self.coords = coords
        self.color = color
        self.is_player = is_player
   
class player_manager:
    def __init__(self):
        self.players = []
        
    
    ## frame num , coords , color, is_player , frame

    def add_player(self,player_id,frame,coords, color, is_player, frame_num):
         
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.players.append(object(player_id,frame,coords, color, is_player, frame_num))
    
    def classify_colors(self):
        # for player in self.players:
        #     h= player.frame.shape[0]
        #     w= player.frame.shape[1]
        #     player.color = detect_color(player.frame[int(h/2)-10:int(h/2)+10, int(w/2)-10:int(w/2)+10 ])

        ## 中心に近いピクセルの平均色を取得
        ## その色を二つのグループに分ける
        ## その二つのグループの中心色を取得
        ## グループの構成員にはその中心色を割り当てる

        
        player_ave_color=[]
         
        for player in self.players:
            if not player.is_player:
                continue
            h= player.frame.shape[0]
            w= player.frame.shape[1]

            central = player.frame[int(h/2)-int(h/4):int(h/2)+int(h/4), int(w/2)-int(w/4):int(w/2)+int(w/4)]
            central = player.frame
            central = central.reshape((central.shape[1]*central.shape[0],3))

            kmeans = KMeans(n_clusters=4)
            s = kmeans.fit(central)

            central_color = kmeans.cluster_centers_ [1]
            
            player_ave_color.append(central_color)
            #print(central_color)


        #選手たちの平均色をK-meansでクラスタリング
        kmeans = KMeans(n_clusters=2)
        player_ave_color = np.array(player_ave_color)
        kmeans.fit(player_ave_color)
        labels = kmeans.labels_
        centroid = kmeans.cluster_centers_
        labels = list(labels)
        ## それぞれのクラスタの中心色で割り当てる
        for i, player in enumerate(self.players):
            if not player.is_player:
                continue
            player.color = (int(centroid[labels[i]][2]), int(centroid[labels[i]][1]), int(centroid[labels[i]][0]))
    
    
    
    
    def draw_circles(self,bg_img,bg_ratio):
        for player in self.players:
            cv2.circle(bg_img, player.coords, bg_ratio+1, player.color, -1)

        
        print("drawn")
    
    def get_players(self):
        return self.players


            



class player_color_manager:
    def __init__(self):
        self.colors = []
        self.final_colors = []
        self.id_limit = 35
        self.id_max= 0
        
        ##set of indexes of the colors of the players

        self.indexes = set()

        for i in range(self.id_limit):
            self.colors.append([])
            self.final_colors.append((0,0,0))
        
         

    
    def add_color(self,player_id,color):
        if player_id > self.id_max:
            self.id_max = player_id
        self.colors[player_id].append(color)
        self.indexes.add(player_id)
    
    def get_most_frequent_color(self,player_id):
        kmeans = KMeans(n_clusters=2)

        colors = self.colors[player_id]

        if len(colors) == 0:
            return (0,0,0)
        colors = np.array(colors)
        kmeans.fit(colors)
        labels = kmeans.labels_
        centroid = kmeans.cluster_centers_



        return centroid[0]
    
    def calculate_final_colors(self):
        
        for i in  self.indexes:
            self.final_colors[i] = self.get_most_frequent_color(i)


            print(i,self.final_colors[i])
             
        
        return self.final_colors
    
    def get_colors(self,id):

        if id not in self.indexes:
            print ("not in indexes")
            return (0,0,0)

        return self.final_colors[id]
    

        


        
        


 
    

    
            


  
 

def get_color(r,g,b):
    list_of_colors = list(pallete.values())
    detected_color = (r,g,b)
    assigned_color = closest_color(list_of_colors, detected_color)[0]
    assigned_color = (int(assigned_color[2]), int(assigned_color[1]), int(assigned_color[0]))
 

    return assigned_color


# Find the closest color to the detected one based on the predefined palette
def closest_color(list_of_colors, color):
    colors = np.array(list_of_colors)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors-color)**2,axis=1))
    index_of_shortest = np.where(distances==np.amin(distances))
    shortest_distance = colors[index_of_shortest]

    return shortest_distance 


def cos_color(r,g,b):
    #　内積から、cos類似度を計算
    # 類似度が高いほど、色が近い

    # 類似度が高い色を返す
    list_of_colors = list(pallete.values())
    
    best_color = [0,0,0]
    best_cos = 0

    for color in list_of_colors:
        color = np.array(color)
        detected_color = np.array([r,g,b])
        cos = np.dot(color,detected_color)/(np.linalg.norm(color)*np.linalg.norm(detected_color))
        if cos > best_cos:
            best_cos = cos
            best_color = color

    
    best_color = (int(best_color[2]), int(best_color[1]), int(best_color[0]))
    
    return best_color


