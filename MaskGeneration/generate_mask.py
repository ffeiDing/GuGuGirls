import cv2
import dlib
import os
import os.path as osp
import numpy
import copy
from scipy.spatial import Delaunay

root_name = "/home/dff/f/data/faces_webface_112x112_raw_image/"
save_path = "/ssd_datasets/dff/GuGuGirls/data"

path = "examples/test.jpg"
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
dets = detector(gray, 1)
for face in dets:
    shape = predictor(img, face)
    count = 0
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        count = count + 1
        if count == 29:
            y_mask_min = int(pt.y)
        if count == 9:
            y_mask_max = int(pt.y)
        if count == 2:
            x_mask_min = int(pt.x)
        if count == 16:
            x_mask_max = int(pt.x)
    size = ((x_mask_max-x_mask_min),(y_mask_max-y_mask_min))
    img2 = cv2.imread('examples/mask.png', cv2.IMREAD_UNCHANGED)
    img2 = cv2.resize(img2,size)
    alpha_channel = img2[:, :, 3]
    _, mask = cv2.threshold(alpha_channel, 100, 255, cv2.THRESH_BINARY)
    color = img2[:, :, :3]
    img2 = cv2.bitwise_not(cv2.bitwise_not(color, mask=mask))
    rows,cols,channels = img2.shape
    roi = img[y_mask_min: y_mask_min + rows, x_mask_min:x_mask_min + cols]
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
    dst = cv2.add(img1_bg,img2_fg)
    img[y_mask_min: y_mask_min + rows, x_mask_min:x_mask_min + cols] = dst

    pointNumber = 45
    points = numpy.zeros((pointNumber, 2)) 
    count = 0
    point_num  = 0
    for pt in shape.parts():
        pt_pos = (pt.x-x_mask_min, pt.y-y_mask_min)
        count = count + 1
        cv2.circle(img, pt_pos, 2, (0, 255, 0), 2)
        cv2.circle(img2, pt_pos, 2, (0, 255, 0), 2)
        if count > 2 and count < 16:
            points[point_num][0] = pt.x
            points[point_num][1] = pt.y
            point_num = point_num + 1
        if count > 28 and count < 37:
            points[point_num][0] = pt.x
            points[point_num][1] = pt.y
            point_num = point_num + 1
        if count > 48 and count < 69:
            points[point_num][0] = pt.x
            points[point_num][1] = pt.y
            point_num = point_num + 1
    points[44][0] = x_mask_min
    points[44][1] = y_mask_min
    points[43][1] = y_mask_min+rows
    points[42][1] = y_mask_min+rows
    points[41][1] = y_mask_min
    points[43][0] = x_mask_min
    points[42][0] = x_mask_min+cols
    points[41][0] = x_mask_min+cols
    tri = Delaunay(points)
    for sim in points[tri.simplices]:
        dot1 = (int(sim[0][0])-x_mask_min, int(sim[0][1])-y_mask_min)
        dot2 = (int(sim[1][0])-x_mask_min, int(sim[1][1])-y_mask_min)
        dot3 = (int(sim[2][0])-x_mask_min, int(sim[2][1])-y_mask_min)

index = [3,4,5,6,7,8,9,10,11,12,13,14,15,29,30,31,32,33,34,35,36,49,
         50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,
         69,70,71,72]

for dirname in os.listdir(root_name):
    directory_name = osp.join(root_name,dirname)
    for filename in os.listdir(directory_name):
        img = cv2.imread(osp.join(directory_name,filename))
        img_mask = copy.deepcopy(img)
        img_mask[:,:] = [0,0,0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        dets = detector(gray, 1)
        if len(dets) == 0:
            continue
        y_max = -1
        for face in dets:
            shape = predictor(img, face)
            if len(shape.parts())==0:
                continue
            count = 0
            for pt in shape.parts():
                pt_pos = (pt.x, pt.y)
                count = count + 1
                if count == 29:
                    y_min = int(pt.y)
                if count == 8 or count == 9 or count == 10:
                    y_max = max(int(pt.y), y_max)
                if count == 2:
                    x_min = int(pt.x)
                if count == 16:
                    x_max = int(pt.x)
            for i in tri.simplices:
                sim = points[i]
                dot1_mask = [int(sim[0][0])-x_mask_min, int(sim[0][1])-y_mask_min]
                dot2_mask = [int(sim[1][0])-x_mask_min, int(sim[1][1])-y_mask_min]
                dot3_mask = [int(sim[2][0])-x_mask_min, int(sim[2][1])-y_mask_min]
                if i[0] == 41:
                    dot1 = [x_max, y_min]
                if i[0] == 42:
                    dot1 = [x_max, y_max]
                if i[0] == 43:
                    dot1 = [x_min, y_max]
                if i[0] == 44:
                    dot1 = [x_min, y_min]
                if i[0] < 41:
                    dot1 = [min(int((shape.parts()[index[i[0]]-1]).x), img.shape[1]), 
                            min(int((shape.parts()[index[i[0]]-1]).y), img.shape[0])] 
                if i[1] == 41:
                    dot2 = [x_max, y_min]
                if i[1] == 42:
                    dot2 = [x_max, y_max]
                if i[1] == 43:
                    dot2 = [x_min, y_max]
                if i[1] == 44:
                    dot2 = [x_min, y_min]
                if i[1] < 41:
                    dot2 = [min(int((shape.parts()[index[i[1]]-1]).x), img.shape[1]),
                            min(int((shape.parts()[index[i[1]]-1]).y), img.shape[0])]       
                if i[2] == 41:
                    dot3 = [x_max, y_min]
                if i[2] == 42:
                    dot3 = [x_max, y_max]
                if i[2] == 43:
                    dot3 = [x_min, y_max]
                if i[2] == 44:
                    dot3 = [x_min, y_min]
                if i[2] < 41:
                    dot3 = [min(int((shape.parts()[index[i[2]]-1]).x), img.shape[1]),
                            min(int((shape.parts()[index[i[2]]-1]).y), img.shape[0])] 
                x_mask_max_ = max(dot1_mask[0],dot2_mask[0],dot3_mask[0])
                x_mask_min_ = min(dot1_mask[0],dot2_mask[0],dot3_mask[0])
                y_mask_max_ = max(dot1_mask[1],dot2_mask[1],dot3_mask[1])
                y_mask_min_ = min(dot1_mask[1],dot2_mask[1],dot3_mask[1])
                dot1_mask[0] = dot1_mask[0]-x_mask_min_
                dot2_mask[0] = dot2_mask[0]-x_mask_min_
                dot3_mask[0] = dot3_mask[0]-x_mask_min_
                dot1_mask[1] = dot1_mask[1]-y_mask_min_
                dot2_mask[1] = dot2_mask[1]-y_mask_min_
                dot3_mask[1] = dot3_mask[1]-y_mask_min_
        
                x_min_ = min(max(min(dot1[0],dot2[0],dot3[0]), 0), img.shape[1])
                x_max_ = max(min(max(dot1[0],dot2[0],dot3[0]), img.shape[1]),0)
                y_min_ = min(max(min(dot1[1],dot2[1],dot3[1]), 0), img.shape[0])
                y_max_ = max(min(max(dot1[1],dot2[1],dot3[1]),img.shape[0]),0)
                dot1[0] = dot1[0]-x_min_
                dot2[0] = dot2[0]-x_min_
                dot3[0] = dot3[0]-x_min_
                dot1[1] = dot1[1]-y_min_
                dot2[1] = dot2[1]-y_min_
                dot3[1] = dot3[1]-y_min_
                pos = numpy.float32([dot1, dot2, dot3])
                pos_mask = numpy.float32([dot1_mask, dot2_mask, dot3_mask])
                M = cv2.getAffineTransform(pos_mask,pos)
        
                mask = img2_fg[y_mask_min_:y_mask_max_, x_mask_min_:x_mask_max_]
                if y_max_ ==  y_min_ or x_max_ == x_min_:
                    continue
                size_new = (x_max_-x_min_, y_max_-y_min_)
                res = cv2.warpAffine(mask, M, size_new, borderValue=(0,0,0))
                indexs = numpy.where(numpy.amin(res, -1)>50)
                img[y_min_:y_max_, x_min_:x_max_][indexs] = res[indexs]
                img[y_min_:y_max_, x_min_:x_max_] = cv2.medianBlur(img[y_min_:y_max_, x_min_:x_max_],3) 
                img_mask[y_min_:y_max_, x_min_:x_max_][indexs] = (255,255,255)
        img_mask = cv2.medianBlur(img_mask,3) 
        name = osp.join(save_path,dirname)
        if not os.path.exists(name):
            os.mkdir(name)
        cv2.imwrite(osp.join(name,filename), img)
        print(osp.join(name,filename))
       
    
