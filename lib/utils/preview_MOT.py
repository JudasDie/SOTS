import os
import cv2
import random
import math
import numpy as np
import shutil

#历遍文件夹
def findfile(path, ret,file_state):
    filelist = os.listdir(path)
    for filename in filelist:
        de_path = os.path.join(path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(file_state):
                ret.append(de_path)
        else:
            findfile(de_path, ret,file_state)

#可视化结果image_MOT
def plot_tracking(filename,tracking_data_root,results, save_image_tracking=False):
    num_zero = ["00000","0000","000","00","0"]
    if not os.path.exists(filename):
        os.mkdir(filename)
    ret = []
    id_color = {}
    id_point = {}
    for i in range(len(os.listdir(tracking_data_root))):
        ret.append(tracking_data_root+"/"+num_zero[len(str(i+1))-1] + str(i+1)+".jpg")
    for i in range(len(results)):
        x = int(float(results[i][2]))
        y = int(float(results[i][3]))
        w = int(float(results[i][4]))
        h = int(float(results[i][5]))
        id = int(results[i][1])
        if id not in id_color.keys():
            id_color[id] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
        if i == 0:
            frame_id = int(results[i][0])
            img = cv2.imread(ret[frame_id-1])
        if frame_id == int(results[i][0]):
            cv2.rectangle(img, (x, y), (x+w, y+h), (id_color[id][0], id_color[id][1], id_color[id][2]), 4)
           # cv2.putText(img, str(id), (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if save_image_tracking:
                if id not in id_point.keys():
                    id_point[id] = [[int(x+w/2),int(y+h)]]
                else:
                    id_point[id] += [[int(x + w / 2), int(y + h)]]
                    for i_Track in range(1,len(id_point[id])):
                        ptStart = (id_point[id][i_Track-1][0],id_point[id][i_Track-1][1])
                        ptEnd = (id_point[id][i_Track][0], id_point[id][i_Track][1])
                        cv2.line(img, ptStart, ptEnd, (id_color[id][0], id_color[id][1], id_color[id][2]), 2, 4)
        else:
            cv2.imwrite(filename+"/"+str(frame_id)+".jpg",img)
            frame_id = int(results[i][0])
            img = cv2.imread(ret[frame_id-1])
            cv2.rectangle(img, (x, y), (x+w, y+h), (id_color[id][0], id_color[id][1], id_color[id][2]), 4)
            #cv2.putText(img, str(id), (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if save_image_tracking:
                if id not in id_point.keys():
                    id_point[id] = [[int(x+w/2),int(y+h)]]
                else:
                    id_point[id] += [[int(x + w / 2), int(y + h)]]
                    for i_Track in range(1,len(id_point[id])):
                        ptStart = (id_point[id][i_Track-1][0],id_point[id][i_Track-1][1])
                        ptEnd = (id_point[id][i_Track - 1][0], id_point[id][i_Track - 1][1])
                        cv2.line(img, ptStart, ptEnd, (id_color[id][0], id_color[id][1], id_color[id][2]), 1, 4)
    if len(results) != 0:
        cv2.imwrite(filename+"/"+str(frame_id)+".jpg",img)
    print('save image to {}'.format(filename))



#将图片合成视频
def image_T_video(im_dir,video_dir,filename):
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    video_dir = video_dir+"/"+filename+".mp4"
    fps = 25
    num = len(os.listdir(im_dir))
    img = cv2.imread(im_dir+"/10.jpg")
    img_size = (len(img[0]),len(img))
    #fourcc = cv2.cv.CV_FOURCC('M','J','P','G')#opencv2.4
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #opencv3.0
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    for i in range(num):
        im_name = im_dir+"/"+str(i+1)+'.jpg'
        frame = cv2.imread(im_name)
        videoWriter.write(frame)
    videoWriter.release()
    print('finish '+str(filename)+".mp4")



if __name__ == '__main__':
    input_root = "MOT20/images/"
    new_root = ""
    output_root = "MOT20/image_result"
    video_root = "MOT20/video"
    tracking_data = "MOT20"
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    if not os.path.exists(video_root):
        os.mkdir(video_root)

    ret = []
    findfile(input_root+"results", ret,".txt")
    for path in ret:
        path_name = path.split("\\")[1].split(".")[0]
        output_root_sub = output_root+"/"+path_name
        input_root_sub = input_root + "/img/" + path_name + "/img1"
        result = []
        with open(path, "a", encoding="utf-8")as f:
            f = open(path, "r", encoding="utf-8")
            for line in f:
                data = line.split(',')
                result.append(data[:6])
        plot_tracking(output_root_sub,input_root_sub, result, False)
        image_T_video(output_root_sub, video_root,path_name)

