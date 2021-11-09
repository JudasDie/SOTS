import os
import cv2
import random
import math
import numpy as np
import shutil
import json
from mpi4py import MPI
import sys

# Traverse folders
def findfile(path, ret,file_state):
    filelist = os.listdir(path)
    for filename in filelist:
        de_path = os.path.join(path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(file_state):
                ret.append(de_path)
        else:
            findfile(de_path, ret,file_state)

# create new directory
def dir_make(path):
    if not os.path.exists(path):
        os.mkdir(path)


# visualization results of image_MOT
def plot_tracking(filename,ret,results, save_image_tracking=False):
    if not os.path.exists(filename):
        os.mkdir(filename)
    id_color = {}
    id_point = {}
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
            cv2.rectangle(img, (x, y), (x+w, y+h), (id_color[id][0], id_color[id][1], id_color[id][2]), 6)
            cv2.putText(img, str(id), (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
            if save_image_tracking:
                if id not in id_point.keys():
                    id_point[id] = [[int(x+w/2),int(y+h)]]
                else:
                    id_point[id] += [[int(x + w / 2), int(y + h)]]
                    for i_Track in range(1,len(id_point[id])):
                        ptStart = (id_point[id][i_Track-1][0],id_point[id][i_Track-1][1])
                        ptEnd = (id_point[id][i_Track][0], id_point[id][i_Track][1])
                        cv2.line(img, ptStart, ptEnd, (id_color[id][0], id_color[id][1], id_color[id][2]), 2, 6)
        else:
            cv2.imwrite(filename+"/"+str(frame_id)+".jpg",img)
            frame_id = int(results[i][0])
            img = cv2.imread(ret[frame_id-1])
            cv2.rectangle(img, (x, y), (x+w, y+h), (id_color[id][0], id_color[id][1], id_color[id][2]), 6)
            cv2.putText(img, str(id), (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
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

def split_img(img="", img_id = 0,seq_id = 0, p_label="", output_size_list=[], over_lap=0.3,save_path_img="",save_path_label="",max_id = 0):

    zero_list = ["00000","0000","000","00","0"]
    h_img = len(img)
    w_img = len(img[0])

    label_dict = []
    with open(p_label, "a", encoding="utf-8") as f:
        f = open(p_label, "r", encoding="utf-8")
        for line in f:
            data = line.split(' ')
            x_w = float(data[2]) * w_img
            y_w = float(data[3]) * h_img
            w = float(data[4]) * w_img
            h = float(data[5][:-2]) * h_img
            label_dict.append([int(data[1]), x_w, y_w, w, h])
    img_id = str(seq_id) + zero_list[len(str(img_id))-1]+str(img_id)

    split_num = 0
    for output_size in output_size_list:
        num_i = int((w_img // (output_size[0] - output_size[0] * over_lap)) + 1)
        num_j = int((h_img // (output_size[1] - output_size[1] * over_lap)) + 1)
        for i in range(num_i):
            if i == 0:
                x_now = 0
            else:
                x_now += (output_size[0] - output_size[0]*over_lap)
            if x_now > w_img - output_size[0]:
                x_now = w_img - output_size[0]
            for j in range(num_j):
                if j == 0:
                    y_now = 0
                else:
                    y_now += (output_size[1] - output_size[1]*over_lap)
                if y_now > h_img - output_size[1]:
                    y_now = h_img - output_size[1]

                label_list = []
                for label in label_dict:
                    if (x_now < label[1]-label[3]/2 < x_now + output_size[0] and y_now < label[2]-label[4]/2 < y_now + output_size[1]) or (x_now < label[1]+label[3]/2 < x_now + output_size[0] and y_now < label[2]+label[4]/2 < y_now + output_size[1]):
                        x_list = sorted([label[1]-label[3]/2-x_now,label[1]+label[3]/2-x_now,0,output_size[0]])
                        y_list = sorted([label[2]-label[4]/2-y_now,label[2]+label[4]/2-y_now,0,output_size[1]])
                        iou = (x_list[2]-x_list[1])*(y_list[2]-y_list[1])/(label[3]*label[4])
                        x_c = (label[1]-x_now)/output_size[0]
                        y_c = (label[2]-y_now)/output_size[1]
                        w_c = label[3]/output_size[0]
                        h_c = label[4]/output_size[1]
                        if output_size == output_size_list[0]:
                            if (w_c <= 0.7 and h_c <= 1) and iou > 0.2:
                                label_list.append([label[0],x_c,y_c,w_c,h_c])
                        else:
                            if (0.01 <= w_c <= 0.7 and 0.01 <= h_c <= 1) and iou > 0.2:
                                label_list.append([label[0],x_c,y_c,w_c,h_c])
                #save train dataset
                if len(label_list) != 0:
                    img_save = img[int(y_now):int(y_now + output_size[1]), int(x_now):int(x_now + output_size[0])]
                    txt_path = save_path_label + "/" + img_id + "_" + zero_list[len(str(split_num))-1]+str(split_num)+".txt"
                    with open(txt_path, "w") as w_txt:
                        for label in label_list:
                            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                                label[0], label[1], label[2], label[3], label[4])
                            w_txt.write(label_str)
                            #cv2.rectangle(img_save, (int((label[1]-label[3]/2)*output_size[0]), int((label[2]-label[4]/2)*output_size[1])),
                            #              (int((label[1]+label[3]/2)*output_size[0]), int((label[2]+label[4]/2)*output_size[1])), (0,0,255), 4)
                    img_save = cv2.resize(img_save, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(save_path_img + "/" + img_id + "_" + zero_list[len(str(split_num))-1]+str(split_num)+".jpg",img_save)
                    split_num += 1

                if y_now == h_img - output_size[1]:
                    break

            if x_now == w_img - output_size[0]:
                break


def catalogue_make(save_path_img):
    ret = []
    findfile(save_path_img, ret, ".jpg")
    with open(save_path+"/panda.train", "w") as w_txt:
        for name in ret:
            w_txt.write(name[9:] + "\n")
    with open(save_path+"/panda.val", "w") as w_txt:
        for name in ret[-2000:]:
            w_txt.write(name[9:] + "\n")

def split_track(i):
    ret = []
    findfile(img_path[i], ret, ".jpg")
    txt_path = []
    label_path = save_path_list[i] + "/labels/"
    ret = sorted(ret)
    for r in ret:
        txt_path.append(label_path + r.split("/")[-1].split(".")[0] + ".txt")
    with open(seqinfo_json[i]) as sr:
        sr_p = json.load(sr)
    w_img = sr_p["imWidth"]
    h_img = sr_p["imHeight"]
    leg = len(ret)
    print(img_path[i])
    print("strat_id:", max_id_list[i])
    sys.stdout.flush()
    for r_i in range(leg):
        #print(img_path[i], r_i / leg)
        #sys.stdout.flush()
        img = cv2.imread(ret[r_i])
        split_img(img=img,
                  img_id=r_i,
                  seq_id=i,
                  p_label=txt_path[r_i],
                  output_size_list=[[2560, 1440], [5120, 2880], [10240, 5760], [w_img, h_img]],
                  over_lap=0.2,
                  save_path_img=save_path_img,
                  save_path_label=save_path_label,
                  max_id=max_id_list[i])

def split_det(i):
    img_path_list = [root+"/panda_round1_train_202104_part1",
                     root+"/panda_round1_train_202104_part2"]
    save_path_det_list = [gt_root+"/panda_round1_train_202104_part1",
                          gt_root+"/panda_round1_train_202104_part2"]
    img_path = img_path_list[i-10]
    ret = []
    findfile(img_path, ret, ".jpg")
    txt_path = []
    ret = sorted(ret)
    for r in ret:
        txt_path.append(save_path_det_list[i-10] + "/" + r.split("/")[-2] + "_labels/" + r.split("/")[-1].split(".")[0] + ".txt")
    leg = len(ret)
    print(img_path)
    for r_i in range(leg):
        if rank == i:
            #print(img_path,r_i / leg)
            #sys.stdout.flush()
            img = cv2.imread(ret[r_i])
            h_img = len(img)
            w_img = len(img[0])
            split_img(img=img,
                      img_id=r_i,
                      seq_id=i,
                      p_label=txt_path[r_i],
                      output_size_list=[[2560, 1440], [5120, 2880], [10240, 5760], [w_img, h_img]],
                      over_lap=0.2,
                      save_path_img=save_path_img,
                      save_path_label=save_path_label,
                      max_id=0)


if __name__ == '__main__':
    root = "../../../tcdata"
    gt_root = "../../../gt"
    seqinfo_json = [root+"/panda_round2_train_annos_20210331/01_University_Canteen/seqinfo.json",
                     root+"/panda_round2_train_annos_20210331/02_OCT_Habour/seqinfo.json",
                     root+"/panda_round2_train_annos_20210331/03_Xili_Crossroad/seqinfo.json",
                     root+"/panda_round2_train_annos_20210331/04_Primary_School/seqinfo.json",
                     root+"/panda_round2_train_annos_20210331/05_Basketball_Court/seqinfo.json",
                     root+"/panda_round2_train_annos_20210331/06_Xinzhongguan/seqinfo.json",
                     root+"/panda_round2_train_annos_20210331/07_University_Campus/seqinfo.json",
                     root+"/panda_round2_train_annos_20210331/08_Xili_Street_1/seqinfo.json",
                     root+"/panda_round2_train_annos_20210331/09_Xili_Street_2/seqinfo.json",
                     root+"/panda_round2_train_annos_20210331/10_Huaqiangbei/seqinfo.json"]

    img_path = [root+"/panda_round2_train_20210331_part1",
                root+"/panda_round2_train_20210331_part2",
                root+"/panda_round2_train_20210331_part3",
                root+"/panda_round2_train_20210331_part4",
                root+"/panda_round2_train_20210331_part5",
                root+"/panda_round2_train_20210331_part6",
                root+"/panda_round2_train_20210331_part7",
                root+"/panda_round2_train_20210331_part8",
                root+"/panda_round2_train_20210331_part9",
                root+"/panda_round2_train_20210331_part10"]

    save_path_list = [gt_root+"/panda_round2_train_20210331_part1",
                      gt_root+"/panda_round2_train_20210331_part2",
                      gt_root+"/panda_round2_train_20210331_part3",
                      gt_root+"/panda_round2_train_20210331_part4",
                      gt_root+"/panda_round2_train_20210331_part5",
                      gt_root+"/panda_round2_train_20210331_part6",
                      gt_root+"/panda_round2_train_20210331_part7",
                      gt_root+"/panda_round2_train_20210331_part8",
                      gt_root+"/panda_round2_train_20210331_part9",
                      gt_root+"/panda_round2_train_20210331_part10"]

    save_path = "../../../data"
    save_path_img = save_path + "/images"
    save_path_label = save_path + "/labels_with_ids"
    dir_make(save_path)
    dir_make(save_path_img)
    dir_make(save_path_label)
    max_id = 0
    max_id_list = []
    max_id_list.append(max_id)
    for i in range(len(img_path)):
        ret = []
        findfile(img_path[i], ret,".jpg")
        txt_path = []
        label_path = save_path_list[i] + "/labels/"
        ret = sorted(ret)
        for r in ret:
            txt_path.append(label_path + r.split("/")[-1].split(".")[0] + ".txt")
        leg = len(ret)
        id_label = []
        for r_i in range(leg):
            with open(txt_path[r_i], "a", encoding="utf-8")as f:
                f = open(txt_path[r_i], "r", encoding="utf-8")
                for line in f:
                    data = line.split(' ')
                    id_label.append(int(data[1]))
        max_id += max(id_label)
        max_id_list.append(max_id)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        split_track(0)
    elif rank == 1:
        split_track(1)
    elif rank == 2:
        split_track(2)
    elif rank == 3:
        split_track(3)
    elif rank == 4:
        split_track(4)
    elif rank == 5:
        split_track(5)
    elif rank == 6:
        split_track(6)
    elif rank == 7:
        split_track(7)
    elif rank == 8:
        split_track(8)
    elif rank == 9:
        split_track(9)
    elif rank == 10:
        split_det(10)
    elif rank == 11:
        split_det(11)

    catalogue_make(save_path_img)





