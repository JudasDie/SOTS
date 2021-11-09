import os
import cv2
import random
import math
import numpy as np
import shutil
import json


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
    else:
        shutil.rmtree(path)
        os.mkdir(path)


def track2_set():
    track_json = [root+"/panda_round2_train_annos_20210331/01_University_Canteen/tracks.json",
                  root+"/panda_round2_train_annos_20210331/02_OCT_Habour/tracks.json",
                  root+"/panda_round2_train_annos_20210331/03_Xili_Crossroad/tracks.json",
                  root+"/panda_round2_train_annos_20210331/04_Primary_School/tracks.json",
                  root+"/panda_round2_train_annos_20210331/05_Basketball_Court/tracks.json",
                  root+"/panda_round2_train_annos_20210331/06_Xinzhongguan/tracks.json",
                  root+"/panda_round2_train_annos_20210331/07_University_Campus/tracks.json",
                  root+"/panda_round2_train_annos_20210331/08_Xili_Street_1/tracks.json",
                  root+"/panda_round2_train_annos_20210331/09_Xili_Street_2/tracks.json",
                  root+"/panda_round2_train_annos_20210331/10_Huaqiangbei/tracks.json"]

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

    dir_make(gt_root)
    for path in save_path_list:
        dir_make(path)
    for i in range(len(img_path)):
        ret = []
        findfile(img_path[i], ret, ".jpg")
        txt_path = []
        for r in sorted(ret):
            txt_path.append(r.split("/")[-1].split(".")[0] + ".txt")
        save_path = save_path_list[i] + "/labels/"
        dir_make(save_path)
        with open(track_json[i]) as tr:
            tr_p = json.load(tr)
        with open(seqinfo_json[i]) as sr:
            sr_p = json.load(sr)
        w = sr_p["imWidth"]
        h = sr_p["imHeight"]
        leg = len(ret)
        frame = {}
        for s_i in range(len(tr_p)):
            for f_i in tr_p[s_i]["frames"]:
                if f_i["frame id"] not in frame.keys():
                    frame[f_i["frame id"]] = []
                frame[f_i["frame id"]].append([tr_p[s_i]["track id"],
                                               (f_i["rect"]["br"]["x"] + f_i["rect"]["tl"]["x"]) / 2,
                                               (f_i["rect"]["br"]["y"] + f_i["rect"]["tl"]["y"]) / 2,
                                               (f_i["rect"]["br"]["x"] - f_i["rect"]["tl"]["x"]),
                                               (f_i["rect"]["br"]["y"] - f_i["rect"]["tl"]["y"])])
        tp_i = 0
        for f_i in sorted(frame.keys()):
            frame[f_i] = sorted(frame[f_i])
            for gt in frame[f_i]:
                label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    gt[0], gt[1], gt[2], gt[3], gt[4])
                with open(save_path + txt_path[tp_i], 'a') as f:
                    f.write(label_str)
            tp_i += 1




def track2_gt():
    track_json = [root+"/panda_round2_train_annos_20210331/01_University_Canteen/tracks.json",
                  root+"/panda_round2_train_annos_20210331/02_OCT_Habour/tracks.json",
                  root+"/panda_round2_train_annos_20210331/03_Xili_Crossroad/tracks.json",
                  root+"/panda_round2_train_annos_20210331/04_Primary_School/tracks.json",
                  root+"/panda_round2_train_annos_20210331/05_Basketball_Court/tracks.json",
                  root+"/panda_round2_train_annos_20210331/06_Xinzhongguan/tracks.json",
                  root+"/panda_round2_train_annos_20210331/07_University_Campus/tracks.json",
                  root+"/panda_round2_train_annos_20210331/08_Xili_Street_1/tracks.json",
                  root+"/panda_round2_train_annos_20210331/09_Xili_Street_2/tracks.json",
                  root+"/panda_round2_train_annos_20210331/10_Huaqiangbei/tracks.json"]

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

    save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    for i in range(len(img_path)):
        ret = []
        findfile(img_path[i], ret, ".jpg")
        txt_path = []
        for r in sorted(ret):
            txt_path.append(r.split("/")[-1].split(".")[0] + ".txt")
        save_path = img_path[i] + "/" + r.split("/")[-2] + "/gt"
        dir_make(save_path)
        with open(track_json[i]) as tr:
            tr_p = json.load(tr)
        with open(seqinfo_json[i]) as sr:
            sr_p = json.load(sr)
        w = sr_p["imWidth"]
        h = sr_p["imHeight"]
        leg = len(ret)
        frame = {}
        for s_i in range(len(tr_p)):
            for f_i in tr_p[s_i]["frames"]:
                if f_i["frame id"] not in frame.keys():
                    frame[f_i["frame id"]] = []
                frame[f_i["frame id"]].append([tr_p[s_i]["track id"],
                                               (f_i["rect"]["tl"]["x"]) * w,
                                               (f_i["rect"]["tl"]["y"]) * h,
                                               (f_i["rect"]["br"]["x"] - f_i["rect"]["tl"]["x"]) * w,
                                               (f_i["rect"]["br"]["y"] - f_i["rect"]["tl"]["y"]) * h])
        for f_i in sorted(frame.keys()):
            frame[f_i] = sorted(frame[f_i])
            for gt in frame[f_i]:
                label_str = save_format.format(frame=f_i, id=gt[0], x1=gt[1], y1=gt[2], w=gt[3], h=gt[4])
                with open(save_path + "/gt.txt", 'a') as f:
                    f.write(label_str)



def det1_set():
    source_p_json = root+"/panda_round1_train_annos_202104/person_bbox_train.json"
    img_path_list = [root+"/panda_round1_train_202104_part1",
                     root+"/panda_round1_train_202104_part2"]
    save_path_list = [gt_root+"/panda_round1_train_202104_part1",
                      gt_root+"/panda_round1_train_202104_part2"]
    for path in save_path_list:
        dir_make(path)
    with open(source_p_json) as sj:
        sj_p = json.load(sj)
    for img_i in range(len(img_path_list)):
        ret = []
        img_path = img_path_list[img_i]
        save_path = save_path_list[img_i]
        findfile(img_path, ret,".jpg")
        txt_path = []
        ret = sorted(ret)
        for r in ret:
            txt_path.append(save_path + "/" + r.split("/")[-2]+"_labels/"+r.split("/")[-1].split(".")[0] + ".txt")
            if not os.path.exists(save_path + "/" + r.split("/")[-2]+"_labels"):
                os.mkdir(save_path + "/" + r.split("/")[-2]+"_labels")
        leg = len(ret)
        for i in range(leg):
            path_sub = ret[i].split("/")
            path_key = path_sub[-2] + "/" + path_sub[-1]
            p_label = sj_p[path_key]
            h = p_label["image size"]["height"]
            w = p_label["image size"]["width"]
            label_dict = []
            for per_label in p_label["objects list"]:
                if per_label["category"] in ["person"]:
                    label_dict.append(
                        [-1,(per_label["rects"]["full body"]["br"]["x"] + per_label["rects"]["full body"]["tl"]["x"])/2,
                         (per_label["rects"]["full body"]["br"]["y"] + per_label["rects"]["full body"]["tl"]["y"])/2,
                         per_label["rects"]["full body"]["br"]["x"] - per_label["rects"]["full body"]["tl"]["x"],
                         per_label["rects"]["full body"]["br"]["y"] - per_label["rects"]["full body"]["tl"]["y"]])
            for gt in label_dict:
                label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    gt[0], gt[1], gt[2], gt[3], gt[4])
                with open(txt_path[i], 'a') as f:
                    f.write(label_str)


if __name__ == '__main__':
    root = "../../../tcdata"
    gt_root = "../../../gt"
    track2_set()
    det1_set()
    save_path = "../../../data_det"
    save_path_img = save_path + "/images"
    save_path_label = save_path + "/labels"
    dir_make(save_path)
    dir_make(save_path_img)
    dir_make(save_path_label)
    save_path = "../../../data"
    save_path_img = save_path + "/images"
    save_path_label = save_path + "/labels_with_ids"
    dir_make(save_path)
    dir_make(save_path_img)
    dir_make(save_path_label)
    print("finish! label clean")
    #track2_gt()




