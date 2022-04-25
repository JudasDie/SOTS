''' Details
Author: Zhipeng Zhang/Chao Liang
Function: visualization helper
Date: 2022.4.7
'''

import os
import random
import cv2
import os.path as osp
import numpy as np


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    plot a bbox in the image
    :param x:
    :param img:
    :param color:
    :param label:
    :param line_thickness:
    :return:
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


id_color = {}
id_point = {}

def plot_mot_tracking(img, results, frameid=0, save=False, name='MOT'):
    """
    plot mot tracking results
    :param img:
    :param results:
    :param frameid:
    :param save:
    :return:
    """

    for re in results:
        cx, cy, w, h = re[2:6]
        x, y = int(cx - w/2), int(cy - h/2)
        w, h = int(w), int(h)
        id = int(re[1])

        if id not in id_color.keys():
            id_color[id] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

            cv2.rectangle(img, (x, y), (x+w, y+h), (id_color[id][0], id_color[id][1], id_color[id][2]), 2)
            cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if save:
                if id not in id_point.keys():
                    id_point[id] = [[int(x+w/2),int(y+h)]]
                else:
                    id_point[id] += [[int(x + w / 2), int(y + h)]]
                    for i_Track in range(1,len(id_point[id])):
                        ptStart = (id_point[id][i_Track-1][0],id_point[id][i_Track-1][1])
                        ptEnd = (id_point[id][i_Track][0], id_point[id][i_Track][1])
                        cv2.line(img, ptStart, ptEnd, (id_color[id][0], id_color[id][1], id_color[id][2]), 2, 6)
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (id_color[id][0], id_color[id][1], id_color[id][2]), 2)
            cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if save:
                if id not in id_point.keys():
                    id_point[id] = [[int(x+w/2),int(y+h)]]
                else:
                    id_point[id] += [[int(x + w / 2), int(y + h)]]
                    for i_Track in range(1,len(id_point[id])):
                        ptStart = (id_point[id][i_Track-1][0],id_point[id][i_Track-1][1])
                        ptEnd = (id_point[id][i_Track - 1][0], id_point[id][i_Track - 1][1])
                        cv2.line(img, ptStart, ptEnd, (id_color[id][0], id_color[id][1], id_color[id][2]), 1, 4)

    cv2.imshow('{}'.format(name).format(), img)
    cv2.waitKey(1)


def plot_mot_tracking_online(img, online_tlwhs, online_ids, frame_id=0, name='MOT', seq_name=None, opt=None):
    """
    plot mot tracking results
    :param img:
    :param results:
    :param frameid:
    :param save:
    :return:
    """

    for box, id in zip(online_tlwhs, online_ids):
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        id = int(id)

        if id not in id_color.keys():
            id_color[id] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

            cv2.rectangle(img, (x, y), (x+w, y+h), (id_color[id][0], id_color[id][1], id_color[id][2]), 2)
            cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if False:
                if id not in id_point.keys():
                    id_point[id] = [[int(x+w/2),int(y+h)]]
                else:
                    id_point[id] += [[int(x + w / 2), int(y + h)]]
                    for i_Track in range(1,len(id_point[id])):
                        ptStart = (id_point[id][i_Track-1][0],id_point[id][i_Track-1][1])
                        ptEnd = (id_point[id][i_Track][0], id_point[id][i_Track][1])
                        cv2.line(img, ptStart, ptEnd, (id_color[id][0], id_color[id][1], id_color[id][2]), 2, 6)
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (id_color[id][0], id_color[id][1], id_color[id][2]), 2)
            cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if False:
                if id not in id_point.keys():
                    id_point[id] = [[int(x+w/2),int(y+h)]]
                else:
                    id_point[id] += [[int(x + w / 2), int(y + h)]]
                    for i_Track in range(1,len(id_point[id])):
                        ptStart = (id_point[id][i_Track-1][0],id_point[id][i_Track-1][1])
                        ptEnd = (id_point[id][i_Track - 1][0], id_point[id][i_Track - 1][1])
                        cv2.line(img, ptStart, ptEnd, (id_color[id][0], id_color[id][1], id_color[id][2]), 1, 4)

    img = cv2.resize(img, (1088, 608))

    if opt and opt.args.save_videos:
        seq_save_path = osp.join(opt.vis_img_root, seq_name)
        if not osp.exists(seq_save_path): os.makedirs(seq_save_path)
        img_save_path = osp.join(seq_save_path, '{:06d}.jpg'.format(frame_id))
        cv2.imwrite(img_save_path, img)

    cv2.imshow('{}'.format(name).format(), img)
    cv2.waitKey(1)