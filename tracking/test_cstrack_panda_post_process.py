import _init_paths 
import os
import os.path as osp
import cv2
import logging
import motmetrics as mm
import numpy as np
from mot_online.log import logger
from mot_online.evaluation import Evaluator
import glob
import random
import argparse
import lap

def write_results(filename, results, data_type="mot"):
    num = 0
    id = []
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
                num += 1
                if track_id not in id:
                    id += [track_id]
    logger.info('save results to {}'.format(filename))


def eval_seq(data_dir,seqs,result_root):
    # eval
    accs = []
    data_type = 'mot'
    for seq in seqs:
        result_filename = os.path.join(result_root, '{}.txt'.format(seq.split("/")[-1]))
        logger.info('Evaluate seq: {}'.format(seq))

        evaluator = Evaluator(data_dir, seq, data_type)
        accs.append(evaluator.eval_file(result_filename,1,234))
    
    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    return strsummary



#Traverse folders
def findfile(path, ret,file_state):
    filelist = os.listdir(path)
    for filename in filelist:
        de_path = os.path.join(path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(file_state):
                ret.append(de_path)
        else:
            findfile(de_path, ret,file_state)

def del_short(ori_dict,leg_thres=1):
    del_num = 0
    label_dict_wo_one = {}
    for id_key in ori_dict.keys():
        if len(ori_dict[id_key]) <= leg_thres:
            del_num += 1
        else:
            label_dict_wo_one[id_key] = ori_dict[id_key]
    print("Num of del_short:", del_num)
    return label_dict_wo_one


def interpolation(ori_dict,leg_thres = 10,frame_thres = 3, interplation_ratio_thres=0.5):
    add_num = 0
    for id_key in ori_dict.keys():
        add_list = []
        if len(ori_dict[id_key]) >= leg_thres:
            for i in range(1,len(ori_dict[id_key])):
                if ori_dict[id_key][i][0] - ori_dict[id_key][i-1][0] != 1:
                    leg = ori_dict[id_key][i][0] - ori_dict[id_key][i-1][0]

                    if leg <= frame_thres and leg <= len(ori_dict[id_key])*interplation_ratio_thres:
                        for add_i in range(1,leg):
                            add_data = [ori_dict[id_key][i - 1][1] + (ori_dict[id_key][i][1] - ori_dict[id_key][i - 1][1]) * add_i / leg,
                                        ori_dict[id_key][i - 1][2] + (ori_dict[id_key][i][2] - ori_dict[id_key][i - 1][2]) * add_i / leg,
                                        ori_dict[id_key][i - 1][3] + (ori_dict[id_key][i][3] - ori_dict[id_key][i - 1][3]) * add_i / leg,
                                        ori_dict[id_key][i - 1][4] + (ori_dict[id_key][i][4] - ori_dict[id_key][i - 1][4]) * add_i / leg]
                            add_list.append([ori_dict[id_key][i-1][0]+add_i]+add_data)

            add_num += len(add_list)
            ori_dict[id_key] = sorted(ori_dict[id_key]+add_list)
    print("Num of interpolation:", add_num)
    return ori_dict


def key_id_to_frame(ori_dict):
    trans_dict = {}
    for key in ori_dict.keys():
        for data in ori_dict[key]:
            if data[0] not in trans_dict.keys():
                trans_dict[data[0]] = []
            trans_dict[data[0]].append([key]+data[1:])
    return trans_dict


def frame_to_results(ori_dict):
    results = []
    for key in ori_dict.keys():
        online_tlwhs = []
        online_ids = []
        for data in ori_dict[key]:
            online_tlwhs.append(data[1:])
            online_ids.append(data[0])
        results.append((key, online_tlwhs, online_ids))
    return results

# Hungary match
def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def tracklet_merge(label_dict, matching_len = 1, body_dis = 3, box_ratio = 0.8, crowd_num = 10):
    #statistics
    fuse_id_num = 0

    # gain start and end index
    frame_start = {}
    frame_end = {}
    for key in label_dict.keys():
        if label_dict[key][0][0] not in frame_start.keys():
            frame_start[label_dict[key][0][0]] = []
        frame_start[label_dict[key][0][0]].append(key)
        if label_dict[key][-1][0] not in frame_end.keys():
            frame_end[label_dict[key][-1][0]] = []
        frame_end[label_dict[key][-1][0]].append(key)

    # matching
    for end_key in sorted(frame_end.keys()):
        for m_i in range(1, matching_len + 1):
            if end_key+m_i in frame_start.keys() and len(frame_start[end_key+m_i]) != 0:
                start_id_list = frame_start[end_key+m_i]
            else:
                continue
            end_tlwh = np.asarray([label_dict[i][-1][1:] for i in frame_end[end_key]])
            start_tlwh = np.asarray([label_dict[i][0][1:] for i in start_id_list])
            cost_matrix = []
            for d_i in range(len(frame_end[end_key])):
                d = (end_tlwh[d_i][:2] - start_tlwh[:,:2])/end_tlwh[d_i][2:]
                d = ((d[:,0])**2+(d[:,1])**2)**0.5/(m_i * body_dis)
                #body dis
                d[d > 1] = np.inf
                d[d < -1] = np.inf
                # box ratio
                wh = end_tlwh[d_i][2:]/start_tlwh[:,2:]
                wh[wh < box_ratio] = np.inf
                wh[wh > 1/box_ratio] = np.inf
                wh = np.minimum(wh[:,0],wh[:,0])
                d[wh==np.inf] = np.inf
                if len(d) - sum(d==np.inf) >= crowd_num:
                    d[d<=1] = np.inf
                cost_matrix.append(d)
            cost_matrix = np.asarray(cost_matrix)
            matches, u_track, u_detection = linear_assignment(cost_matrix, thresh=1)

            #merge tracklet
            start_remove_list = []
            end_remove_list = []
            for iend, istrat in matches:
                fuse_id_num += 1

                id_end = frame_end[end_key][iend]
                id_start = frame_start[end_key+m_i][istrat]
                label_dict[id_end] = label_dict[id_end]+label_dict[id_start]

                frame_end[label_dict[id_start][-1][0]][frame_end[label_dict[id_start][-1][0]].index(id_start)] = id_end
                frame_end[label_dict[id_start][-1][0]] = sorted(frame_end[label_dict[id_start][-1][0]])
                label_dict.pop(id_start)
                start_remove_list.append(id_start)
                end_remove_list.append(id_end)
            for i in range(len(start_remove_list)):
                frame_start[end_key+m_i].remove(start_remove_list[i])
                frame_end[end_key].remove(end_remove_list[i])

    print("Num of merge:",fuse_id_num)
    return label_dict


def post_process(opt,seq,post_results_root):
    label_dict = {}
    with open(seq, "r", encoding="utf-8") as f:
        for line in f:
            data = line.split(',')
            if int(data[1]) not in label_dict.keys():
                label_dict[int(data[1])] = []
            label_dict[int(data[1])].append([int(data[0]), float(data[2]), float(data[3]), float(data[4]), float(data[5])])

    #tracklet_merge
    label_dict = tracklet_merge(label_dict, matching_len = opt.matching_len, body_dis = opt.body_dis, box_ratio = opt.box_ratio, crowd_num = opt.crowd_num)
    #del_short
    label_dict = del_short(label_dict,leg_thres=opt.def_leg_thres)
    #Linear interpolation
    label_dict = interpolation(label_dict,leg_thres=opt.add_leg_thres,frame_thres=opt.frame_thres,interplation_ratio_thres=opt.interplation_ratio_thres)

    label_dict = key_id_to_frame(label_dict)

    results = frame_to_results(label_dict)
    write_results(post_results_root, results, data_type="mot")
    #print(label_dict.keys())


#visualization results of image_MOT
def plot_tracking(filename,tracking_data_root,results, save_image_tracking=False):
    if not os.path.exists(filename):
        os.mkdir(filename)
    ret = []
    id_color = {}
    id_point = {}
    findfile(tracking_data_root, ret, "jpg")
    ret =sorted(ret)
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Tracklet Merge
    parser.add_argument('--matching_len', type=float, default=1, help='matching length limit of gap')
    parser.add_argument('--body_dis', type=float, default=3, help='merge distance limit')
    parser.add_argument('--box_ratio', type=float, default=0.8, help='merge box ratio limit, 0-1')
    parser.add_argument('--crowd_num', type=float, default=3, help='crowd limit')
    #del_short
    parser.add_argument('--def_leg_thres', type=float, default=3, help='the tracklet len')
    # Linear interpolation
    parser.add_argument('--add_leg_thres', type=float, default=5, help='Interpolation length limit of tracklet')
    parser.add_argument('--frame_thres', type=float, default=10, help='Interpolation length limit')
    parser.add_argument('--interplation_ratio_thres', type=float, default=0.3, help='Interpolation length ratio limit, 0-1')

    opt = parser.parse_args()
    print(opt)

    results_root = '../results'
    post_results_root = '../results'

    ret = []
    findfile(results_root, ret, "txt")
    ret = sorted(ret)
    for seq in ret:
        post_process(opt, seq, os.path.join(post_results_root,seq.split("/")[-1]))

    '''
    ### visualization
    input_root_sub = '/home/mfx2/tcdata'
    data_dir = '/home/mfx2/tcdata'
    seqs = ["panda_round2_train_20210331_part7/07_University_Campus",
            "panda_round2_train_20210331_part10/10_Huaqiangbei"]
    output_root_sub = "vis"
    for seq in seqs:
        path = os.path.join(post_results_root,seq.split("/")[-1])+".txt"
        result = []
        with open(path, "a", encoding="utf-8")as f:
            f = open(path, "r", encoding="utf-8")
            for line in f:
                data = line.split(',')
                result.append(data[:6])
        plot_tracking(output_root_sub, os.path.join(input_root_sub,seq), result, True)



    ### eval mot performance
    data_dir = '/home/mfx2/tcdata'
    seqs = ["panda_round2_train_20210331_part7/07_University_Campus",
            "panda_round2_train_20210331_part10/10_Huaqiangbei"]
    #eval_seq(data_dir,seqs,results_root)
    eval_seq(data_dir, seqs, post_results_root)
    '''


