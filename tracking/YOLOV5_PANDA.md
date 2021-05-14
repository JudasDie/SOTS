## YOLOV5_PANDA tutorial

This is a repo for [PANDA(Gigapixel-Level Human-Centric Video Dataset) competition](https://tianchi.aliyun.com/competition/entrance/531855/introduction)

We assume the root path is **$SOTS**, e.g. `/home/chaoliang/SOTS`

## Set up environment

If you have set up environment according to **CSTrack_PANDA tutorial**, please skip here. 

```bash
conda create -n CSTrack python=3.8
source activate CSTrack
conda install pytorch=1.7.0 torchvision cudatoolkit=11.0 -c pytorch
conda install -c anaconda mpi4py==3.0.3 --yes
cd $SOTS/lib/tutorial/CSTrack_panda/
pip install -r requirements.txt
```

## Testing

### Prepare data and models

1. Download the **yolov5 model**[[Baidu NetDisk(q8j9)](https://pan.baidu.com/s/1lqByflTMAdhgYUzjW4Fr5g)] trained on tianchi-PANDA preliminary competition datasets to `$SOTS/yolov5_panda/weights`
2. Download the **PANDA datasets**[[Baidu NetDisk(ecxm)]](https://pan.baidu.com/s/1yVl-fHxyF7mhDYwsmdNTUA)  to `$SOTS/tcdata`. e.g. `$SOTS/tcdata/panda_round2_train_20210331_part10`

### Inference

#### For yolov5_panda inference

```bash
cd $SOTS/yolov5_panda
mpirun -np 1 python detect_mpi.py --iou_thres 0.5 --conf_thres 0.4 --weights weights/yolov5_panda.pt 
```

## Training

### Prepare data and models

1. Download the [yolov5 pretrained model](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt) which pretrains on COCO dataset to `$SOTS/yolov5_panda/weights`
2. Download  [tianchi-PANDA  preliminary competition datasets](https://tianchi.aliyun.com/competition/entrance/531855/information) to `$SOTS/tcdata`

### Data preprocessing

**Note:** yolov5 training datasets are mere detection datasets of the preliminary competition

```
cd $SOTS/lib/utils/panda
python label_clean.py
mpirun -np 2 python split_det.py
```

### Training

#### For yolov5_panda training

```bash
cd $SOTS/yolov5_panda
python train.py --device 0,1 --batch-size 48
```

