## CSTrack_PANDA tutorial

This is a repo for [PANDA (Gigapixel-Level Human-Centric Video Dataset) competition](https://tianchi.aliyun.com/competition/entrance/531855/introduction)

We assume the root path is **$SOTS**, e.g. `/home/chaoliang/SOTS`

## Set up environment

**$conda_path** denotes your anaconda path, e.g. `/home/chaoliang/anaconda3` 

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

2. Download the **CSTrack model**[[Baidu NetDisk(8m67)]](https://pan.baidu.com/s/1mlivPz3hyPENLTeJxMfjjQ) trained on all tianchi-PANDA datasets to `$SOTS/weights`
3. Download  parts of **tianchi-PANDA datasets**[[Baidu NetDisk(ecxm)]](https://pan.baidu.com/s/1yVl-fHxyF7mhDYwsmdNTUA)  to`$SOTS/tcdata`. e.g. `$SOTS/tcdata/panda_round2_train_20210331_part10`

### Inference

#### For CSTrack_panda inference

```bash
cd $SOTS/tracking
# "--vis_state 1" denotes visualization of detection results 
mpirun -np 1 python test_cstrack_panda_mpi.py   --test_panda True --det_results ../yolov5_panda --nms_thres 0.5 --conf_thres 0.5 --weights ../weights/cstrack_panda.pt  --vis_state 1 
```

**Note:** We provide model fusion between detection results  of CSTrack  and other detection results  such as yolov5.

**yolov5_panda** will produce extra detection results, and `--det_results ../yolov5_panda` acquires the yolov5 detection results. You can fuse detection results of other detectors for

further model fusion.  (if you only want to test CSTrack, please ignore the argument `--det_results ../yolov5_panda` )

## Training

### Prepare data and models

1. Download the CSTrack pretrained model which pretrains on COCO dataset [[Google Drive\]](https://drive.google.com/file/d/1qJHNlEXPVirDVmWL7hHeU4-P9amWHJHR/view?usp=sharing)[[Baidu NetDisk(ba1g)\]](https://pan.baidu.com/s/1S04i6-yxQ3QHtfUDDtd1Kw) to `$SOTS/weights`.
2. Download  [tianchi-PANDA datasets](https://tianchi.aliyun.com/competition/entrance/531855/information) to `$SOTS/tcdata`

### Data preprocessing

**Note:** CSTrack training datasets are tracking datasets of the final competition

```
cd $SOTS/lib/utils/panda
python label_clean.py
mpirun -np 12 python split.py
```

### Training

#### For  CSTrack_panda training

```
cd $SOTS/tracking
python train_cstrack_panda.py --device 0,1 --batch_size 32
```

## Docker

This is our docker implementation for [PANDA competition](https://tianchi.aliyun.com/competition/entrance/531855/introduction) 

1. Install curl and docker 

   ```bash
   # for ubnutu18.04
   sudo apt install curl
   sudo curl -sS https://get.docker.com/ | sh
   ```

2. Activate [Aliyun docker image service](https://tianchi.aliyun.com/competition/entrance/531863/tab/253?spm=5176.12586973.0.0.52d56567ZO368y)

3. Find "**Dockerfile**" and "**test_panda.sh**" in the path of `$SOTS/experiments/run`  and put them to root path, i.e. `$SOTS`

4. Prepare data and models according to above-mentioned **CSTrack_PANDA tutorial**

5. Build and push docker image

   ```bash
   #for PANDA competition
   1) cd $SOTS
   2) docker login registry.cn-shanghai.aliyuncs.com --username your_username  --password your_password
   # for example: docker build -t registry.cn-shanghai.aliyuncs.com/panda_tracking/panda_submit:1.0 . 
   3) docker build -t registry.cn-shanghai.aliyuncs.com/xxx/xxx:1.0 . 
   4) docker push registry.cn-shanghai.aliyuncs.com/xxx/xxx:1.0
   ```

   

