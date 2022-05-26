# Tutorial for Single Object Tracking (SOT)
## Testing

We assume your path is $SOTS, e.g. `/home/zpzhang/SOTS`
### Set up environment

```
cd $SOTS/lib/tutorial/install
bash install.sh $conda_path SOTS
cd $SOTS
conda activate SOTS
export PYTHONPATH=${PYTHONPATH}:$SOTS:$SOTS/lib
```


### Prepare data and models
1. Download the pretrained [PyTorch model](https://drive.google.com/drive/folders/1VP9UsBYpqSVbLwXSr6PkHeuUxJjG0QsT?usp=sharing) to `$SOTS/snapshot`. (Some trackers may contain multiple pretrained models. Please refer to `readme.txt` in their directories.)
2. Download [json](https://drive.google.com/drive/folders/1iXqDQH6duadH9TIa8GK1oybBAyS9nHbk?usp=sharing) files of testing data and put them in `$SOTS/dataset`.
3. Download testing data e.g. VOT2019 and put them in `$SOTS/dataset`. Please download each data from their official websites, and the directories should be named like `VOT2019`, `OTB2015`, `GOT10K`, `LASOTTEST` (see `experiment/xx.yaml` for their name of each benchmark).


### Testing
In root path `$SOTS`,

```
python tracking/test_sot.py --cfg experiments/TransInMo.yaml --resume snapshot/AutoMatch.pth --dataset OTB2015
or
python tracking/test_sot.py --cfg experiments/AutoMatch.yaml --resume snapshot/AutoMatch.pth --dataset OTB2015
or
python tracking/test_sot.py --cfg experiments/Ocean.yaml --resume snapshot/Ocean.pth --dataset OTB2015
or
python tracking/test_sot.py --cfg experiments/SiamDW.yaml --resume snapshot/SiamDW.pth --dataset OTB2015
```


### Evaluation
```
python tracking/eval_sot.py --dataset OTB2015
```
This will test all trackers under `result/OTB2015`, if you want to test specific trackers, please use

```
python tracking/eval_sot.py --dataset OTB2015 --trackers tracker1, tracker2
```

- Note: We support 9 typical SOT benchmarks: `OTB`, `GOT10K`, `LaSOT`, `VOT`, `TNL2K`, `TOTB`, `NFS30`, `TC128`, `UAV123`. If you want to plot success curves or get more detailed scores, you can run with `pysot-toolkit`.




:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:
## Training
#### prepare data
- Please download training data from [BaiduDrive(urxq)](https://pan.baidu.com/s/1jGPEJieir5OWqCmibV3yrQ), and then put them in `$SOTS/data`. You can also process data following `pysot` benchmark.
 
- For splited files in BaiduDrive, please use `cat got10k.tar.*  | tar -zxv` to merge and unzip.


#### prepare pretrained model
Please download the pretrained model on ImageNet [here](https://drive.google.com/drive/folders/1ppVTE4oeQuXWKNxTf-mfHBNiNGpk_L6t?usp=sharing), and then put it in `$SOTS/pretrain`.

#### modify settings
Please modify the training settings in `$SOTS/experiments/XX.yaml`. The default number of GPU and batch size in paper are 8 and 32 for `Ocean` and `AutoMatch`.

- The training on 4*GTX 2080Ti of `AutoMatch` takes about 20 hours.


#### run
In root path $SOTS,
```
python tracking/onekey.py --cfg experiments/AutoMatch.yaml
```

if you want to use DDP, please change `DDP` in `experiments/xx.yaml` to `True`
```
python -m torch.distributed.launch --nproc_per_node $GPU_NUMS trackng/train_sot.py
```

This script integrates **train**, **epoch test** and **tune**. It is suggested to run them one by one when you are not familiar with our whole framework (modify the key `ISTRUE` in `$SOTS/experiments/XX.yaml`). When you know this framework well, simply run this one-key script.

## Notice
- The new codebase will not support `OceanPlus` and `TensorRT` testing. You could find them in the `v0` branch.
```
git clone -b v0 https://github.com/JudasDie/SOTS 
```
- The results for `Ocean` and `SiamDW` are reproduced with this new codebase. They may be sightly better or worse than that in the paper. 

- We current split SOT and MOT to separated branches. Clone MOT branch with:

```
git clone -b MOT https://github.com/JudasDie/SOTS 
```
