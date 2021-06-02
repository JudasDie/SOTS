# OceanPlus tutorial
## Testing

We assume the root path is $SOTS, e.g. `/home/zpzhang/SOTS`

### Set up environment

```
cd $TracKit/lib/tutorial
bash install.sh $conda_path TracKit
cd $TracKit
conda activate TracKit
python setup.py develop
```
`$conda_path` denotes your anaconda path, e.g. `/home/zpzhang/anaconda3`


**Note:**  all the results for VOT2020 in the paper (including other methods) are performed with `vot-toolkit=0.2.0`.


### Prepare data and models

1. Following the official [guidelines](https://www.votchallenge.net/howto/tutorial_python.html) to set up VOT workspace.

2. Download from [GoogleDrive](https://drive.google.com/drive/folders/1_uagYRFpQmYoWAc0oeiAY49gHwQxztrN?usp=sharing) and put them in `$tracker_path/snapshot`


### Testing

#### For VOT2020

1. Modify scripts
Set the model path in line81 of `$tracker_path/tracking/vot_wrap.py` and `$tracker_path/tracking/vot_wrap_mms.py`.

2. run

- for model without MMS network:
```
set running script in vot2020 workspace (i.e. trackers.ini) to `vot_wrap.py`
```
- for model with MMS network:
```
set running script in vot2020 workspace (i.e. trackers.ini) to `vot_wrap_mms.py`
```
- Note: We provided a reference of `trackers.ini` in `$tracker_path/trackers.ini`. Please find more running guidelines in VOT official [web](https://www.votchallenge.net/howto/tutorial_python.html).

#### For VOS (DAVIS/YTBVOS)
Coming soon ...

:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:

The training code will be released after accepted. Thanks for your interest!
 

