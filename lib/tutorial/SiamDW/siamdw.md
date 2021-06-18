# SiamDW tutorial
## Testing

We assume the root path is $SOTS, e.g. `/home/zpzhang/SOTS`
### Set up environment
Please follow [readme of Ocean](../Ocean/ocean.md) to install the environment.

### Prepare data and models
1. Download the pretrained [PyTorch model](https://drive.google.com/drive/folders/1QhNlhsatD0ufdz7Gxd2hgyRMxR4k_5z-?usp=sharing) to `$TracKit/snapshot`.
2. Download [json](https://drive.google.com/open?id=1S-RkzyMVRFWueWW91NmZldUJuDyhGdp1) files of testing data and put thme in `$TracKit/dataset`.
3. Download testing data e.g. VOT2017 and put them in `$SOTS/dataset`. 

### Testing
In root path `$SOTS`,
```
python tracking/test_siamdw.py --arch Ocean --resume snapshot/siamdw_res22w.pth --dataset VOT2017
```

### Training
1. Download the imagenet-pretrained [model](https://drive.google.com/drive/folders/13qgXymmi6u_YgHljU7D3BikSa4wx314U?usp=sharing) to `$SOTS/pretrain`.


In root path `$SOTS`,
```
python tracking/onekey_siamdw.py
```
