# SOTS and MOT

### Codes and comparison of recent single object tracking and segmentation (VOT and VOTS), and multiple object tracking (MOT).

## News
:boom: **Repo Init:** Official implementation of [OceanPlus]() is uploaded. [OceanPlus](), [Ocean]() and [SiamDW]() are supported now. 

## Supported Trackers

### Single-Object Tracking (SOT)

- [x] [**OceanPlus**]()
- [x] [**Ocean**]()
- [x] [**SiamDW**]()
- [ ] [**SiamMask**]()
- [ ] [**SiamRPN++**]()
- [ ] [**SiamFC++**]()
- [ ] [**SiamFC**]()
- [ ] [**ATOM**]()
- [ ] [**DiMP**]() 
- [ ] [**PrDiMP**]()
- [ ] [**KYS**]()
- [ ] [**LWL**]()
- [ ] [**SiamBAN**]()
- [ ] [**SiamAttn**]()

### Multi-Object Tracking (MOT)
- [x] [**CSTrack**]()

## Results Comparison
- [x] [**Comparison**](https://github.com/JudasDie/Comparison)

## Talks about VOT and VOTS (video)

[**[SiamFC]**](https://www.bilibili.com/video/BV1ck4y1B7Yv?from=search&seid=15454527879078953284)[**[SiamRPN series]**](https://www.bilibili.com/video/BV1tJ411K7iQ?from=search&seid=16794822988427286264)[**[SiamDW]**](https://www.bilibili.com/video/BV1ut411L7Ru?from=search&seid=8271716311900472376)[**[SiamMask]**](https://www.bilibili.com/video/BV1Kt411u7CT?from=search&seid=684552033902530378)[**[ATOM]**](https://www.bilibili.com/video/BV1Lt411n7mK?from=search&seid=8307889874986411460)[**[Ocean]**](https://www.bilibili.com/video/BV1354y1e7wU?from=search&seid=15378680533874208460)[**[SiamBAN]**](https://www.youtube.com/watch?v=HW61L2GI7Kc)

## Achievements of the related trackers
- SiamRPN and its variants achieve VOT2018-RT winner.
- SiamDW-T achieves runner-up of VOT2019-RGBT and 1st of VOT2020 RGBT (re-submitted by the committee).
- OceanPlus and its variants achieve the runner-ups for both VOT2020-ST and VOT2020-RT.


## Tracker Details
### OceanPlus [Arxiv now]
**[[Paper]]() [[Raw Results]]() [[Training and Testing Tutorial]](https://github.com/JudasDie/SOTS/tree/master/lib/tutorial/OceanPlus/oceanplus.md) [[Demo]]()** <br/>
Official implementation of the OceanPlus tracker. It proposes an attention retrieval network (ARN) to perform soft spatial constraints on backbone features. Concretely, we first build a look-up-table (LUT) with the ground-truth mask in the starting frame, and then retrieve the LUT to obtain a target-aware attention map for suppressing the negative influence of background clutter. Furthermore, we introduce a multi-resolution multi-stage segmentation network (MMS) to ulteriorly weaken responses of background clutter by reusing the predicted mask to filter backbone features.


</div>
<img src="https://github.com/JudasDie/SOTS/blob/master/demo/oceanplu_overview.png"  alt="OceanPlus"/><br/>
</div>

### Ocean [ECCV2020]
**[[Paper]](https://arxiv.org/abs/2006.10721) [[Raw Results]](https://drive.google.com/file/d/1vDp4MIkWzLVOhZ-Yt2Zdq8Z_Z0rz6y0R/view?usp=sharing) [[Training and Testing Tutorial]](https://github.com/JudasDie/SOTS/tree/master/lib/tutorial/Ocean/ocean.md) [[Demo]](https://www.youtube.com/watch?v=83-XCEsQ1Kg&feature=youtu.be)** <br/>

Ocean proposes a general anchor-free based tracking framework. It includes a pixel-based anchor-free regression network to solve the weak rectification problem of RPN, and an object-aware classification network to learn robust target-related representation. Moreover, we introduce an effective multi-scale feature combination module to replace heavy result fusion mechanism in recent Siamese trackers. This work also serves as the baseline model of OceanPlus. An additional **TensorRT** toy demo is provided in this repo.
<div align="left">
  <img src="https://github.com/JudasDie/SOTS/blob/master/demo/Ocean_overview.jpg" height="300" alt="Ocean"/><br/>
  <!-- <p>Example SiamFC, SiamRPN and SiamMask outputs.</p> -->
</div>

### SiamDW [CVPR2019]
**[[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Deeper_and_Wider_Siamese_Networks_for_Real-Time_Visual_Tracking_CVPR_2019_paper.pdf) [[Raw Results]](https://github.com/researchmm/SiamDW) [[Training and Testing Tutorial]](https://github.com/JudasDie/SOTS/tree/master/lib/tutorial/SiamDW/siamdw.md) [[Demo]]()** <br/>
SiamDW is one of the pioneering work using deep backbone networks for Siamese tracking framework. Based on sufficient analysis on network depth, output size, receptive field and padding mode, we propose guidelines to build backbone networks for Siamese tracker. Several deeper and wider networks are built following the guidelines with the proposed CIR module. 

<img src="https://github.com/JudasDie/SOTS/blob/master/demo/siamdw_overview.jpg" height="250" alt="SiamDW"/><br/>

### CSTrack [Arxiv now]
**[[Paper]](https://arxiv.org/abs/2010.12138) [[Training and Testing Tutorial]](https://github.com/JudasDie/SOTS/blob/master/lib/tutorial/CSTrack/cstrack.md) [[Demo]](https://motchallenge.net/method/MOT=3601&chl=10)** <br/>
CSTrack proposes a strong ReID based one-shot MOT framework. It includes a novel cross-correlation network that can effectively impel the separate branches to learn task-dependent representations, and a scale-aware attention network that learns discriminative embeddings to improve the ReID capability. This work also provides an analysis of the weak data association ability in one-shot MOT methods. Our improvements make the data association ability of our one-shot model is comparable to two-stage methods while running more faster.

<img src="https://github.com/JudasDie/SOTS/blob/master/demo/CSTrack_CCN.jpg" height="300" alt="CSTrack"/><br/>

This version can achieve the performance described in the paper (70.7 MOTA on MOT16, 70.6 MOTA on MOT17). The new version will be released soon. If you are interested in our work or have any questions, please contact me at 201921060415@std.uestc.edu.cn.


Other trackers, coming soon ...


:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:


## Structure
- `experiments:` training and testing settings
- `demo:` figures for readme
- `dataset:` testing dataset
- `data:` training dataset
- `lib:` core scripts for all trackers
- `snapshot:` pre-trained models 
- `pretrain:` models trained on ImageNet (for training)
- `tutorials:` guidelines for training and testing
- `tracking:` training and testing interface

```
$TrackSeg
|—— experimnets
|—— lib
|—— snapshot
  |—— xxx.model/xxx.pth
|—— dataset
  |—— VOT2019.json 
  |—— VOT2019
     |—— ants1...
  |—— VOT2020
     |—— ants1...
|—— ...

```

## References
```
[1] Bhat G, Danelljan M, et al. Learning discriminative model prediction for tracking. ICCV2019.
[2] Chen, Kai and Wang, et.al. MMDetection: Open MMLab Detection Toolbox and Benchmark.
...
```
## Contributors
- **[Zhipeng Zhang](https://github.com/JudasDie)**
- **[Chao Liang]()**








