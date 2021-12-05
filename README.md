# SOTS -- MOT branch

### Codes of our multiple object tracking paper.

## News
:boom: [OMC](https://arxiv.org/abs/2104.09441) is accepted by AAAI2022. The training and testing code has been released in this codebase.

:boom: The improved version of [CSTrack_panda](https://github.com/JudasDie/SOTS/blob/MOT/CSTrack/lib/tutorial/CSTrack_panda/CSTrack_PANDA.md) has been released, containing the end-to-end tranining codes on PANDA. It is a strong baseline for [Gigavison](http://gigavision.cn/index.html) MOT tracking. Our tracker takes the **5th** place in **Tianchi Global AI Competition (天池—全球人工智能技术创新大赛[赛道二])**, with the score of **A-0.6712/B-0.6251 (AB榜)**, which surprisingly outperforms the baseline tracker JDE with score of A-0.32/B-0.34. More details about CSTrack_panda can be found [here](https://blog.csdn.net/qq_34919792/article/details/116792954?spm=1001.2014.3001.5501).

[![MOT Tracking on Panda](https://res.cloudinary.com/marcomontalbano/image/upload/v1622981850/video_to_markdown/images/youtube--zRCRgsrW71s-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=zRCRgsrW71s "")



## Supported Trackers (MOT)

### Multi-Object Tracking (MOT)
- [x] [**[AAAI2022] OMC**](https://arxiv.org/abs/2104.09441)
- [x] [**CSTrack**](https://arxiv.org/abs/2010.12138)

## Results Comparison
- [x] [**Comparison**](https://github.com/JudasDie/Comparison)


## Tracker Details
### OMC [AAAI2022]
**[[Paper]](https://arxiv.org/abs/2104.09441) [[Training and Testing Tutorial]](https://github.com/JudasDie/SOTS/blob/MOT/OMC/lib/tutorial/omc.md)** <br/>
AutoMatch replaces the essence of Siamese tracking, i.e. the cross-correlation and its variants, to a learnable matching network. The underlying motivation is that heuristic matching network design relies heavily on expert experience. Moreover, we experimentally find that one sole matching operator is difficult to guarantee stable tracking in all challenging environments. In this work, we introduce six novel matching operators from the perspective of feature fusion instead of explicit similarity learning, namely Concatenation, Pointwise-Addition, Pairwise-Relation, FiLM, Simple-Transformer and Transductive-Guidance, to explore more feasibility on matching operator selection. The analyses reveal these operators' selective adaptability on different environment degradation types, which inspires us to combine them to explore complementary features. We propose binary channel manipulation (BCM) to search for the optimal combination of these operators. 

<img src="https://github.com/JudasDie/SOTS/blob/MOT/demo/OMC.jpg" height="600" alt="OMC"/><br/>

### CSTrack [Arxiv now]
**[[Paper]](https://arxiv.org/abs/2010.12138) [[Training and Testing Tutorial]](https://github.com/JudasDie/SOTS/blob/MOT/CSTrack/lib/tutorial/CSTrack/cstrack.md) [[Demo]](https://motchallenge.net/method/MOT=3601&chl=10)** <br/>
CSTrack proposes a strong ReID based one-shot MOT framework. It includes a novel cross-correlation network that can effectively impel the separate branches to learn task-dependent representations, and a scale-aware attention network that learns discriminative embeddings to improve the ReID capability. This work also provides an analysis of the weak data association ability in one-shot MOT methods. Our improvements make the data association ability of our one-shot model is comparable to two-stage methods while running more faster.

<img src="https://github.com/JudasDie/SOTS/blob/MOT/demo/CSTrack_CCN.jpg" height="300" alt="CSTrack"/><br/>

<!-- :cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:
 -->

<!-- ## Structure
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
|—— ... -->

```









