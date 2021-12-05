#MOT16\MOT17
# First stage：CSTrack training
python train_omc.py --weights ../weights/yolov5l_coco.pt --cfg ../experiments/model_set/CSTrack_l.yaml --batch-size 8 --data ../lib/dataset/mot/cfg/data_ch.json --name l-all --device 0
# Second stage：Train with re-check network
python train_omc.py --weights ../runs/train/l-all/weights/best.pt --cfg ../experiments/model_set/CSTrack_l.yaml --batch-size 8 --data ../lib/dataset/mot/cfg/mot17.json --project ../runs/train_w_recheck  --name l-mot17 --device 0 --recheck --noautoanchor

#MOT20
# First stage：CSTrack training
python train_omc.py --weights ../runs/train/l-all/weights/last.pt --cfg ../experiments/model_set/CSTrack_l.yaml --batch-size 8 --data ../lib/dataset/mot/cfg/mot20.json --name l-mot20 --device 0
# Second stage：Train with re-check network
python train_omc.py --weights ../runs/train/l-mot20/weights/best.pt --cfg ../experiments/model_set/CSTrack_l.yaml --batch-size 8 --data ../lib/dataset/mot/cfg/mot20.json --project ../runs/train_w_recheck  --name l-mot20 --device 0 --recheck --noautoanchor
