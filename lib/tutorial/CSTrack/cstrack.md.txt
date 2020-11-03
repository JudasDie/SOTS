conda create -n CSTrack python=3.8
source activate CSTrack
cd CSTrack/
pip install -r requirements.txt

#Train
python train.py --batch_size 10 --device 0 --data cfg/data_ch.json

#Test
python track.py --nms_thres 0.6
                         --conf_thres 0.5
                         --weights /CSTrack/runs/exp0_mot_test/weights/last_mot_test.pt
                         --device 0
                         --test_mot16 True
