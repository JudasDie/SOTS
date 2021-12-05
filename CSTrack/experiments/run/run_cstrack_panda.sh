# Data preprocessing
cd ../../lib/utils/panda
python label_clean.py
mpirun -np 2 python split_det.py
mpirun -np 12 python split.py

cd ../../../yolov5_panda
# offline
python train.py --device 0,1 --batch-size 48
mpirun -np 2 python detect_mpi.py   --iou_thres 0.5 \
									--conf_thres 0.4 \
                                    --weights runs/train/1/weights/last.pt

# online
#python train.py --device 0,1,2,3 --batch-size 128
#mpirun -np 2 python detect_mpi.py --device 0 --weights runs/train/1/weights/last.pt

# no training
#mpirun -np 2 python detect_mpi.py --device 0

cd ..
cd tracking
# offline
python train_cstrack_panda.py --device 0,1 --batch_size 32
mpirun -np 2 python test_cstrack_panda_mpi.py   --test_panda True  \
                                                --det_results ../yolov5_panda \
                                                --nms_thres 0.5 \
                                                --conf_thres 0.5 \
                                                --weights ../runs/exp0_mot_test/weights/last_mot_test.pt

# online
#python train_cstrack.py --device 0,1,2,3 --batch_size 64
#mpirun -np 2 python test_cstrack_panda_mpi.py --test_panda True --device 0 --weights ../runs/exp0_mot_test/weights/last_mot_test.pt

# no training
#mpirun -np 2 python test_cstrack_panda_mpi.py --test_panda True --device 0

python test_cstrack_panda_post_process.py
zip -q -r ../results.zip ../results