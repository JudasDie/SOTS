#for yolov5  test
cd ../../yolov5_panda
mpirun -np 1 python detect_mpi.py --iou_thres 0.5 --conf_thres 0.4 --weights weights/yolov5_panda.pt
#for CSTrack test
cd ../tracking
mpirun -np 1 python test_cstrack_panda_mpi.py --test_panda True --det_results ../yolov5_panda --nms_thres 0.5 --conf_thres 0.5 --weights ../weights/cstrack_panda.pt
python test_cstrack_panda_post_process.py
zip -q -r ../results.zip ../results
