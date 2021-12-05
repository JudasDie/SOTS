#test with recheck
python test_omc.py --weights ../model/OMC_mot17.pt --cfg ../experiments/model_set/CSTrack_l.yaml --name l-mot17-test --test_mot17 True --output_root runs/test_w_recheck
python test_omc.py --weights ../model/OMC_mot20.pt --cfg ../experiments/model_set/CSTrack_l.yaml --name l-mot20-test --test_mot20 True --output_root runs/test_w_recheck