CUDA_VISIBLE_DEVICES=0,1 python ./tools/train_net.py --config-file ./configs/own/faster_rcnn_R_50_FPN_0.yaml --num-gpus 2

CUDA_VISIBLE_DEVICES=4,5,6,7 python ./tools/train_net.py --config-file ./configs/own/faster_rcnn_R_50_FPN_1_one.yaml --num-gpus 4

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_net.py --config-file ./configs/own/faster_rcnn_R_50_FPN_2.yaml --num-gpus 4 

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./tools/train_net.py --config-file ./configs/own/faster_rcnn_R_50_FPN_3.yaml --num-gpus 4 

# CUDA_VISIBLE_DEVICES=0,1 python ./tools/train_net.py --config-file ./configs/own/faster_rcnn_R_50_FPN_4.yaml --num-gpus 2


# CUDA_VISIBLE_DEVICES=2,3 python ./tools/train_net.py --config-file ./configs/own/base.yaml --num-gpus 2
# 
# CUDA_VISIBLE_DEVICES=0,1 python ./tools/train_net.py --config-file ./configs/own/faster_rcnn_R_50_FPN_4.yaml --num-gpus 2