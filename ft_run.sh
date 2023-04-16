CUDA_VISIBLE_DEVICES=4,5 python ./tools/train_net.py --config-file ./configs/own/finetune0.yaml --num-gpus 2 --dist-url tcp://127.0.0.1:54200

CUDA_VISIBLE_DEVICES=4,5 python ./tools/train_net.py --config-file ./configs/own/finetune1.yaml --num-gpus 2 --dist-url tcp://127.0.0.1:54200

CUDA_VISIBLE_DEVICES=4,5 python ./tools/train_net.py --config-file ./configs/own/finetune2.yaml --num-gpus 2 --dist-url tcp://127.0.0.1:54200 

CUDA_VISIBLE_DEVICES=4,5 python ./tools/train_net.py --config-file ./configs/own/finetune3.yaml --num-gpus 2 --dist-url tcp://127.0.0.1:54200

