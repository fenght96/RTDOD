CUDA_VISIBLE_DEVICES=6,7 python ./tools/train_net.py --config-file ./configs/own/base_night.yaml --num-gpus 2 --dist-url tcp://127.0.0.1:54300

CUDA_VISIBLE_DEVICES=6,7 python ./tools/train_net.py --config-file ./configs/own/base_evening.yaml --num-gpus 2 --dist-url tcp://127.0.0.1:54300

CUDA_VISIBLE_DEVICES=6,7 python ./tools/train_net.py --config-file ./configs/own/base_day.yaml --num-gpus 2 --dist-url tcp://127.0.0.1:54300

