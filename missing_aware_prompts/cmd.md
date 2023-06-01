python make_arrow.py --dataset food101 --root /home/workshop/dataset/

python run.py with data_root=/home/zhangjiaqing/missing_aware_prompts-main/datasets/Food101 num_gpus=1  num_nodes=1 per_gpu_batchsize=2  task_finetune_food101 load_path=vilt_200k_mlm_itm.ckpt exp_name=food101


ARROW_ROOT=./datasets/mmimdb
NUM_GPUS=2
NUM_NODES=1
BS_FITS_YOUR_GPU=2
PRETRAINED_MODEL_PATH=./pretrained_weight/vilt_200k_mlm_itm.ckpt
EXP_NAME=mmimdb