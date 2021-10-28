# if permission error, run
# chmod +x run/sample.sh

set -x

PY_ARGS=${@:1}

python -m torch.distributed.launch \
       --nproc_per_node=8 \
    --use_env src/main.py \
    --wandb_pname Sample \
    --wandb_entity #YOUR_WANDB_ID# \
    --dataset cifar10 \
    --data_dir data/cifar10/ \
    --model sample \
    --use_wandb \
    ${PY_ARGS}