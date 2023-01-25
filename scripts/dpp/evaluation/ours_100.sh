export CUDA_VISIBLE_DEVICES=0
python run.py \
    --problem dpp \
    --model device_transformer \
    --resume [model_location]
    --eval_only \