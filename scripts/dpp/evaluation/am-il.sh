export CUDA_VISIBLE_DEVICES=0
python run.py \
    --problem dpp \
    --model attention \
    --resume [model_location]
    --eval_only \