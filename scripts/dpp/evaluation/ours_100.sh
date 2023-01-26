export CUDA_VISIBLE_DEVICES=0
python run.py \
    --problem dpp \
    --model devformer \
    --resume [model_location]
    --eval_only \