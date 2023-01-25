export CUDA_VISIBLE_DEVICES=0
python run.py \
    --problem tsp \
    --model attention \
    --training_mode IL \
    --resume [location] \
    --eval_only \

