export CUDA_VISIBLE_DEVICES=0
python run.py \
    --problem dpp \
    --model attention \
    --training_mode RL \
    --baseline rollout \
    --epoch_size 50000\
    --n_epochs 10 \
    --train_dataset data/dpp/training_50000.pkl \
    --checkpoint_epochs 1 \

