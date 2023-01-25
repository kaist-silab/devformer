export CUDA_VISIBLE_DEVICES=0
python run.py \
    --problem dpp \
    --model attention \
    --training_mode IL \
    --train_dataset data/dpp/training_2000_new.pkl\
    --guiding_action data/dpp/guiding_2000_new.pkl\
