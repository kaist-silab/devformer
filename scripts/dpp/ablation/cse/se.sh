export CUDA_VISIBLE_DEVICES=0
python run.py \
    --problem dpp \
    --N_aug 4 \
    --model devformer \
    --training_mode IL \
    --train_dataset data/dpp/training_100_new.pkl\
    --guiding_action data/dpp/guiding_100_new.pkl\
    --SE \