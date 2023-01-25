export CUDA_VISIBLE_DEVICES=0
python run.py \
    --problem dpp \
    --N_aug 4 \
    --model device_transformer \
    --training_mode IL \
    --train_dataset data/dpp/training_2000_new.pkl\
    --guiding_action data/dpp/guiding_2000_new.pkl\
    --EE \
    --SE \
    --batch_size 200 \