export CUDA_VISIBLE_DEVICES=0

python run.py \
    --problem tsp \
    --N_aug 4 \
    --model attention \
    --training_mode IL \
    --val_dataset data/tsp/tsp100_test_seed1111.pkl \
    --train_dataset data/tsp/tsp100_problem.pkl \
    --guiding_action data/tsp/tsp100_solution.pkl \
    --seed 5



