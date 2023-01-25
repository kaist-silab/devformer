export CUDA_VISIBLE_DEVICES=0

for i in {1..5}
do
    python run.py \
        --problem tsp \
        --N_aug 4 \
        --model attention \
        --training_mode IL \
        --val_dataset data/tsp/tsp20_test_seed1111.pkl \
        --train_dataset data/tsp/tsp20_problem.pkl \
        --guiding_action data/tsp/tsp20_solution.pkl \
        --EE \
        --SE \
        --seed $i
done
