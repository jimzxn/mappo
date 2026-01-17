#!/bin/sh
env="StarCraft2"
map="3s_vs_4z"
algo="mappo"
exp="check"
seed_max=20
seed_min=1
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"


for seed in $(seq ${seed_min} ${seed_max});
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 --lr 1e-3 --critic_lr 5e-4  \
    --num_env_steps 10000000 --ppo_epoch 15 --use_value_active_masks --use_eval --eval_episodes 32 --clip_param 0.1 --wandb_name "mappo" --user_name "jimzhao0422"  
done
