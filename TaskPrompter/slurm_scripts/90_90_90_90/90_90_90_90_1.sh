#!/bin/bash
#SBATCH -A m4277
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 23:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J 90_90_90_90_same_0_3
#SBATCH --exclusive
#SBATCH --output=%x_%j.out

export SLURM_CPU_BIND="cores"


cd /pscratch/sd/h/hwchen/code/Multi-Task-Transformer/TaskPrompter

# semseg
srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=55G python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/90_90_90_90_same/exp0.yml' --run_mode train &
srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=55G python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/90_90_90_90_same/exp1.yml' --run_mode train &
srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=55G python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/90_90_90_90_same/exp2.yml' --run_mode train &
srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=55G python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/90_90_90_90_same/exp3.yml' --run_mode train &
wait