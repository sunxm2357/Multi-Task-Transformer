#!/bin/bash
#SBATCH -A m4277
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 23:59:59
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J all_baselines
#SBATCH --exclusive
#SBATCH --output=%x_%j.out


export SLURM_CPU_BIND="cores"

cd /pscratch/sd/h/hwchen/code/Multi-Task-Transformer/TaskPrompter

srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-cpu=55G python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/0_0_0_570/exp0.yml' --run_mode train &
srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-cpu=55G python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/0_0_0_570/exp1.yml' --run_mode train &
srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-cpu=55G python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/0_0_0_570/exp2.yml' --run_mode train &
srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-cpu=55G python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/0_0_0_570/exp3.yml' --run_mode train &
wait