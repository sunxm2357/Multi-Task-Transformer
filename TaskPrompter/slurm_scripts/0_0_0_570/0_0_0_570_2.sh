#!/bin/bash
#SBATCH -A m4277
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 23:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J 0_0_0_570_4_7
#SBATCH --exclusive
#SBATCH --output=%x_%j.out


export SLURM_CPU_BIND="cores"


cd /pscratch/sd/h/hwchen/code/Multi-Task-Transformer/TaskPrompter

# semseg
srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=55G python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/0_0_0_570/exp4.yml' --run_mode train &
srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=55G python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/0_0_0_570/exp5.yml' --run_mode train &
srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=55G python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/0_0_0_570/exp6.yml' --run_mode train &
srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=55G python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/0_0_0_570/exp7.yml' --run_mode train &
wait