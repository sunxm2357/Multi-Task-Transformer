#!/bin/bash
#SBATCH -A m4277
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 23:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J 43_142_142_142_2


export SLURM_CPU_BIND="cores"

conda activate sunxm
module load pytorch

cd /pscratch/sd/h/hwchen/code/Multi-Task-Transformer/TaskPrompter

# semseg
srun --ntasks=1 python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/43_142_142_142/exp4.yml' --run_mode train &
srun --ntasks=1 python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/43_142_142_142/exp5.yml' --run_mode train &
srun --ntasks=1 python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/43_142_142_142/exp6.yml' --run_mode train &
srun --ntasks=1 python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/43_142_142_142/exp7.yml' --run_mode train &
wait