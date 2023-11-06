#!/bin/bash
#SBATCH -A m4277
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -n 1
#SBATCH --gpus-per-task=4
#SBATCH --gpu-bind=none
#SBATCH -J init_set

export SLURM_CPU_BIND="cores"

conda activate sunxm
module load pytorch

cd /pscratch/sd/h/hwchen/code/Multi-Task-Transformer/TaskPrompter

CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/nyud_vitLp16_taskprompter_no_overlap_initset.yml' --run_mode train  #--trained_model  pretrained_mode0l_path