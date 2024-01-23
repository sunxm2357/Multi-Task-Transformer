#!/bin/bash -l

#$ -N m93_95_165_2

#$ -m bea

#$ -M sunxm@bu.edu

# Set SCC project
#$ -P ivc-ml

# Request my job to run on Buy-in Compute group hardware my_project has access to
#$ -l buyin

# Request 4 CPUs
#$ -pe omp 3

# Request 2 GPU
#$ -l gpus=1

# Specify the minimum GPU compute capability
#$ -l gpu_c=7.5

#$  -l gpu_memory=48G

#$ -l h_rt=48:00:00

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="



module load miniconda
module load cuda/12.2
module load gcc
conda activate taskprompter3
#conda install -c conda-forge opencv

cd /projectnb/ivc-ml/sunxm/code/Multi-Task-Transformer/TaskPrompter

  python -m torch.distributed.launch --nproc_per_node=1  --master_port=$((RANDOM%1000+12000))  main_non_overlap_data.py --config_exp './configs/nyud/93_95_165/exp2.yml' --run_mode train
