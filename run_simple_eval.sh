#!/bin/bash
#SBATCH --job-name=nlp
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --account=qli-lab
#SBATCH --qos=normal
#SBATCH --mail-type=END,FAIL         
#SBATCH --mail-user=lan0908@iastate.edu

# Simple CrossEncoder Evaluation Script

echo "CrossEncoder Evaluation"
echo "======================="

# Activate environment
eval $(/work/LAS/qli-lab/nhat/anaconda3/bin/conda shell.bash hook)
source /work/LAS/qli-lab/nhat/anaconda3/etc/profile.d/conda.sh
conda activate /work/LAS/qli-lab/nhat/conda_envs/blink37_v2

export PYTHONPATH=.

# Run evaluation
python categorize_test_data.py
