#!/bin/bash
#SBATCH --job-name=nlp
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00
#SBATCH --account=qli-lab
#SBATCH --qos=normal
#SBATCH --mail-type=END,FAIL         
#SBATCH --mail-user=lan0908@iastate.edu

eval $(/work/LAS/qli-lab/nhat/anaconda3/bin/conda shell.bash hook)
source /work/LAS/qli-lab/nhat/anaconda3/etc/profile.d/conda.sh
conda activate /work/LAS/qli-lab/nhat/conda_envs/blink37

# Navigate to the BLINK directory
cd /work/LAS/qli-lab/nhat/KANG_BLINK/BLINK_rasel

# Run the cross-encoder training script
export PYTHONPATH=.

# context_input shape:  torch.Size([246, 128])
# candidate_input shape:  torch.Size([246, 64, 128])
# label_input shape:  torch.Size([246])

# Change data path, output path, context length
python blink/crossencoder/train_cross.py \
  --data_path  models/ncbi/top64_candidates_pretrained_original_def_64 \
  --output_path models/ncbi/crossencoder_original_def_64 \
  --path_to_model models/crossencoder_wiki_large.bin \
  --learning_rate 2e-05 \
  --num_train_epochs 5 \
  --max_context_length 64 \
  --max_cand_length 192 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --bert_model bert-large-uncased \
  --type_optimization all_encoder_layers \
  --add_linear \
  --setting original_def \
  --data_parallel



# Always use 128 for candidate length, change context length + top64 directory location only
# python blink/biencoder/eval_biencoder.py \
#   --path_to_model models/biencoder_wiki_large.bin \
#   --entity_dict_path data/bc5cdr/blink_format/kb.jsonl \
#   --cand_encode_path models/bc5cdr/original_def_encodings_64.t7 \
#   --max_context_length 64 \
#   --max_cand_length 128 \
#   --data_path data/bc5cdr/blink_format \
#   --output_path models/ncbi \
#   --encode_batch_size 8 --eval_batch_size 1 --top_k 64 --save_topk_result \
#   --bert_model bert-large-uncased --mode train,test,valid

# python test.py