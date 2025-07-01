# BLINK NCBI Entity Linking Pipeline Documentation

## Overview

This repository implements a complete pipeline for biomedical entity linking using the BLINK framework. The system processes NCBI biomedical text and trains neural models for accurate entity disambiguation through a two-stage architecture: fast biencoder candidate retrieval followed by precise cross-encoder reranking.

## Pipeline Architecture

The complete pipeline consists of 5 main stages:
1. **Data Preparation & Preprocessing**
2. **Biencoder Training & Entity Encoding** 
3. **Candidate Retrieval & Filtering**
4. **Cross-Encoder Training**
5. **Evaluation & Inference**

---

## Stage 1: Data Preparation & Preprocessing

### Purpose
Convert raw NCBI/biomedical datasets to BLINK-compatible format and prepare training data.

### Input Data Structure
```
data/ncbi/blink_format/
├── entity.jsonl          # 355,213 entity entries (Knowledge Base)
├── kb.jsonl              # 355,213 knowledge base entities  
├── train.jsonl           # 251 training examples
├── valid.jsonl           # 251 validation examples
├── test.jsonl            # 251 test examples
├── train_ho.jsonl        # 830 high-overlap training examples
└── biencoder/            # Biencoder-specific data
    ├── train.jsonl       # Biencoder training data
    ├── valid.jsonl       # Biencoder validation data
    ├── test.jsonl        # Biencoder test data
    └── kb.jsonl          # Entity knowledge base
```

### Data Format Example
```json
{
  "context_left": "...dominant inheritance of ",
  "context_right": " APC gene mutations...",
  "mention": "adenomatous polyposis coli", 
  "label": "A polyposis syndrome due to...",
  "label_id": 10641,
  "label_title": "Familial Colonic Hyperplasia Syndrome",
  "world": "ncbi"
}
```

### Scripts
- **Primary**: `convert_data.py`
- **Purpose**: Convert raw datasets to BLINK format

### HPC Resources Required
- **CPU**: 1-2 cores
- **Memory**: 4-8 GB RAM
- **Time**: 10-30 minutes
- **Storage**: ~500 MB for processed data

### Execution
```bash
python convert_data.py
```

---

## Stage 2: Biencoder Training & Entity Encoding

### Purpose
Train dual-encoder model for fast candidate retrieval and pre-compute entity embeddings.

### Architecture
- **Biencoder**: Separate BERT encoders for mentions and entities
- **Model**: BERT-large-uncased (768 dimensions)
- **Training**: Contrastive learning with in-batch negatives

### Scripts
- **Training**: `blink/biencoder/train_biencoder.py`
- **Encoding**: `blink/biencoder/eval_biencoder.py`
- **Shell Scripts**: `train.sh`, `get_encoding.sh`

### HPC Resources Required

#### Biencoder Training
- **GPU**: 1-2 NVIDIA V100/A100 (16-32 GB VRAM)
- **CPU**: 8-16 cores
- **Memory**: 32-64 GB RAM
- **Time**: 2-6 hours (5 epochs)
- **Storage**: ~2-5 GB for model checkpoints

#### Entity Encoding Generation
- **GPU**: 1 NVIDIA V100/A100 (16 GB VRAM)
- **CPU**: 4-8 cores  
- **Memory**: 16-32 GB RAM
- **Time**: 30-60 minutes (355K entities)
- **Storage**: ~1-2 GB for embeddings

### Execution

#### Train Biencoder (Optional - can use pre-trained)
```bash
python blink/biencoder/train_biencoder.py \
  --data_path data/ncbi/blink_format/biencoder \
  --output_path models/ncbi/biencoder \
  --learning_rate 1e-05 \
  --num_train_epochs 5 \
  --max_context_length 128 \
  --max_cand_length 128 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --bert_model bert-large-uncased \
  --type_optimization all_encoder_layers
```

#### Generate Entity Encodings
```bash
./get_encoding.sh
# OR
python blink/biencoder/eval_biencoder.py \
  --path_to_model models/ncbi/biencoder/pytorch_model.bin \
  --entity_dict_path data/ncbi/blink_format/kb.jsonl \
  --cand_encode_path models/ncbi/entity_encodings_by_biencoder.t7 \
  --encode_batch_size 8 \
  --only_get_ent_encoding true
```

### Output Files
- `models/ncbi/biencoder/pytorch_model.bin` - Trained biencoder model
- `models/ncbi/entity_encodings_by_biencoder*.t7` - Pre-computed entity embeddings

---

## Stage 3: Candidate Retrieval & Filtering

### Purpose
Use biencoder to retrieve top-K candidates for each mention and filter training examples.

### Process
1. **Retrieval**: Cosine similarity between mention and entity embeddings
2. **Filtering**: Remove examples where gold entity not in top-64 candidates
3. **Impact**: Reduces training data from 251 → 68 examples (27% retention rate)

### Scripts
- **Primary**: `get_biencoder_candidate.sh`
- **Core Module**: `blink/biencoder/eval_biencoder.py`
- **Filtering Logic**: `blink/biencoder/nn_prediction.py`

### HPC Resources Required
- **GPU**: 1 NVIDIA V100/A100 (8-16 GB VRAM)
- **CPU**: 4-8 cores
- **Memory**: 16-32 GB RAM
- **Time**: 15-30 minutes
- **Storage**: ~100-500 MB for candidate files

### Execution
```bash
./get_biencoder_candidate.sh
# OR
python blink/biencoder/eval_biencoder.py \
  --path_to_model models/ncbi/biencoder/pytorch_model.bin \
  --entity_dict_path data/ncbi/blink_format/kb.jsonl \
  --cand_encode_path models/ncbi/entity_encodings_by_biencoder.t7 \
  --data_path data/ncbi/blink_format \
  --output_path models/ncbi \
  --encode_batch_size 8 \
  --eval_batch_size 1 \
  --top_k 64 \
  --save_topk_result \
  --bert_model bert-base-uncased \
  --mode test
```

### Output Files
- `models/ncbi/top64_candidates/train.t7` - Training candidates [68, 64, 128]
- `models/ncbi/top64_candidates/valid.t7` - Validation candidates 
- `models/ncbi/top64_candidates/test.t7` - Test candidates

### Data Structure
- **Shape**: [num_examples, 64, 128]
- **Content**: 68 examples, 64 candidates each, 128 tokens per candidate
- **Format**: 
  - `context_vecs`: [68, 128] mention contexts
  - `candidate_vecs`: [68, 64, 128] candidate entities
  - `labels`: [68] gold candidate indices (0-63)

---

## Stage 4: Cross-Encoder Training

### Purpose
Train a reranker model to accurately rank the top-64 candidates from the biencoder.

### Architecture
- **Model**: Single BERT encoder for concatenated (context, candidate) pairs
- **Training**: Cross-entropy loss over 64 candidates per example
- **Input**: Each example produces 64 (context, candidate) pairs
- **Output**: Single score per pair → softmax over 64 scores

### Scripts
- **Primary**: `slurm.sh` (SLURM job submission)
- **Core Module**: `blink/crossencoder/train_cross.py`
- **Architecture**: `blink/crossencoder/crossencoder.py`

### HPC Resources Required
- **GPU**: 1-2 NVIDIA V100/A100 (16-32 GB VRAM)
- **CPU**: 8-16 cores
- **Memory**: 32-64 GB RAM
- **Time**: 2-4 hours (5 epochs, 68 examples)
- **Storage**: ~1-3 GB for model checkpoints
- **Queue**: Use GPU partition with SLURM

### SLURM Configuration
```bash
#SBATCH --job-name=cross_encoder_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
```

### Execution
```bash
# Submit SLURM job
sbatch slurm.sh

# OR direct execution
python blink/crossencoder/train_cross.py \
  --data_path models/ncbi/top64_candidates/ \
  --output_path models/ncbi/crossencoder \
  --learning_rate 2e-05 \
  --num_train_epochs 5 \
  --max_context_length 128 \
  --max_seq_length 128 \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --bert_model bert-base-uncased \
  --type_optimization all_encoder_layers \
  --add_linear
```

### Training Parameters
```bash
--learning_rate 2e-05          # Learning rate for AdamW
--num_train_epochs 5           # Number of training epochs  
--train_batch_size 2           # Batch size (limited by GPU memory)
--eval_batch_size 2            # Evaluation batch size
--max_context_length 128       # Max tokens for mention context
--max_seq_length 128           # Max tokens for full sequence
--gradient_accumulation_steps 1 # Gradient accumulation
--warmup_proportion 0.1        # Warmup steps proportion
--max_grad_norm 1.0           # Gradient clipping
```

### Output Files
- `models/ncbi/crossencoder/epoch_0/` - Epoch 0 checkpoint
- `models/ncbi/crossencoder/epoch_1/` - Epoch 1 checkpoint  
- `models/ncbi/crossencoder/epoch_X/` - Best performing epoch
- `models/ncbi/crossencoder/training_time.txt` - Training duration
- `models/ncbi/crossencoder/training_params.txt` - Training configuration

---

## Stage 5: Evaluation & Inference

### Purpose
Run end-to-end entity linking pipeline and evaluate performance.

### Process
1. **Biencoder**: Retrieve top-64 candidates using fast similarity search
2. **Cross-encoder**: Rerank candidates using trained cross-encoder model  
3. **Evaluation**: Compute accuracy metrics at both stages

### Scripts
- **Primary**: `blink/main_dense.py`
- **Evaluation**: Built-in evaluation functions

### HPC Resources Required
- **GPU**: 1 NVIDIA V100/A100 (8-16 GB VRAM)
- **CPU**: 4-8 cores
- **Memory**: 16-32 GB RAM  
- **Time**: 10-30 minutes
- **Storage**: Minimal (~10 MB for results)

### Execution
```bash
python blink/main_dense.py \
  --biencoder_model models/ncbi/biencoder/pytorch_model.bin \
  --biencoder_config models/ncbi/biencoder/config.json \
  --entity_catalogue data/ncbi/blink_format/kb.jsonl \
  --entity_encoding models/ncbi/entity_encodings_by_biencoder.t7 \
  --crossencoder_model models/ncbi/crossencoder/epoch_X/pytorch_model.bin \
  --crossencoder_config models/ncbi/crossencoder/epoch_X/config.json \
  --test_mentions data/ncbi/blink_format/test.jsonl \
  --output_path output/ncbi_results \
  --top_k 64
```

### Evaluation Metrics
- **Biencoder Accuracy**: Top-1 accuracy in retrieved candidates
- **Biencoder Recall@64**: Fraction of gold entities in top-64
- **Cross-encoder Accuracy**: Top-1 accuracy after reranking
- **Overall Accuracy**: Final system performance

---

## Complete Execution Workflow

### Full Training Pipeline
```bash
# 1. Prepare data (if needed)
python convert_data.py

# 2. Train biencoder (optional - can use pre-trained)
python blink/biencoder/train_biencoder.py \
  --data_path data/ncbi/blink_format/biencoder \
  --output_path models/ncbi/biencoder

# 3. Generate entity encodings
./get_encoding.sh

# 4. Generate top-64 candidates
./get_biencoder_candidate.sh

# 5. Train cross-encoder
sbatch slurm.sh

# 6. Evaluate end-to-end performance
python blink/main_dense.py \
  --biencoder_model models/ncbi/biencoder/pytorch_model.bin \
  --entity_encoding models/ncbi/entity_encodings_by_biencoder.t7 \
  --crossencoder_model models/ncbi/crossencoder/epoch_X/pytorch_model.bin \
  --test_mentions data/ncbi/blink_format/test.jsonl
```

### Resource Summary
| Stage | GPU | CPU | Memory | Time | Storage |
|-------|-----|-----|--------|------|---------|
| Data Prep | - | 1-2 cores | 4-8 GB | 10-30 min | 500 MB |
| Biencoder Train | 1-2 V100/A100 | 8-16 cores | 32-64 GB | 2-6 hrs | 2-5 GB |
| Entity Encoding | 1 V100/A100 | 4-8 cores | 16-32 GB | 30-60 min | 1-2 GB |
| Candidate Retrieval | 1 V100/A100 | 4-8 cores | 16-32 GB | 15-30 min | 500 MB |
| Cross-encoder Train | 1-2 V100/A100 | 8-16 cores | 32-64 GB | 2-4 hrs | 1-3 GB |
| Evaluation | 1 V100/A100 | 4-8 cores | 16-32 GB | 10-30 min | 10 MB |

---

## Using Pre-trained BLINK Models (No Fine-tuning)

If you want to use the pre-trained BLINK models without fine-tuning on your specific data, follow this simplified workflow:

### Overview
This approach uses the original BLINK models trained on Wikipedia data for entity linking. It's faster to deploy but may have lower accuracy on domain-specific (e.g., biomedical) text.

### Prerequisites
1. Download pre-trained BLINK models
2. Prepare your data in BLINK format
3. Set up the inference pipeline

### Step 1: Download Pre-trained Models

```bash
# Download pre-trained models (if not already available)
./download_blink_models.sh

# This downloads:
# - models/biencoder_wiki_large.bin (Biencoder model)
# - models/biencoder_wiki_large.json (Biencoder config)  
# - models/crossencoder_wiki_large.bin (Cross-encoder model)
# - models/crossencoder_wiki_large.json (Cross-encoder config)
# - models/entity.jsonl (Wikipedia entity catalog)
# - models/all_entities_large.t7 (Pre-computed entity embeddings)
```

### HPC Resources Required
- **CPU**: 4-8 cores
- **Memory**: 8-16 GB RAM
- **Time**: 30-60 minutes (download)
- **Storage**: 10-20 GB for models and data

### Step 2: Prepare Your Data

Convert your mention data to BLINK format:

```json
{
  "id": 0,
  "label": "unknown",
  "label_id": -1, 
  "context_left": "Text before the mention",
  "mention": "entity mention",
  "context_right": "Text after the mention"
}
```

Save as JSONL file (e.g., `data/your_dataset/test.jsonl`)

### Step 3: Generate Entity Encodings (Optional)

If using custom entity catalog:

```bash
python blink/biencoder/eval_biencoder.py \
  --path_to_model models/biencoder_wiki_large.bin \
  --entity_dict_path your_custom_entities.jsonl \
  --cand_encode_path models/your_entity_encodings.t7 \
  --encode_batch_size 8 \
  --only_get_ent_encoding true
```

### HPC Resources Required
- **GPU**: 1 NVIDIA V100/A100 (8-16 GB VRAM)
- **CPU**: 4-8 cores
- **Memory**: 16-32 GB RAM
- **Time**: 1-3 hours (depending on entity catalog size)

### Step 4: Run Inference

#### Option A: End-to-End Pipeline
```bash
python blink/main_dense.py \
  --biencoder_model models/biencoder_wiki_large.bin \
  --biencoder_config models/biencoder_wiki_large.json \
  --entity_catalogue models/entity.jsonl \
  --entity_encoding models/all_entities_large.t7 \
  --crossencoder_model models/crossencoder_wiki_large.bin \
  --crossencoder_config models/crossencoder_wiki_large.json \
  --test_mentions data/your_dataset/test.jsonl \
  --output_path output/results \
  --top_k 64
```

#### Option B: Biencoder Only (Faster)
```bash
python blink/main_dense.py \
  --biencoder_model models/biencoder_wiki_large.bin \
  --biencoder_config models/biencoder_wiki_large.json \
  --entity_catalogue models/entity.jsonl \
  --entity_encoding models/all_entities_large.t7 \
  --test_mentions data/your_dataset/test.jsonl \
  --output_path output/results \
  --top_k 10 \
  --fast
```

### HPC Resources Required
- **GPU**: 1 NVIDIA V100/A100 (8-16 GB VRAM) 
- **CPU**: 4-8 cores
- **Memory**: 16-32 GB RAM
- **Time**: 10-60 minutes (depending on dataset size)
- **Storage**: Minimal (~10-100 MB for results)

### Step 5: Interactive Mode (Optional)

For real-time entity linking:

```bash
python blink/main_dense.py \
  --biencoder_model models/biencoder_wiki_large.bin \
  --entity_catalogue models/entity.jsonl \
  --entity_encoding models/all_entities_large.t7 \
  --crossencoder_model models/crossencoder_wiki_large.bin \
  --interactive
```

### Expected Performance

#### With Wikipedia Models on General Text:
- **Biencoder Accuracy**: 70-85%
- **Cross-encoder Accuracy**: 85-95%
- **Speed**: ~10-100 mentions/second

#### With Wikipedia Models on Biomedical Text:
- **Biencoder Accuracy**: 40-60% (domain mismatch)
- **Cross-encoder Accuracy**: 60-80%  
- **Recommendation**: Fine-tune for better performance

### Advantages of Pre-trained Approach
- ✅ **Fast deployment**: No training required
- ✅ **Large entity coverage**: Wikipedia-scale knowledge base
- ✅ **General purpose**: Works across domains
- ✅ **Lower resource requirements**: No GPU training needed

### Disadvantages of Pre-trained Approach  
- ❌ **Domain mismatch**: Lower accuracy on specialized text
- ❌ **Fixed entity set**: Limited to Wikipedia entities
- ❌ **No adaptation**: Cannot learn domain-specific patterns

### When to Use Each Approach

**Use Pre-trained Models When:**
- Working with general domain text
- Need quick deployment
- Limited computational resources
- Prototyping or proof-of-concept

**Use Fine-tuned Models When:**
- Working with specialized domains (biomedical, legal, etc.)
- Have domain-specific training data
- Need highest possible accuracy
- Have sufficient computational resources

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--train_batch_size 1 --eval_batch_size 1`
   - Use gradient accumulation: `--gradient_accumulation_steps 4`

2. **Small Training Set After Filtering**
   - Increase top-k candidates: `--top_k 128`
   - Use data augmentation techniques
   - Check biencoder performance

3. **Slow Entity Encoding**
   - Increase batch size: `--encode_batch_size 16`
   - Use multiple GPUs if available

4. **SLURM Job Failures**
   - Check resource requests match available hardware
   - Verify file paths and permissions
   - Monitor job logs: `squeue` and `sacct`

### Performance Optimization

1. **For Faster Training:**
   - Use mixed precision: `--fp16`
   - Increase batch size on larger GPUs
   - Use multiple GPUs with `--data_parallel`

2. **For Better Accuracy:**
   - Increase sequence length: `--max_seq_length 256`
   - Try different learning rates: `1e-05`, `2e-05`, `3e-05`
   - Increase training epochs: `--num_train_epochs 10`

3. **For Memory Optimization:**
   - Use gradient checkpointing
   - Reduce maximum sequence length
   - Process data in smaller batches

---

## File Structure

```
BLINK_rasel/
├── README_PIPELINE.md          # This documentation
├── slurm.sh                    # Cross-encoder training script  
├── train.sh                    # Biencoder evaluation script
├── get_biencoder_candidate.sh  # Candidate generation script
├── get_encoding.sh             # Entity encoding script
├── convert_data.py             # Data preprocessing
├── data/
│   └── ncbi/
│       └── blink_format/       # BLINK-formatted data
├── models/
│   └── ncbi/
│       ├── biencoder/          # Biencoder models
│       ├── crossencoder/       # Cross-encoder models
│       └── top64_candidates/   # Candidate files
├── blink/
│   ├── biencoder/              # Biencoder modules
│   ├── crossencoder/           # Cross-encoder modules
│   ├── main_dense.py           # End-to-end pipeline
│   └── ...
└── output/                     # Results and logs
```

## References

- [BLINK: Better Language Understanding with Entity Linking](https://arxiv.org/abs/1911.03814)
- [Zero-shot Entity Linking with Efficient Long Range Sequencing](https://arxiv.org/abs/1902.10137)
- [Original BLINK Repository](https://github.com/facebookresearch/BLINK)

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@inproceedings{wu2020zero,
  title={Zero-shot Entity Linking with Efficient Long Range Sequencing},
  author={Wu, Ledell and Petroni, Fabio and Josifoski, Martin and Riedel, Sebastian and Zettlemoyer, Luke},
  booktitle={ICLR},
  year={2020}
}
```
