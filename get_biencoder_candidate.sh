source venv/bin/activate
export PYTHONPATH=.

# entity_encodings_by_biencoder.t7 has shape (num_entities, embedding_dim)
python blink/biencoder/eval_biencoder.py \
  --path_to_model models/biencoder_wiki_large.bin \
  --entity_dict_path data/ncbi/blink_format/prime_def/kb.jsonl \
  --cand_encode_path models/ncbi/entity_encodings_by_biencoder.t7 \
  --data_path data/ncbi/blink_format/prime_def \
  --output_path models/ncbi \
  --encode_batch_size 8 --eval_batch_size 1 --top_k 64 --save_topk_result \
  --bert_model bert-large-uncased --mode train,test,valid 

  