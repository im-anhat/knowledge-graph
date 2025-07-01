source venv/bin/activate
export PYTHONPATH=.

# python blink/biencoder/train_biencoder.py \
#   --data_path data/ncbi/blink_format/biencoder \
#   --output_path models/ncbi/biencoder  \
#   --learning_rate 1e-05 \
#   --num_train_epochs 5 \
#   --max_context_length 128 \
#   --max_cand_length 128 \
#   --train_batch_size 16 \
#   --eval_batch_size 16 \
#   --num_train_epochs 5 \
#   --bert_model bert-large-uncased \
#   --type_optimization all_encoder_layers \
#   # --data_parallel \
#   # --debug \
# #   --data_parallel \

python blink/biencoder/eval_biencoder.py \
  --path_to_model models/ncbi/biencoder/pytorch_model.bin \
  --entity_dict_path data/ncbi/blink_format/biencoder/kb.jsonl \
  --cand_encode_path models/ncbi/entity_encodings_by_biencoder_wiki_large_tuned_prime.t7 \
  --data_path data/ncbi/blink_format/biencoder \
  --output_path models/ncbi/biencoder \
  --encode_batch_size 8 --eval_batch_size 1 --top_k 64 --save_topk_result \
  --bert_model bert-large-uncased --mode train,valid,test \
  # --zeshel \
  # --debug \
# # #   --data_parallel

# python blink/crossencoder/train_cross.py \
#   --data_path  models/zeshel/top64_candidates/ \
#   --output_path models/zeshel/crossencoder \
#   --learning_rate 2e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 10 \
#   --train_batch_size 2 --eval_batch_size 2 --bert_model bert-large-uncased \
#   --type_optimization all_encoder_layers --add_linear \
#   --zeshel \
# #   --debug \
# #   --data_parallel \

