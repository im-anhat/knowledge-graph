source venv/bin/activate
log_file="eval_biencoder_log.txt"
error_log_file="eval_biencoder_error_log.txt"
export PYTHONPATH=.
python blink/biencoder/eval_biencoder.py \
    --path_to_model models/biencoder_wiki_large.bin \
    --data_path examples/bc5cdr \
    --entity_dict_path examples/bc5cdr/kb.jsonl \
    --cand_encode_path models/bc5cdr/entity_encodings_by_biencoder_wiki_large.t7 \
    --output_path models/bc5cdr \
    --encode_batch_size 8 \
    --eval_batch_size 1 \
    --top_k 64 \
    --save_topk_result \
    --bert_model bert-large-uncased \
    --mode test \
    --only_get_ent_encoding true \
    # --data_parallel \
    # --debug 

#     # --zeshel 
    
#    --data_parallel >> $log_file 2>> $error_log_file

# --debug \
# --data_parallel >> $log_file 2>> $error_log_file
# --entity_dict_path examples/ncbi/ncbi.json \


# python  blink/biencoder/eval_biencoder.py \
#     --path_to_model models/biencoder_wiki_large.bin \
#     --data_path examples/bc5 \
#     --entity_dict_path examples/bc5/kb.jsonl \
#     --cand_encode_path models/bc5/entity_encodings_by_biencoder.t7 \
#     --output_path models/bc5 \
#     --encode_batch_size 64 \
#     --eval_batch_size 1 \
#     --top_k 64 \
#     --save_topk_result \
#     --bert_model bert-large-uncased \
#     --mode test \

# python  blink/biencoder/eval_biencoder.py \
#     --path_to_model models/biencoder_wiki_large.bin \
#     --data_path examples/cometa \
#     --entity_dict_path examples/cometa/kb.jsonl \
#     --cand_encode_path models/cometa/entity_encodings_by_biencoder.t7 \
#     --output_path models/cometa \
#     --encode_batch_size 64 \
#     --eval_batch_size 1 \
#     --top_k 64 \
#     --save_topk_result \
#     --bert_model bert-large-uncased \
#     --mode test \


# python  blink/biencoder/eval_biencoder.py \
#     --path_to_model models/biencoder_wiki_large.bin \
#     --data_path examples/medmentions \
#     --entity_dict_path examples/medmentions/kb.jsonl \
#     --cand_encode_path models/medmentions/entity_encodings_by_biencoder.t7 \
#     --output_path models/medmentions \
#     --encode_batch_size 64 \
#     --eval_batch_size 1 \
#     --top_k 64 \
#     --save_topk_result \
#     --bert_model bert-large-uncased \
#     --mode test \
