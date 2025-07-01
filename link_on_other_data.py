import io
import json
import os
import blink.main_dense as main_dense
import argparse

from performence import save_report


def link_ent(onto):
    
    models_path = "models/"
    data_path = f'examples/{onto}/'
    mentions_file = f'{data_path}test_all.jsonl'
    config = {
        "test_entities": f'{data_path}/kb.jsonl',
        "test_mentions": mentions_file,
        "interactive": False,
        "top_k": 64,
        "biencoder_model": models_path+f"biencoder_wiki_large.bin",
        # "biencoder_model": models_path+f"{onto}/biencoder/pytorch_model.bin",
        "biencoder_config": models_path+"biencoder_wiki_large.json",
        "entity_catalogue": f"{data_path}/entity.jsonl",
        "entity_encoding": f"models/{onto}/entity_encodings_by_biencoder_wiki_large.t7",
        "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
        "crossencoder_config": models_path+"crossencoder_wiki_large.json",
        "fast": False, # set this to be true if speed is a concern
        "output_path": "logs/" # logging directory
    }
    args = argparse.Namespace(**config)
    models = main_dense.load_models(args, logger=None)
    cross_preds = []
    bi_and_cross_pred = []
    c = 0
    c_er = 0
    with io.open(mentions_file, mode="r", encoding="utf-8") as file:
        for line in file:
            try:
                data_to_link = [json.loads(line.strip())]
                biencoder_accuracy, recall_at, crossencoder_normalized_accuracy, overall_unormalized_accuracy, samplesaa_len, bi_and_cross, predictions, scores, = main_dense.run(args, 
                                                                                                                                                            None, *models, test_data=data_to_link)
                bi_and_cross_pred.append(bi_and_cross)
                cross_preds.append(predictions)
                # if c>=5:
                #     break
                # else:
                #     c+=1
            except Exception as e:
                c_er+=1
    
    print(f'failed {c_er}')
    
    out_path = f'output/{onto}/men_nm_only/'
    os.makedirs(out_path, exist_ok=True)
            
    with open(f'{out_path}predictions.json', 'w') as f:
        json.dump(bi_and_cross_pred, f,  default=str, indent=1)

    save_report(onto, f'{out_path}predictions.json')

onto = 'ncbi'
link_ent(onto)

# onto = 'bc5cdr'
# link_ent(onto)

# onto = 'cometa'
# link_ent(onto)

# onto = 'medmentions'
# link_ent(onto)


