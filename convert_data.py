import json
import os
from tqdm import tqdm

# dp = 'examples/zeshel/data/zeshel/blink_format/'
# sm = utils.read_dataset('test', dp)

# print(json.dumps(sm[0], indent=1))



def grag_to_blink(
        source_dir, 
        source_file, 
        kb_file_path,
        title_key, 
        defi_key, 
        world, 
        out_file,
        gt_title_key='title',
        mtokens=['[MENTION_START]', '[MENTION_END]'],
        only_ho=True):
    with open(source_dir+source_file, 'r') as f:
        corpus = json.load(f)

    with open(kb_file_path, 'r') as f:
        kb = json.load(f)

    
    onto_path = f'data/{world}/blink_format'
    os.makedirs(onto_path, exist_ok=True)

    map_dict = {}
    ent_list = []
    id_map={}
    with open(f'{onto_path}/kb.jsonl', 'w') as f:
        for i, id in enumerate(kb):
            inc_id = i+1
            id_map[inc_id]=id
            map_dict[id]=inc_id
            kb[id]['id'] =inc_id
            d = {'id':inc_id, 'title':kb[id][title_key],'text':kb[id][defi_key]}
            json_str = json.dumps(d)
            f.write(json_str + "\n")
            dn = {'idx':f"{world}?curid={inc_id}", 'entity':kb[id][title_key], 'title':kb[id][title_key],'text':kb[id][defi_key]}
            ent_list.append(dn)

    with open(f'{onto_path}/id_map.json', 'w') as f:
        json.dump(id_map,f)

    with open(f'{onto_path}/entity.jsonl', 'w') as f:
        for ent in ent_list:
            json_str = json.dumps(ent)
            f.write(json_str + "\n")
    
    c_missed = 0
    c_done = 0
    with open(f'{onto_path}/{out_file}', 'w') as f:
        for doc in tqdm(corpus):
            try:
                ml = doc['mention'].lower().strip()
                gttl = doc['ground_truth']['title'].lower().strip()
                if only_ho:
                    if ml == gttl:
                        context_left = doc['mention_context'].split(mtokens[0])[0]
                        context_right = doc['mention_context'].split(mtokens[1])[1]
                        m = doc['mention']
                        label_id = doc['ground_truth']['id']
                        defi = kb[label_id][defi_key]
                        d = {
                            "context_left": context_left,
                            "context_right": context_right,
                            "mention":m,
                            "label": defi,
                            "label_id": map_dict[label_id],
                            "label_title": doc['ground_truth'][gt_title_key],
                            "onto": world
                            }
                        f.write(json.dumps(d) + "\n")
                else:
                    context_left = doc['mention_context'].split(mtokens[0])[0]
                    context_right = doc['mention_context'].split(mtokens[1])[1]
                    m = doc['mention']
                    label_id = doc['ground_truth']['id']
                    defi = kb[label_id][defi_key]
                    d = {
                        "context_left": context_left,
                        "context_right": context_right,
                        "mention":m,
                        "label": defi,
                        "label_id": map_dict[label_id],
                        "label_title": doc['ground_truth'][gt_title_key],
                        "onto": world
                        }
                    f.write(json.dumps(d) + "\n")
                c_done+=1
            
            except Exception as e:
                print(e)
                c_missed+=1
    if c_missed>0:
        print(f'{c_done} is done, but {c_missed} are not able to convert! this a big issue!')
        
def grag_to_blink_prime_def(
        source_dir, 
        source_file, 
        kb_file_path,
        title_key, 
        defi_key, 
        world, 
        out_file,
        gt_title_key='title',
        mtokens=['[MENTION_START]', '[MENTION_END]'],
        only_ho=True):

    with open(source_dir+source_file, 'r') as f:
        corpus = json.load(f)

    with open(kb_file_path, 'r') as f:
        kb = json.load(f)
    
    onto_path = f'data/{world}/blink_format/prime_def'
    # os.makedirs(onto_path, exist_ok=True)

    with open(f'{onto_path}/ncbi_prime_test_newly_generated.json', 'r') as f:
        prime_def = json.load(f)
        kb_prime_def = {}
        for i in prime_def:
            kb_prime_def[i['document_id']] = i


    map_dict = {}
    ent_list = []
    id_map={}
    with open(f'{onto_path}/kb.jsonl', 'w') as f:
        for i, id in enumerate(kb):
            inc_id = i+1
            id_map[inc_id]=id
            map_dict[id]=inc_id
            kb[id]['id'] =inc_id
            d = {'id':inc_id, 'title':kb[id][title_key],'text':kb[id][defi_key]}
            json_str = json.dumps(d)
            f.write(json_str + "\n")
            dn = {'idx':f"{world}?curid={inc_id}", 'entity':kb[id][title_key], 'title':kb[id][title_key],'text':kb[id][defi_key]}
            ent_list.append(dn)

    with open(f'{onto_path}/id_map.json', 'w') as f:
        json.dump(id_map,f)

    with open(f'{onto_path}/entity.jsonl', 'w') as f:
        for ent in ent_list:
            json_str = json.dumps(ent)
            f.write(json_str + "\n")
    
    c_missed = 0
    c_done = 0
    with open(f'{onto_path}/{out_file}', 'w') as f:
        for doc in tqdm(corpus):
            try:
                ml = doc['mention'].lower().strip()
                gttl = doc['ground_truth']['title'].lower().strip()
                if only_ho:
                    if ml == gttl:
                        context_left = doc['mention_context'].split(mtokens[0])[0]
                        context_right = doc['mention_context'].split(mtokens[1])[1]
                        m = doc['mention']
                        label_id = doc['ground_truth']['id']
                        defi = kb[label_id][defi_key]
                        d = {
                            "context_left": context_left,
                            "context_right": context_right,
                            "mention":m,
                            "label": defi,
                            "label_id": map_dict[label_id],
                            "label_title": doc['ground_truth'][gt_title_key],
                            "onto": world
                            }
                        f.write(json.dumps(d) + "\n")
                else:
                    context_left = doc['mention_context'].split(mtokens[0])[0]
                    context_right = doc['mention_context'].split(mtokens[1])[1]
                    m = doc['mention']
                    label_id = doc['ground_truth']['id']
                    defi = kb[label_id][defi_key]
                    d = {
                        "context_left": context_left,
                        "context_right": context_right,
                        "mention":m,
                        "label": defi,
                        "label_id": map_dict[label_id],
                        "label_title": doc['ground_truth'][gt_title_key],
                        "onto": world
                        }
                    f.write(json.dumps(d) + "\n")
                c_done+=1
            
            except Exception as e:
                print(e)
                c_missed+=1
    if c_missed>0:
        print(f'{c_done} is done, but {c_missed} are not able to convert! this a big issue!')
        
def fix_entity_catalogue(filename):
    c = 0
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            try:
                ent = json.loads(line)
                if 'title' not in ent:
                    ent['title'] = ent['entity']
                    # input('title : ')
                    print(ent)
                    c+=1
            except Exception as e:
                print(e)
                print(line)
    print(c)


# dir = '/work/LAS/qli-lab/rasel/graphrag/related_work/datasets/'
# world = 'ncbi'
# kb_file_path = '/work/LAS/qli-lab/rasel/graphrag/related_work/datasets/onto/mesh/merged_onto.json'
# grag_to_blink_prime_def(dir, 'ncbi-disease/test_grag.json', kb_file_path,'name', 'def', world, 'prime_def/test_all.jsonl', only_ho=False)

# dir = '/work/LAS/qli-lab/rasel/graphrag/related_work/datasets/'
# world = 'ncbi'
# kb_file_path = '/work/LAS/qli-lab/rasel/graphrag/related_work/datasets/onto/mesh/merged_onto.json'
# grag_to_blink(dir, 'ncbi-disease/test_grag_generated_ho.json', kb_file_path,'name', 'def', world, 'train.jsonl', 'no_cntx')
# grag_to_blink(dir, 'ncbi-disease/test_grag.json', kb_file_path,'name', 'def', world, 'valid.jsonl')
# grag_to_blink(dir, 'ncbi-disease/test_grag.json', kb_file_path,'name', 'def', world, 'test.jsonl')
# grag_to_blink(dir, 'ncbi-disease/test_grag.json', kb_file_path,'name', 'def', world, 'test_all.jsonl', only_ho=False)
# grag_to_blink(dir, 'ncbi-disease/train_grag.json', kb_file_path,'name', 'def', world, 'train_ho.jsonl', only_ho=True)

# dir = '/work/LAS/qli-lab/rasel/graphrag/related_work/datasets/'
# world = 'bc5cdr'
# kb_file_path = '/work/LAS/qli-lab/rasel/graphrag/related_work/datasets/onto/mesh/merged_onto.json'
# grag_to_blink(dir, 'bc5cdr/test_grag_generated_ho.json', kb_file_path,'name', 'def', world, 'train.jsonl', 'no_cntx')
# grag_to_blink(dir, 'bc5cdr/test_grag.json', kb_file_path,'name', 'def', world, 'valid.jsonl')
# grag_to_blink(dir, 'bc5cdr/test_grag.json', kb_file_path,'name', 'def', world, 'test.jsonl')
# grag_to_blink(dir, 'bc5cdr/test_grag.json', kb_file_path,'name', 'def', world, 'test_all.jsonl', only_ho=False)

# grag_to_blink(dir, 'bc5cdr/train_grag.json', kb_file_path,'name', 'def', world, 'train_ho.jsonl', only_ho=True)


# dir = '/work/LAS/qli-lab/rasel/graphrag/related_work/datasets/'
# world = 'bc5'
# kb_file_path = '/work/LAS/qli-lab/rasel/graphrag/related_work/datasets/onto/mesh/merged_onto.json'
# grag_to_blink(dir, 'bc5cdr/test_grag.json', kb_file_path,'name', 'def', world)

# dir = '/work/LAS/qli-lab/rasel/graphrag/related_work/datasets/cometa/Prompt-BioEL/'
# world = 'cometa'
# kb_file_path = '/work/LAS/qli-lab/rasel/graphrag/related_work/datasets/cometa/snomedct_kb.json'
# grag_to_blink(dir, 'test_grag.json', kb_file_path,'title', 'text', world)

# dir = '/work/LAS/qli-lab/rasel/graphrag/related_work/datasets/MedMentions/full/data/'
# world = 'medmentions'
# kb_file_path = '/work/LAS/qli-lab/rasel/graphrag/related_work/datasets/MedMentions/full/data/umls_kb.json'
# grag_to_blink(dir, 'corpus_pubtator_test.json', kb_file_path,'title', 'text', world, mtokens=['[E1]', '[\\E1]'])

# fix_entity_catalogue(f"examples/ncbi/entity.jsonl")



