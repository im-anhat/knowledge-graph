import io
import json

from category_eval import Evaluation


def save_report(onto, pfile):
    with open(pfile) as f:
        pred = json.load(f)
    with open(f'examples/{onto}/id_map.json', 'r') as f:
        id_map = json.load(f)
    converted = []
    for p in pred:
        m = p['test_data'][0]
        b = p['bi'][0]
        mention = m['mention']
        mention_context = m['context_left'] + '[MENTION_START]'+mention+'[MENTION_END]'+m['context_right']
        unique_triple = {}
        for bitem in b:
            # eid = bitem['id']
            eid = id_map[bitem['id']]
            unique_triple[eid]=bitem
        candidates = p['cross'][0]
        retrieved_candidates = []
        for c in candidates:
            c['id'] = id_map[c['id']]
            retrieved_candidates.append(c)

        d = {'mention_id' : '111111',
            'mention': mention,
            'mention_context':mention_context,
            'ground_truth': {'id':id_map[str(m['label_id'])], 'title':m['label_title']},
            'retrieved_candidates':retrieved_candidates,
            'unique_triple':unique_triple
            }
        converted.append(d)
    
    with open(f"{pfile.replace('.json', '_grag.json')}", 'w') as f:
        json.dump(converted, f, indent=1)
    
    eval_bi = Evaluation(converted, for_retrieval=True)
    not_none_data, none_data = eval_bi.get_report()

    eval_cross = Evaluation(converted, for_retrieval=False)
    not_none_data, none_data = eval_cross.get_report()

    

    report = f'Bi-Encoder\n{"_"*20}\n{eval_bi.text_report}\n\nCross-Encoder\n{"_"*20}\n{eval_cross.text_report}'


    with open(f"{pfile.replace('.json', '_eval.txt')}", 'w') as f:
        f.write(report)


    with open(f"{pfile.replace('.json', '_category_info.json')}", 'w') as f:
        json.dump(eval_cross.cat_wise_gt_matched, f, indent=1)

    with open(f"{pfile.replace('.json', '_retri_and_rerank.json')}", 'w') as f:
        json.dump(converted, f, indent=1)

def eval_grag(file):
    with open(file) as f:
        pred = json.load(f)
    eval_bi = Evaluation(pred, for_retrieval=True)
    not_none_data, none_data = eval_bi.get_report()
    print(eval_bi.text_report)

# eval_grag('output/bc5cdr/zshel_tarined/predictions_grag.json')

# save_report('bc5cdr', 'output/bc5cdr/zshel_tarined/predictions.json')
# save_report('ncbi', 'output/ncbi/zshel_tarined/predictions.json')

# save_report('examples/cometa/test_ho.jsonl', 'output/cometa/predictions.json')
# save_report('examples/bc5/test_ho.jsonl', 'output/bc5/predictions.json')
# save_report('examples/medmentions/test_ho.jsonl', 'output/medmentions/predictions.json')
