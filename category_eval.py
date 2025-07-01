import json
import random
from copy import deepcopy

class Evaluation:
    def __init__(self, eval_data, error_analysis=False, for_retrieval=True):
        data = deepcopy(eval_data)
        self.data = data
        self.error_analysis = error_analysis
        self.for_retrieval = for_retrieval
        self.final_data = []
        self.mention_matched_at_k = {}
        self.cat_wise_gt_matched = {}
        self.mention_wise_triple_matched = {}   
        self.text_report = "\n\n\n"

        for i in data:
            if 'retrieved_candidates' not in i:
                continue
            self.final_data.append(i)
        
        removed_count = len(data) - len(self.final_data) 
        if removed_count > 0:
            self.text_report+=f"Removed {removed_count} items since they did not have retrieved_candidates key\n"
        
        self.none_data = [] 
        self.not_none_data = []
        for item in self.data:
            ground_truth_id = item["ground_truth"]["id"]
            if ground_truth_id.lower() == "none":
                self.none_data.append(item) 
            else:
                self.not_none_data.append(item)
        self.text_report+=f"Number of items with ground truth 'None': {len(self.none_data)}\n"
        self.text_report+=f"Number of items with ground truth not 'None': {len(self.not_none_data)}\n"

    def get_report(self):
        mrr = self.calculate_mrr()
        recall_at_1 = self.calculate_recall_at_k_for_graph_retriever(1)
        # recall_at_5 = self.calculate_recall_at_k_for_graph_retriever(5)
        # recall_at_10 = self.calculate_recall_at_k_for_graph_retriever(10)
        recall_at_10 = self.calculate_recall_at_k_for_graph_retriever(63)
        # none_accuracy = self.get_none_accuracy()

        return self.not_none_data, self.none_data

                
    
    def get_none_accuracy(self):
        data = self.none_data
        correct_count = 0
        for item in data:
            retrieved_candidates = item["retrieved_candidates"]
            if len(retrieved_candidates) == 0:
                correct_count += 1
            else:
                try:
                    for i in retrieved_candidates:
                        if i["id"].lower() == "none" or not i["id"]:
                            correct_count += 1
                            break
                except Exception as e:
                    if isinstance(retrieved_candidates[0], str):
                        if retrieved_candidates[0].lower() == "none":
                            correct_count += 1
                        
        accuracy = correct_count / len(data)
        self.text_report+=f"Total {len(data)} items was actually NONE, among them {correct_count} items correctly predicted as NONE.\nSo, accuracy for items with ground truth 'None' : {accuracy}\n"
        return accuracy
    

    def calculate_mrr(self):
        data = self.not_none_data
        reciprocal_ranks = []
        for item in data:
            ground_truth_id = item["ground_truth"]["id"]
            retrieved_candidates = item["retrieved_candidates"]
            
            # Find the rank of the first relevant item
            rank = 0
            for idx, candidate in enumerate(retrieved_candidates, start=1):
                if candidate["id"] == ground_truth_id:
                    rank = idx
                    break
            
            # Calculate reciprocal rank; if not found, reciprocal rank is 0
            reciprocal_rank = 1 / rank if rank > 0 else 0
            reciprocal_ranks.append(reciprocal_rank)
        
        # Calculate MRR by averaging all reciprocal ranks
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        self.text_report+=f"\n\nCalculated MRR for {len(data)} items. \nSo, MRR : {mrr}\n\n"
        return mrr

    def calculate_recall_at_k(self, k):
        data = self.not_none_data
        relevant_count = 0
        for item in data:
            ground_truth_id = item["ground_truth"]["id"]
            retrieved_candidates = item["retrieved_candidates"][:k]  # Consider top-k candidates
            
            # Check if the relevant item is within the top-k candidates
            if any(candidate["id"] == ground_truth_id for candidate in retrieved_candidates):
                relevant_count += 1
        
        # Calculate recall@k
        recall_at_k = relevant_count / len(data)
        self.text_report+=f"\n\nFound {relevant_count} items out of {len(data)} within top-{k} candidates \nSo, recall@{k} : {relevant_count}/{len(data)}={recall_at_k}\n\n"
        return recall_at_k
    

    def category_specific_count(self, item, is_matched, k):
        mention_id=item["mention_id"]
        mention=item["mention"]
        title= item["ground_truth"]['title']
        candidates = ' | '.join([i['title'] for i in item["retrieved_candidates"]])
        
        mention_lower = mention.lower()
        title_lower = title.lower()
        
        if mention_lower == title_lower:
            self.cat_wise_gt_matched[k]["HO"]['count'] += 1
            self.cat_wise_gt_matched[k]["HO"]['items'].append({
                'mention_id':mention_id, 
                'mention':mention, 
                'gt_title':title, 
                'matched': is_matched,
                'retrieved_candidates': candidates
                })
            if is_matched:
                self.cat_wise_gt_matched[k]["HO"]['matched'] += 1
        elif mention_lower in title_lower and title_lower != mention_lower:
            self.cat_wise_gt_matched[k]["MINT"]['count'] += 1
            self.cat_wise_gt_matched[k]["MINT"]['items'].append({
                'mention_id':mention_id, 
                'mention':mention, 
                'gt_title':title, 
                'matched': is_matched,
                'retrieved_candidates': candidates
                })
            if is_matched:
                self.cat_wise_gt_matched[k]["MINT"]['matched'] += 1
        else:
            lo = False
            mention_words = mention_lower.split()
            for word in mention_words:
                if word in title_lower:
                    lo = True
                    break
            if lo:
                self.cat_wise_gt_matched[k]["LO"]['count'] += 1
                self.cat_wise_gt_matched[k]["LO"]['items'].append({
                'mention_id':mention_id, 
                'mention':mention, 
                'gt_title':title, 
                'matched': is_matched,
                'retrieved_candidates': candidates
                })
                if is_matched:
                    self.cat_wise_gt_matched[k]["LO"]['matched'] += 1
            else:
                self.cat_wise_gt_matched[k]["NO"]['count'] += 1
                self.cat_wise_gt_matched[k]["NO"]['items'].append({
                'mention_id':mention_id, 
                'mention':mention, 
                'gt_title':title, 
                'matched': is_matched,
                'retrieved_candidates': candidates
                })
                if is_matched:
                    self.cat_wise_gt_matched[k]["NO"]['matched'] += 1


    
    def calculate_recall_at_k_for_graph_retriever(self, k):
        data = self.not_none_data
        self.gt_matched = []
        self.gt_did_not_matched = []

        self.cat_wise_gt_matched[k] = {
            "HO":{'count':0, 'matched':0, 'items':[]}, 
            "MINT":{'count':0, 'matched':0, 'items':[]}, 
            "LO":{'count':0, 'matched':0, 'items':[]},
            "NO":{'count':0, 'matched':0, 'items':[]}
        }
        
        self.mention_matched_at_k[k] = []
        
        relevant_count = 0

        if k==10:
            self.mention_wise_triple_matched[k]=[]

        for item in data:
            ground_truth_id = item["ground_truth"]["id"]
            
            if not self.for_retrieval:
                unique_triple = {}
                for id in item["retrieved_candidates"]:
                    unique_triple[id['id']] = id
                item["unique_triple"] = unique_triple

            retrieved_candidates = list(item["unique_triple"].keys())
            top_k = retrieved_candidates[:k] # Consider top-k candidates
            
            is_matched = False
            # Check if the relevant item is within the top-k candidates
            for i, candidate in enumerate(top_k):
                # if item["mention"] == "liver disease":
                #     print(0)
                if ground_truth_id in candidate:
                    relevant_count += 1
                    is_matched = True
                    self.mention_matched_at_k[k].append(item["mention_id"])
                    
                    matched_dict = item["unique_triple"][candidate]
                    self.gt_matched.append({
                        'mention_id':item["mention_id"],
                        'mention':item["mention"],
                        'mention_context':item["mention_context"],
                        'ground_truth':item["ground_truth"],
                        'rank':i+1,
                        'matched_triple': {**{'triple':candidate}, **matched_dict},
                        'unique_triple':item["unique_triple"]
                        })
                    break

            if k==10:
                if is_matched:
                    triple_and_aug = []
                    for i in top_k:
                        if isinstance(item["unique_triple"], dict):
                            break

                        item["unique_triple"][i]['score']
                        triple_and_aug.append({'triple':i, 'aug':item["unique_triple"][i]})

                    self.mention_wise_triple_matched[k].append(
                        {
                            'mention_id':item["mention_id"],
                            'mention':item["mention"],
                            'ground_truth':item["ground_truth"],
                            f'top_{k}':triple_and_aug,
                        }
                    )
            
            # if not self.for_retrieval:
            self.category_specific_count(item,
                        is_matched=is_matched,
                        k=k
                    )
            
            if not is_matched:
                self.gt_did_not_matched.append({
                            'mention_id':item["mention_id"],
                            'mention':item["mention"],
                            'mention_context':item["mention_context"],
                            'ground_truth':item["ground_truth"],
                            'rank':-1,
                            'matched_triple':None,
                            'unique_triple':item["unique_triple"]
                            })
        
        # Calculate recall@k
        recall_at_k = relevant_count / len(data)
        self.text_report+=f"\n\nFound {relevant_count} items out of {len(data)} within top-{k} candidates \nSo, recall@{k} : {relevant_count}/{len(data)}={recall_at_k}\n\n"
        # self.text_report+=f"\n\n"
        for cat in self.cat_wise_gt_matched[k]:
            count = self.cat_wise_gt_matched[k][cat]['count']
            matched = self.cat_wise_gt_matched[k][cat]['matched']
            score = matched / count if count > 0 else 0
            self.text_report+=f"{cat}, count: {count}, matched: {matched}, score: {score}\n"

        return recall_at_k


def cat_eval(acc, filepath, kbpath):
    with open(filepath) as f:
        preds = json.load(f)
    with open(kbpath) as f:
        kb = json.load(f)
    converted = []
    
    for p in preds:
        mention_context = p['text'][0]
        gtid = p['mention_data']['kb_id'][0]
        gttitle = kb[gtid]['title']
        candidates = [{'id':i[0], 'title':''} for i in p['mention_data']['candidates'] if i[0]!=gtid ]
        retrieved_candidates = []
        if p['linked'] == 1.0:
            retrieved_candidates.append({'id':gtid, 'title':gttitle})
            retrieved_candidates+=random.sample(candidates, 9)
        else:
            retrieved_candidates+=random.sample(candidates, 10)

        mention = mention_context.split("[E1]")[1].split("[\E1]")[0].strip()
        
        d = {'mention_id' : '111111',
            'mention': mention,
            'mention_context':mention_context,
            'ground_truth': {'id':gtid, 'title':gttitle},
            'retrieved_candidates':retrieved_candidates}
        
        converted.append(d)

    
    evaluation = Evaluation(converted, for_retrieval=False)
    not_none_data, none_data = evaluation.get_report()

    report = f'Accuracy : {acc}\n\n{evaluation.text_report}'
    with open(f"{filepath.replace('.json', '_eval.txt')}", 'w') as f:
        f.write(report)
    with open(f"{filepath.replace('.json', '_category_info.json')}", 'w') as f:
        json.dump(evaluation.cat_wise_gt_matched, f, indent=1)


