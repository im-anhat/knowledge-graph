
import json


def compare_blink_with_res(idmap_file, predictions_file, res_file):

    with open(predictions_file) as f:
        predictions = json.load(f)
        bi_pred = predictions['predictions']['cross']
        
    with open(idmap_file) as f:
        id_map = json.load(f)
    with open(res_file) as f:
        res_data = json.load(f)
    matched = []
    l = {}
    if len(bi_pred) == len(res_data):
        for resdata, bdata in zip(res_data, bi_pred):
            cands = resdata['mention_data']['candidates']
            l[len(cands)]=None
            ands = cands[:63]
            # print(bdata[0])
            preds = [id_map[i['id']] for i in bdata[0] ][:63]
            nmc = 0
            for c, p in zip(cands, preds):
                if c==p:
                    pass
                else:
                    nmc+=1
            
            matched.append(nmc)

        print(matched)
        print(l)
    else:
        print('cant comapre!')

def compare_id_map(idmap_file_1, idmap_file_2):

    with open(idmap_file_1) as f:
        id_map1 = json.load(f)
    with open(idmap_file_2) as f:
        id_map2 = json.load(f)
    
    if len(id_map1)==len(id_map2):
        print('both has same number of items')
    cnm_key = 0
    cnm_val = 0
    for m1, m2 in zip(id_map1, id_map2):
        if not m1==m2:
            cnm_key+=1
        if not id_map1[m1]==id_map2[m2]:
            cnm_val+=1
    print(f'didnt match key {cnm_key}')
    print(f'didnt match key {cnm_val}')

def compare_cw_result(files):
    cw_list = []
    with open(files[0]) as f:
        cw_list.append(json.load(f))
    with open(files[1]) as f:
        cw_list.append(json.load(f))
    with open(files[2]) as f:
        cw_list.append(json.load(f))

    if len(cw_list[0]) == len(cw_list[1]) and len(cw_list[1]) == len(cw_list[2]):

        
        for z, t, s in zip(cw_list[0],cw_list[1],cw_list[2]):
            if z['mention'] == t['mention'] == s['mention']:
                print(f"Mention : { z['mention']}, Ground truth : {z['ground_truth']['title']}")
                print(f'Context 0 | Context 32 |Context 64 ')
                cw0 = z['unique_triple']
                cw32 = t['unique_triple']
                cw64 = s['unique_triple']
                c = 1
                for zbi, tbi, sbi in zip(cw0,cw32, cw64):
                    print(f"{c} --> {cw0[zbi]['title']} | {cw32[tbi]['title']} | {cw64[sbi]['title']}")
                    c+=1
                print('_'*100)
            else:
                print('mentions are not same!')
                return
    else:
        print('files are not same!')

    

files = ['output/ncbi/zshel_trained_cw_0/predictions_retri_and_rerank.json',
'output/ncbi/zshel_trained_cw_32/predictions_retri_and_rerank.json',
'output/ncbi/zshel_tarined_cw_64/predictions_retri_and_rerank.json']

compare_cw_result(files)


# compare_id_map('examples/ncbi/id_map.json', 'data/ncbi/blink_format/id_map.json')

# onto = 'ncbi'
# idmap_file = 'data/ncbi/blink_format/id_map.json'
# res_dir = '/lustre/hdd/LAS/qli-lab/rasel/graphrag/related_work/Read-and-Select/data/blink/'
# compare_blink_with_res(idmap_file, 'output/ncbi/predictions.json', res_dir+'ncbi_test.json')


