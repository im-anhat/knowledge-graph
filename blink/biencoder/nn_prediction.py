# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import logging
import torch
from tqdm import tqdm

import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, Stats


def get_topk_predictions(
    reranker,
    train_dataloader,
    candidate_pool, # (# entities, max_seq_len)
    cand_encode_list, # (# entities, embedding_dim)
    silent,
    logger,
    top_k=10,
    is_zeshel=False,
    save_predictions=False,
):
    reranker.model.eval()
    device = reranker.device
    logger.info("Getting top %d predictions." % top_k)
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)

    nn_context = []
    nn_candidates = []
    nn_candidates_prime = [] # nhat
    nn_labels = []
    nn_labels_idx = [] # nhat
    nn_worlds = []
    stats = {}

    if is_zeshel:
        world_size = len(WORLDS)
    else:
        # only one domain
        world_size = 1
        candidate_pool = [candidate_pool]
        cand_encode_list = [cand_encode_list]

    logger.info("World size : %d" % world_size)

    for i in range(world_size):
        stats[i] = Stats(top_k)

    # rasel
    matched = 0
    # rasel
    
    oid = 0
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input, candidate_input, srcs, label_ids = batch

        src = srcs[0].item()

        scores = reranker.score_candidate(
            context_input, 
            None, 
            cand_encs=cand_encode_list[src].to(device)
        )
        values, indicies = scores.topk(top_k)

        old_src = src
        for i in range(context_input.size(0)):
            oid += 1
            inds = indicies[i]

            if srcs[i] != old_src:
                src = srcs[i].item()
                # not the same domain, need to re-do
                new_scores = reranker.score_candidate(
                    context_input[[i]], 
                    None,
                    cand_encs=cand_encode_list[src].to(device)
                )
                _, inds = new_scores.topk(top_k)
                inds = inds[0]

            pointer = -1
            for j in range(top_k):
                if inds[j].item() == label_ids[i].item():
                    pointer = j
                    break
            stats[src].add(pointer)

            # rasel
            if pointer != -1:
                matched += 1
            # rasel

            # Nhat: If gold entity not found in top-k, replace the last candidate with gold entity
            # This ensures every mention has its gold entity in the candidate set for training
            else:
                # Replace the last (worst) candidate with the gold entity
                inds = inds.clone()  # Make a copy to avoid modifying original
                inds[-1] = label_ids[i].item()  # Replace 64th candidate with gold
                pointer = top_k - 1  # Gold entity is now at position 63 (0-indexed)
            
            if not save_predictions:
                continue

            # add examples in new_data
            cur_candidates = candidate_pool[src][inds]
            nn_context.append(context_input[i].cpu().tolist())
            nn_candidates_prime.append(candidate_input[i].cpu().tolist()) # nhat
            nn_candidates.append(cur_candidates.cpu().tolist())
            nn_labels.append(pointer)
            nn_labels_idx.append(label_ids[i].item()) # nhat
            nn_worlds.append(src)

    res = Stats(top_k)
    for src in range(world_size):
        if stats[src].cnt == 0:
            continue
        if is_zeshel:
            logger.info("In world " + WORLDS[src])
        output = stats[src].output()
        logger.info(output)
        res.extend(stats[src])

    logger.info(res.output())

    nn_context = torch.LongTensor(nn_context)
    nn_candidates = torch.LongTensor(nn_candidates)
    nn_candidates_prime = torch.LongTensor(nn_candidates_prime) # nhat
    nn_labels = torch.LongTensor(nn_labels)
    nn_labels_idx = torch.LongTensor(nn_labels_idx) # nhat
    nn_data = {
        'context_vecs': nn_context,
        'candidate_vecs': nn_candidates,
        'labels': nn_labels,
        'labels_idx': nn_labels_idx,  # nhat
        'candidates_prime': nn_candidates_prime,  # nhat
    }

    # rasel
    print(f'len {len(train_dataloader)}, matched : {matched}')
    print(f'accuracy is : {matched/len(train_dataloader)}')
    # rasel

    if is_zeshel:
        nn_data["worlds"] = torch.LongTensor(nn_worlds)
    
    return nn_data

