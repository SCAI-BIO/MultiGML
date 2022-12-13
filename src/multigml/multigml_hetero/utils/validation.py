# -*- coding: utf-8 -*-

"""Validate the model."""

import logging
from functools import singledispatch
from typing import Dict, Tuple

import torch as th

logger = logging.getLogger(__name__)


def create_all_labels_and_scores(
    pos_score: Dict[Tuple[str, str, str], th.Tensor],
    neg_score: Dict[Tuple[str, str, str], th.Tensor],
    device: th.device,
) -> Tuple[th.Tensor, th.Tensor]:
    """Create labels and scores for all edges in positive and negative graph.

    Args:
        pos_score (dict): The dictionary of canonical edge type to scores for positive graph.
        neg_score (dict): The dictionary of canonical edge type to scores for positive graph.
        device (th.device): The torch device.

    Returns:
        (tuple): tuple containing:
            scores_val (th.Tensor): The scores for the validation edges.
            labels_val (th.Tensor): The labels for the validation edges.
    """
    # create labels

    scores_pos = th.cat([pos_score[etype] for etype in pos_score.keys()], dim=0)
    scores_neg = th.cat([neg_score[etype] for etype in neg_score.keys()], dim=0)
    score_neg_balanced = th.cat([neg_score[etype][:pos_score[etype].shape[0]] for etype in neg_score.keys()], dim=0)

    # create labels
    labels_pos = th.ones(scores_pos.size()[0], dtype=th.int, device=device)
    labels_neg = th.zeros(scores_neg.size()[0], dtype=th.int, device=device)
    labels_neg_balanced = th.zeros(score_neg_balanced.size()[0], dtype=th.int, device=device)

    # merge together:
    scores_val = th.cat([
        scores_pos,
        scores_neg,
    ])
    labels_val = th.cat([
        labels_pos,
        labels_neg,
    ])

    scores_val_balanced = th.cat([
        scores_pos,
        score_neg_balanced,
    ])
    labels_val_balanced = th.cat([
        labels_pos,
        labels_neg_balanced,
    ])

    scores_val = scores_val.squeeze()
    scores_val_balanced = scores_val_balanced.squeeze()

    etype_index_dict = {etype: idx for idx, etype in enumerate(list(pos_score.keys()))}

    # indexes for MRR and Hits at k according to etype 
    l_pos = [[etype_index_dict[etype] for x in range(pos_score[etype].shape[0])] for etype in pos_score]
    l_neg = [[etype_index_dict[etype] for x in range(neg_score[etype].shape[0])] for etype in neg_score]
    merged_l = [item for sublist in l_pos for item in sublist]
    merged_l.extend([item for sublist in l_neg for item in sublist])
    indexes = th.tensor(
        merged_l,
        dtype=th.long, device=device
        )

    return scores_val_balanced, labels_val_balanced, scores_val, labels_val, indexes


def create_all_labels_and_scores_etype_specific(
    pos_score: Dict[Tuple[str, str, str], th.Tensor],
    neg_score: Dict[Tuple[str, str, str], th.Tensor],
    device: th.device,
    validation_etype = None,
) -> Tuple[Dict[Tuple[str, str, str], th.Tensor], Dict[Tuple[str, str, str], th.Tensor]]:
    """Create labels and scores for all edges in positive and negative graph.

    Args:
        pos_score (dict): The dictionary of canonical edge type to scores for positive graph.
        neg_score (dict): The dictionary of canonical edge type to scores for positive graph.
        device (th.device): The torch device.

    Returns:
        (tuple): tuple containing:
            scores_val (th.Tensor): The scores for the validation edges.
            labels_val (th.Tensor): The labels for the validation edges.
            indexes (th.Tensor): The indexes for the retrieval metrics.
    """
    scores = dict()
    labels = dict()
    scores_balanced = dict()
    labels_balanced = dict()
    indexes = dict()

    # only calculate scores for specified validation etype
    if validation_etype:

        scores = th.cat([
            pos_score[validation_etype],
            neg_score[validation_etype],
        ]).view(-1)
        labels = th.cat([
            th.ones(pos_score[validation_etype].size()[0], dtype=th.int, device=device),
            th.zeros(neg_score[validation_etype].size()[0], dtype=th.int, device=device),
        ])

        scores_balanced = th.cat([
            pos_score[validation_etype],
            neg_score[validation_etype][:pos_score[validation_etype].shape[0]],
        ]).view(-1)
        labels_balanced = th.cat([
            th.ones(pos_score[validation_etype].size()[0], dtype=th.int, device=device),
            th.zeros(neg_score[validation_etype][:pos_score[validation_etype].shape[0]].size()[0], dtype=th.int, device=device),
        ])

        indexes = th.tensor([0 for x in range(labels.shape[0])], dtype=th.long, device=device)

        return scores_balanced, labels_balanced, scores, labels, indexes


    # calculate scores for all etypes
    else:
        etype_index_dict = {etype: idx for idx, etype in enumerate(list(pos_score.keys()))}

        for etype in pos_score:
            
            scores[etype] = th.cat([
                pos_score[etype],
                neg_score[etype],
            ]).view(-1)
            labels[etype] = th.cat([
                th.ones(pos_score[etype].size()[0], dtype=th.int, device=device),
                th.zeros(neg_score[etype].size()[0], dtype=th.int, device=device),
            ])

            scores_balanced[etype] = th.cat([
                pos_score[etype],
                neg_score[etype][:pos_score[etype].shape[0]],
            ]).view(-1)
            labels_balanced[etype] = th.cat([
                th.ones(pos_score[etype].size()[0], dtype=th.int, device=device),
                th.zeros(neg_score[etype][:pos_score[etype].shape[0]].size()[0], dtype=th.int, device=device),
            ])

            indexes[etype] = th.tensor([etype_index_dict[etype] for x in range(labels[etype].shape[0])], dtype=th.long, device=device)

        return scores_balanced, labels_balanced, scores, labels, indexes


@singledispatch
def keys_to_strings(ob):
    return ob

@keys_to_strings.register
def _handle_dict(ob: dict):
    return {str(k): keys_to_strings(v) for k, v in ob.items()}

@keys_to_strings.register
def _handle_list(ob: list):
    return [keys_to_strings(v) for v in ob]
