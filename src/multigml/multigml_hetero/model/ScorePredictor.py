# -*- coding: utf-8 -*-


"""Score predictor for the RGCN link prediction.

Taken from tutorial and adapted:

https://docs.dgl.ai/guide/minibatch-link.html

"""

import logging
from typing import Dict, List, Tuple

import dgl
import torch as th
import torch.nn as nn

logger = logging.getLogger(__name__)

def bilinear_msg(eweight):
    def msg_func(edges):

        src_x, dst_x = edges.src['x'], edges.dst['x']
        return {'score': th.sigmoid(th.einsum('bn,nm,bm->b', src_x, eweight, dst_x))}

    return msg_func

def bilinear_msg_without_edge():
    def msg_func(edges):
        src_x, dst_x = edges.src['x'], edges.dst['x']
        return {'score': th.sigmoid(th.mul(src_x, dst_x))}

    return msg_func


class ScorePredictor(nn.Module):
    """The decoder for the link prediction task that predicts the scores for all edges."""

    def __init__(
        self,
        canonical_edge_types: List[Tuple[str]],
        out_dim: int,
        device: th.device,
    ):
        """
        Initialize score predictor

        Args:
            canonical_edge_types (List[Tuple[str]]): The canonical edge types in the heterograph.
            out_dim (int): The output dimension of the encoder.
            device (th.device): Current device (TODO: do we really need it?)
        """
        super().__init__()
        # trainable weight matrices for each edge type of shape dxd with d being the dimension of the latent space
        self.weights_edges = th.nn.ParameterDict({
            str(edge_type): nn.Parameter(th.Tensor(out_dim, out_dim))# TODO REVERT.to("cuda:0"))
            for edge_type in canonical_edge_types
        })
        for k, v in self.weights_edges.items():
            self.weights_edges[k] = nn.init.xavier_uniform_(
                v,
                gain=nn.init.calculate_gain('relu'),
            )


    def forward(
        self,
        edge_subgraph: dgl.DGLHeteroGraph,
        node_feature_tensor: Dict[str, th.Tensor],
        use_edge_weight: bool = False,
    ) -> Dict[Tuple[str, str, str], Dict[str, th.Tensor]]:
        """Perform score prediction only on the evaluation edge type.

        Args:
            edge_subgraph (dgl.DGLHeteroGraph): The subgraph to be evaluated.
            node_feature_tensor (th.Tensor): Node feature tensor for every node type .
            edge_feature_tensor (th.Tensor): Edge feature tensor for every edge type.
            use_edge_weight (bool): True for using the edge representation in the calculation, False otherwise.

        Returns:
            results_dict (dict): A dictionary of canonical edge types mapping to the score of the corresponding edges.

        """
        result_dict = {}
        with edge_subgraph.local_scope():

            edge_subgraph.ndata['x'] = node_feature_tensor

            for canonical_etype in edge_subgraph.canonical_etypes:
                srctype, etype, dsttype = canonical_etype

                if srctype not in edge_subgraph.ndata['x'].keys() or dsttype not in edge_subgraph.ndata['x'].keys():
                    continue

                # check that edge type has values
                if edge_subgraph.num_edges(etype) <= 0:
                    continue

                # sum of all edge embeddings of edge type and apply diagonal hr
                # multiply with constant for edge type
                if use_edge_weight:
                    edge_weight = self.weights_edges[str(canonical_etype)]
                    # apply message function

                    edge_subgraph.apply_edges(
                        bilinear_msg(edge_weight),
                        etype=etype,
                    )
                else:

                    edge_subgraph.apply_edges(
                        (dgl.function.u_dot_v('x', 'x', 'rawscore')), etype=etype
                    )

                    # apply logsigmoid function
                    edge_subgraph[(srctype, etype, dsttype)].edata['score'] = th.sigmoid(
                        edge_subgraph[
                            (srctype, etype, dsttype)
                        ].edata['rawscore'].type(th.float)
                    )

                result_dict[canonical_etype] = edge_subgraph.edges[etype].data['score']

        return result_dict
