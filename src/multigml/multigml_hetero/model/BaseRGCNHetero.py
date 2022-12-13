# -*- coding: utf-8 -*-

"""Base RGCN layer implementation for a heterograph."""

import logging
from timeit import default_timer as timer
from typing import Dict, List, Tuple

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from multigml.multigml_hetero.model.MultiModalEmbedding import \
    MultiModalEmbeddingLayer
from multigml.multigml_hetero.model.RelGraphConv import RelGraphConvLayer
from multigml.multigml_hetero.model.UniModalEmbedding import UniModalEmbeddingLayer

#from dgl.nn.pytorch.conv import RelGraphConv


logger = logging.getLogger(__name__)


class BaseRGCNHetero(nn.Module):
    """The base class for the relational graph convolutional neural network."""
 
    def __init__(self,
                 g,
                 input_dim,
                 h_dim,
                 out_dim,
                 modality_sizes_dict: Dict[str, int],
                 num_hidden_layers=1,
                 dropout=0,
                 dropout_multimodal=0,
                 use_self_loop=True,
                 which_features: str = 'reduced',
                 num_bases: int = None,
                 ):
        super(BaseRGCNHetero, self).__init__()
        self.name = 'RGCN'
        self.g = g
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.out_dim = out_dim

        self.rel_names = list(set(g.canonical_etypes))
        self.rel_names.sort()
        self.num_bases = num_bases
        self.num_rels = len(list(set(g.canonical_etypes)))
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.dropout_multimodal = dropout_multimodal
        self.use_self_loop = use_self_loop
        self.layers: nn.ModuleList = nn.ModuleList()
        self.which_features = which_features
        self.activation = F.relu
        self.modality_sizes_dict = modality_sizes_dict

        # ==================================
        # Create multimodal embeddings
        # ==================================
        if self.which_features in ['unimodal', 'random', 'identity', 'mayaan']:
            self.uni_modal_embed = th.nn.ModuleDict({
                node_type: UniModalEmbeddingLayer(
                    node_type=node_type,
                    input_sizes_dict=modality_sizes_dict[node_type],
                    output_size=self.input_dim,
                    dropout_multimodal=dropout_multimodal,
                )
                for node_type in self.g.ntypes
            })
        else:
            self.multi_modal_embed = th.nn.ModuleDict({
                node_type: MultiModalEmbeddingLayer(
                    node_type=node_type,
                    input_sizes_dict=modality_sizes_dict[node_type],
                    output_size=self.input_dim,
                    dropout_multimodal=dropout_multimodal,
                )
                for node_type in self.g.ntypes
            })

        # Embedding to Input layer
        self.layers.append(RelGraphConvLayer(
            in_feat=self.input_dim,
            out_feat=self.h_dim,
            rel_names=self.rel_names,
            num_bases=self.num_bases,
            activation=self.activation,
            self_loop=self.use_self_loop,
            dropout=self.dropout,
            )
        )

        # hidden to hidden layer
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                in_feat=self.h_dim,
                out_feat=self.h_dim,
                rel_names=self.rel_names,
                num_bases=self.num_bases,
                activation=self.activation,
                self_loop=self.use_self_loop,
                dropout=self.dropout,
            )
        )

        # hidden to output layer
        self.layers.append(RelGraphConvLayer(
            in_feat=self.h_dim,
            out_feat=self.out_dim,
            rel_names=self.rel_names,
            num_bases=self.num_bases,

            activation=None,
            self_loop=self.use_self_loop,
            dropout=self.dropout,
            )
        )


    def forward(
        self,
        h: Dict[str, Dict[str, th.Tensor]],
        blocks: List[dgl.DGLHeteroGraph] = None,
        g: dgl.DGLHeteroGraph = None,
        verbose: bool = False,
    ) -> Tuple[Dict[str, th.Tensor], Dict[str, th.Tensor], Dict[str, th.Tensor]] or Dict[str, th.Tensor] :
        """Do a forward pass of the features of the source and destination nodes of the block.

        Args:
            g (dgl.DGLHeteroGraph): The DGL heterograph.
            blocks (list): List of blocks.
            h (dict): The current features of the source and destination nodes of the block.
                Mapping from node type to modality name to tensor.
            verbose (bool): True for printing time stamps, false otherwise.

        Returns:
            (tuple): tuple containing:
                h (dict): From  CustomHeteroGraphConv, RelGraphConv and RelGraphEmbed,
                    a mapping from node type to current feature.
                pos_edge_embedding (dict): Embedding of positive edges of the current graphs.
                neg_edge_embedding (dict): Embedding of negative edges of the current graphs.
        """
        t0 = timer()

        if self.which_features in ['unimodal', 'random', 'identity', 'mayaan']:
            for node_type in h:
                h[node_type] = self.uni_modal_embed[node_type](list(h[node_type].values())[0])

        # convert unimodal features for every node type to a multi modal embedding
        else:
            for node_type in h:
                h[node_type] = self.multi_modal_embed[node_type](h[node_type])

        t1 = timer()
        if verbose:
            logger.info('Embedded multimodal: {}'.format(t1 - t0))


        if blocks != None:
            for idx, layer in enumerate(self.layers):
                # the first layer handles the first block, and so on
                h = layer.forward(
                    g=blocks[idx],
                    inputs=h,
                )  # layer_number=idx)
        else:
            for idx, layer in enumerate(self.layers):
                # the first layer handles the first block, and so on
                h = layer.forward(
                    g=g,
                    inputs=h,
                ) 
        t2 = timer()
        if verbose:
            logger.info('Passed through rgcn layers: {}'.format(t2 - t1))

        return h
