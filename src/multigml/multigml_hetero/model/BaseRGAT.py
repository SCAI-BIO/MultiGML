# -*- coding: utf-8 -*-

"""Base RGAT layer implementation for a heterograph."""


import logging
from collections import defaultdict
from timeit import default_timer as timer
from typing import Dict, List, Tuple

import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from multigml.multigml_hetero.model.MultiModalEmbedding import \
    MultiModalEmbeddingLayer
from multigml.multigml_hetero.model.UniModalEmbedding import UniModalEmbeddingLayer
from dgl import apply_each
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch.hetero import HeteroGraphConv

logger = logging.getLogger(__name__)

class BaseRGAT(nn.Module):
    """The base class for the relational graph attention neural network."""
    def __init__(self,
                 g,
                 input_dim,
                 h_dim,
                 out_dim,
                 modality_sizes_dict: Dict[str, int],
                 num_heads: int,
                 num_hidden_layers=1,
                 dropout=0,
                 dropout_multimodal=0,
                 attention_dropout: float = 0.0,
                 use_self_loop=True,
                 which_features: str = 'reduced',
                 ):
        super(BaseRGAT, self).__init__()
        self.name = 'RGAT'
        #: The heterograph to perform convolution on.
        self.g = g
        #: The input dimension.
        self.input_dim = input_dim
        #: The hidden dimension of the RGCN.
        self.h_dim = h_dim
        #: The output dimension of the RGCN.
        self.out_dim = out_dim
        self.rel_names = list(set(g.canonical_etypes))
        self.rel_names.sort()
        #: The number of hidden layers in the RGCN.
        self.num_hidden_layers = num_hidden_layers
        #: The dropout rate.
        self.dropout = dropout
        #: The dropout rate for the multimodal embedding layer.
        self.dropout_multimodal = dropout_multimodal
        #: Attention dropout rate.
        self.attention_dropout = attention_dropout
        #: True for using the node's own representation from the previous layer, false otherwise.
        self.use_self_loop = use_self_loop
        self.layers: nn.ModuleList = nn.ModuleList()
        self.which_features = which_features
        self.activation = F.relu
        #: Input dimensions of modalities.
        self.modality_sizes_dict = modality_sizes_dict
        #: Number of attention heads.
        self.num_heads = num_heads

        # ==================================
        # Create multimodal embeddings
        # ==================================
        if self.which_features in ['unimodal', 'random', 'identity', 'mayaan']:
            self.uni_modal_embed = th.nn.ModuleDict({
                node_type: UniModalEmbeddingLayer(
                    node_type=node_type,
                    input_sizes_dict=modality_sizes_dict[node_type],
                    output_size=self.input_dim,
                    #droupout_modalities=dropout_modalities,
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
                    #droupout_modalities=dropout_modalities,
                    dropout_multimodal=dropout_multimodal,
                )
                for node_type in self.g.ntypes
            })

        # Embedding to Input layer
        self.layers.append(RGATLayer(
            in_feat=self.input_dim,
            out_feat=self.h_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            rel_names=self.rel_names,
            activation=self.activation,
            self_loop=self.use_self_loop,
            merge='cat',
            )
        )

        # hidden to hidden layer
        for i in range(self.num_hidden_layers):
            self.layers.append(RGATLayer(
                in_feat=self.h_dim,
                out_feat=self.h_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                rel_names=self.rel_names,
                activation=self.activation,
                self_loop=self.use_self_loop,
                merge='cat',
            )
        )

        # hidden to output layer
        self.layers.append(RGATLayer(
            in_feat=self.h_dim,
            out_feat=self.out_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            rel_names=self.rel_names,
            activation=None,
            self_loop=self.use_self_loop,
            merge='mean',
            )
        )

    def forward(
        self,
        h: Dict[str, Dict[str, th.Tensor]],
        blocks: List[dgl.DGLHeteroGraph] = None,
        g: dgl.DGLHeteroGraph = None,
        verbose: bool = False,
        return_attention: bool = False,
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
                #if h[node_type].nelement() != 0:
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
                if return_attention:
                    if idx == len(self.layers)-1:
                        # the first layer handles the first block, and so on
                        h, attention_weights = layer.forward(
                            g=blocks[idx],
                            inputs=h,
                            return_attention=return_attention,
                        )  
                    
                else:
                    # the first layer handles the first block, and so on
                    h = layer.forward(
                        g=blocks[idx],
                        inputs=h,
                    )  
        else:
            for idx, layer in enumerate(self.layers):
                if return_attention:
                    if idx == len(self.layers)-1:
                        # the first layer handles the first block, and so on
                        h, attention_weights = layer.forward(
                            g=g,
                            inputs=h,
                            return_attention=return_attention,
                        ) 
                else:
                    h = layer.forward(
                        g=g,
                        inputs=h,
                    )      
        
        if return_attention == True:
            return h, attention_weights
        else:
            return h
        

class RGATLayer(nn.Module):
    r"""Relational graph attention layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    num_heads : int
        Number of attention heads.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_heads,
                 rel_names,
                 *,
                 merge: str = 'mean',
                 weight=True,
                 activation=F.relu,
                 self_loop=False,
                 dropout=0.0,
                 attention_dropout: float = 0.0,
                 ):
        super(RGATLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_heads = num_heads
        self.rel_names = rel_names
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.use_weight = weight
        self.self_loop = self_loop
        self.merge = merge

        self.conv = HeteroGraphConv({
                rel : GATConv(
                                in_feats=self.in_feat,
                                out_feats=self.out_feat // self.num_heads,
                                num_heads=self.num_heads,
                                feat_drop=self.dropout,
                                attn_drop=self.attention_dropout,
                                activation=self.activation,
                                )
                for src, rel, dst in rel_names
            })

        if self.use_weight:
            self.weight = nn.Parameter(th.Tensor(len(self.rel_names), in_feat, out_feat))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))


        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs, return_attention=False,):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            wdict = {self.rel_names[i] : {'weight' : w.squeeze(0)}
                     for i, w in enumerate(th.split(self.weight, 1, dim=0))}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            # inputs_src = inputs_dst = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
            inputs = (inputs, inputs_dst)

        def _apply(ntype, h):
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        if return_attention:
            outputs = {nty : [] for nty in g.dsttypes}
            attention_collector = defaultdict(lambda: defaultdict(list))
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if isinstance(inputs, tuple):
                    src_inputs, dst_inputs = inputs
                else:
                    src_inputs = inputs
                    dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
                
                dstdata, attention_weights = self.conv.mods[etype](
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype]),
                    get_attention=True,
                    )
                outputs[dtype].append(dstdata)

                for src_id, dst_id, attn_weight in zip(rel_graph.edges()[0], rel_graph.edges()[1], attention_weights.view(-1,1)):
                    attention_collector[(stype, etype, dtype)][(src_id, dst_id)] = attn_weight

        hs = self.conv(g, inputs, mod_kwargs=wdict)

        # merge over attention head outputs
        # code taken from: https://github.com/dmlc/dgl/blob/744896e2d8ff802420f9e89e52d5d73a736275d6/examples/pytorch/rgat/rgat.py#L8
        concatenated_hs = apply_each(hs, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2]))

        if return_attention:
            return {ntype : _apply(ntype, h) for ntype, h in concatenated_hs.items()}, attention_collector
        else:
            return {ntype : _apply(ntype, h) for ntype, h in concatenated_hs.items()}
        

