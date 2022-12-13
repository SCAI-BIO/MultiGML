# -*- coding: utf-8 -*-

"""Preprocessing the heterograph."""

from __future__ import print_function

import logging
from typing import Dict, Set, Tuple

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from multigml.constants import (FUNCTIONAL_INTERACTION, GENETIC_ASSOCIATION,
                                PHYSICAL_INTERACTION, RELATION_TYPE,
                                SIGNALLING_INTERACTION, SOURCE_IDENTIFIER,
                                SOURCE_NODE_TYPE, TARGET_IDENTIFIER,
                                TARGET_NODE_TYPE)
from tqdm import tqdm

INTERACTIONS = {SIGNALLING_INTERACTION, GENETIC_ASSOCIATION, PHYSICAL_INTERACTION, FUNCTIONAL_INTERACTION}

logger = logging.getLogger(__name__)


def get_graph_df(path: str) -> pd.DataFrame:
    """Read the graph from the data file and create a dataframe.

    For the construction of the heterograph, all drugs from DrugBank are included, therefore also unapproved drugs.
    This is the case because the drugs used in the LINCS L1000 assay, that are relevant for the features of the
    drug nodes, also include drugs that are now withdrawn or not approved.
    Nevertheless, if you want to only include approved drugs, please set the path variable to 'GRAPH_FILE_APPROVED'.
    Bare in mind that this will reduce the number of drugs in the graph from 7485 to 888.

    Args:
        path: graph file name

    Returns:
        df (pd.DataFrame): graph as a dataframe with columns 'source_identifier', 'target_identifier', 'bel_relation',
       'relation_type', 'source_database'

    """
    df = pd.read_csv(path, sep='\t', dtype='str', keep_default_na=False).drop(
        ['Unnamed: 0'], axis=1)
    return df


def get_heterograph(
    graph_dict: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]],
    num_nodes_dict: Dict,
    device: torch.device,
) -> dgl.DGLHeteroGraph:
    """Construct heterograph from all subgraphs corresponding to an edge type. Also get a mapping from the original
    node String labels (drug, protein, disease identifiers) to the corresponding integer labels.

    Args:
        graph_dict (dict): dictionary with key = edge type (string) and value = edge list
        num_nodes_dict (dict): dictionary with key = node type, value = number of nodes
        device: torch device

    Returns:
        het_graph (dgl.heterograph): heterograph with different node and edge types

    """
    het_graph = dgl.heterograph(data_dict=graph_dict, num_nodes_dict=num_nodes_dict, device=device)
    return het_graph


def read_graph_inconsecutive_labels(
    path: str,
    edge_attr: str = RELATION_TYPE,
    source: str = SOURCE_IDENTIFIER,
    target: str = TARGET_IDENTIFIER,
    source_node_type: str = SOURCE_NODE_TYPE,
    target_node_type: str = TARGET_NODE_TYPE,
) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict]:
    """Get the individual graphs of each edge type as node indices and their corresponding mapping to node labels.

    Args:
        path: Path to graph file.
        edge_attr: Name of column of graph file that contains the edge attribute (type).
        source: Name of column of graph file that contains the source node.
        target: Name of column of graph file that contains the target node.
        source_node_type: Name of column for source node type.
        target_node_type: Name of column for target node type.
        source_database: Name of column for source database.
        device: Torch device ('cuda' or 'cpu').

    Returns:
        (tuple): tuple containing:
            graph_dict_label (dict): Dictionary with key = edge type (string) and value = edge list with node labels.
            graph_dict_int (dict): Dictionary with key = edge type (string) and value = edge list with node indices.
            label_idx_mapping (dict): Mapping for each node type from node id label to node integer label.
            idx_label_mapping (dict): Inverse mapping for each node type from node integer label : node id label.
            label_idx_mapping_all (dict):  Mapping of all node labels to node indices.
            node_label_to_type (dict):  Mapping from node labels to node types.
            node_tracker (dict): Node tracker with key = type, value = set of node labels.
            num_nodes_dict (dict): Number of nodes dictionary with key = type, value = number of nodes.
            graph_dict_tensor (dict): Dictionary of tuple of tensors.

    """
    logger.info('Reading graph file: {}'.format(path))
    # empty dictionary to construct heterograph
    graph_dict_int = {}
    graph_dict_label = {}
    # empty dictionaries for Tuple of Tensors
    graph_dict_tensor = {}
    # node count to keep track of converting node labels to integers
    label_idx_mapping = {}
    idx_label_mapping = {}
    df = get_graph_df(path)
    values_data = df.values
    g = nx.empty_graph(0, create_using=nx.DiGraph)

    # index of columns
    source_index = df.columns.get_loc(source)
    source_node_type_index = df.columns.get_loc(source_node_type)
    target_index = df.columns.get_loc(target)
    target_node_type_index = df.columns.get_loc(target_node_type)
    edge_attr_index = df.columns.get_loc(edge_attr)

    # get unique node types in graph
    unique_node_types = list(np.unique((df[source_node_type], df[target_node_type])))
    # get unique canonical relation types in graph
    unique_edge_types_df = df.groupby([source_node_type, edge_attr, target_node_type]).size().reset_index()
    unique_edge_types = []
    for index, values in unique_edge_types_df.iterrows():
        unique_edge_types.append((values[source_node_type], values[edge_attr], values[target_node_type]))
        inv_edge_type = '{}_inverse'.format(values[edge_attr])
        unique_edge_types.append((values[target_node_type], inv_edge_type, values[source_node_type]))
    for edge_type in unique_edge_types:
        graph_dict_label[edge_type] = []
        graph_dict_int[edge_type] = []
        # NEW: add inverse edge type

    # generate placeholder to store nodes for each node type
    node_tracker = dict()
    for node_type in unique_node_types:
        node_tracker[node_type] = set()

    # row values
    for value_list in tqdm(values_data, desc='Adding nodes with node type to graph'):
        # add node (of source and destination) to dictionary with key = node type, value = node label
        node_tracker[value_list[source_node_type_index]].add(value_list[source_index])
        node_tracker[value_list[target_node_type_index]].add(value_list[target_index])

    # create num_nodes_dict with key = node type, value = number of nodes for this node type
    num_nodes_dict = {type: len(nodes) for type, nodes in node_tracker.items()}

    # create mapping of node label to node index for each node type seperately after sorting alphabetically
    for node_type in unique_node_types:
        label_idx_mapping[node_type] = {label: idx for idx, label in enumerate(sorted(node_tracker[node_type]))}
        idx_label_mapping[node_type] = {idx: label for idx, label in enumerate(sorted(node_tracker[node_type]))}

    # merged dictionary for label to idx
    label_idx_mapping_all = dict()
    for node_type in label_idx_mapping:
        label_idx_mapping_all = {**label_idx_mapping_all, **label_idx_mapping[node_type]}

    # node label to type placeholders
    node_label_to_type = dict()

    # add edges to graph
    for value_list in tqdm(values_data, desc='Adding edges with edge type to graph'):
        # label
        g.add_edge(value_list[source_index], value_list[target_index])
        g[value_list[source_index]][value_list[target_index]].update(zip(edge_attr, value_list[edge_attr_index]))
        # index not consecutive
        source_int = label_idx_mapping[value_list[source_node_type_index]][value_list[source_index]]
        target_int = label_idx_mapping[value_list[target_node_type_index]][value_list[target_index]]

        # add edges to graph dictionaries
        # label
        graph_dict_label[(
            value_list[source_node_type_index], value_list[edge_attr_index], value_list[target_node_type_index]
        )].append((value_list[source_index], value_list[target_index]))

        # index

        graph_dict_int[(
            value_list[source_node_type_index], value_list[edge_attr_index], value_list[target_node_type_index]
        )].append((source_int, target_int))

        # NEW: add inverse edge (label)
        inv_edge_type = '{}_inverse'.format(value_list[edge_attr_index])
        graph_dict_label[(
            value_list[target_node_type_index], inv_edge_type, value_list[source_node_type_index]
        )].append((value_list[target_index], value_list[source_index]))
        # NEW: add inverse edge (index)
        graph_dict_int[(
            value_list[target_node_type_index], inv_edge_type, value_list[source_node_type_index]
        )].append((target_int, source_int))

        # add nodes to node type dictionaries
        node_label_to_type.update({value_list[source_index]: value_list[source_node_type_index]})
        node_label_to_type.update({value_list[target_index]: value_list[target_node_type_index]})

    # create dictionary for Tuple of tensors from array
    for rel in graph_dict_int:
        src_dst_tuple_idx = list(zip(*graph_dict_int[rel]))
        src_tensor = torch.tensor(data=np.array(src_dst_tuple_idx[0]))
        dst_tensor = torch.tensor(data=np.array(src_dst_tuple_idx[1]))
        # src_tensor = torch.tensor(data=np.array(src_dst_tuple_idx[0]), device=device)
        # dst_tensor = torch.tensor(data=np.array(src_dst_tuple_idx[1]), device=device)
        graph_dict_tensor[rel] = (src_tensor, dst_tensor)

    # make copy of graph to convert into integer labels
    g_int = g.copy()
    nx.relabel_nodes(g_int, label_idx_mapping_all, copy=True)

    return graph_dict_label, graph_dict_int, label_idx_mapping, idx_label_mapping, label_idx_mapping_all, \
           node_label_to_type, node_tracker, num_nodes_dict, graph_dict_tensor
