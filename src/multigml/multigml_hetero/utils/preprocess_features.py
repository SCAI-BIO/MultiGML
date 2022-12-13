# -*- coding: utf-8 -*-

"""Preprocessing the feature of the heterograph."""

from __future__ import print_function

import logging
from typing import Dict, Set

import dgl
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def get_graph_df(path: str) -> pd.DataFrame:
    """Read the graph from the data file and create a dataframe.

    For the construction of the heterograph, all drugs from DrugBank are included, therefore also unapproved drugs.
    This is the case because the drugs used in the LINCS L1000 assay, that are relevant for the features of the
    drug nodes, also include drugs that are now withdrawn or not approved.
    Nevertheless, if you want to only include approved drugs, please set the path variable to 'GRAPH_FILE_APPROVED'.
    Bare in mind that this will reduce the number of drugs in the graph from 7485 to 888.

    Args:
        path (str): The path to graph file.

    Returns:
        df (pd.DataFrame): The graph as a dataframe with columns 'source_identifier', 'target_identifier', 'bel_relation',
       'relation_type', 'source_database'.
    """
    df = pd.read_csv(path, sep='\t', dtype='str', keep_default_na=False).drop(
        ['Unnamed: 0'], axis=1)

    return df


def process_feature_matrix(
    file: str,
    label_idx_mapping: Dict[str, int],
    nodes: Set,
    index_name: str = None,
    index_mapping: bool = True,
) -> pd.DataFrame:
    """Process the feature matrix of a node type.

    The index is converted from the original string node label to the corresponding integer node label and sorted
    accordingly. The nodes that do not have real-valued features are initialized with a zero vector of length of the
    other vectors. The nodes that are in the feature matrix index, but are not in the node_label_mapping of the graph
    are dropped.

    Args:
        file (str): The path to feature matrix file with rows = string node labels and columns = vector indices.
        label_idx_mapping (dict): The dictionary with key = node string label, value = node integer label.
        nodes (set): The nodes of the specified node type.
        index_name (str): The name of the index column.
        index_mapping (bool): True if index (string label) is mapped to corresponding integer, False otherwise.

    Returns:
        full_feature_df (pd.DataFrame): The dataframe of processed feature matrix.
    """
    feature_df = pd.read_csv(file, sep='\t')

    if index_name:
        feature_df = feature_df.set_index(index_name)

    # Set column type to int64 (otherwise the columns will not be recognized as the same in both dfs and will be
    # concatenated onto axis=1 (columns)
    feature_df.columns = feature_df.columns.astype('int64')

    # Drop all columns of nodes that are not in the specified set of nodes existent in the current graph
    feature_df = feature_df[feature_df.index.isin(nodes)]

    # Nodes (string labels) with features
    nodes_with_features = set(feature_df.index)

    # Nodes (string labels) without features
    nodes_without_features = nodes.difference(nodes_with_features)

    # in case of duplicates in the df (lincs cyt profiling because of Broad CPD ID to Drugbank Mapping) drop the duplicates
    if len(nodes_with_features) != feature_df.shape[0]:
        feature_df = feature_df[~feature_df.index.duplicated(keep='first')]

    # Create dataframe of node features + df of nodes without features, initialize with 0
    drug_nodes_empty_features_df = pd.DataFrame(index=nodes_without_features, columns=np.arange(feature_df.shape[1]), dtype=float)
    drug_nodes_empty_features_df.fillna(0.0, inplace=True)

    # combine dataframe with existing gene expression feature values and empty dataframe
    full_feature_df = pd.concat([feature_df, drug_nodes_empty_features_df], sort=False, axis=0)

    # drop nodes without mapping (= no edges) from feature df
    # if nodes are not in the node label mapping, they have not been added to the graph
    full_feature_df = full_feature_df[full_feature_df.index.isin(label_idx_mapping)]

    if index_mapping:
        # Map drug_feature_df (drug fold changes) index (drugbank ID) to integer
        full_feature_df_int = full_feature_df.copy()
        full_feature_df_int.index = full_feature_df_int.index.map(label_idx_mapping)

        # sort feature matrix according to mapped node integer label (value of node_label_mapping)
        full_feature_df_int.sort_index(inplace=True)
        #assert full_feature_df_int.shape[0] != 0, 'Feature matrix is empty'
        return full_feature_df_int
    else:
        return full_feature_df


def add_node_features(
    graph: dgl.DGLHeteroGraph,
    node_type: str,
    modality_number: str,
    feature_matrix: np.array,
    device: torch.device,
) -> dgl.DGLHeteroGraph:
    """Add features to nodes of the graph.

    Args:
        graph (dgl.DGLHeteroGraph): The heterograph.
        node_type (str): The node type name.
        modality_number (str): The number of the current modality.
        feature_matrix (np.array): The corresponding feature matrix for the modality.

    Returns:

    """
    graph.nodes[node_type].data['feature_{}'.format(modality_number)] = torch.tensor(
        feature_matrix.to_numpy(dtype='float32'),
        device=device,
    )

    return graph

