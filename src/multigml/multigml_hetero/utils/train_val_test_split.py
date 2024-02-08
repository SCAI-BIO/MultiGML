# -*- coding: utf-8 -*-

"""Utility functions for splitting the data into train, validation and test sets."""

import os
import logging
from typing import Tuple, Dict, List

import dgl
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from collections import defaultdict, OrderedDict
from tqdm import tqdm

from multigml.multigml_hetero.heterograph import Heterograph
from multigml.multigml_hetero.utils.storing import open_graph, save_graph

logger = logging.getLogger(__name__)


def get_edge_ids_and_labels(heterograph, dgl_heterograph: dgl.DGLHeteroGraph) -> Tuple[Dict[Tuple[str], np.ndarray], Dict[Tuple[str], np.ndarray]]:
    """Get the edge ids and edge labels according to the edge types as a dictionary.
    
    Args:
        heterograph (Heterograph): The heterograph.

    Returns:
        (tuple): tuple containing:
                edge_id_array (dict): The edge ids for each edge type.
                edge_label_array (dict): The edge labels for each edge type.
    """
    # collector for edge ids for new subgraphs
    edge_id_collector = OrderedDict()
    edge_label_collector = OrderedDict()
    # get edge id array for every edge type
    for canonical_etype in dgl_heterograph.canonical_etypes:
        # edge ids are specific to the edge type, all starting at 0
        edge_ids = dgl_heterograph.edges(form='eid', etype=canonical_etype)

        edge_id_collector[canonical_etype] = edge_ids
        edge_label_collector[canonical_etype] = np.array([
            heterograph.canonical_edge_label_idx_mapping[canonical_etype]
            for _ in range(edge_ids.size()[0])
        ])

    # Edge id collector values start with 0 for every edge type
    # create array of edge ids and edge labels
    edge_id_list = []
    for x in edge_id_collector.values():
        edge_id_list.extend(x.numpy())
    edge_id_array = np.array(edge_id_list)

    edge_label_list = []
    for x in edge_label_collector.values():
        edge_label_list.extend(x)
    edge_label_array = np.array(edge_label_list)

    return edge_id_array, edge_label_array

def stratified_train_val_split(
    heterograph: Heterograph,
    dgl_heterograph: dgl.DGLHeteroGraph,
    seed: int,
    n_splits: int = 5,
) -> Tuple[List[dgl.DGLHeteroGraph], List[dgl.DGLHeteroGraph]]:
    """Do a stratified split of training and validation edges with equal proportions of each
    edge type occurring in each graph.

    Args:
        heterograph (Heterograph): The heterograph.
        dgl_heterograph (dgl.DGLHeteroGraph): The dgl heterograph.
        seed (int): The seed for shuffling.
        n_splits (int): The number of splits.

    Returns:
        (tuple): tuple containing:
            training_graph (dgl.DGLHeteroGraph): The training graph.
            validation_graph (dgl.DGLHeteroGraph): The validation graph.
    """
    # create a Stratified K-Folds cross-validator
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    # lists to store dgl.DGLHeteroGraphs of train and val graphs
    train_graph_list = []
    test_graph_list = []

    edge_id_array, edge_label_array = get_edge_ids_and_labels(dgl_heterograph=dgl_heterograph)

    # get train and val edges for each split
    for train_index, val_index in tqdm(skf.split(edge_id_array, edge_label_array), desc='Creating Train and Validation CV Splits'):

        # edges
        train_edges = edge_id_array[train_index]
        val_edges = edge_id_array[val_index]

        # edge labels as indices
        train_labels = edge_label_array[train_index]
        val_labels = edge_label_array[val_index]

        # create dictionary mapping edge type to edge ids
        edge_dict_train = defaultdict(list)
        for label, edge in zip(train_labels, train_edges):
            edge_label = heterograph.canonical_edge_idx_label_mapping[label]
            edge_dict_train[edge_label].append(edge)

        edge_dict_val = defaultdict(list)
        for label, edge in zip(val_labels, val_edges):
            edge_label = heterograph.canonical_edge_idx_label_mapping[label]
            edge_dict_val[edge_label].append(edge)
            
        # create edge subgraph
        train_subgraph = dgl_heterograph.edge_subgraph(
            edges=edge_dict_train,
            preserve_nodes=True,
            store_ids=True,
        )

        test_subgraph = dgl_heterograph.edge_subgraph(
            edges=edge_dict_val,
            preserve_nodes=True,
            store_ids=True,
        )

        train_graph_list.append(train_subgraph)
        test_graph_list.append(test_subgraph)

    return train_graph_list, test_graph_list


def single_train_test_split(
    heterograph,
    dgl_heterograph: dgl.DGLHeteroGraph,
    test_size: float,
    train_size: float,
    seed: int,
) -> Tuple[dgl.DGLHeteroGraph, dgl.DGLHeteroGraph]:
    """Do a stratified split of training and validation edges with equal proportions of each
    edge type occurring in each graph.

    Args:
        heterograph (Heterograph): The heterograph.
        dgl_heterograph (dgl.DGLHeteroGraph): The dgl heterograph to use for splitting edges.
        seed (int): The seed for shuffling.
        test_size (float): The ratio of the test set.
        train_size (float): The ratio of the training set.
        n_splits (int): The number of splits.
        stratify (bool): True for doing a stratified split.

    Returns:
        (tuple): tuple containing:
            training_graph (dgl.DGLHeteroGraph): The training graph.
            validation_graph (dgl.DGLHeteroGraph): The validation graph.
    """
    edge_id_array, edge_label_array = get_edge_ids_and_labels(heterograph=heterograph, dgl_heterograph=dgl_heterograph)

    # shuffle and split edge id tensors
    sss = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=test_size, 
        train_size=train_size,
        random_state=0
        )

    # split into stratified train and test set
    for train_index, test_index in sss.split(edge_id_array, edge_label_array):
     
        X_train, X_test = edge_id_array[train_index], edge_id_array[test_index]
        y_train, y_test = edge_label_array[train_index], edge_label_array[test_index]

    # TODO: do you needto sort ?
    edge_dict_train = defaultdict(list)
    for label, edge in zip(y_train, X_train):
        edge_label = heterograph.canonical_edge_idx_label_mapping[label]
        edge_dict_train[edge_label].append(edge)

    edge_dict_test = defaultdict(list)
    for label, edge in zip(y_test, X_test):
        edge_label = heterograph.canonical_edge_idx_label_mapping[label]
        edge_dict_test[edge_label].append(edge)

    # create edge subgraph
    train_subgraph = dgl.edge_subgraph(
        graph=dgl_heterograph,
        edges=edge_dict_train,
        preserve_nodes=True,
        store_ids=True,
    )
    test_subgraph = dgl.edge_subgraph(
        graph=dgl_heterograph,
        edges=edge_dict_test,
        preserve_nodes=True,
        store_ids=True,
    )
    return train_subgraph, test_subgraph


# ======================================
# Main Functions
# ======================================


def split_data(
    run_dir: str,
    heterograph: Heterograph,
    seed: int,
    n_splits: int,
    returns: bool = True,
) -> Tuple[List[dgl.DGLHeteroGraph], List[dgl.DGLHeteroGraph], List[dgl.DGLHeteroGraph]]:
    """Split the dataset.

    Args:
        run_dir (str): The current run directory.
        n_crossval_outer (int): The number of outer cvs.
        heterograph (Heterograph): The heterograph.
        device (th.device): The device.
        seed (int): The seed for shuffling.
        n_splits (int): The number of CV folds.
        returns (bool): True for returning the graphs, False for None,

    Returns:
        (tuple): Tuple containing:
            train_graph (list): The training graph(s).
            val_graph (list): The validation graph(s).
            test_graph (list): The test graph(s).

    """
    trainval_graph, test_graph = single_train_test_split(
        heterograph=heterograph,
        seed=seed,
        dgl_heterograph=heterograph.g,
        test_size=heterograph.test_ratio,
        train_size=(heterograph.training_ratio + heterograph.validation_ratio),
    )

    if n_splits == 1:

        total_ratio = heterograph.validation_ratio + heterograph.training_ratio
        validation_ratio = np.round_(heterograph.validation_ratio / total_ratio, decimals=1)
        training_ratio = np.round_(heterograph.training_ratio / total_ratio, decimals=1)
        # Plural, but is only one train graph and one val graph
        train_graph, val_graph = single_train_test_split(
            heterograph=heterograph,
            seed=seed,
            dgl_heterograph=trainval_graph,
            test_size=validation_ratio,
            train_size=training_ratio,
        )

        train_graphs = [train_graph]
        val_graphs = [val_graph]

        # save graphs for train and val split to file
        train_graph_file = os.path.join(run_dir, 'trainsplit_cv{}.pkl'.format(0))
        save_graph(G=train_graph, path=train_graph_file)

        val_graph_file = os.path.join(run_dir, 'valsplit_cv{}.pkl'.format(0))
        save_graph(G=val_graph, path=val_graph_file)

        test_graph_file = os.path.join(run_dir, 'testsplit_cv{}.pkl'.format(0))
        save_graph(G=test_graph, path=test_graph_file)

        logger.info('Saved data set splits to folder: {}'.format(run_dir))

        return train_graphs, val_graphs, test_graph

    # split into cross validation training and validation graph, stratified split
    else:
        train_graphs, val_graphs = stratified_train_val_split(
            heterograph=heterograph,
            dgl_heterograph=trainval_graph,
            seed=seed,
            n_splits=n_splits,
        )

        # save graphs for train and val split to file
        for i, train_graph in enumerate(train_graphs):
            train_graph_file = os.path.join(run_dir, 'trainsplit_cv{}.pkl'.format(i))
            save_graph(G=train_graph, path=train_graph_file)

        for i, val_graph in enumerate(val_graphs):
            val_graph_file = os.path.join(run_dir, 'valsplit_cv{}.pkl'.format(i))
            save_graph(G=val_graph, path=val_graph_file)

        test_graph_file = os.path.join(run_dir, 'testsplit_cv{}.pkl'.format(0))
        save_graph(G=test_graph, path=test_graph_file)

        logger.info('Saved data set splits to folder: {}'.format(run_dir))

        return train_graphs, val_graphs, test_graph


def load_data_split(
    data_set_split: str,
    n_splits: int,
    cv_out: int = 0,
    selected_cv_fold: int = None,
) -> Tuple[List[dgl.DGLHeteroGraph], List[dgl.DGLHeteroGraph], List[dgl.DGLHeteroGraph]]:
    """Load the already split graphs for training and validation.

    Args:
        data_set_split (str): Path to the data set split directory.
        cv_out (int): Current outer CV fold.
        n_splits (int): Number of CV Splits.
        selected_cv_fold (int): The selected data set split of a CV fold that is used in this run.

    Returns:
        (tuple): Tuple containing:
            train_graph (list): The training graphs.
            val_graph (list): The validation graph.
            test_graph (dgl.DGLHeteroGraph): The test graph.
    """
    if selected_cv_fold:
        test_graph_file = os.path.join(data_set_split, 'testsplit_cv{}.pkl'.format(selected_cv_fold))
        test_graph = open_graph(test_graph_file)
        training_graphs_file = os.path.join(data_set_split, 'trainsplit_cv{}.pkl'.format(selected_cv_fold))
        validation_graphs_files = os.path.join(data_set_split, 'valsplit_cv{}.pkl'.format(selected_cv_fold))

        # list of dgl.DGLHeteroGraphs for train and val sets
        train_graph = open_graph(path=training_graphs_file)
        val_graph = open_graph(path=validation_graphs_files)

        return [train_graph], [val_graph], [test_graph]
    else:

        test_graph_file = os.path.join(data_set_split, 'testsplit_cv{}.pkl'.format(cv_out))
        test_graph = open_graph(test_graph_file)

        if n_splits == 1:

            training_graphs_file = os.path.join(data_set_split, 'trainsplit_cv{}.pkl'.format(cv_out))
            validation_graphs_files = os.path.join(data_set_split, 'valsplit_cv{}.pkl'.format(cv_out))

            # list of dgl.DGLHeteroGraphs for train and val sets
            train_graph = open_graph(path=training_graphs_file)
            val_graph = open_graph(path=validation_graphs_files)

            return [train_graph], [val_graph], [test_graph]
        else:

            training_graphs_files = [
                os.path.join(data_set_split, 'trainsplit_cv{}.pkl'.format(str(i)))
                for i in range(n_splits)
            ]
            validation_graphs_files = [
                os.path.join(data_set_split, 'valsplit_cv{}.pkl'.format(str(i)))
                for i in range(n_splits)
            ]

            # list of dgl.DGLHeteroGraphs for train and val sets
            train_graphs = [open_graph(path=file) for file in training_graphs_files]
            val_graphs = [open_graph(path=file) for file in validation_graphs_files]

        return train_graphs, val_graphs, [test_graph]


