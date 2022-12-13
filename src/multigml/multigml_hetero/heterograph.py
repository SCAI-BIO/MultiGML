# -*- coding: utf-8 -*-

"""Create DGL (Deep Graph Library) data object for heterographs."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import dgl
import numpy as np
import pandas as pd
import torch as th
from multigml.constants import (DST_NODE_TYPE, RELATION_TYPE,
                                SOURCE_IDENTIFIER, SOURCE_NODE_TYPE,
                                SRC_NODE_TYPE, TARGET_IDENTIFIER,
                                TARGET_NODE_TYPE)
from multigml.multigml_hetero.utils.preprocess_features import (
    add_node_features, process_feature_matrix)
from multigml.multigml_hetero.utils.preprocess_heterograph import (
    get_graph_df, get_heterograph, read_graph_inconsecutive_labels)

logger = logging.getLogger(__name__)


@dataclass
class Heterograph():
    """A class for graphs with heterogeneous nodes and edges and multi-modal input for the nodes."""

    def __init__(
        self,
        graph_file: str,
        device: th.device,
        feature_file_list: str = None,
        feature_name_list: str = None,
        feature_node_type_list: str = None,
        eval_edge_type: str = None,
        eval_edge_type_inv: str = None,
        training_ratio: float = 0.7,
        validation_ratio: float = 0.2,
        test_ratio: float = 0.1,
    ) -> None:
        """Initialize the heterograph.

        Args:
            graph_file (str): The file path of the heterograph.
            feature_file_list (list): List of files of features, has to correspond to feature_node_type_list and feature_name_list.
            feature_name_list (list): List of names of features, has to correspond to features_file_list and feature_node_type_list.
            feature_node_type_list (list): List of node types of features, has to correspond to features_file_list and feature_name_list.
            eval_edge_type (str): The edge type to evaluate.
            eval_edge_type_inv (str): The inverse edge type to evaluate.
            training_ratio (float): The ratio of training edges.
            validation_ratio (float): The ratio of validation edges.
            test_ratio (float): The ratio of test edges.
            device (torch.device): The device to run on.
        """
        self.device = device 
        if not graph_file:
            raise ValueError('No graph file provided.')

        self.graph_file = graph_file
        self.graph_df: pd.DataFrame = get_graph_df(path=self.graph_file)
        self.feature_file_list = feature_file_list
        self.feature_name_list = feature_name_list
        self.feature_node_type_list = feature_node_type_list
        self.feature_files = self.create_feature_file_dict(
            feature_file_list=self.feature_file_list,
            feature_name_list=self.feature_name_list,
            feature_node_type_list=self.feature_node_type_list,
        )

        self.training_ratio = training_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.best_auc = 0

        self.interaction_types = self.get_interaction_types()
        self.eval_edge_type = (
            self.interaction_types[eval_edge_type][SRC_NODE_TYPE],
            eval_edge_type,
            self.interaction_types[eval_edge_type][DST_NODE_TYPE]
        )
        self.eval_edge_type_inv = (
            self.interaction_types[eval_edge_type_inv][SRC_NODE_TYPE],
            eval_edge_type_inv,
            self.interaction_types[eval_edge_type_inv][DST_NODE_TYPE]
        )

        self.test_edge_types = [(
            self.interaction_types[eval_edge_type][SRC_NODE_TYPE],
            eval_edge_type,
            self.interaction_types[eval_edge_type][DST_NODE_TYPE]
        )]

        # load data and get mappings from node labels to indices
        graph_dict_label, graph_dict_int, label_idx_mapping_type_specific, idx_label_mapping, label_idx_mapping, \
        node_label_to_node_type, node_type_labels, num_nodes_dict, graph_dict_tensor = read_graph_inconsecutive_labels(
            path=self.graph_file,
        )
        self.label_idx_mapping_type_specific = label_idx_mapping_type_specific
        self.num_nodes_dict = num_nodes_dict
        self.label_idx_mapping = label_idx_mapping

        self.idx_label_mapping = idx_label_mapping

        self.graph_dict_int = graph_dict_int
        self.graph_dict_label = graph_dict_label
        self.graph_dict_tensor = graph_dict_tensor
        self.node_type_labels = node_type_labels

        self.node_indices_to_type = self.get_node_indices_to_type()
        self.node_labels_to_type = node_label_to_node_type
        
        # create dgl heterograph
        self.g = get_heterograph(
            graph_dict=self.graph_dict_tensor,
            num_nodes_dict=num_nodes_dict,
            device=self.device,
        )
        self.node_type_labels_generic = self.get_node_type_labels_generic()
        # add features to heterograph
        self.g = self.featurize_heterograph_generic(
            g=self.g,
        )

        # node types
        self.node_types = self.g.ntypes

        # total number of nodes
        num_nodes = 0
        for n_type in self.node_types:
            num_nodes += self.g.number_of_nodes(n_type)
        self.num_nodes = num_nodes

        # edge types
        self.edge_types = self.g.etypes

        # node and edge tracker for test, train val sets
        self.edge_tracker = defaultdict(dict)
        self.node_tracker = defaultdict(dict)
        for etype in self.edge_types:
            self.edge_tracker[etype]['test'] = {}
            self.edge_tracker[etype]['train'] = {}
            self.edge_tracker[etype]['val'] = {}
        for ntype in self.node_types:
            self.node_tracker[ntype]['test'] = {}
            self.node_tracker[ntype]['train'] = {}
            self.node_tracker[ntype]['val'] = {}

        # edge label to index mapping
        self.edge_label_idx_mapping = {
            label: idx for idx, label in enumerate(self.edge_types, start=1)
        }

        # edge index to label mapping
        self.edge_idx_label_mapping = {
            idx: label for idx, label in enumerate(self.edge_types, start=1)
        }

        # canonical edge types
        self.canonical_edges = self.g.canonical_etypes

        # canonical edge label to index and index to label mapping
        self.canonical_edge_label_idx_mapping = dict()
        self.canonical_edge_idx_label_mapping = dict()
        for edge_label, edge_index in self.edge_label_idx_mapping.items():
            for canonical_edge in self.canonical_edges:
                if edge_label == canonical_edge[1]:
                    self.canonical_edge_label_idx_mapping[canonical_edge] = edge_index
                    self.canonical_edge_idx_label_mapping[edge_index] = canonical_edge

        # add  feature matrix dictionary
        self.feature_matrices = self.create_feature_matrices_dict()

        # node feature sizes
        self.modality_sizes_dict = self.create_modality_sizes_dict()

    def get_edges_indices(
        self,
        edge_type: Tuple[str, str, str],
    ) -> List[Tuple[int, int]]:
        """Get the edge list of the specified edge type from node indices.

        Args:
            edge_type (tuple): The edge label of the edge type.

        Returns:
            (list): The edges for the specified edge type.
        """
        return self.graph_dict_int[edge_type]

    def get_edges_labels(
        self,
        edge_type: Tuple[str, str, str]
    ) -> List[Tuple[str, str]]:
        """Get the edge list of the specified edge type from node labels.

        Args:
            edge_type (tuple): The edge label of the edge type.

        Returns:
            (list): A list of edges for the specified edge type.
        """
        return self.graph_dict_label[edge_type]

    def get_node_indices_to_type(self) -> Dict[int, str]:
        """Get a dictionary with node indices as keys and node type as value."""
        node_type_labels = self.get_node_type_labels_generic()
        node_indices_to_type_dict = {}
        for ntype, nodes in node_type_labels.items():
            for node in nodes:
                node_indices_to_type_dict[self.label_idx_mapping[node]] = ntype

        return node_indices_to_type_dict

    def get_node_type_labels_generic(self) -> Dict[str, Set]:
        """Get set of nodes of all node types in dataframe.

        Returns:
            nodes_set (dict): A dictionary with key = node type, value = set of node type labels.
        """
        nodes = {}
        all_nodes = []
        for interaction in self.interaction_types:
            src_ntype = self.interaction_types[interaction][SRC_NODE_TYPE]
            dst_ntype = self.interaction_types[interaction][DST_NODE_TYPE]

            nodes[src_ntype] = []
            nodes[dst_ntype] = []

        for interaction in self.interaction_types:
            # get source and dst node type for each interaction
            src_type = self.interaction_types[interaction][SRC_NODE_TYPE]
            dst_type = self.interaction_types[interaction][DST_NODE_TYPE]

            source_nodes = self.graph_df.loc[self.graph_df[RELATION_TYPE] == interaction][SOURCE_IDENTIFIER]
            dst_nodes = self.graph_df.loc[self.graph_df[RELATION_TYPE] == interaction][TARGET_IDENTIFIER]

            nodes[src_type].extend(list(source_nodes))
            nodes[dst_type].extend(list(dst_nodes))
            all_nodes.extend(list(source_nodes))
            all_nodes.extend(list(dst_nodes))

        all_nodes = set(all_nodes)

        nodes_set = defaultdict(set)

        for key, vals in nodes.items():
            # values are nodes for source node  and destination node type
            for val in vals:
                nodes_set[key].add(val)

        total_nodes = 0

        for nodes in nodes_set.values():
            total_nodes += len(nodes)

        return nodes_set

    def get_feature_matrix_generic(
        self,
        node_type: str,
        modality_name: str,
        feature_file: str,
        index_name: str,
        index_mapping: bool = True,
    ) -> pd.DataFrame:
        """Get the feature matrix dataframe of the specified node type.

        Args:
            node_type (str): The node type for which the generic feature matrix should get produced.
            modality_name (str): The name of the modality.
            feature_file (str): The path of the feature file.
            index_name (str): The name of the index column.
            index_mapping (bool): True for mapping index (string label) to corresponding integer, False otherwise.

        Returns:
            feature_matrix (pd.DataFrame): The feature matrix of the specified node type.
        """
        logger.debug('feature file: {}, node type: {}'.format(feature_file, node_type))
        if modality_name in self.feature_files[node_type]:
            nodes = set([self.idx_label_mapping[node_type][x.item()] for x in self.g.nodes(node_type)])
            feature_matrix = process_feature_matrix(
                file=self.feature_files[node_type][modality_name],
                label_idx_mapping=self.label_idx_mapping,
                nodes=nodes,
                index_name=index_name,
                index_mapping=index_mapping,
            )
            return feature_matrix
        else:
            logger.info(f'Node type {node_type} is not known.')

    def get_feature_matrix_fast(
        self,
        node_type: str,
        feature_file: str,
    ) -> pd.DataFrame:
        """Get the feature matrix dataframe of the specified node type."""
        logger.debug('(FAST) feature file: {}, node type: {}'.format(feature_file, node_type))
        if feature_file in self.feature_files[node_type]:
            feature_matrix = self.feature_matrices[node_type][feature_file]
            return feature_matrix
        else:
            logger.debug(f'Node type {node_type} is not known.')

    def map_from_adj_idx_to_node_type(
        self,
        adj_idx: int,
        adj_idx_to_label: Dict[int, str],
    ) -> str:
        """Map from adjacency index to node type.

        Args:
            adj_idx (int): The index of node in adjacency matrix.
            adj_idx_to_label (dict): The mapping from adjacency matrix indices to node label.

        Returns:
            node_type (str): The node type of adjacency node index.
        """
        node_label = adj_idx_to_label[adj_idx]
        node_type = self.node_labels_to_type[node_label]
        return node_type

    def map_from_adj_idx_to_node_idx(
        self,
        adj_idx: int,
        adj_idx_to_label: Dict[int, str],
    ) -> int:
        """Map from adjacency index to node index.

        Args:
            adj_idx (int): The index of node in adjacency matrix.
            adj_idx_to_label: The mapping from adjacency matrix index to node label.

        Returns:
            node_idx (int): The node label of adjacency node index.
        """
        # map  to node label
        node_label = adj_idx_to_label[adj_idx]
        # map to original node index
        node_idx = self.label_idx_mapping[node_label]
        return node_idx

    def featurize_heterograph_generic(
        self,
        g: dgl.DGLHeteroGraph,
        index_mapping: bool = True,
        test_triples: np.array = None,
        test_node_types: np.array = None,
        test_graph: bool = False,
    ) -> dgl.DGLHeteroGraph:
        """Add the features to the nodes of the given dgl.heterograph.

        Args:
            g (dgl.DGLHeteroGraph): The heterograph to add features to.
            index_mapping (bool): True for mapping index (string label) to corresponding integer, False otherwise.
            test_triples (np.array): The triples of source node, edge type and destination node.
            test_node_types (np.array): The node types of the source and destination node of the triples.
            test_graph (bool): True if the test graph is used, False otherwise.
            device (torch.device): The torch device ('cpu' or 'cuda').

        Returns:
            g (dgl.DGLHeteroGraph): The heterograph with features.
        """
        logger.info('Adding features to graph.')

        self.modality_number_dict = defaultdict(dict)

        if test_graph:
            src_type = np.unique(test_node_types[:, 0])[0]  # drug
            dst_type = np.unique(test_node_types[:, 1])[0]  # disease
            unique_src_nodes = np.unique(test_triples[:, 0])  # drug nodes original index
            unique_dst_nodes = np.unique(test_triples[:, 2])  # disease nodes original index
            src_idx = list(np.unique(test_triples[:, 0]))
            dst_idx = list(np.unique(test_triples[:, 2]))
            for node_type, unique_nodes_idx in zip(
                [src_type, dst_type],
                [unique_src_nodes, unique_dst_nodes]
            ):  # drug disease

                # current modality_number
                current_modality_num = 0

                for modality_name, feature_file in self.feature_files[node_type]:
                    self.modality_number_dict[node_type][current_modality_num] = modality_name

                    feature_matrix = self.get_feature_matrix_generic(
                        node_type=node_type,
                        modality_name=modality_name,
                        feature_file=feature_file,
                        index_name='Unnamed: 0',
                        index_mapping=index_mapping,
                    )
                    # original indices

                    selected_feature_matrix = feature_matrix.iloc[unique_nodes_idx, :]

                    # Add features to nodes
                    g = add_node_features(
                        graph=g,
                        node_type=node_type,
                        modality_number=str(current_modality_num),
                        feature_matrix=selected_feature_matrix,
                        device=self.device,
                    )

                    # update current modality number
                    current_modality_num += 1
        # not test  graph
        else:
            for node_type in self.feature_files:
                # add features only if node type exists
                if node_type in g.ntypes:

                    # current modality number (for adding feature to g.data)
                    current_modality_num = 0

                    for modality_name, feature_file in self.feature_files[node_type].items():
                        self.modality_number_dict[node_type][current_modality_num] = modality_name

                        logger.info('Featurizing with {} from {}'.format(modality_name, feature_file))

                        feature_matrix = self.get_feature_matrix_generic(
                            node_type=node_type,
                            modality_name=modality_name,
                            feature_file=feature_file,
                            index_name='Unnamed: 0',
                            index_mapping=index_mapping,
                        )
                        # select drug features of nodes that are in training graph
                        node_index_list = list(g.nodes(node_type))
                        
                        if type(node_index_list[0]) != int:
                           node_index_list = [tensor.item() for tensor in list(g.nodes(node_type))]
                        else:
                           node_index_list = list(g.nodes(node_type))

                        selected_feature_matrix = feature_matrix.iloc[node_index_list, :]

                        # Add features to nodes
                        g = add_node_features(
                            graph=g,
                            node_type=node_type,
                            modality_number=str(current_modality_num),
                            feature_matrix=selected_feature_matrix,
                            device=self.device,
                        )

                        # update current modality number
                        current_modality_num += 1
                else:
                    continue
        return g

    def create_feature_matrices_dict(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Create a dictionary mapping node types to the feature names to the feature matrix.

        Returns:
            (dict): Mapping of node types to feature names to feature matrices.
        """
        feature_matrices = dict()
        # for every node type, add the corresponding feature name and features
        for ntype, modalities in self.feature_files.items():

            # create modality dictionary for every node type
            feature_matrices[ntype] = defaultdict(dict)
            # get all node labels for that node type
            all_nodes = self.node_type_labels_generic[ntype]

            for modality_name, file in modalities.items():
                matrix = process_feature_matrix(
                    file=self.feature_files[ntype][modality_name],
                    label_idx_mapping=self.label_idx_mapping,
                    nodes=all_nodes,
                    index_name='Unnamed: 0',
                    index_mapping=True,
                )
                feature_matrices[ntype][modality_name] = matrix

        return feature_matrices

    def get_interaction_types(self) -> Dict[str, Dict[str, str]]:
        """Read the unique interaction types from the file."""
        interaction_df = self.graph_df[[RELATION_TYPE, SOURCE_NODE_TYPE, TARGET_NODE_TYPE]].drop_duplicates()
        interaction_types = dict()
        for index, row in interaction_df.iterrows():
            interaction_types[row[RELATION_TYPE]] = dict()
            interaction_types[row[RELATION_TYPE]][SRC_NODE_TYPE] = row[SOURCE_NODE_TYPE]
            interaction_types[row[RELATION_TYPE]][DST_NODE_TYPE] = row[TARGET_NODE_TYPE]
            # ADD inverse edge
            inv_edge_type = '{}_inverse'.format(row[RELATION_TYPE])
            interaction_types[inv_edge_type] = dict()
            interaction_types[inv_edge_type][SRC_NODE_TYPE] = row[TARGET_NODE_TYPE]
            interaction_types[inv_edge_type][DST_NODE_TYPE] = row[SOURCE_NODE_TYPE]
        return interaction_types

    def create_feature_file_dict(
        self,
        feature_file_list: List[str],
        feature_name_list: List[str],
        feature_node_type_list: List[str],
    ) -> Dict[str, Dict[str, str]]:
        """Create feature file dict with feature file for every node type. If there are multiple files for each
        node type, input them as a list.

        Args:
            feature_file_list (list): List of files of features, has to correspond to feature_node_type_list and feature_name_list.
            feature_name_list (list): List of names of features, has to correspond to features_file_list and feature_node_type_list.
            feature_node_type_list (list): List of node types of features, has to correspond to features_file_list and feature_name_list.

        Returns:
            (dict): Mapping of node type of feature to feature name to feature file path.
        """
        feature_files = defaultdict(dict)

        assert (len(feature_file_list) == len(feature_name_list) == len(feature_node_type_list)), \
            "The number of feature names has to correspond to the number of feature files and node types."

        for file, name, ntype in zip(feature_file_list, feature_name_list, feature_node_type_list):
            feature_files[ntype][name] = file

        return feature_files

    def create_modality_sizes_dict(self) -> Dict[str, Dict[str, int]]:
        """For a node type, create a dictionary mapping modality to feature size.

        Returns:
            (dict): Mapping from modality name to feature size.
        """
        modality_sizes_dict = defaultdict(dict)
        # for every node type, add the corresponding feature name and features
        for ntype, modalities in self.feature_matrices.items():

            for modality_name, df in modalities.items():
                modality_sizes_dict[ntype][modality_name] = df.shape[1]
                # previously: (self.g.nodes[ntype].data['feature']).size()

        return modality_sizes_dict

