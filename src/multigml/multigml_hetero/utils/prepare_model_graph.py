# -*- coding: utf-8 -*-

"""Prepare the model and the graph used in the deployer."""

import itertools
import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import torch as th
from multigml.constants import (CONDITION, CONDITION_GRAPH_FILE, DRUG,
                                FEATURE_FILE_DISEASE_CUI2VEC_NAME,
                                FEATURE_FILE_DISEASE_RANDOM_NAME,
                                FEATURE_FILE_DRUG_CYT_NAME,
                                FEATURE_FILE_DRUG_FC_NAME,
                                FEATURE_FILE_DRUG_FCFP_NAME,
                                FEATURE_FILE_DRUG_RANDOM_NAME,
                                FEATURE_FILE_PROTEIN_ESM_NAME,
                                FEATURE_FILE_PROTEIN_GO_NAME,
                                FEATURE_FILE_PROTEIN_RANDOM_NAME, FILE_NAME,
                                FULL_CUI2VEC_DISGENET_FILE,
                                FULL_DRUG_COUNT_FCFP_FILE,
                                FULL_DRUG_CYTOLOGICAL_FILE,
                                FULL_ESM_PROT_EMBEDDINGS,
                                FULL_LINCS_DRUG_FC_FILE,
                                FULL_PROTEIN_GENE_ONTOLOGY_FINGERPRINT,
                                INDICATION, INDICATION_INVERSE, NODE_TYPE,
                                PROTEIN, RANDOM_DISEASE_FEATURE_FILE,
                                RANDOM_DRUG_FEATURE_FILE,
                                RANDOM_PROTEIN_FEATURE_FILE)
from multigml.multigml_hetero.heterograph import Heterograph
from multigml.multigml_hetero.utils.configure import configure
from multigml.multigml_hetero.utils.train_val_test_split import split_data

logger = logging.getLogger(__name__)

feature_dict = {
    # DRUG
    FEATURE_FILE_DRUG_RANDOM_NAME: {NODE_TYPE: DRUG, FILE_NAME: RANDOM_DRUG_FEATURE_FILE},
    FEATURE_FILE_DRUG_CYT_NAME: {NODE_TYPE: DRUG, FILE_NAME: FULL_DRUG_CYTOLOGICAL_FILE},
    FEATURE_FILE_DRUG_FC_NAME: {NODE_TYPE: DRUG, FILE_NAME: FULL_LINCS_DRUG_FC_FILE},
    FEATURE_FILE_DRUG_FCFP_NAME: {NODE_TYPE: DRUG, FILE_NAME: FULL_DRUG_COUNT_FCFP_FILE},
    # PROTEIN
    FEATURE_FILE_PROTEIN_RANDOM_NAME: {NODE_TYPE: PROTEIN, FILE_NAME: RANDOM_PROTEIN_FEATURE_FILE},
    FEATURE_FILE_PROTEIN_ESM_NAME: {NODE_TYPE: PROTEIN, FILE_NAME: FULL_ESM_PROT_EMBEDDINGS},
    FEATURE_FILE_PROTEIN_GO_NAME: {NODE_TYPE: PROTEIN, FILE_NAME: FULL_PROTEIN_GENE_ONTOLOGY_FINGERPRINT},
    # DISEASE
    FEATURE_FILE_DISEASE_RANDOM_NAME: {NODE_TYPE: CONDITION, FILE_NAME: RANDOM_DISEASE_FEATURE_FILE},
    FEATURE_FILE_DISEASE_CUI2VEC_NAME: {NODE_TYPE: CONDITION, FILE_NAME: FULL_CUI2VEC_DISGENET_FILE},
}

def create_heterograph(
    which_graph: str,
    device=th.device,
    graph_file: str = None,
    feature_file_list: List[str] = None,
    feature_name_list: List[str] = None,
    feature_node_type_list: List[str] = None,
    features: str = None,
    eval_edge_type: str = INDICATION,
    eval_edge_type_inv: str = INDICATION_INVERSE,
    training_ratio: float = 0.6,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.3,
    which_features: str = 'reduced',
    modality_combinations: Dict[str, List[str]] = None,
    use_uva: bool = None,
) -> Heterograph:
    """Initialize the heterograph.

    Args:
        graph_file (str): The path of the graph file.
        feature_file_list (list): List of files of features, has to correspond to feature_node_type_list and feature_name_list.
        feature_name_list (list): List of names of features, has to correspond to features_file_list and feature_node_type_list.
        feature_node_type_list (list): List of node types of features, has to correspond to features_file_list and feature_name_list.
        features (str): The names of the features used as string in list format.
        eval_edge_type (str): Edge type to evaluate.
        eval_edge_type_inv: Inverse edge type to evaluate.
        training_ratio (float): The float that determines the size of the training set.
        validation_ratio (float): The float that determines the size of the validation set.
        test_ratio (float): The float that determines the size of the test set.
        which_graph (str): The name of the graph to use: '50' or 'complete'.
        modality_combinations (dict): The modality combinations to use for each node type.
    Returns:
        heterograph (Heterograph): The heterograph.
    """
    graph_file = choose_graph_file(which_graph=which_graph, graph_file=graph_file)

    feature_file_list, feature_name_list, feature_node_type_list = choose_features(
        which_graph=which_graph,
        feature_name_list=feature_name_list,
        feature_node_type_list=feature_node_type_list,
        feature_file_list=feature_file_list,
        which_features=which_features,
        modality_combinations=modality_combinations,
    )
    logger.info('Generating heterograph.')
    logger.info('Graph file: {}'.format(graph_file))
    if use_uva:
        device = 'cpu'
    # instantiate heterograph
    heterograph = Heterograph(
        graph_file=graph_file,
        feature_file_list=feature_file_list,
        feature_name_list=feature_name_list,
        feature_node_type_list=feature_node_type_list,
        training_ratio=training_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        eval_edge_type=eval_edge_type,
        eval_edge_type_inv=eval_edge_type_inv,
        device=device,
    )
    logger.info("Heterograph Info: {}".format(heterograph.g.ntypes, heterograph.g.etypes))

    return heterograph


def load_graph_configuration(
    which_graph: str,
    device: th.device,
    graph_file: str = None,
    feature_file_list: List[str] = None,
    feature_name_list: List[str] = None,
    feature_node_type_list: List[str] = None,
    features: str = None,
    eval_edge_type: str = INDICATION,
    eval_edge_type_inv: str = INDICATION_INVERSE,
    training_ratio: float = 0.7,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.2,
    directed: bool = False,
    which_features: str = 'reduced',
    modality_combinations: Dict[str, List[str]] = None,
    use_uva: bool = None,
) -> Tuple[Heterograph, nx.Graph]:
    """Create the heterograph and the current run directory.

    Args:
        graph_file (str): The path to the graph file.
        which_graph (str): The name of the graph, '50' or 'complete'.
        feature_file_list (list): List of files of features, has to correspond to feature_node_type_list and feature_name_list.
        feature_name_list (list): List of names of features, has to correspond to features_file_list and feature_node_type_list.
        feature_node_type_list (list): List of node types of features, has to correspond to features_file_list and feature_name_list.
        features (str): The names of the features used as string in list format.
        eval_edge_type (str): Edge type to evaluate.
        eval_edge_type_inv: Inverse edge type to evaluate.
        training_ratio (float): The float that determines the size of the training set.
        validation_ratio (float): The float that determines the size of the validation set.
        test_ratio (float): The float that determines the size of the test set.
        use_stored_data (bool): True for using previously stored data of the graph, False otherwise.
        directed (bool): True for using nx.DiGraph, False for using nx.Graph.
        which_features (str): Which features to use, one of: 'reduced' (features after PCA transformation),
         'full' (full features without PCA), 'random' (random features), 'unimodal' (one modality for each node type).
         modality_combinations: The combinations of modalities to use for the features.

    Returns:
        (tuple): tuple containing:
            heterograph (Heterograph): The heterograph.
            nx_graph (nx.Graph): The networkX version of the heterograph.
    """
    configure()

    heterograph = create_heterograph(
        graph_file=graph_file,
        feature_file_list=feature_file_list,
        feature_name_list=feature_name_list,
        feature_node_type_list=feature_node_type_list,
        features=features,
        eval_edge_type=eval_edge_type,
        eval_edge_type_inv=eval_edge_type_inv,
        training_ratio=training_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        which_graph=which_graph,
        which_features=which_features,
        modality_combinations=modality_combinations,
        device=device,
        use_uva=use_uva,
    )

    if directed:
        nx_graph = nx.from_pandas_edgelist(
            df=heterograph.graph_df,
            source='source_identifier',
            target='target_identifier',
            edge_attr='relation_type',
            create_using=nx.DiGraph,
        )
    else:
        nx_graph = nx.from_pandas_edgelist(
            df=heterograph.graph_df,
            source='source_identifier',
            target='target_identifier',
            edge_attr='relation_type',
            create_using=nx.Graph,
        )

    return heterograph, nx_graph


def make_split(
    run_dir: str,
    device: th.device,
    graph_file: str = None,  # GRAPH_FILE,
    feature_file_list: List[str] = None,
    feature_name_list: List[str] = None,
    feature_node_type_list: List[str] = None,
    features: str = None,
    eval_edge_type: str = INDICATION,
    eval_edge_type_inv: str = INDICATION_INVERSE,
    training_ratio: float = 0.7,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.2,
    which_graph: str = None,
    seed: int = 43,
    n_splits: int = 5,
    which_features: str = 'full',
    modality_combinations: Dict = None,
):
    """Run link prediction.

    Args:
        graph_file (str): The path of the graph file.
        feature_file_list (list): List of files of features, has to correspond to feature_node_type_list and feature_name_list.
        feature_name_list (list): List of names of features, has to correspond to features_file_list and feature_node_type_list.
        feature_node_type_list (list): List of node types of features, has to correspond to features_file_list and feature_name_list.
        features (str): The feature names as string in list format.
        eval_edge_type (str): The edge type that will be used for the evaluation.
        eval_edge_type_inv (str): The inverse edge type for evaluation.
        training_ratio (float): The ratio of training edges.
        validation_ratio (float): The ratio of validation edges.
        test_ratio (float): The ratio of test edges.
        use_cuda (bool): True to use CUDA, False otherwise.
        which_graph (str): The name of the graph.
        seed (int): The seed for shuffling the train val split.
        n_splits (int): The number of splits for the cross-validation split.

    Returns:
        None
    """
    heterograph, nx_graph = load_graph_configuration(
        graph_file=graph_file,
        feature_file_list=feature_file_list,
        feature_name_list=feature_name_list,
        feature_node_type_list=feature_node_type_list,
        features=features,
        which_graph=which_graph,
        eval_edge_type=eval_edge_type,
        eval_edge_type_inv=eval_edge_type_inv,
        training_ratio=training_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        device=device,
        which_features=which_features,
        modality_combinations=modality_combinations,
    )
    logger.info('Run directory: {}'.format(run_dir))
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    logger.info('Heterograph: Num nodes: {}, num edges : {}'.format(heterograph.g.num_nodes(), heterograph.g.num_edges()))
    split_data(
        run_dir=run_dir,
        #n_crossval_outer=n_crossval_outer,
        heterograph=heterograph,
        seed=seed,
        n_splits=n_splits,
        returns=False,
    )

def get_modality_dict(feature_dict):
    """Get a mapping from node type to possible modalities."""
    modality_dict = defaultdict(list)
    for modality, dicts in feature_dict.items():
        if 'random' not in modality:
            modality_dict[dicts['node_type']].append(modality)
    return modality_dict

def get_modality_combinations(feature_dict):
    """From a list of possible modalities per node type, get all possible combinations."""
    modality_dict = get_modality_dict(feature_dict)
    combinations = defaultdict(dict)
    for ntype, values in modality_dict.items():
        all_combinations = []
        for r in range(len(values)+1):
            c = itertools.combinations(values, r)
            c = list(c)
            result = [list(x) for x in c]
            all_combinations += result
        all_combinations = all_combinations[1:]
        for i in range(len(all_combinations)):
            combinations[ntype][i] = all_combinations[i]

    return combinations

def choose_graph_file(which_graph: str, graph_file: str):
    # select graph file
    if which_graph == 'complete':
        graph_file = CONDITION_GRAPH_FILE
    elif which_graph == 'custom':
        if graph_file != None:
            graph_file = graph_file
        else:
            raise Exception("The argument `which_graph='custom'` is selected, but no graph path is given via `graph_path`.")
    else:
        raise Exception("Please select one of the graphs: 'complete'")
    return graph_file

def choose_features(
    which_graph: str,
    which_features: str = 'full',
    feature_file_list: List[str] = None,
    feature_name_list: List[str] = None,
    feature_node_type_list: List[str] = None,
    modality_combinations: Dict[str, List[str]] = None,
):
    """Choose the graph features depending on the given parameters."""
    # select features
    # for the hyperparameter optimization, where different combinations of modalities are being tested
    
    if which_graph == 'custom':
        if which_features != None:
            feature_file_list = [
                os.path.join(which_features, 'drugbank_count_fcfp.tsv'),
                os.path.join(which_features, 'lincs_cytological_profiling_drug_features.tsv'),
                os.path.join(which_features, 'lincs_drug_fc.tsv'),
                os.path.join(which_features, 'protein_embeddings_esm.tsv'),
                os.path.join(which_features, 'protein_go_fingerprint.tsv'),
                os.path.join(which_features, 'cui2vec_disgenet.tsv'),
            ]
            feature_name_list = [
                FEATURE_FILE_DRUG_FCFP_NAME,
                FEATURE_FILE_DRUG_CYT_NAME,
                FEATURE_FILE_DRUG_FC_NAME,
                FEATURE_FILE_PROTEIN_ESM_NAME,
                FEATURE_FILE_PROTEIN_GO_NAME,
                FEATURE_FILE_DISEASE_CUI2VEC_NAME,
            ]
            feature_node_type_list = [DRUG, DRUG, DRUG, PROTEIN, PROTEIN, CONDITION] 
        else:
            raise Exception("You specified a custom graph, but did not provide custom graph features folder path via `which_features`.")

    else:
        if which_features == 'random':
            feature_file_list = [
                RANDOM_DRUG_FEATURE_FILE,
                RANDOM_PROTEIN_FEATURE_FILE,
                RANDOM_DISEASE_FEATURE_FILE,
            ]
            feature_name_list = [
                FEATURE_FILE_DRUG_RANDOM_NAME,
                FEATURE_FILE_PROTEIN_RANDOM_NAME,
                FEATURE_FILE_DISEASE_RANDOM_NAME,
            ]
            feature_node_type_list = [DRUG, PROTEIN, CONDITION]

        elif which_features == 'full':
            feature_file_list = [
                FULL_DRUG_COUNT_FCFP_FILE,
                FULL_DRUG_CYTOLOGICAL_FILE,
                FULL_LINCS_DRUG_FC_FILE,
                FULL_ESM_PROT_EMBEDDINGS,
                FULL_PROTEIN_GENE_ONTOLOGY_FINGERPRINT,
                FULL_CUI2VEC_DISGENET_FILE,
            ]
            feature_name_list = [
                FEATURE_FILE_DRUG_FCFP_NAME,
                FEATURE_FILE_DRUG_CYT_NAME,
                FEATURE_FILE_DRUG_FC_NAME,
                FEATURE_FILE_PROTEIN_ESM_NAME,
                FEATURE_FILE_PROTEIN_GO_NAME,
                FEATURE_FILE_DISEASE_CUI2VEC_NAME,
            ]
            feature_node_type_list = [DRUG, DRUG, DRUG, PROTEIN, PROTEIN, CONDITION]    

        else:
            raise Exception("Please select one of the following feature types: 'full', 'random'.")

    return feature_file_list, feature_name_list, feature_node_type_list

def get_modalities_dict_from_which_features(which_graph: str, which_features: str):
    """Get a mapping from ntype to modality only from the parameter of 'which_features'."""
    _, modalities, ntypes = choose_features(which_graph, which_features)
    modality_dict = dict()
    for mod, ntype in zip(modalities, ntypes):
        modality_dict[ntype] = [mod]
    return modality_dict

