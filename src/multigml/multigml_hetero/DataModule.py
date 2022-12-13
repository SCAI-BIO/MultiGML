# -*- coding: utf-8 -*-

"""The data module for the RGCN model."""

import logging
from typing import Dict, List, Tuple

import dgl
import numpy as np
import pytorch_lightning as pl
import torch
import torch as th

from multigml.constants import INDICATION, INDICATION_INVERSE
from multigml.multigml_hetero.utils.prepare_model_graph import \
    load_graph_configuration
from multigml.multigml_hetero.utils.train_val_test_split import (load_data_split,
                                                             split_data)
from multigml.multigml_hetero.utils.storing import open_graph                                                             

logger = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int,
        n_layers: int,
        run_dir: str,
        device: th.device,
        create_split: bool = False,
        data_set_split: str = None,
        seed: int = 43,
        which_graph: str = None,
        graph_file: str = None,  # GRAPH_FILE,
        feature_file_list: List[str] = None,
        feature_name_list: List[str] = None,
        feature_node_type_list: List[str] = None,
        features: str = None,
        eval_edge_type: str = INDICATION,
        eval_edge_type_inv: str = INDICATION_INVERSE,
        training_ratio: float = 0.7,
        validation_ratio: float = 0.2,
        test_ratio: float = 0.1,
        n_crossval_outer: int = 1,
        n_splits: int = 1,
        which_features: str = 'reduced',
        ratio: str = 'equal',
        modality_combinations: Dict[str, List[str]] = None,
        factor: int = None,
        use_uva: bool = None,
        full_training: bool = False,
        neg_sample_size: int = 1,
        test_only_eval_etype: bool = False,
        test_graph_file: str = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.device = device
        self.create_split = create_split
        self.seed = seed
        self.data_set_split = data_set_split
        self.run_dir = run_dir
        self.which_graph = which_graph
        self.graph_file = graph_file
        self.feature_file_list = feature_file_list
        self.feature_name_list = feature_name_list
        self.feature_node_type_list = feature_node_type_list
        #: The names of the features used as string in list format.
        self.features = features
        #: edge type to be used for testing and evalutation
        self.eval_edge_type = eval_edge_type
        #: edge type to be used for testing and evalutation
        self.eval_edge_type_inv = eval_edge_type_inv
        #: Float number that determines the size of the training set
        self.training_ratio = training_ratio
        #: Float number that determines the size of the validation set
        self.validation_ratio = validation_ratio
        #: Float number that determines the size of the test set
        self.test_ratio = test_ratio
        #: Number of outer cross validations.
        self.n_crossval_outer = n_crossval_outer
        #: Number of inner cross validations.
        self.n_splits = n_splits
        #: Features to use.
        self.which_features = which_features
        self.modality_combinations = modality_combinations
        self.use_uva = use_uva
        # True for using training and validation set for training, False for only using training dataset.
        self.full_training = full_training
        self.neg_sample_size = neg_sample_size
        self.test_only_eval_etype = test_only_eval_etype

        heterograph = self.get_heterograph()
        self.heterograph = heterograph
        train_graphs, val_graphs, test_graphs = self.get_train_val_test_graphs()
        #(heterograph, train_graphs, val_graphs, test_graphs) = self.prepare_data()
        self.train_graphs = train_graphs
        self.val_graphs = val_graphs
        self.trainval_graphs = [
            dgl.merge([train_graphs[i], val_graphs[i]]) for i in range(len(train_graphs))]
        self.test_graphs = test_graphs
        #: The number of neighbors to sample for each edge type.
        self.fanouts = self.create_fanout_dict(factor=factor, ratio=ratio)
        self.test_graph_file = test_graph_file

    def get_heterograph(self):

        heterograph, nx_graph = load_graph_configuration(
            graph_file=self.graph_file,
            feature_file_list=self.feature_file_list,
            feature_name_list=self.feature_name_list,
            feature_node_type_list=self.feature_node_type_list,
            features=self.features,
            which_graph=self.which_graph,
            eval_edge_type=self.eval_edge_type,
            eval_edge_type_inv=self.eval_edge_type_inv,
            training_ratio=self.training_ratio,
            validation_ratio=self.validation_ratio,
            test_ratio=self.test_ratio,
            which_features=self.which_features,
            modality_combinations=self.modality_combinations,
            device=self.device,
            use_uva=self.use_uva,
        )

        logger.info('Run directory: {}'.format(self.run_dir))

        return heterograph

    def get_train_val_test_graphs(self):

        if self.create_split:
            train_graphs, val_graphs, test_graphs = split_data(
                heterograph=self.heterograph,
                seed=self.seed,
                run_dir=self.run_dir,
                n_splits=self.n_splits,
            )

        elif self.data_set_split:
            train_graphs = []
            val_graphs = []
            test_graphs = []

            for cv_out in range(self.n_crossval_outer):

                train_graph, val_graph, test_graph = load_data_split(
                    data_set_split=self.data_set_split,
                    cv_out=cv_out,
                    n_splits=self.n_splits,
                )
                train_graphs.extend(train_graph)
                val_graphs.extend(val_graph)
                test_graphs.extend(test_graph)
            
            return train_graphs, val_graphs, test_graphs

        else:
            raise Exception("Please provide a directory with data set splits to the --data_set_split option "
                            "or set the option --create_split to True to create a new data split.")

        return train_graphs, val_graphs, test_graphs

    def train_dataloader(self) -> dgl.dataloading.EdgeDataLoader:
        """train_collator = ExternalEdgeCollator(
            self.heterograph.g,
            train_idx,
            sampler=self.sampler,
            paper_offset,
            feats,
            label
        )"""
        if self.full_training:
            train_dataloader = self.generate_edgeloader(
                heterograph=self.heterograph.g,
                # TODO: change for cross validation
                subgraph=self.trainval_graphs[0],
                fanouts=self.fanouts,
                n_layers=self.n_layers,
                batch_size=self.batch_size,
                modality_combinations=self.modality_combinations,
                device=self.device,
                reproducible_dataloading=True,
            )
        else:
            train_dataloader = self.generate_edgeloader(
                heterograph=self.heterograph.g,
                # TODO: change for cross validation
                subgraph=self.train_graphs[0],
                fanouts=self.fanouts,
                n_layers=self.n_layers,
                batch_size=self.batch_size,
                modality_combinations=self.modality_combinations,
                device=self.device,
                reproducible_dataloading=True,
            )
        return train_dataloader

    def val_dataloader(self) -> dgl.dataloading.EdgeDataLoader:
        val_dataloader = self.generate_edgeloader(
            heterograph=self.heterograph.g,
            # TODO: change for cross validation
            subgraph=self.val_graphs[0],
            fanouts=self.fanouts,
            n_layers=self.n_layers,
            batch_size=self.batch_size,
            modality_combinations=self.modality_combinations,
            device=self.device,
            shuffle=False,
            reproducible_dataloading=True,
        )
        return val_dataloader

    def test_dataloader(self) -> dgl.dataloading.EdgeDataLoader:
        if self.test_graph_file:
            test_graph = open_graph(self.test_graph_file)
            subgraph=None
        else:
            test_graph = self.heterograph.g
            subgraph=self.test_graphs[0]
        test_dataloader = self.generate_edgeloader(
            heterograph=test_graph,
            # TODO: change for cross validation
            subgraph=subgraph,
            fanouts=self.fanouts,
            n_layers=self.n_layers,
            batch_size=self.batch_size,
            eval_edge_type=self.heterograph.eval_edge_type,
            eval_edge_type_inv=self.heterograph.eval_edge_type_inv,
            test_only_eval_etype=self.test_only_eval_etype,
            modality_combinations=self.modality_combinations,
            device=self.device,
            shuffle=False,
            neg_sample_size=self.neg_sample_size,
            reproducible_dataloading=False,
        )
        return test_dataloader

    def create_fanout_dict(self, factor: int, ratio: str = 'equal') -> Dict[Tuple[str, str, str], int] or int:
        """Create dictionary mapping edge type to number of edges to sample in batch sampling.

        Args:
            ratio (str, optional): Defines how many edges per edge type are sampled. If 'equal', 
            takes the same amount for every edge type. If 'proportional', takes proportionate numbers for the number of edges per type. 
            Defaults to 'equal'.
            factor (int): The number of edge per edge type, or factor to multiply the ratios with.

        Returns:
            Dict[Tuple[str, str, str], int]: _description_
        """
        edge_count = {
            etype: self.heterograph.g.num_edges(etype) for etype in self.heterograph.g.canonical_etypes
        }
        # Proportional
        if ratio == 'proportional':
            # Get total number of edges of each edge type in training

            total_edge_count = sum(edge_count.values())
            # Get ratio for each edge type
            ratios = {
                etype: edge_count[etype] / total_edge_count for etype in edge_count
            }
            # create dict
            # Note: number of edges per edge type are being balanced out here if number of edges > 10
            # Since side effect edge ratio is very small
            # Otherwise, number of edges would be very high for protein protein edges  etc.
            fanouts = {
                etype: int(ratios[etype] * factor)
                if (int(ratios[etype] * factor) < 10) else 10
                for etype in ratios
            }
        # Equal
        elif ratio == 'equal':
            #fanouts = factor
            fanouts = {etype: factor for etype in edge_count}
        else:
            raise Exception

        return fanouts

    def generate_edgeloader(
        self,
        heterograph: dgl.DGLHeteroGraph,
        subgraph: dgl.DGLHeteroGraph,
        fanouts: Dict[Tuple[str, str, str], int] or int,
        n_layers: int,
        batch_size: int,
        modality_combinations: Dict[str, List[str]],
        device: torch.device,
        eval_edge_type: Tuple[str, str, str] = None,
        eval_edge_type_inv: Tuple[str, str, str] = None,
        g_sampling: dgl.DGLHeteroGraph = None,
        test_only_eval_etype: bool = False,
        shuffle: bool = True,
        neg_sample_size: int = 1,
        reproducible_dataloading: bool = False,
    ) -> dgl.dataloading.EdgeDataLoader:
        """Generate data loader.

        Args:
            heterograph (dgl.DGLHeteroGraph): The heterograph .
            subgraph (dgl.DGLHeteroGraph): The graph, (either training or validation or test graph) .
            n_layers (int): The total number of layers.
            batch_size (int): The batch size for sampling.
            modality_combinations (dict): Mapping from node type to modality list.
            g_sampling (dgl.DGLHeteroGraph): The graph to use for sampling.
            eval_edge_type (tuple): The evaluation edge type.
            eval_edge_type_inv (tuple): The inverse evaluation edge type.
            device (torch.device): The device to run on.

        Returns:
            train_loader (dgl.dataloading.EdgeDataLoader): The data loader for the edges.
        """
        if reproducible_dataloading:
            dgl.random.seed(0)
        # New method to split data from DGL
        # edge IDs used to compute the output
        # get original edge types (not inverse)
        # create train dict for these
        # create inverse train dict
        # merge original and inverse train dict
        if subgraph == None:
            subgraph = heterograph
        if test_only_eval_etype:
            evaluation_edge_types = [eval_edge_type, eval_edge_type_inv]
            train_eid_dict = {
                canonical_etype: torch.arange(subgraph.num_edges(
                    canonical_etype[1]), dtype=torch.int64, device=device)
                for canonical_etype in evaluation_edge_types
            }
            train_eid_dict_inverse = {
                canonical_etype: torch.arange(subgraph.num_edges(
                    canonical_etype[1]), dtype=torch.int64, device=device)
                for canonical_etype in evaluation_edge_types if 'inverse' in canonical_etype[1]
            }
            reverse_etypes = {
                (canonical_etype[1] if 'inverse' not in canonical_etype[1] else canonical_etype[1]): ('{}_inverse'.format(canonical_etype[1]) if 'inverse' not in canonical_etype[1] else canonical_etype[1].split('_')[0])
                for canonical_etype in evaluation_edge_types
            }
            replace = True
        else:
            train_eid_dict = {
                canonical_etype: torch.arange(subgraph.num_edges(
                    canonical_etype[1]), dtype=torch.int64, device=device)
                for canonical_etype in subgraph.canonical_etypes
            }
            train_eid_dict_inverse = {
                canonical_etype: torch.arange(subgraph.num_edges(
                    canonical_etype[1]), dtype=torch.int64, device=device)
                for canonical_etype in subgraph.canonical_etypes if 'inverse' in canonical_etype[1]
            }
            reverse_etypes = {
                (canonical_etype[1] if 'inverse' not in canonical_etype[1] else canonical_etype[1]): ('{}_inverse'.format(canonical_etype[1]) if 'inverse' not in canonical_etype[1] else canonical_etype[1].split('_')[0])
                for canonical_etype in subgraph.canonical_etypes

            }
            replace = False

        # create node features that should be prefetched
        # dictionary mapping from node type to list of feature names with 'feature_x', with x being the number of the modality
        node_feats = dict()
        for ntype, modalities in modality_combinations.items():
            feature_names = []
            for i in range(len(modalities)):
                feature_names.append('feature_{}'.format(i))
            node_feats[ntype] = feature_names

        # train sampler
        sampler = dgl.dataloading.NeighborSampler(
            fanouts=[fanouts] * n_layers, prefetch_node_feats=node_feats, replace=replace)
        # pick one negative edge per positive edge
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(neg_sample_size)

        # iterates over a set of edges in minibatches,
        # yielding the subgraph induced by the edge minibatch and blocks to be consumed by the module above.
        # PyTorch DataLoader that iterates over the training edge ID array train_eids in batches,
        # putting the list of generated blocks onto GPU.

        sampler = dgl.dataloading.as_edge_prediction_sampler(
            sampler=sampler,
            exclude='reverse_types',
            reverse_etypes=reverse_etypes,
            # reverse_eids=train_eid_dict_inverse,
            negative_sampler=neg_sampler)

        # Preserving reproducibility
        if reproducible_dataloading:
            g = torch.Generator()
            g.manual_seed(0)
        else:
            g = None

        train_loader = dgl.dataloading.DataLoader(
            graph=heterograph,
            indices=train_eid_dict,
            graph_sampler=sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0,
            device=device,
            use_uva=self.use_uva,
            generator=g,
        )

        return train_loader
