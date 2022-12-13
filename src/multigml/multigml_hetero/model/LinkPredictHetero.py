# -*- coding: utf-8 -*-

"""RGCN layer implementation.

Adapted from the EntityClassify class.

The original code was taken from the DGL Heterograph Implementation.

https://github.com/dmlc/dgl/tree/0b902d032ab92f9c4fd6d3d8b283129851fb43d2/examples/pytorch/rgcn-hetero

"""
import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import dgl
import plotly_express as px
import pytorch_lightning as pl
import torch as th
import torch.nn as nn
from multigml.multigml_hetero.model.BaseRGAT import BaseRGAT
from multigml.multigml_hetero.model.BaseRGCNHetero import BaseRGCNHetero
from multigml.multigml_hetero.model.ScorePredictor import ScorePredictor
from multigml.multigml_hetero.utils.storing import (
    append_to_list_in_nested_defaultdict, to_json)
from multigml.multigml_hetero.utils.validation import (
    create_all_labels_and_scores,
    create_all_labels_and_scores_etype_specific,
    keys_to_strings)
from mlflow.tracking import MlflowClient
from pytorch_lightning.core.memory import ModelSummary
from torchmetrics import (AUROC, ROC, Accuracy, AveragePrecision, F1Score,
                          MetricCollection, Precision, PrecisionRecallCurve,
                          Recall, RetrievalHitRate, RetrievalMRR)

logger = logging.getLogger(__name__)

class LinkPredictHetero(pl.LightningModule):
    """The main class for the link prediction model."""

    def __init__(self,
                 hparams: Dict,
                 ):
        super().__init__()
        #: The heterograph to perform convolution on.
        self.g = hparams['g']
        #: The input dimension.
        self.input_dim = hparams['input_dim']
        #: The hidden dimension of the RGCN.
        self.h_dim = hparams['h_dim']
        #: The output dimension of the RGCN.
        self.out_dim = hparams['out_dim']
        #: The hidden dimension of the edge embedding.
        self.rel_names = list(set(hparams['g'].etypes))
        self.rel_names.sort()
        #: The number of hidden layers in the RGCN.
        self.num_hidden_layers = hparams['num_hidden_layers']
        #: The dropout rate.
        self.dropout = hparams['dropout']
        #: The dropout rate for the multimodal layer.
        self.dropout_multimodal = hparams['dropout_multimodal']
        #: Dictionary mapping modality to dropout ratio. / single dropout ratio previously
        #self.dropout_modalities = hparams['dropout_modalities']
        #: True for using the node's own representation from the previous layer, false otherwise.
        self.use_self_loop = hparams['use_self_loop']
        #: Mini batch size.
        self.batch_size = hparams['batch_size']
        #: The binary cross-entropy with logits loss function.
        self.n_layers = hparams['n_layers']
        #: The learning rate for the optimizer.
        self.learning_rate = hparams['learning_rate']
        #: The weight decay for the Adam optimizer.
        self.weight_decay = hparams['weight_decay']
        #: The term added to the denominator to improve numerical stability
        self.epsilon = hparams['epsilon']
        #: True for using MLPEdge, False otherwise.
        self.use_edge_weight = hparams['use_edge_weight']
        #: The run directory.
        self.run_dir = hparams['run_dir']
        #: Method to calculate loss: 'sum', 'mean' or 'weighted'.
        self.average = hparams['average']
        #: Method to calculate metrics: 'weighted' or 'micro'.
        self.metrics_average = hparams['metrics_average']
        #: Current validation epoch.
        self.current_val_epoch = 0
        #: The local logger.
        self.local_logger = logging.getLogger(__name__)
        #: The features to use.
        self.which_features = hparams['which_features']
        #: The collector for the evaluation edges and scores.
        self.test_score_collector = defaultdict(list)
        #: The input dimensions of the modalities.
        self.modality_sizes_dict = hparams['modality_sizes_dict']
        #: The canoncial edges types.
        self.canonical_edges = hparams['canonical_edges']
        #: Dictionary of modalities to indeces.
        self.modality_number_dict = hparams['modality_number_dict']
        #: Evaluation edge type.
        self.eval_edge_type = hparams['eval_edge_type']
        #: Inverse evaluation edge type.
        self.eval_edge_type_inv = hparams['eval_edge_type_inv']
        #: Index label mapping.
        self.idx_label_mapping = hparams['idx_label_mapping']
        #: Device to create tensors on
        self.tensor_device = hparams['tensor_device']
        #: The modalities used for this model.
        self.used_modalities = hparams['used_modalities']
        #: Boolean for calculating edge type specific metrics.
        self.calculate_etype_metrics = hparams['calculate_etype_metrics']
        #: Reduction parameter for the loss.
        self.reduction = hparams['reduction']
        #: True for using heterogeneous graph attention, False otherwise.
        self.use_attention = hparams['use_attention']
        #: The attention dropout rate.
        self.attention_dropout = hparams['attention_dropout']
        #: The number of attention heads.
        self.num_heads = hparams['num_heads']
        #: True for returning attention weight.
        self.return_attention = hparams['return_attention']
        #: Test graph file used.
        self.test_graph_file = hparams['test_graph_file']

        self.count = 0

        # Remove parameters that do not need to be hyperparameters
        for p in ['g','modality_sizes_dict', 'modality_number_dict', 'canonical_edges', 'idx_label_mapping', 'eval_edge_type', 'eval_edge_type_inv']:
            del hparams[p]

        #: Dictionary with all class variables
        self.save_hyperparameters()

        # List of etypes in the current batch
        self.current_used_etypes = []


        # Metrics
        if self.metrics_average == 'weighted':
            self.metrics = MetricCollection([
                Accuracy(average='weighted', num_classes=2),
                AUROC(average='weighted'),
                PrecisionRecallCurve(),
                F1Score(average='weighted', num_classes=2),
                Precision(average='weighted', num_classes=2),
                Recall(average='weighted', num_classes=2),
                AveragePrecision(pos_label=1, ),
                ROC(pos_label=1),
            ])
            self.retrieval_metrics = MetricCollection({
                'RetrievalMRR': RetrievalMRR(),
                'RetrievalHitRate_1': RetrievalHitRate(k=1),
                'RetrievalHitRate_3': RetrievalHitRate(k=3),
                'RetrievalHitRate_5': RetrievalHitRate(k=5),
                'RetrievalHitRate_10': RetrievalHitRate(k=10),
            })

        elif self.metrics_average == 'micro':
            self.metrics = MetricCollection([
                Accuracy(),
                AUROC(),
                PrecisionRecallCurve(),
                F1Score(),
                Precision(),
                Recall(),
                AveragePrecision(pos_label=1,),
                ROC(pos_label=1),
            ])
            self.retrieval_metrics = MetricCollection({
                'RetrievalMRR': RetrievalMRR(),
                'RetrievalHitRate_1': RetrievalHitRate(k=1),
                'RetrievalHitRate_3': RetrievalHitRate(k=3),
                'RetrievalHitRate_5': RetrievalHitRate(k=5),
                'RetrievalHitRate_10': RetrievalHitRate(k=10),
            })
        else:
            raise Exception("Please choose one of the following arguments for 'average_metrics': 'weighted' or 'micro'.")

        self.train_metrics = self.metrics.clone(prefix='train_')
        self.validation_metrics = self.metrics.clone(prefix='val_')
        self.test_metrics = self.metrics.clone(prefix='test_')
        self.train_retrieval_metrics = self.retrieval_metrics.clone(prefix='train_')
        self.validation_retrieval_metrics = self.retrieval_metrics.clone(prefix='val_')
        self.test_retrieval_metrics = self.retrieval_metrics.clone(prefix='test_')

        # List of etypes in the current batch
        self.current_used_etypes = []

        self.train_retrieval_metrics = self.retrieval_metrics.clone(prefix='train_')
        self.validation_retrieval_metrics = self.retrieval_metrics.clone(prefix='val_')
        self.test_retrieval_metrics = self.retrieval_metrics.clone(prefix='test_')

        self.compute_val_etype_metrics = False

        # Edge type specific metrics
        # Dictionaries mapping edge type to corresponding metric collection

        self.train_metrics_etypes = dict()
        self.val_metrics_etypes = dict()
        self.test_metrics_etypes = dict()
        
        self.train_retrieval_metrics_etypes = dict()
        self.val_retrieval_metrics_etypes = dict()
        self.test_retrieval_metrics_etypes = dict()

        if self.calculate_etype_metrics:
            for etype in self.g.etypes:
                self.train_metrics_etypes[etype] = self.metrics.clone(prefix='train_{}_'.format(etype)).to(self.tensor_device)
                self.val_metrics_etypes[etype] = self.metrics.clone(prefix='val_{}_'.format(etype)).to(self.tensor_device)
                self.test_metrics_etypes[etype] = self.metrics.clone(prefix='test_{}_'.format(etype)).to(self.tensor_device)

                self.train_retrieval_metrics_etypes[etype] = self.retrieval_metrics.clone(prefix='train_{}_'.format(etype)).to(self.tensor_device) 
                self.val_retrieval_metrics_etypes[etype] = self.retrieval_metrics.clone(prefix='val_{}_'.format(etype)).to(self.tensor_device)
                self.test_retrieval_metrics_etypes[etype] = self.retrieval_metrics.clone(prefix='test_{}_'.format(etype)).to(self.tensor_device)
        # only evaluation edge type    
        else:
            self.train_metrics_etypes[self.eval_edge_type[1]] = self.metrics.clone(prefix='train_{}_'.format(self.eval_edge_type[1])).to(self.tensor_device)
            self.val_metrics_etypes[self.eval_edge_type[1]] = self.metrics.clone(prefix='val_{}_'.format(self.eval_edge_type[1])).to(self.tensor_device)
            self.test_metrics_etypes[self.eval_edge_type[1]] = self.metrics.clone(prefix='test_{}_'.format(self.eval_edge_type[1])).to(self.tensor_device)

            self.train_retrieval_metrics_etypes[self.eval_edge_type[1]] = self.retrieval_metrics.clone(prefix='train_{}_'.format(self.eval_edge_type[1])).to(self.tensor_device) 
            self.val_retrieval_metrics_etypes[self.eval_edge_type[1]] = self.retrieval_metrics.clone(prefix='val_{}_'.format(self.eval_edge_type[1])).to(self.tensor_device)
            self.test_retrieval_metrics_etypes[self.eval_edge_type[1]] = self.retrieval_metrics.clone(prefix='test_{}_'.format(self.eval_edge_type[1])).to(self.tensor_device)
 
            
        # loss function
        self.loss_func = nn.BCELoss(reduction=self.reduction)

        #### Initialize the Encoder
        if self.use_attention:
            self.encoder = BaseRGAT(
                g=self.g,
                input_dim=self.input_dim,
                h_dim=self.h_dim,
                out_dim=self.out_dim,
                num_hidden_layers=self.num_hidden_layers,
                dropout=self.dropout,
                dropout_multimodal=self.dropout_multimodal,
                attention_dropout=self.attention_dropout,
                num_heads=self.num_heads,
                use_self_loop=self.use_self_loop,
                which_features=self.which_features,
                modality_sizes_dict=self.modality_sizes_dict,
            )
        else:
            self.encoder = BaseRGCNHetero(
                g=self.g,
                input_dim=self.input_dim,
                h_dim=self.h_dim,
                out_dim=self.out_dim,
                num_hidden_layers=self.num_hidden_layers,
                dropout=self.dropout,
                dropout_multimodal=self.dropout_multimodal,
                use_self_loop=self.use_self_loop,
                which_features=self.which_features,
                modality_sizes_dict=self.modality_sizes_dict,
            )

        # Predictor
        self.predictor = ScorePredictor(
            canonical_edge_types=self.canonical_edges,
            out_dim=self.out_dim,
            device=self.tensor_device,
        )

        ##### Number of relations
        self.num_rels = len(set(self.g.canonical_etypes))

    def configure_optimizers(self) -> th.optim.Optimizer:
        # optimizer
        optimizer = th.optim.AdamW([x for x in self.parameters(recurse=True)], lr=self.learning_rate, weight_decay=self.weight_decay, eps=self.epsilon)
        return optimizer

    def on_train_start(self) -> None:
        super().on_train_start()

        if self.global_rank == 0 and isinstance(self.logger.experiment, MlflowClient):
            self.mlflow: MlflowClient = self.logger.experiment

            # Save model description to mlflow artifacts
            self.mlflow.log_text(self.logger.run_id, str(ModelSummary(self)), #, mode=ModelSummary.MODE_FULL)),
                                 "./model/model_summary.txt")
            self.mlflow.log_text(self.logger.run_id, str(self), "./model/model_summary_with_params.txt")

    def training_step(self, train_batch, batch_idx):

        _, positive_graph, negative_graph, blocks = train_batch

        # mapping of node type to modality name to tensor
        # has all node types
        input_features = self.create_src_features(blocks=blocks)

        # Feed the list of blocks and the input node features to the RGCN and get the outputs
        pos_score, neg_score = self.forward(
            positive_graph=positive_graph,
            negative_graph=negative_graph,
            blocks=blocks,
            h=input_features,
            use_edge_weight=self.use_edge_weight,
        )

        loss = self.compute_loss(
            pos_score=pos_score,
            neg_score=neg_score,
            average=self.average,
        )

        # Compute AUC
        preds, target, preds_retrieval, target_retrieval, indexes = create_all_labels_and_scores(
            pos_score=pos_score,
            neg_score=neg_score,
            device=self.tensor_device,
        )

        self.train_metrics.update(preds, target)
        self.train_retrieval_metrics.update(preds_retrieval, target_retrieval, indexes=indexes)

        if self.calculate_etype_metrics:
            preds_etypes, target_etypes, preds_etypes_retrieval, target_etypes_retrieval, indexes_etypes = create_all_labels_and_scores_etype_specific(
                pos_score=pos_score,
                neg_score=neg_score,
                device=self.tensor_device,
            )
            for etype in list(preds_etypes.keys()):
                if etype not in self.current_used_etypes:
                    self.current_used_etypes.append(etype)

            for etype in list(preds_etypes.keys()):
                if etype not in self.current_used_etypes:
                    self.current_used_etypes.append(etype)

            for etype in preds_etypes:
                self.train_metrics_etypes[etype[1]].update(preds_etypes[etype], target_etypes[etype])
                self.train_retrieval_metrics_etypes[etype[1]].update(preds_etypes_retrieval[etype], target_etypes_retrieval[etype], indexes_etypes[etype])
    
            # logs metrics for each training_step,
            # and the average across the epoch, to the progress bar and logger

            self.log_dict({
                'train_loss_{}'.format(etype[1]): value for etype, value in loss.items()
            },
                on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
            )

        self.log('loss', loss['total'], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

        return {
            'loss': loss['total'],
        }

    def training_epoch_end(self, outs):

        result = self.train_metrics.compute()
        result_retrieval = self.train_retrieval_metrics.compute()
        self.train_metrics.reset()
        self.train_retrieval_metrics.reset()
        # Remove metrics so that mlflow works
        result.pop('train_ROC', None)
        result.pop('train_PrecisionRecallCurve', None)
        self.log_dict(result)
        self.log_dict(result_retrieval)

        if self.calculate_etype_metrics:
            for etype in self.current_used_etypes:
                result_etype = self.train_metrics_etypes[etype[1]].compute()
                self.train_metrics_etypes[etype[1]].reset()
                result_etype.pop('train_{}_ROC'.format(etype[1]), None)
                result_etype.pop('train_{}_PrecisionRecallCurve'.format(etype[1]), None)
                self.log_dict(result_etype)

                result_retrieval_etype = self.train_retrieval_metrics_etypes[etype[1]].compute()
                self.train_retrieval_metrics_etypes[etype[1]].reset()
                self.log_dict(result_retrieval_etype)

        self.current_used_etypes = []

    def validation_step(self, val_batch, batch_idx):
        input_nodes, positive_graph, negative_graph, blocks = val_batch

        input_features = self.create_src_features(blocks=blocks)

        # forward pass and predict the scores for the validation set of  positive and negative graph
        pos_score, neg_score = self.forward(
            positive_graph=positive_graph,
            negative_graph=negative_graph,
            blocks=blocks,
            h=input_features,
            use_edge_weight=self.use_edge_weight,
        )

        loss = self.compute_loss(
            pos_score=pos_score,
            neg_score=neg_score,
            evaluation_edge_types=[self.eval_edge_type, self.eval_edge_type_inv],
            average=self.average,
        )

        preds, target, preds_retrieval, target_retrieval, indexes = create_all_labels_and_scores(
            pos_score=pos_score,
            neg_score=neg_score,
            device=self.tensor_device,
        )

        self.validation_metrics.update(preds, target)
        self.validation_retrieval_metrics.update(preds_retrieval, target_retrieval, indexes=indexes)
        self.log('val_loss', loss['total'], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

        if self.calculate_etype_metrics:
            preds_etypes, target_etypes, preds_etypes_retrieval, target_etypes_retrieval, indexes_etypes = create_all_labels_and_scores_etype_specific(
                pos_score=pos_score,
                neg_score=neg_score,
                device=self.tensor_device,
            )

            for etype in list(preds_etypes.keys()):
                if etype not in self.current_used_etypes:
                    self.current_used_etypes.append(etype)
            
            for etype in list(preds_etypes.keys()):
                self.val_metrics_etypes[etype[1]].update(preds_etypes[etype], target_etypes[etype])
                self.val_retrieval_metrics_etypes[etype[1]].update(preds_etypes_retrieval[etype], target_etypes_retrieval[etype], indexes_etypes[etype])
    
            self.log_dict({
                'val_loss_{}'.format(etype[1]): value for etype, value in loss.items()
            },
                on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
            )
        else:
            # Check if validation etype is in current batch
            if self.eval_edge_type in pos_score:
                preds_val_etype, target_val_etype, preds_val_etype_retrieval, target_val_etype_retrieval, indexes_val_etype = create_all_labels_and_scores_etype_specific(
                    pos_score=pos_score,
                    neg_score=neg_score,
                    device=self.tensor_device,
                    validation_etype=self.eval_edge_type
                )
                self.val_metrics_etypes[self.eval_edge_type[1]].update(preds_val_etype, target_val_etype)
                self.val_retrieval_metrics_etypes[self.eval_edge_type[1]].update(preds_val_etype_retrieval, target_val_etype_retrieval, indexes_val_etype)
                self.compute_val_etype_metrics = True

        return {
            'val_loss': loss['total'],
        }

    def validation_epoch_end(self, outs: List[Dict[str, th.Tensor]]):

        val_loss = th.stack([x['val_loss'] for x in outs]).mean()
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)

        # Compute the metrics of this epoch
        result = self.validation_metrics.compute()
        result_retrieval = self.validation_retrieval_metrics.compute()

        # Remove metrics so that mlflow works
        result.pop('val_ROC', None)
        result.pop('val_PrecisionRecallCurve', None)
        self.log_dict(result, on_epoch=True)
        self.log_dict(result_retrieval, on_epoch=True)

        # reset metrics for each epoch
        self.validation_metrics.reset()
        self.validation_retrieval_metrics.reset()

        # Etype specific metrics
        if self.calculate_etype_metrics:
            for etype in self.current_used_etypes:
                result_etype = self.val_metrics_etypes[etype[1]].compute()
                self.val_metrics_etypes[etype[1]].reset()
                result_etype.pop('val_{}_ROC'.format(etype[1]), None)
                result_etype.pop('val_{}_PrecisionRecallCurve'.format(etype[1]), None)
                self.log_dict(result_etype, on_epoch=True)

                result_retrieval_etype = self.val_retrieval_metrics_etypes[etype[1]].compute()
                self.val_retrieval_metrics_etypes[etype[1]].reset()
                self.log_dict(result_retrieval_etype, on_epoch=True)
        else:
            # Calculate validation edge type metrics
            if self.compute_val_etype_metrics:
                result_val_etype = self.val_metrics_etypes[self.eval_edge_type[1]].compute()
                self.val_metrics_etypes[self.eval_edge_type[1]].reset()
                result_val_etype.pop('val_{}_ROC'.format(self.eval_edge_type[1]), None)
                result_val_etype.pop('val_{}_PrecisionRecallCurve'.format(self.eval_edge_type[1]), None)
                self.log_dict(result_val_etype, on_epoch=True)

                result_retrieval_val_etype = self.val_retrieval_metrics_etypes[self.eval_edge_type[1]].compute()
                self.val_retrieval_metrics_etypes[self.eval_edge_type[1]].reset()
                self.log_dict(result_retrieval_val_etype, on_epoch=True)

        # increase validation epoch
        self.current_val_epoch += 1
        self.current_used_etypes = []
        self.compute_val_etype_metrics = False

    def test_step(self, test_batch, batch_idx):

        input_nodes, positive_graph, negative_graph, blocks = test_batch

        h = self.create_src_features(blocks=blocks)

        # forward pass and predict the scores for the validation set of  positive and negative graph
        pos_score_indices, neg_score_indices = self.forward(
            positive_graph=positive_graph,
            negative_graph=negative_graph,
            blocks=blocks,
            h=h,
            use_edge_weight=self.use_edge_weight,
            return_attention=self.return_attention,
        )

        # Map edges to the node labels
        pos_score, neg_score = self.translate_test_scores(
            negative_graph=negative_graph,
            positive_graph=positive_graph,
            neg_score=neg_score_indices,
            pos_score=pos_score_indices,
            test_graph_file=self.test_graph_file,
        )

        loss = self.compute_loss(
            pos_score=pos_score_indices,
            neg_score=neg_score_indices,
            #evaluation_edge_types=[self.eval_edge_type, self.eval_edge_type_inv],
            average=self.average,
        )

        preds, target, preds_retrieval, target_retrieval, indexes = create_all_labels_and_scores(
            pos_score=pos_score_indices,
            neg_score=neg_score_indices,
            device=self.tensor_device,
        )

        self.test_metrics.update(preds, target)
        self.test_retrieval_metrics.update(preds_retrieval, target_retrieval, indexes)
        self.log('test_loss', loss['total'], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

        if self.calculate_etype_metrics:
            preds_etypes, target_etypes, preds_etypes_retrieval, target_etypes_retrieval, indexes_etypes = create_all_labels_and_scores_etype_specific(
                pos_score=pos_score_indices,
                neg_score=neg_score_indices,
                device=self.tensor_device,
            )

            for etype in list(preds_etypes.keys()):
                if etype not in self.current_used_etypes:
                    self.current_used_etypes.append(etype)

            for etype in list(preds_etypes.keys()):
                self.test_metrics_etypes[etype[1]].update(preds_etypes[etype], target_etypes[etype])
                self.test_retrieval_metrics_etypes[etype[1]].update(preds_etypes_retrieval[etype], target_etypes_retrieval[etype], indexes_etypes[etype])


            self.log_dict({
                'test_loss_{}'.format(etype[1]): value for etype, value in loss.items()
            },
                on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
            )

        # Convert for JSON
        for indices in [pos_score_indices, neg_score_indices]:
            for k,v in indices.items():
                indices[k] = v.flatten().tolist()

        evaluation_edges_result = {
            "preds": preds.cpu().numpy().tolist(),
            "target": target.cpu().numpy().tolist(),
            "preds_retrieval": preds_retrieval.cpu().numpy().tolist(),
            "target_retrieval": target_retrieval.cpu().numpy().tolist(),
            "eval_edges_pos_label": keys_to_strings(pos_score),
            "eval_edges_neg_label": keys_to_strings(neg_score),
            "eval_edges_pos_indices": keys_to_strings(pos_score_indices),
            "eval_edges_neg_indices": keys_to_strings(neg_score_indices)
        }

        self.test_score_collector['preds'].append(evaluation_edges_result['preds'])
        self.test_score_collector['target'].append(evaluation_edges_result['target'])
        self.test_score_collector['preds_retrieval'].append(evaluation_edges_result['preds_retrieval'])
        self.test_score_collector['target_retrieval'].append(evaluation_edges_result['target_retrieval'])
        self.test_score_collector['eval_edges_pos_indices'].append(evaluation_edges_result['eval_edges_pos_indices'])
        self.test_score_collector['eval_edges_neg_indices'].append(evaluation_edges_result['eval_edges_neg_indices'])

        append_to_list_in_nested_defaultdict(nested_dd = self.test_score_collector, key = 'eval_edges_pos_label', original_d = evaluation_edges_result)
        append_to_list_in_nested_defaultdict(nested_dd = self.test_score_collector, key = 'eval_edges_neg_label', original_d = evaluation_edges_result)

        return {
            'test_loss': loss['total'],
        }

    def test_epoch_end(self, outs):
        # this out is now the full size of the batch

        test_loss = th.stack([x['test_loss'] for x in outs]).mean()
        self.log('test_loss', test_loss)

        result = self.test_metrics.compute()
        result_retrieval = self.test_retrieval_metrics.compute()

        # Create curves for ROC and PR
        self.log_roc_graph(result=result, mode=self.test_metrics.prefix)
        self.log_prc_graph(result=result, mode=self.test_metrics.prefix)

        # Remove metrics so that mlflow works
        result.pop('test_ROC', None)
        result.pop('test_PrecisionRecallCurve', None)
        self.log_dict(result, on_epoch=True)
        self.log_dict(result_retrieval, on_epoch=True)

        # reset metrics for each epoch
        self.test_metrics.reset()
        self.test_retrieval_metrics.reset()

        # Etype specific metrics
        if self.calculate_etype_metrics:
            for etype in self.current_used_etypes:
                result_etype = self.test_metrics_etypes[etype[1]].compute()
                self.test_metrics_etypes[etype[1]].reset()
                result_etype.pop('test_{}_ROC'.format(etype[1]), None)
                result_etype.pop('test_{}_PrecisionRecallCurve'.format(etype[1]), None)
                self.log_dict(result_etype, on_epoch=True)

                result_retrieval_etype = self.test_retrieval_metrics_etypes[etype[1]].compute()
                self.test_retrieval_metrics_etypes[etype[1]].reset()
                self.log_dict(result_retrieval_etype, on_epoch=True)

        to_json(os.path.join(self.run_dir, 'evaluation_results.json'), self.test_score_collector)


    def log_roc_graph(self, result, mode="val") -> None:
        fpr, tpr, thresholds = result[mode + "ROC"]  # self.val_metric_roc.compute()

        fpr = fpr.cpu().numpy() if isinstance(fpr, th.Tensor) else fpr
        tpr = tpr.cpu().numpy() if isinstance(tpr, th.Tensor) else tpr
        thresholds = thresholds.cpu().numpy() if isinstance(thresholds, th.Tensor) else thresholds

        fig = px.area(
            x=fpr, y=tpr,
            #            title=f'ROC Curve (AUC={self.val_metric_auroc.compute():.4f})',
            title=f'ROC Curve (AUROC={result[mode + "AUROC"]:.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            # hover_name=thresholds, # doesn't work as the size is different
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        # log fig
        if self.global_rank == 0 and isinstance(self.logger.experiment, MlflowClient):
            artifact_file = "./roc/epoch_" + str(self.current_epoch) + "/roc_curve_epoch_" + str(
                self.current_epoch) + "_val_" + str(self.current_val_epoch) + ".html"
            if mode == "test":
                artifact_file = "./roc/epoch_" + str(self.current_epoch) + "/roc_curve_test.html"
            self.local_logger.debug("Saving roc curve to: %s", artifact_file)
            self.logger.experiment.log_figure(self.logger.run_id, fig, artifact_file)

    def log_prc_graph(self, result, mode="val") -> None:
        precision, recall, thresholds = result[mode + "PrecisionRecallCurve"]  # self.val_metric_prc.compute()

        precision = precision.cpu().numpy() if isinstance(precision, th.Tensor) else precision
        recall = recall.cpu().numpy() if isinstance(recall, th.Tensor) else recall
        thresholds = thresholds.cpu().numpy() if isinstance(thresholds, th.Tensor) else thresholds

        fig = px.area(
            x=recall, y=precision,
            #            title=f'Precision-Recall Curve (AUC={self.val_metric_auc.compute():.4f}, AvePrec={self.val_metric_ap.compute():.4f})',
            title=f'Precision-Recall Curve (AvePrec={result[mode + "AveragePrecision"]:.4f})',
            labels=dict(x='Recall', y='Precision'),
            # hover_name=thresholds, # doesn't work as the size is different
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )


        # log fig
        if self.global_rank == 0 and isinstance(self.logger.experiment, MlflowClient):
            artifact_file = "./prc/epoch_" + str(self.current_epoch) + "/prc_curve_epoch_" + str(
                self.current_epoch) + "_val_" + str(self.current_val_epoch) + ".html"
            if mode == "test":
                artifact_file = "./prc/epoch_" + str(self.current_epoch) + "/roc_curve_test.html"
            self.local_logger.debug("Saving prc curve to: %s", artifact_file)
            self.logger.experiment.log_figure(self.logger.run_id, fig, artifact_file)

    def translate_test_scores(
        self,
        negative_graph: dgl.DGLHeteroGraph,
        positive_graph: dgl.DGLHeteroGraph,
        neg_score: Dict[Tuple[str, str, str], th.Tensor],
        pos_score: Dict[Tuple[str, str, str], th.Tensor],
        test_graph_file: str = None,
        
    ) -> Tuple[Dict[Tuple[str, str, str], th.Tensor], Dict[Tuple[str, str, str], th.Tensor]]:
        """Map the test edges with their corresponding scores to the original node labels.

        Args:
            negative_graph (dgl.DGLHeteroGraph): The negative graph.
            positive_graph (dgl.DGLHeteroGraph): The positive graph.
            neg_score (dict): The scores for the negative graph.
            pos_score (dict): The scores for the positive graph.

        Returns:
            (tuple): Tuple containing:
                (dict): Scores for the positive graph with node labels.
                (dict): Scores for the negative graph with node labels.

        """
        # Validate only on evaluation edge types
        evaluation_edge_types = [self.eval_edge_type, self.eval_edge_type_inv]
        # NEGATIVE GRAPH (has only node IDs)

        tmp_pair_score_collector_neg = self.decode_negative_edges(
            neg_score=neg_score,
            negative_graph=negative_graph
        )

        # POSITIVE GRAPH (has original edge IDs)

        # When a test graph file is used, the edges in the positive graph are assumed to be negative edges that were constructed
        # Therefore the same code has to be used as for the negative graph
        if test_graph_file is not None:
            tmp_pair_score_collector_pos = self.decode_negative_edges(
                neg_score=pos_score,
                negative_graph=positive_graph,
            )
        else:
            tmp_pair_score_collector_pos = self.decode_positive_edges(
                pos_score=pos_score,
                positive_graph=positive_graph,
            )

        return tmp_pair_score_collector_pos, tmp_pair_score_collector_neg

    def decode_negative_edges(self, neg_score, negative_graph):
        tmp_pair_score_collector_neg = defaultdict(lambda: defaultdict(list))
        for etype in neg_score:
            if negative_graph.edges(etype=etype)[0].nelement() == 0 and \
                negative_graph.edges(etype=etype)[1].nelement() == 0:
                continue
            else:
                # indices of source and destination nodes of negative graph indices
                src_node_indices, dst_node_indices = negative_graph.edges(etype=etype)
                # node types of src and dest
                src_node_type = etype[0]
                dst_node_type = etype[2]

                # indices of original graph nodes
                detached_src_nids = negative_graph.ndata[dgl.NID][src_node_type].detach().cpu().numpy()
                detached_dst_nids = negative_graph.ndata[dgl.NID][dst_node_type].detach().cpu().numpy()
                src_node_indices_orig = [
                    detached_src_nids[i]
                    for i in src_node_indices
                ]
                dst_node_indices_orig = [
                    detached_dst_nids[i]
                    for i in dst_node_indices
                ]

                # get original labels
                src_node_labels = [
                    self.idx_label_mapping[src_node_type][src_id] for src_id in src_node_indices_orig
                ]
                dst_node_labels = [
                    self.idx_label_mapping[dst_node_type][dst_id] for dst_id in dst_node_indices_orig
                ]

                # Create edge pairs
                pairs = list(zip(src_node_labels, dst_node_labels))

                score_list = neg_score[etype].flatten().tolist()
                tmp_score_dict = {pair: score for pair, score in zip(pairs, score_list)}
                for pair, score in tmp_score_dict.items():
                    tmp_pair_score_collector_neg[etype][pair].append(score)

        return tmp_pair_score_collector_neg

    def decode_positive_edges(self, pos_score, positive_graph):
        tmp_pair_score_collector_pos = defaultdict(lambda: defaultdict(list))
        for etype in pos_score:
            # placeholder dictionary for mapping edge type to pair to score

            # for etype, srcdst in edges.items():
            # if no edges of etype, then skip
            if etype not in positive_graph.edata[dgl.EID] or \
                positive_graph.edata[dgl.EID][etype].size()[0] == 0:
                # if srcdst[0].nelement() == 0:
                continue
            else:
                # check if you need the NID of the original graph, since this is the
                # positive subgraph
                # edge id of original graph
                eids = positive_graph.edata[dgl.EID][etype]#.to('cpu')

                # list to collect node pairs
                pairs = []

                for eid in eids:
                    # node ids of original graph
                    src_id, dst_id = self.g.find_edges(eid=eid, etype=etype)
                    src_id, dst_id = int(src_id), int(dst_id)
                    # map ids of source and destination node to the node labels via node type
                    src_node_type = etype[0]
                    dst_node_type = etype[2]
                    # get original labels
                    src_node_label = self.idx_label_mapping[src_node_type][src_id]
                    dst_node_label = self.idx_label_mapping[dst_node_type][dst_id]
                    # list of tuple integers
                    pair = (src_node_label, dst_node_label)
                    pairs.append(pair)

                # pairs = list(zip(srcdst[0].tolist(), srcdst[1].tolist()))
                score_list = pos_score[etype].flatten().tolist()
                tmp_score_dict = {pair: score for pair, score in zip(pairs, score_list)}
                for pair, score in tmp_score_dict.items():
                    tmp_pair_score_collector_pos[etype][pair].append(score)

    def translate_positive_negative_nodes(
        self,
        negative_graph: dgl.DGLHeteroGraph,
        positive_graph: dgl.DGLHeteroGraph,
    ) -> Tuple[Dict[Tuple[str, str, str], List[Tuple[str, str]]], Dict[Tuple[str, str, str], List[Tuple[str, str]]]]:
        """Map the test edges with their corresponding scores to the original node labels.

        Args:
            negative_graph (dgl.DGLHeteroGraph): The negative graph.
            positive_graph (dgl.DGLHeteroGraph): The positive graph.

        Returns:
            (tuple): Tuple containing:
                (dict): Positive graph with node labels.
                (dict): Negative graph with node labels.

        """
        tmp_pair_score_collector_pos = defaultdict(lambda: dict())
        tmp_pair_score_collector_neg = defaultdict(lambda: dict())
        for graph, collector in zip(
            [positive_graph, negative_graph],
            [tmp_pair_score_collector_pos, tmp_pair_score_collector_neg]
        ):
            for etype in graph.canonical_etypes:

                if graph.edges(etype=etype)[0].nelement() == 0 and \
                    graph.edges(etype=etype)[1].nelement() == 0:
                    continue
                else:
                    # indices of source and destination nodes of negative graph indices
                    src_node_indices, dst_node_indices = graph.edges(etype=etype)
                    # node types of src and dest
                    src_node_type = etype[0]
                    dst_node_type = etype[2]

                    # indices of original graph nodes
                    src_node_indices_orig = [
                        int(graph.nodes[src_node_type].data[dgl.NID][i].detach())
                        for i in src_node_indices
                    ]
                    dst_node_indices_orig = [
                        int(graph.nodes[dst_node_type].data[dgl.NID][i].detach())
                        for i in dst_node_indices
                    ]

                    # get original labels
                    src_node_labels = [
                        self.idx_label_mapping[src_node_type][src_id] for src_id in src_node_indices_orig
                    ]
                    dst_node_labels = [
                        self.idx_label_mapping[dst_node_type][dst_id] for dst_id in dst_node_indices_orig
                    ]

                    # Create edge pairs
                    pairs_labels = list(zip(src_node_labels, dst_node_labels))
                    pairs_int = list(zip(src_node_indices_orig, dst_node_indices_orig))

                    collector[etype]['labels'] = pairs_labels
                    collector[etype]['indices'] = pairs_int
                    # update pair collector for every node type with pair with score

        return tmp_pair_score_collector_pos, tmp_pair_score_collector_neg

    def flatten_scores(
        self,
        pos_score,
        neg_score,
    ):
        pos_tensor = self.flatten_dict(pos_score)
        neg_tensor = self.flatten_dict(neg_score)

    def flatten_dict(self, scores: Dict[Tuple[str], th.Tensor]) -> th.Tensor:
        for etype in scores.keys():
            score_input = th.cat([scores[etype]])

    def forward(
        self,
        positive_graph: dgl.DGLHeteroGraph,
        h: Dict[str, th.Tensor],
        blocks: List[dgl.DGLHeteroGraph] = None,
        negative_graph: dgl.DGLHeteroGraph = None,
        verbose: bool = False,
        use_edge_weight: bool = False,
        return_attention: bool = False,
    ) -> Tuple[
        Dict[Tuple[str, str, str], th.Tensor],
        Dict[Tuple[str, str, str],th.Tensor]
    ]:
        """Custom forward method with the BaseRGCN as the encoder and the score predictor as the decoder.

        Args:
            g (dgl.DGLHeteroGraph): The DGL graph.
            positive_graph (dgl.DGLHeteroGraph): The sampled heterograph made out of positive edges.
            negative_graph (dgl.DGLHeteroGraph): The sampled heterograph made out of negative edges
            blocks (list): The list of mini-batched heterographs from the given big graph
            h (dict): Dictionary mapping node type to the feature of the src node type.
            eval_edge_types (list): The canonical edge types to be evaluated on (direct and inverse).
            verbose (bool): True for printing time stamps, False otherwise.
            use_edge_weight (bool): True for using the edge representation in the calculation, False otherwise.

        Returns:
            (tuple): tuple containing:
                pos_score (dict): The scores for positive graph as dictionary mapping edge type to scores of edges.
                neg_score (dict): The scores for the negative graph as dictionary mapping edge type to scores of edges.

        """
        if blocks != None:
            if return_attention:
                node_outputs, attention_weights = self.encoder.forward(
                    blocks=blocks,
                    h=h,
                    verbose=verbose,
                    return_attention=return_attention,
                )
            else:
                node_outputs = self.encoder.forward(
                    blocks=blocks,
                    h=h,
                    verbose=verbose
                )
        else:
            if return_attention:
                node_outputs, attention_weights = self.encoder.forward(
                    h=h,
                    g=positive_graph,
                    return_attention=return_attention,
                ) 
            else:               
                node_outputs = self.encoder.forward(
                    h=h,
                    g=positive_graph,
                    verbose=verbose,
                )
        # Normalize outputs 
        node_outputs_scaled = self.scale_outputs(node_outputs)
        # predictor
        pos_score = self.predictor(
            edge_subgraph=positive_graph,
            node_feature_tensor=node_outputs_scaled,
        )
        if negative_graph:
            neg_score = self.predictor(
                edge_subgraph=negative_graph,
                node_feature_tensor=node_outputs_scaled,
            )

            return pos_score, neg_score
        
        else:
            if return_attention:
                return pos_score, attention_weights
            else:
                return pos_score

    def compute_loss(
        self,
        pos_score: Dict[Tuple[str, str, str], th.Tensor],
        neg_score: Dict[Tuple[str, str, str], th.Tensor],
        evaluation_edge_types: List[Tuple[str, str, str]] = None,
        average: str = 'sum',
    ) -> Dict[str, th.Tensor]:
        """Compute the cross-entropy loss.

        Args:
            pos_score (dict): A dictionary mapping canonical edge type to scores of edges for this edge type.
            neg_score (dict): A dictionary mapping canonical edge type to scores of edges for this edge type.
            heterograph (Heterograph): The heterograph.
            evaluation_edge_types (list): The edge types to evaluate on.

        Returns:
            losses (dict): The predicted cross-entropy loss for all edge types.
        """
        losses = dict()
        # collect loss for every edge type
        loss_collector = dict()
        if evaluation_edge_types:
            etypes = evaluation_edge_types
        else:
            etypes = pos_score.keys()
        # calculate cross-entropy loss for every edge type

        if len(pos_score.values()) == 0:
            return None
        else:

            if average == 'sum' or average == 'mean':
                try:
                    pos_scores = th.cat(list(pos_score.values()))
                except:

                    logger.info('Failed to concatenate pos_score')
                    raise Exception
                neg_scores = th.cat(list(neg_score.values()))
                scores = th.stack((pos_scores, neg_scores))
                scores = scores.flatten()

                target = th.cat([
                    th.ones(size=(pos_scores.size()[0], 1), device=self.tensor_device),
                    th.zeros(size=(neg_scores.size()[0], 1), device=self.tensor_device)
                ]).flatten()

                predicted_loss = self.loss_func(input=scores, target=target)

                losses['total'] = predicted_loss

            # etype loss
            elif average == 'weighted':

                edge_count = {
                    etype: len(pos_score[etype]) + len(neg_score[etype]) for etype in pos_score
                }
                for etype in pos_score:
                    # tensor of evaluation scores of positive and negative graph
                    score_input = th.cat([pos_score[etype], neg_score[etype]])
                    # tensor of target labels (0 for negative and 1 for positive edge)
                    target_input = th.cat([
                        th.ones(size=pos_score[etype].shape, device=self.tensor_device),
                        th.zeros(size=neg_score[etype].shape, device=self.tensor_device)
                    ])
                    # calculate binary cross-entropy loss
                    etype_loss = self.loss_func(input=score_input, target=target_input)
                    # norm loss by number of edges per edge type
                    # etype_loss = (1 / (g.num_edges(etype))) * etype_loss

                    loss_collector[etype] = etype_loss / edge_count[etype]

                    losses[etype] = etype_loss / edge_count[etype]

                # sum loss of all edge types
                # create one tensor of all losses
                stacked_loss = th.stack(list(loss_collector.values()))

                predicted_loss = th.sum(stacked_loss, dim=0)
                losses['total'] = predicted_loss
            else:
                raise Exception("Please provide one of the following methods for calculating the loss: 'sum', 'weighted', 'mean'.")

            return losses #+ regularization_loss

    def scale_outputs(
        self,
        outputs: Dict[str, th.Tensor],
    ) -> Dict[str, th.Tensor]:
        """Scale the outputs to 0 and 1.

        Args:
            outputs (dict): The outputs for each node type to be scaled.

        Returns:
            outputs_scaled (dict): The scaled outputs for each node type.

        """
        outputs_scaled = {}
        for ntype, feature in outputs.items():
            # flatten tensor
            flattened_tensor = th.flatten(feature)
            # get mean and standard deviation
            mean = flattened_tensor.mean(0)
            std = flattened_tensor.std(0)
            if th.isnan(std).any() or std == 0:
                #std = th.empty(1, device=device).fill_(1)
                normalized_tensor = (feature - mean)
            # normalize tensor
            else:
                normalized_tensor = (feature - mean) / std
            outputs_scaled[ntype] = normalized_tensor
        return outputs_scaled

    def create_input_features(
        self,
        blocks: List[dgl.DGLHeteroGraph],
    ) -> Tuple[Dict[str, Dict[str, th.Tensor]], Dict[str, Dict[str, th.Tensor]]]:
        """Create the features of source and destination nodes of the current block.

        Args:
            blocks (list): DGL Heterographs.

        Returns:
            (tuple): Tuple containing:
                (dict): Mapping of node type to modality name to tensor for source nodes of current block.
                (dict): Mapping of node type to modality name to tensor for destination nodes of current block.
        """
        src_features = self.create_src_features(blocks)
        dst_features = self.create_dst_features(blocks)
        
        input_features = (src_features, dst_features)

        return input_features
        
    def create_src_features(self,
        blocks: List[dgl.DGLHeteroGraph],
    ) -> Dict[str, Dict[str, th.Tensor]]:
        """Create the features of source nodes of the current block.

        Args:
            blocks (list): DGL Heterographs.

        Returns:
            (dict): Mapping of node type to modality name to tensor for source nodes of current block.
                
        """
        src_features = defaultdict(dict)

        # the features of input nodes of the first block,
        # which is identical to all the necessary nodes needed for computing the final representations
        for node_type in set(blocks[0].srctypes):

            # first modality
            if node_type in dict(blocks[0].srcdata['feature_0']):
                # extract modality name
                modality_name = self.modality_number_dict[node_type][0]
                src_features[node_type][modality_name] = dict(blocks[0].srcdata['feature_0'])[node_type]
            if node_type in dict(blocks[0].srcdata['feature_1']):
                # extract modality name
                modality_name = self.modality_number_dict[node_type][1]
                src_features[node_type][modality_name] = dict(blocks[0].srcdata['feature_1'])[node_type]
            if node_type in dict(blocks[0].srcdata['feature_2']):
                # extract modality name
                modality_name = self.modality_number_dict[node_type][2]
                src_features[node_type][modality_name] = dict(blocks[0].srcdata['feature_2'])[node_type]

        return src_features

    def create_dst_features(
        self,
        blocks: List[dgl.DGLHeteroGraph],
        ) -> Dict[str, Dict[str, th.Tensor]]:
        """Create the features of source and destination nodes of the current block.

        Args:
            blocks (list): DGL Heterographs.

        Returns:
            (dict): Mapping of node type to modality name to tensor for destination nodes of current block.
        """
        dst_features = defaultdict(dict)
        for node_type in set(blocks[0].dsttypes):

            # first modality
            if node_type in dict(blocks[0].srcdata['feature_0']):
                # extract modality name
                modality_name = self.modality_number_dict[node_type][0]
                dst_features[node_type][modality_name] = dict(blocks[0].dstdata['feature_0'])[node_type]

            if node_type in dict(blocks[0].srcdata['feature_1']):
                # extract modality name
                modality_name = self.modality_number_dict[node_type][1]
                dst_features[node_type][modality_name] = dict(blocks[0].dstdata['feature_1'])[node_type]

            if node_type in dict(blocks[0].srcdata['feature_2']):
                # extract modality name
                modality_name = self.modality_number_dict[node_type][2]
                dst_features[node_type][modality_name] = dict(blocks[0].dstdata['feature_2'])[node_type]

        return dst_features

