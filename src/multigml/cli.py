# -*- coding: utf-8 -*-

"""Console script for multigml."""

import logging
from typing import List

import click
import pandas as pd
import torch

from multigml.constants import (CONDITION_GRAPH_FILE, INDICATION,
                                INDICATION_INVERSE)
from multigml.multigml_hetero.deployer_hyperopt import run_bayesian_hyperopt
from multigml.multigml_hetero.deployer_train import run_multigml

logger = logging.getLogger(__name__)

# ======================================
# Shared arguments for the CLI
# ======================================
graph_file_argument = click.option(
    "-g",
    "--graph_file",
    default=CONDITION_GRAPH_FILE,
    type=str,
    help="the path to the graph tsv file",
)
feature_file_list_argument = click.option(
    "-ffile",
    "--feature_file_list",
    default=None,
    type=str,
    help='File paths for feature files.'
)
feature_name_list_argument = click.option(
    "-fname",
    "--feature_name_list",
    default=None,
    type=str,
    help='Names for features.'
)
feature_node_type_list_argument = click.option(
    "-fntype",
    "--feature_node_type_list",
    default=None,
    type=str,
    help='Node types of features.'
)
features_argument = click.option(
    "-fs",
    "--features",
    default=None,
    type=str,
    help="The names of the features used."
)
eval_edge_type_argument = click.option(
    "-e",
    "--eval_edge_type",
    default=INDICATION,
    type=str,
    help="edge type to be evaluated"
)
eval_edge_type_inv_argument = click.option(
    "-e_inv",
    "--eval_edge_type_inv",
    default=INDICATION_INVERSE,
    type=str,
    help="edge type inverse to be evaluated"
)
training_ratio_argument = click.option(
    "-tr",
    "--training_ratio",
    default=0.7,
    type=float,
    help="training ratio"
)
validation_ratio_argument = click.option(
    "-vr",
    "--validation_ratio",
    default=0.1,
    type=float,
    help="validation ratio"
)
test_ratio_argument = click.option(
    "-tsr",
    "--test_ratio",
    default=0.2,
    type=float,
    help="test ratio"
)
test_run_argument = click.option(
    "-r",
    "--test_run",
    default=False,
    type=bool,
    help="do a test run with test data"
)
h_dim_argument = click.option(
    "-h",
    "--h_dim",
    default=64,
    type=int,
    help="hidden dimension"
)
h_out_argument = click.option(
    "-ho",
    "--h_out",
    default=32,
    type=int,
    help="output dimension"
)
hidden_embedding_dimension_argument = click.option(
    "-hemb",
    "--h_emb_dim",
    default=256,
    type=int,
    help="hidden embedding dimension"
)
output_embedding_dimension_argument = click.option(
    "-oemb",
    "--out_emb_dim",
    default=108,
    type=int,
    help="output embedding dimension"
)
num_hidden_layers_argument = click.option(
    "-nh",
    "--num_hidden_layers",
    default=1,
    type=int,
    help="number of hidden layers"
)
edge_hidden_dim_argument = click.option(
    "-ehd",
    "--edge_hidden_dim",
    default=54,
    type=int,
    help="hidden edge dimension"
)
n_epochs_argument = click.option(
    "-ne",
    "--n_epochs",
    default=1,
    type=int,
    help="number of epochs"
)
n_cv_outer_argument = click.option(
    "-ncv_out",
    "--n_crossval_outer",
    default=1,
    type=int,
    help="number of outer cross validations"
)

device_argument = click.option(
    "-d",
    "--device",
    default=torch.device('cuda'),
    type=torch.device,
    help="device"
)
learning_rate_argument = click.option(
    "-lr",
    "--learning_rate",
    default=0.01,
    type=float,
    help="size of learning rate"
)
weight_decay_argument = click.option(
    "-wd",
    "--weight_decay",
    default=0.0,
    type=float,
    help="weight decay"
)
evaluate_every_argument = click.option(
    "-ev",
    "--evaluate_every",
    default=1,
    type=int,
    help="evaluate every n epochs"
)
use_cuda = click.option(
    "-cuda",
    "--use_cuda",
    default=True,
    type=bool,
    help="use CUDA"
)
dropout_argument = click.option(
    "-do",
    "--dropout",
    default=0.0,
    type=float,
    help="dropout ratio"
)
use_self_loop_argument = click.option(
    "-sl",
    "--use_self_loop",
    default=True,
    type=bool,
    help="use self loop"
)
reg_param_argument = click.option(
    "-rp",
    "--reg_param",
    default=0.0,
    type=float,
    help="regularization parameter strength"
)
output_option = click.option(
    '--output',
    help="Path to output file",
    type=click.Path(file_okay=True, dir_okay=False, exists=False)
)
use_stored_data = click.option(
    '-usd',
    '--use_stored_data',
    default=False,
    type=bool,
    help='use previously stored data'
)
which_graph_argument = click.option(
    '-wg',
    '--which_graph',
    type=str,
    help="which graph to use ('50' or 'withoutstringdb' or 'withstringdb')",
)
batch_size_argument = click.option(
    "-b",
    "--batch_size",
    default=None,
    type=int,
    help='Size of batch'
)

bayesian_hyperopt_argument = click.option(
    "-hopt",
    "--hyper_optimization",
    default=True,
    type=bool,
    help='Doing Bayesian Hyperparameter Optimization',
)

n_trials_argument = click.option(
    "-nt",
    "--n_trials",
    default=10,
    type=int,
    help='Number of trials for optuna study',
)

timeout_argument = click.option(
    "-to",
    "--timeout",
    default=None,
    type=int,
    help="Stop study after the given number of second(s)"
)

best_params_file_argument = click.option(
    "-bp",
    "--best_params",
    default=None,
    type=str,
    help="File path for best hyperparameters"
)
n2v_embedding_file_name = click.option(
    "-n2ve",
    "--n2v_embedding",
    default=None,
    type=str,
    help="File path for node2vec embedding"
)
n2v_model_file_name = click.option(
    "-n2vm",
    "--n2v_model",
    default=None,
    type=str,
    help="File path for node2vec model"
)
n2v_results_file_name = click.option(
    "-n2vr",
    "--n2v_results",
    default=None,
    type=str,
    help="File path for node2vec results",
)
fanout_argument = click.option(
    "-fa",
    "--fanout",
    default=4,
    type=int,
    help=" The number of neighbors to sample for each node for the neighbor-sampling.",
)
n_layers_argument = click.option(
    "-nl",
    "--n_layers",
    default=3,
    type=int,
    help="The total number of layers.",
)
create_split_argument = click.option(
    "-cs",
    "--create_split",
    default=False,
    type=bool,
    help="True for creating a test, train, val split, False otherwise.",
)
seed_argument = click.option(
    "-s",
    "--seed",
    default=43,
    type=int,
    help="The seed for shuffling in cross-validation split.",
)
n_splits_argument = click.option(
    "-ns",
    "--n_splits",
    default=1,
    type=int,
    help="The number of cross-validation splits.",
)
data_set_split_argument = click.option(
    "-d",
    "--data_set_split",
    default=None,
    type=str,
    help="The path to the directory containing the data set splits"
)
verbose_argument = click.option(
    "-v",
    "--verbose",
    default=False,
    type=bool,
    help="True for printing time stamps, False otherwise."
)
data_loader_dir = click.option(
    "-dld",
    "--data_loader_dir",
    default=None,
    type=str,
    help="Path to the directory with the data for each epoch from the dataloader."
)
nvme_data_loader_dir = click.option(
    "-nvme",
    "--nvme_data_loader_dir",
    default=None,
    type=str,
    help="Path to the NVME directory with the data for each epoch from the dataloader."
)
pretrained_model_argument = click.option(
    "-ptm",
    "--pretrained_model",
    default=None,
    type=str,
    help="File path for the pretrained model (pickle)."
)
patience_argument = click.option(
    "-pc",
    "--patience",
    default=10,
    type=int,
    help="Patience for early stopping."
)
delta_argument = click.option(
    "-dt",
    "--delta",
    default=0.05,
    type=float,
    help="Minimum change for early stopping."
)
selected_cv_fold_argument = click.option(
    "-scv",
    "--selected_cv_fold",
    default=None,
    type=int,
    help="The selected CV fold for the data set split."
)
use_edge_weight_argument = click.option(
    "-ew",
    "--use_edge_weight",
    default=True,
    type=bool,
    help="True for using edge weight in message passing, False otherwise."
)
min_resource_argument = click.option(
    "-minr",
    "--min_resource",
    default=1,
    type=int,
    help="The minimum resource allocated to a trial."
)
max_resource_argument = click.option(
    "-maxr",
    "--max_resource",
    default=100,
    type=int or str,
    help="The maximum resource allocated to a trial."
)
reduction_factor_argument = click.option(
    "-rf",
    "--reduction_factor",
    default=4,
    type=int,
    help="The reduction factor of promotable trials."
)
pruner_argument = click.option(
    "-p",
    "--pruner",
    default='hyperband',
    type=str,
    help="The pruner to use for the hyperparameter optimization."
)
log_every_argument = click.option(
    "-log",
    "--log_every",
    default=100,
    type=int,
    help="Number of steps after which to report values by logger."
)
num_nodes_argument = click.option(
    "-nn",
    "--num_nodes",
    default=1,
    type=int,
    help="Number of nodes to use for training."
)
accelerator_argument = click.option(
    "-acc",
    "--accelerator",
    default=None,
    type=str,
    help="The accelerator backend to use."
)
use_db_server_argument = click.option(
    "-dbs",
    "--use_db_server",
    default=False,
    type=bool,
    help="Use db server."
)
epsilon_argument = click.option(
    "-eps",
    "--epsilon",
    default=1e-08,
    type=float,
    help="Epsilon parameter for Adamw"
)
average_argument = click.option(
    "-av",
    "--average",
    default='sum',
    type=str,
    help='Method to calculate the loss.',
)
metrics_average_argument = click.option(
    "-mav",
    "--metrics_average",
    default='micro',
    type=str,
    help='Method to calculate the metrics.',
)
optuna_study_name_argument = click.option(
    "-osn",
    "--optuna_study_name",
    default='optuna_test',
    type=str,
    help='The study name for optuna.',
)
mlflow_experiment_name_argument = click.option(
    "-mlf",
    "--mlflow_experiment_name",
    default='hyperopt_test',
    type=str,
    help='Mlflow experiment name.',
)
which_features_argument = click.option(
    "-wf",
    "--which_features",
    default='reduced',
    type=str,
    help="Which features to use, one of: 'reduced' (features after PCA transformation), "
         "'full' (full features without PCA), 'random' (random features), 'unimodal' (one modality for each node type)."
)
limit_training_batches_argument = click.option(
    "-ltb",
    "--limit_training_batches",
    default=1.0,
    type=float or int,
    help="How many training batches to use."
)
workers_argument = click.option(
    "-w",
    "--workers",
    default=8,
    type=int,
    help="How many workers to use."
)
current_run_dir_argument = click.option(
    "-crd",
    "--current_run_dir",
    default=None,
    type=str,
    help="Directory to use especially for submitting multiple slurm scripts for one optuna study."
)
calculate_etype_metrics_argument = click.option(
    "-cem",
    "--calculate_etype_metrics",
    default=False,
    type=bool,
    help="True for calculating edge type specific metrics, False otherwise."
)
use_attention_argument = click.option(
    "-att",
    "--use_attention",
    default=False,
    type=bool,
    help="True for using the attention mechanism via RGAT, False for using RGCN."
)
gpu_devices_argument = click.option(
    "-dev",
    "--gpu_devices",
    default=None,
    type=int,
    help="Number of GPUs to use."
)
explainer_argument = click.option(
    "-ex",
    "--explainer",
    default=None,
    type=str,
    help="The explainability method from captum to use."
)
neg_sample_size_argument = click.option(
    "-ns",
    "--neg_sample_size",
    default=1,
    type=int,
    help="Number of negative samples per positive sample in testing."
)
test_only_eval_etype_argument = click.option(
    "-te",
    "--test_only_eval_etype",
    default=False,
    type=bool,
    help="True for testing only eval etype, false for testing all etypes."
)
test_graph_file_argument = click.option(
    "-tgf",
    "--test_graph_file",
    default=None,
    type=str,
    help="Test graph file for test graph to use other than default test graph"
)

@click.group()
def main():
    """MultiGML CLI."""


@main.group()
def linkpredict():
    """Do link prediction."""


# =======================================
# Link Prediction with RGCN
# =======================================

@linkpredict.command()
@graph_file_argument
@feature_file_list_argument
@feature_name_list_argument
@feature_node_type_list_argument
@features_argument
@eval_edge_type_argument
@eval_edge_type_inv_argument
@training_ratio_argument
@validation_ratio_argument
@test_ratio_argument
@h_dim_argument
@h_out_argument
@hidden_embedding_dimension_argument
@output_embedding_dimension_argument
@num_hidden_layers_argument
@n_epochs_argument
@n_cv_outer_argument
@learning_rate_argument
@weight_decay_argument
@epsilon_argument
@evaluate_every_argument
@use_cuda
@test_run_argument
@dropout_argument
@use_self_loop_argument
@batch_size_argument
@which_graph_argument
@create_split_argument
@seed_argument
@n_splits_argument
@data_set_split_argument
@best_params_file_argument
@pretrained_model_argument
@use_edge_weight_argument
@average_argument
@metrics_average_argument
@which_features_argument
@limit_training_batches_argument
@mlflow_experiment_name_argument
@calculate_etype_metrics_argument
@use_attention_argument
@gpu_devices_argument
@neg_sample_size_argument
@test_only_eval_etype_argument
@test_graph_file_argument
def run(
    graph_file: str = None,  # GRAPH_FILE,
    feature_file_list: List[str] = None,
    feature_name_list: List[str] = None,
    feature_node_type_list: List[str] = None,
    features: str = None,
    eval_edge_type: str = INDICATION,
    eval_edge_type_inv = INDICATION_INVERSE,
    training_ratio: float = 0.7,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.1,
    h_dim: int = 64,
    h_out: int = 32,
    h_emb_dim: int = 256,
    out_emb_dim: int = 108,
    num_hidden_layers: int = 1,
    n_epochs: int = 1,
    n_crossval_outer: int = 1,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0,
    epsilon:float = 1e-08,
    evaluate_every: int = 1,
    use_cuda: bool = True,
    test_run: bool = False,
    dropout: float = 0.0,
    use_self_loop: bool = False,
    batch_size: int = None,
    which_graph: str = None,
    create_split: bool = False,
    seed: int = 43,
    n_splits: int = 5,
    data_set_split: str = None,
    best_params: str = None,
    pretrained_model: str = None,
    use_edge_weight: bool = True,
    average: str = 'sum',
    metrics_average: str ='micro',
    which_features: str = 'reduced',
    limit_training_batches: int or float = 1.0,
    mlflow_experiment_name: str = 'training_multigml',
    calculate_etype_metrics: bool = False,
    use_attention: bool = False,
    gpu_devices: int = None,
    neg_sample_size: int = 1,
    test_only_eval_etype: bool = False,
    test_graph_file: str = None,
):
    """Run link prediction.

    Args:
        graph_file (str): The path of the graph file.
        feature_file_list (list): List of files of features, has to correspond to feature_node_type_list and feature_name_list.
        feature_name_list (list): List of names of features, has to correspond to features_file_list and feature_node_type_list.
        feature_node_type_list (list): List of node types of features, has to correspond to features_file_list and feature_name_list.
        features (str): The feature names as a string in list format.
        eval_edge_type (str): The edge type that will be used for the evaluation.
        training_ratio (float): The ratio of training edges.
        validation_ratio (float): The ratio of validation edges.
        test_ratio (float): The ratio of test edges.
        h_dim (int): The dimension of the hidden layer.
        h_out (int): The dimension of the output layer.
        num_hidden_layers (int): The number of hidden layers.
        n_epochs (int): The number of epochs.
        n_crossval (int): The number of cross validations.
        learning_rate (float): The learning rate.
        weight_decay (float): The L2 norm coefficient.
        evaluate_every (int): The number of epochs after which the model is validated again.
        use_cuda (bool): True to use CUDA, False otherwise.
        test_run (bool): True if this is a test run and the test files should be used.
        dropout (float): The dropout ratio.
        use_self_loop (bool): True for using the node's own feature of the previous layer, False otherwise.
        use_stored_data (bool): True for using previously stored data.
        batch_size (int): The batch size for mini-batching.
        which_graph (str): The name of the graph.
        hyper_optimization (bool): True for applying hyperparameter optimization, False otherwise.
        data_set_split (str): The path to the directory storing the data set splits.
        data_loader_dir (str): Path to the directory with the data for each epoch from the dataloader.
        pretrained_model (str): File path to the pretrained model.
        patience (int): The patience for early stopping.
        delta (float): Minimum change to be considered for early stopping.
        use_edge_weight (bool): True for using the edge representation in the calculation, False otherwise.
        limit_training_batches (int or float): How many training batches to use.

    Returns:
        None
    """
    run_multigml(
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
        h_dim=h_dim,
        h_out=h_out,
        num_hidden_layers=num_hidden_layers,
        n_epochs=n_epochs,
        n_crossval_outer=n_crossval_outer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epsilon=epsilon,
        evaluate_every=evaluate_every,
        use_cuda=use_cuda,
        test_run=test_run,
        use_self_loop=use_self_loop,
        dropout=dropout,
        batch_size=batch_size,
        which_graph=which_graph,
        create_split=create_split,
        seed=seed,
        n_splits=n_splits,
        data_set_split=data_set_split,
        best_params=best_params,
        pretrained_model_file=pretrained_model,
        use_edge_weight=use_edge_weight,
        average=average,
        metrics_average=metrics_average,
        which_features=which_features,
        limit_training_batches=limit_training_batches,
        mlflow_experiment_name=mlflow_experiment_name,
        calculate_etype_metrics=calculate_etype_metrics,
        use_attention=use_attention,
        gpu_devices=gpu_devices,
        neg_sample_size=neg_sample_size,
        test_only_eval_etype=test_only_eval_etype,
        test_graph_file=test_graph_file,
    )


@linkpredict.command()
@graph_file_argument
@feature_file_list_argument
@feature_name_list_argument
@feature_node_type_list_argument
@features_argument
@eval_edge_type_argument
@eval_edge_type_inv_argument
@training_ratio_argument
@validation_ratio_argument
@test_ratio_argument
@n_epochs_argument
@n_cv_outer_argument
@evaluate_every_argument
@use_cuda
@use_self_loop_argument
@which_graph_argument
@n_trials_argument
@timeout_argument
@create_split_argument
@seed_argument
@n_splits_argument
@data_set_split_argument
@use_edge_weight_argument
@min_resource_argument
@max_resource_argument
@reduction_factor_argument
@pruner_argument
@num_nodes_argument
@accelerator_argument
@use_db_server_argument
@average_argument
@metrics_average_argument
@optuna_study_name_argument
@mlflow_experiment_name_argument
@which_features_argument
@limit_training_batches_argument
@current_run_dir_argument
@calculate_etype_metrics_argument
@use_attention_argument
@gpu_devices_argument
def run_opt(
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
    n_epochs: int = 1,
    n_crossval_outer: int = 1,
    evaluate_every: int = 5,
    use_cuda: bool = True,
    use_self_loop: bool = False,
    which_graph: str = None,
    n_trials: int = 10,
    timeout: int = 600,
    create_split: bool = False,
    seed: int = 43,
    n_splits: int = 1,
    data_set_split: str = None,
    use_edge_weight: bool = True,
    min_resource: int = 1,
    max_resource: int or str = 100,
    reduction_factor: int = 4,
    pruner: str = 'hyperband',
    num_nodes: int = 1,
    accelerator: str = None,
    use_db_server: bool = False,
    average: str = 'sum',
    metrics_average: str = 'micro',
    optuna_study_name: str = 'optuna_test',
    mlflow_experiment_name: str = 'hyperopt_test',
    which_features: str = 'reduced',
    limit_training_batches: int or float = 1.0,
    current_run_dir: str = None,
    calculate_etype_metrics: bool = False,
    use_attention: bool = False,
    gpu_devices: int = None,
):
    """Run bayesian hyperparameter optimization."""
    run_bayesian_hyperopt(
        n_trials=n_trials,
        which_graph=which_graph,
        timeout=timeout,
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
        n_epochs=n_epochs,
        n_crossval_outer=n_crossval_outer,
        evaluate_every=evaluate_every,
        use_cuda=use_cuda,
        use_self_loop=use_self_loop,
        create_split=create_split,
        seed=seed,
        n_splits=n_splits,
        data_set_split=data_set_split,
        use_edge_weight=use_edge_weight,
        min_resource=min_resource,
        max_resource=max_resource,
        reduction_factor=reduction_factor,
        pruner=pruner,
        num_nodes=num_nodes,
        accelerator=accelerator,
        use_db_server=use_db_server,
        average=average,
        metrics_average=metrics_average,
        optuna_study_name=optuna_study_name,
        mlflow_experiment_name=mlflow_experiment_name,
        which_features=which_features,
        limit_training_batches=limit_training_batches,
        current_run_dir=current_run_dir,
        calculate_etype_metrics=calculate_etype_metrics,
        use_attention=use_attention,
        gpu_devices=gpu_devices,
    )


if __name__ == '__main__':
    main()
