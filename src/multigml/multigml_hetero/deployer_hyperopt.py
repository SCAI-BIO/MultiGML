# -*- coding: utf-8 -*-

"""This module generates the heterograph, passes it to the relational graph convolutional neural network and does
the link prediction on the edges of the heterograph.

=============================
Best Params
============================
N splits                          5           1
best_trial_number          6.000000   12.000000
batch_size               256.000000  256.000000
dropout                    0.400000    0.100000
dropout_disease_cui2vec    0.300000    0.500000
dropout_drug_cyt           0.500000    0.000000
dropout_drug_fc            0.300000    0.100000
dropout_drug_fp            0.500000    0.400000
dropout_multimodal         0.500000    0.500000
dropout_protein_esm        0.000000    0.300000
dropout_protein_fc         0.400000    0.400000
dropout_protein_go         0.300000    0.000000
edge_hidden_dim          100.000000  404.000000
h_dim                    168.000000  160.000000
h_emb_dim                368.000000  448.000000
h_out                     40.000000   48.000000
learning_rate              0.204155    0.036751
out_emb_dim              112.000000   68.000000
weight_decay               0.000041    0.000004

"""

import logging
import os
from datetime import timedelta
from timeit import default_timer as timer
from typing import List

import joblib
import optuna
import pytorch_lightning as pl
import torch as th
from multigml.constants import INDICATION, INDICATION_INVERSE
from multigml.multigml_hetero.utils.storing import to_json
from multigml.multigml_hetero.DataModule import DataModule
from multigml.multigml_hetero.model.LinkPredictHetero import LinkPredictHetero
from multigml.multigml_hetero.utils.configure import (get_device_gpus,
                                                  get_hostname,
                                                  get_slurm_job_id)
from multigml.multigml_hetero.utils.make_dir import make_run_dir
from multigml.multigml_hetero.utils.prepare_model_graph import (
    choose_features, feature_dict, get_modalities_dict_from_which_features,
    get_modality_combinations)
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.profiler import AdvancedProfiler

logger = logging.getLogger(__name__)

def run_bayesian_hyperopt(
    n_trials: int,
    which_graph: str or None,
    timeout: int or None = 200,
    graph_file: str or None = None,
    feature_file_list: List[str] = None,
    feature_name_list: List[str] = None,
    feature_node_type_list: List[str] = None,
    features: str = None,
    eval_edge_type: str = INDICATION,
    eval_edge_type_inv: str = INDICATION_INVERSE,
    training_ratio: float = 0.7,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.2,
    n_epochs: int = 1,
    n_crossval_outer: int = 1,
    evaluate_every: int = 1,
    use_cuda: bool = True,
    use_self_loop: bool = False,
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
    optuna_study_name: str = 'testing',
    mlflow_experiment_name: str = 'testing',
    which_features: str = 'custom',
    ratio: str = 'equal',
    limit_training_batches: int or float = 1.0,
    limit_val_batches: int or float = 1.0,
    limit_test_batches: int or float = 1.0,
    profiler: str = None,
    use_uva: bool = None,
    current_run_dir: str = None,
    calculate_etype_metrics: bool = False,
    reduction: str = 'mean',
    use_attention: bool = False,
    gpu_devices: int = None,
    return_attention: bool = False,
    test_graph_file: str = None,
) -> str:
    """Run link prediction with bayesian hyperparameter optimization.

    Args:
        n_trials (int): The number of optuna study trials to perform.
        timeout (int): The number of seconds after which to stop the process.
        which_graph (str): The name of the graph.
        graph_file (str): The path to the graph file.
        feature_file_list (list): List of files of features, has to correspond to feature_node_type_list and feature_name_list.
        feature_name_list (list): List of names of features, has to correspond to features_file_list and feature_node_type_list.
        feature_node_type_list (list): List of node types of features, has to correspond to features_file_list and feature_name_list.
        features (str): The names of the features used as a string in list format, will be evaluated.
        eval_edge_type (str): The edge type that will be used for the evaluation.
        eval_edge_type_inv (str): The inverse edge type that will be used for the evaluation.
        training_ratio (float): The ratio of training edges.
        validation_A parameter for specifying reduction factor of promotable trials ratio (float): The ratio of validation edges.
        test_ratio (float): The ratio of test edges.
        num_hidden_layers (int): The number of hidden layers.
        n_epochs (int): The number of epochs.
        n_crossval_outer (int): The number of outer cross validations (split into test / train+val set).
        evaluate_every (int): The number of epochs after which the model is validated again
        use_cuda (bool): True to use CUDA, False otherwise.
        use_self_loop (bool): True for using the node's own feature of the previous layer, False otherwise.
        use_stored_data (bool): True for using previously stored data.
        seed (int): Seed for shuffling in stratified k-fold split of train val data.
        n_splits (int): Number of k-fold splits for cross-validation.
        n_layers (int): The total number of layers.
        data_set_split (str): The path to the directory storing the data set splits.
        verbose (bool): True for printing time stamps, False otherwise.
        data_loader_dir (str): Path to the file with the data for each epoch from the dataloader.
        patience (int): The number of validation counts to wait for early stopping.
        delta (float): Minimum change to be considered in the early stopping.
        selected_cv_fold (int): The selected cv fold of the data split to use.
        use_edge_weight (bool): True for using the edge representation in the calculation, False otherwise.
        min_resource (int): A parameter for specifying the minimum resource allocated to a trial.
        max_resource (int): A parameter for specifying the maximum resource allocated to a trial.
        reduction_factor (int): A parameter for specifying reduction factor of promotable trials .
        pruner (str): The pruner to use for the optuna hyperparameter optimization, either 'hyperband' or 'median'.
        num_nodes (int): The number of nodes to use for training.
        accelerator (str): The accelerator backend to use.
        average (str): Method to calculate the loss.
        which_features (str): Which features to use, one of: 'reduced' (features after PCA transformation),
         'full' (full features without PCA), 'random' (random features), 'unimodal' (one modality for each node type).
        use_uva (bool): True for doing UVA sampling, False otherwise.
        current_run_dir (str): Directory to use especially for submitting multiple slurm scripts for one optuna study.
        calculate_etype_metrics (str): True for calculating edge type specific metrics, False otherwise.
        reduction (str): The reduction parameter for the loss function.

    Returns:
        (str): File path for best hyperparameter sets for each cv fold.

    """
    # create directories
    if current_run_dir:
        run_dir = current_run_dir
    else:
        run_dir = make_run_dir(which_graph=which_graph)
    study_path = os.path.join(run_dir, 'optuna_study.pkl')
    best_params_file = os.path.join(run_dir, 'best_params.json')

    if profiler == 'advanced':
        profiler = AdvancedProfiler(output_filename=os.path.join(run_dir, 'profiler.out'))
        
    # dictionary mapping each cv fold number to the set of best hyperparameters for this fold
    best_hyperparameter_cv = {}

    t_start = timer()

    # Use different combinations of modalities, leaving individual modalities out
    # to test their feature importance
    modality_combinations = get_modality_combinations(feature_dict)

    _, feature_name_list, _ = choose_features(which_features=which_features, which_graph=which_graph)

    for cv_out in range(n_crossval_outer):
        # find best hyperparameter set for each cross validation split (into train and test set)
        for cv_inner in range(n_splits):

            def objective(
                trial: optuna.trial.Trial,
            ):
                """Run link prediction with bayesian optimization."""
                # Attention
                if use_attention:
                    num_heads = trial.suggest_int("num_heads", low=1, high=4, step=1)
                    attention_dropout = trial.suggest_discrete_uniform("attention_dropout", low=0.0, high=0.5, q=0.1)
                else:
                    num_heads = None
                    attention_dropout = None

                # input dimension for RGCN, output dimension for multimodal embed
                # correct for dimension given the number of heads in RGAT
                if use_attention:
                    if num_heads == 2 or num_heads == 4:
                        low = 8
                        step = 8
                    elif num_heads == 3: 
                        low = 9
                        step = 15
                    else:
                        low = 10
                        step = 10
                else:
                    low = 10
                    step = 10

                input_dim = trial.suggest_int("input_dim", low=low, high=500, step=step)
                h_dim = trial.suggest_int("h_dim", low=low, high=300, step=step)
                h_out = trial.suggest_int("h_out", low=low, high=100, step=step)

                num_hidden_layers = trial.suggest_int("num_hidden_layers", low=1, high=7)

                dropout = trial.suggest_discrete_uniform("dropout", low=0.0, high=0.5, q=0.1)
                dropout_multimodal = trial.suggest_discrete_uniform("dropout_multimodal", low=0.0, high=0.4, q=0.1)

                learning_rate = trial.suggest_loguniform("learning_rate", low=10e-5, high=10e-4)
                weight_decay_exponent = trial.suggest_int(name='weight_decay', low=4, high=6)
                weight_decay = float("1e-"+str(weight_decay_exponent))

                # size for mini batches
                
                if which_graph == '50':
                    batch_size = trial.suggest_int("batch_size", low=2, high=100, step=10)
                else:
                    batch_size = trial.suggest_int("batch_size", low=100, high=800, step=100)
        
                if ratio == 'proportional':
                    factor = trial.suggest_int("factor", low=5000, high=30000, step=5000)
                else:
                    factor = trial.suggest_int("factor", low=6, high=32, step=4)

                epsilon = trial.suggest_float("epsilon", low=1e-8, high=1e-6)

                # multimodal case
                if which_features in ['full', 'reduced', 'variance_selection']:

                    drug_modality_key = trial.suggest_categorical('drug_modality', choices=list(modality_combinations['drug'].keys()))
                    protein_modality_key = trial.suggest_categorical('protein_modality', choices=list(modality_combinations['protein'].keys()))
                    drug_modality = modality_combinations['drug'][drug_modality_key]
                    protein_modality = modality_combinations['protein'][protein_modality_key]
                    condition_modality = modality_combinations['condition'][0]

                    used_modalities = {
                        'drug': drug_modality,
                        'protein': protein_modality,
                        'condition': condition_modality,
                    }
                # unimodal case
                else:
                    used_modalities = get_modalities_dict_from_which_features(which_graph=which_graph, which_features=which_features)

                gpus, precision, device = get_device_gpus(use_cuda=use_cuda, gpu_devices=gpu_devices)

                # define datamodule
                datamodule = DataModule(
                    batch_size=batch_size,
                    n_layers=(num_hidden_layers+2),
                    device=device,
                    data_set_split=data_set_split,
                    create_split=create_split,
                    seed=seed,
                    run_dir=run_dir,
                    which_graph=which_graph,
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
                    n_crossval_outer=n_crossval_outer,
                    n_splits=n_splits,
                    which_features=which_features,
                    modality_combinations=used_modalities,
                    factor=factor,
                    ratio=ratio,
                    use_uva=use_uva,
                )
                # Move g explicitely to cuda, otherwise it stays on cpu 
                if th.cuda.is_available:
                    datamodule.heterograph.g = datamodule.heterograph.g.to(device)

                hparams = {
                    'device': device,
                    'g': datamodule.heterograph.g,
                    'input_dim': input_dim,
                    'h_dim': h_dim,
                    'out_dim': h_out,
                    'num_hidden_layers': num_hidden_layers,
                    'dropout': dropout,
                    'dropout_multimodal': dropout_multimodal,
                    'use_self_loop': use_self_loop,
                    'batch_size': batch_size,
                    'n_layers': (num_hidden_layers + 2),
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'use_edge_weight': use_edge_weight,
                    'epsilon': epsilon,
                    'run_dir': run_dir,
                    'average': average,
                    'metrics_average': metrics_average,
                    'which_features': which_features,
                    'tensor_device': device,
                    'modality_sizes_dict': datamodule.heterograph.modality_sizes_dict,
                    'modality_number_dict': datamodule.heterograph.modality_number_dict,
                    'canonical_edges': datamodule.heterograph.canonical_edges,
                    'eval_edge_type': datamodule.heterograph.eval_edge_type,
                    'eval_edge_type_inv': datamodule.heterograph.eval_edge_type_inv,
                    'idx_label_mapping': datamodule.heterograph.idx_label_mapping,
                    'used_modalities': used_modalities,
                    'calculate_etype_metrics': calculate_etype_metrics,
                    'reduction': reduction,
                    'use_attention': use_attention,
                    'num_heads': num_heads,
                    'attention_dropout': attention_dropout,
                    'return_attention': return_attention,
                    'test_graph_file': test_graph_file,
                }

                # define model
                model: LinkPredictHetero = LinkPredictHetero(
                    hparams=hparams,
                )
                if th.cuda.is_available():
                    model.to(device)
                    datamodule.heterograph.g = datamodule.heterograph.g.to(device)

                pl.seed_everything(seed=42)

                mlflow_hostname = get_hostname()

                slurm_job_id = get_slurm_job_id()
                
                tags={
                    'optuna_trial_id': str(trial.number),
                    'results_directory': str(run_dir),
                    'loss_method': str(average),
                    'metrics_method': str(metrics_average),
                    'which_features': str(which_features),
                }
                if type(slurm_job_id) == str:
                    tags['slurm_job_id'] =  str(slurm_job_id)
                elif type(slurm_job_id) == tuple:              
                    tags['slurm_job_id'] = '{}_{}'.format(slurm_job_id[0], slurm_job_id[1])
                else:
                    tags['slurm_job_id'] = None

                mlflow_logger = pl.loggers.MLFlowLogger(
                    experiment_name=mlflow_experiment_name,
                    tracking_uri="file:./ml-runs",
                    tags=tags,
                )

                trainer = pl.Trainer(
                    default_root_dir=datamodule.run_dir,
                    logger=mlflow_logger,
                    check_val_every_n_epoch=evaluate_every,
                    max_epochs=n_epochs,
                    num_nodes=num_nodes,
                    accelerator=accelerator,
                    gpus=gpus,
                    deterministic=True,
                    precision=precision,
                    callbacks=[
                        #Pruning
                        PyTorchLightningPruningCallback(trial, monitor="val_AveragePrecision"),
                    ],
                    limit_train_batches=limit_training_batches,
                    limit_val_batches=limit_val_batches,
                    limit_test_batches=limit_test_batches,
                    profiler=profiler,
                )

                trainer.fit(
                    model=model,
                    datamodule=datamodule,
                )

                logger.info('Parameters for trial {}: {}'.format(trial.params, trial.number))
                logger.info('Trial report: {}'.format(trial.report))

                return trainer.callback_metrics['val_AveragePrecision']

        if pruner == 'hyperband':
            # BAYESIAN HYPERPARAMETER OPTIMIZATION
            study = optuna.create_study(
                direction="maximize",
                study_name=optuna_study_name,
                storage='sqlite:///{}/gpu_final.db'.format(run_dir),
                load_if_exists=True,
                pruner=optuna.pruners.HyperbandPruner(
                    min_resource=min_resource,
                    max_resource=max_resource,
                    reduction_factor=reduction_factor,
                )
            )

        elif pruner == 'median':
            # BAYESIAN HYPERPARAMETER OPTIMIZATION
            study = optuna.create_study(
                direction="maximize",
                study_name=optuna_study_name,
                storage='sqlite:///{}/gpu_final.db'.format(run_dir),
                load_if_exists=True,
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=3,
                    n_warmup_steps=5,
                    interval_steps=1,
                )
            )
        else:
            raise Exception(
                "Please select a pruner from the following: 'hyperband' (default) or 'median'."
            )
            
        study.optimize(func=objective, n_trials=n_trials, timeout=timeout)

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        logger.info("Study statistics: ")
        logger.info("  Number of finished trials: {}".format(len(study.trials)))
        logger.info("  Number of pruned trials: {}".format(len(pruned_trials)))
        logger.info("  Number of complete trials: {}".format(len(complete_trials)))

        logger.info("Best trial: {}".format(study.best_trial.number))
        best_trial = study.best_trial

        logger.info("  Value: {}".format(best_trial.value))

        logger.info("  Params: ")
        for key, value in best_trial.params.items():
            logger.info("    {}: {}".format(key, value))

        best_params = {'best_trial_number': study.best_trial.number}
        best_params = {**best_params, **best_trial.params}

        # store best hps for this fold in dict
        best_hyperparameter_cv[cv_out] = best_params

        # store study for each cv
        joblib.dump(study, study_path)

    to_json(file=best_params_file, obj=best_hyperparameter_cv)

    t_end = timer()

    logger.info("Total time for training: {}".format(str(timedelta(seconds=(t_end-t_start)))))
    return best_params_file
