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
import random
from typing import List

import torch as th
from multigml.constants import INDICATION, INDICATION_INVERSE
from multigml.multigml_hetero.DataModule import DataModule
from multigml.multigml_hetero.model.LinkPredictHetero import LinkPredictHetero
from multigml.multigml_hetero.utils.configure import (get_device_gpus,
                                                  get_hostname,
                                                  get_slurm_job_id)
from multigml.multigml_hetero.utils.evaluate_hyperopt import get_best_hps
from multigml.multigml_hetero.utils.make_dir import make_run_dir
from multigml.multigml_hetero.utils.prepare_model_graph import (
    feature_dict, get_modality_combinations)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


def run_multigml(
    graph_file: str = None,
    feature_file_list: List[str] = None,
    feature_name_list: List[str] = None,
    feature_node_type_list: List[str] = None,
    features: str = None,
    eval_edge_type: str = INDICATION,
    eval_edge_type_inv: str = INDICATION_INVERSE,
    training_ratio: float = 0.7,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.1,
    input_dim: int = 64,
    h_dim: int = 64,
    h_out: int = 64,
    num_hidden_layers: int = 1,
    n_epochs: int = 1,
    n_crossval_outer: int = 1,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    epsilon: float = 1e-08,
    evaluate_every: int = 1,
    use_cuda: bool = True,
    test_run: bool = False,
    dropout: float = 0.2,
    dropout_multimodal: float = 0.0,
    use_self_loop: bool = True,
    batch_size: int = None,
    which_graph: str = None,
    create_split: bool = False,
    seed: int = 43,
    n_splits: int = 1,
    data_set_split: str = None,
    best_params: str = None,
    pretrained_model_file: str = None,
    use_edge_weight: bool = False,
    average: str = 'sum',
    metrics_average: str = 'micro',
    which_features: str = 'reduced',
    limit_training_batches: int or float = 1.0,
    limit_val_batches: int or float = 1.0,
    limit_test_batches: int or float = 1.0,
    mlflow_experiment_name: str = 'training_multigml',
    calculate_etype_metrics: bool = False,
    reduction: str = 'mean',
    use_attention: bool = False,
    ratio: str = 'equal',
    full_training: bool = False,
    use_uva: bool = False,
    gpu_devices: int = None,
    neg_sample_size: int = 1,
    test_only_eval_etype: bool = False,
    return_attention: bool = False,
    test_graph_file: str = None,
):
    """Run link prediction.

    Args:
        graph_file (str): The path of the graph file.
        feature_file_list (list): List of files of features, has to correspond to feature_node_type_list and feature_name_list.
        feature_name_list (list): List of names of features, has to correspond to features_file_list and feature_node_type_list.
        feature_node_type_list (list): List of node types of features, has to correspond to features_file_list and feature_name_list.
        eval_edge_type (str): The edge type that will be used for the evaluation.
        eval_edge_type_inv (str): The inverse edge type for evaluation.
        training_ratio (float): The ratio of training edges.
        validation_ratio (float): The ratio of validation edges.
        test_ratio (float): The ratio of test edges.
        h_dim (int): The dimension of the hidden layer.
        h_out (int): The dimension of the output layer.
        h_emb_dim (int): The hidden dimension of the embedding layer.
        out_emb_dim (int): The output dimension of the embedding layer.
        edge_hidden_dim (int): The hidden dimension of the edge embedding.
        num_hidden_layers (int): The number of hidden layers.
        n_epochs (int): The number of epochs.
        n_crossval_outer (int): The number of outer cross validations (split into test / train+val set).
        learning_rate (float): The learning rate.
        weight_decay (float): The L2 norm coefficient.
        lambda_l2_reg (float): The parameter for the L2 regularization.
        evaluate_every (int): The number of epochs after which the model is validated again.
        use_cuda (bool): True to use CUDA, False otherwise.
        test_run (bool): True if this is a test run and the test files should be used.
        dropout (float): The dropout ratio.
        use_self_loop (bool): True for using the node's own feature of the previous layer, False otherwise.
        use_stored_data (bool): True for using previously stored data.
        batch_size (int): The batch size for mini-batching.
        which_graph (str): The name of the graph.
        hyper_optimization (bool): True for applying hyperparameter optimization, False otherwise.
        fanout (int): The number of node to sample in the neighborhood sampling.
        n_layers (int): The total number of layers.
        seed (int): The seed for shuffling the train val split.
        n_splits (int): The number of splits for the cross-validation split.
        data_set_split (str): The path to the directory with the data set splits.
        verbose (bool): True for printing time stamps, False otherwise.
        data_loader_dir (str): Path to the file with the data for each epoch from the dataloader.
        best_params (str): Path to the file with the best hyperparameters.
        pretrained_model_file (str): Path to the file with the pre-trained model.
        patience (int): The patience for early stopping.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement for early stopping.
        use_edge_weight (bool): True for using the edge representation in the calculation, False otherwise.
        average (str): Method to calculate the loss.
        use_random_features (bool): True for using random features, false for using default features.
        limit_training_batches (int or float): How many batches to use for training.
        full_training (bool): True for using training and validation set for training, False for only using training dataset.


    Returns:
        None
    """
    pl.seed_everything(42)
    # create directories
    run_dir = make_run_dir(which_graph=which_graph)

    # Use best hyperparameters found

    hyperparams = get_best_hps(best_params_file=best_params)

    input_dim = hyperparams['input_dim']
    h_dim = hyperparams['h_dim']
    h_out = hyperparams['h_out']

    num_hidden_layers = hyperparams['num_hidden_layers']

    # set best parameters found
    if batch_size == None:
        batch_size = hyperparams["batch_size"]

    # Dropouts
    dropout = hyperparams["dropout"]
    dropout_multimodal = hyperparams["dropout_multimodal"]

    # Optimizer
    learning_rate = hyperparams["learning_rate"]
    weight_decay_exponent =  hyperparams["weight_decay"]
    weight_decay = float("1e-" + str(weight_decay_exponent))
    epsilon = hyperparams["epsilon"]

    factor = hyperparams['factor']

    # modalities
    modality_combinations = get_modality_combinations(feature_dict)
    if which_features in ['random', 'unimodal', 'full_unimodal', 'identity']:
        drug_modality = modality_combinations['drug'][0]#'{}_drug'.format(which_features)
        protein_modality = modality_combinations['protein'][0]#'{}_protein'.format(which_features)
        condition_modality = modality_combinations['condition'][0]#'{}_condition'.format(which_features)
    else:
        drug_modality = modality_combinations['drug'][hyperparams['drug_modality']]
        protein_modality = modality_combinations['protein'][hyperparams['protein_modality']]
        condition_modality = modality_combinations['condition'][0]

    used_modalities = {
        'drug': drug_modality,
        'protein': protein_modality,
        'condition': condition_modality,
    }
        
    # attention
    if use_attention:
        num_heads = hyperparams['num_heads'] 
        attention_dropout = hyperparams['attention_dropout']
    else:
        num_heads = None
        attention_dropout = None

    gpus, precision, device = get_device_gpus(use_cuda=use_cuda, gpu_devices=gpu_devices)

    # ===========================================
    # Datamodule definition
    # ==========================================
    datamodule = DataModule(
        batch_size=batch_size,
        n_layers=(num_hidden_layers + 2),
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
        full_training=full_training,
        use_uva=use_uva,
        neg_sample_size=neg_sample_size,
        test_only_eval_etype=test_only_eval_etype,
        test_graph_file=test_graph_file,
    )

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
    
    if th.cuda.is_available():
        datamodule.heterograph.g = datamodule.heterograph.g.to(device)

    mlflow_hostname = get_hostname()

    slurm_job_id = get_slurm_job_id()

    mlflow_logger = pl.loggers.MLFlowLogger(
        experiment_name=mlflow_experiment_name,
        tracking_uri="file:./ml-runs",
        tags={
            'slurm_job_id': str(slurm_job_id),
            'results_directory': str(run_dir),
            'loss_method': str(average),
            'metrics_method': str(metrics_average),
            'which_features': str(which_features),
            'best_params': str(best_params),
            'pretrained_model': str(pretrained_model_file),
            'data_set_split':str(data_set_split),
            'neg_sample_size':str(neg_sample_size),
        }
    )
    logger.info("========================= \nMLflow run id: {}\n=========================".format(mlflow_logger.run_id))

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=run_dir,
            filename='epoch={epoch}-val-loss={val_loss:.2f}-val-averageprecision={val_AveragePrecision:.2f}',
            monitor="val_side-effect_AveragePrecision",
            mode="max",
            verbose=True,
            save_top_k=1,
        ),
        EarlyStopping(
            monitor='val_AveragePrecision',
            min_delta=0.005,
            patience=5,
            )
    ]
    if use_cuda:
        accelerator = 'gpu'
        gpus = 1
    else:
        accelerator = None
        gpus = None
    trainer = pl.Trainer(
        default_root_dir=datamodule.run_dir,
        logger=mlflow_logger,
        check_val_every_n_epoch=evaluate_every,
        max_epochs=n_epochs,
        accelerator=accelerator,
        gpus=gpus,
        deterministic=True,
        callbacks=callbacks,
        limit_train_batches=limit_training_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )

    # ==================
    # Testing 
    # =================

    # Case: Testing pre-trained model
    if test_run and pretrained_model_file:
        logger.info("Start testing ...")

        model = LinkPredictHetero.load_from_checkpoint(pretrained_model_file, hparams=hparams)
        if th.cuda.is_available():
            model.to(device)
        logger.info(model)  

        # Test with different negative samples each time
        random.seed(None)
        new_seed = random.randint(0, 1000)
        pl.utilities.seed.seed_everything(new_seed)
        
        trainer.test(
            model=model,
            datamodule=datamodule,
        )
    # Case: Testing immediately after training model
    elif test_run and pretrained_model_file == None:

        model: LinkPredictHetero = LinkPredictHetero(
            hparams=hparams
        )
        if th.cuda.is_available():
            model.to(device)
        logger.info(model)
 
        logger.info("Start training ...")
        trainer.fit(
            model=model,
            datamodule=datamodule,
        )
        logger.info("Start testing ...")
        trainer.test(
            datamodule=datamodule,
        )

    # Case: Training model without testing 
    else:
        model: LinkPredictHetero = LinkPredictHetero(
            hparams=hparams
        )
        if th.cuda.is_available():
            model.to(device)
        logger.info(model)
        logger.info("Start training ...")
        trainer.fit(
            model=model,
            datamodule=datamodule,
     )


