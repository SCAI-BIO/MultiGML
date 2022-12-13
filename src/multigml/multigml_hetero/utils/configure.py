# -*- coding: utf-8 -*-

"""Set configurations."""

import logging
from operator import mod
import socket
import platform
import os 
import torch as th
import numpy as np
import random
from typing import Dict, Any, List

from multigml.multigml_hetero.utils.make_dir import make_results_dir


def configure():
    """Set configurations for the script."""
    logging.basicConfig(level=logging.INFO)
    make_results_dir()


def get_hostname():
    if socket.gethostname().endswith("localdomain"):
        home_base_dir = '/home/skrix/Documents/multigml_docs/trained_models'
        BASE_DATA_DIR = home_base_dir + "/data"
        mlflow_hostname = 'localhost'
    elif platform.system() == "Linux" and (
            socket.gethostname().startswith("l") or socket.gethostname().startswith("d")):
        home_base_dir = '/home/skrix/Documents/multigml_docs/trained_models'
        BASE_DATA_DIR = home_base_dir + "/data"
        # IP von leo1: "leo1.leo.scai.fraunhofer.de"
        mlflow_hostname = "10.118.41.21"
    elif platform.system() == 'Darwin':
        home_base_dir = '/home/skrix/Documents/multigml_docs/trained_models'
        mlflow_hostname = 'localhost'
    else:
        raise Exception('No correct hostname found for the mlflow logger. \n Platform: {} \n Hostname: {}'.format(
            platform.system(), socket.gethostname()
        ))
        logging.info("MLFLOW Hostname: {}".format(mlflow_hostname))
    return mlflow_hostname

def get_slurm_job_id():
    
    if "SLURM_ARRAY_JOB_ID" in os.environ:
        slurm_array_job_id = os.environ["SLURM_ARRAY_JOB_ID"]
        slurm_array_task_id = os.environ["SLURM_ARRAY_TASK_ID"]
        return slurm_array_job_id, slurm_array_task_id

    elif "SLURM_JOB_ID" in os.environ:
        slurm_job_id = os.environ["SLURM_JOB_ID"]
        return slurm_job_id
        
    else:
        slurm_job_id = "None"
        return slurm_job_id

def get_model_checkpoint_dir(TRAINED_MODELS_DIR: str, run_dir:str):
    if "SLURM_JOB_ID" in os.environ:
        model_checkpoint_dir = os.path.join(TRAINED_MODELS_DIR, slurm_job_id)
    else:
        model_checkpoint_dir = os.path.join(TRAINED_MODELS_DIR, run_dir.split('/')[-1])
    return model_checkpoint_dir

def get_device_gpus(use_cuda: bool = None, gpu_devices: str or int or List = None):
    if use_cuda and th.cuda.is_available():
        if gpu_devices != None:
            gpus = gpu_devices
        else:
            gpus = [0]
        precision = 32
        device = 'cuda'
    else:
        gpus = None
        precision = 32
        device = 'cpu'

    return gpus, precision, device

def seed_worker(worker_id):
    worker_seed = th.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def update_hparams(
    hparams: Dict[str, Any],
    use_attention: bool,
    num_heads: int,
    attention_dropout: float,
    ):

        hparams.update({
            'num_heads': num_heads,
            'use_attention': use_attention,
            'attention_dropout': attention_dropout,
            })
        return hparams

