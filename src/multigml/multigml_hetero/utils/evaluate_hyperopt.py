# -*- coding: utf-8 -*-

"""Evaluate the parameters from the bayesian hyperparameter optimization."""

import operator
from collections import defaultdict
from typing import Dict

import numpy as np

from multigml.multigml_hetero.utils.storing import read_json


def get_best_hps(
    best_params_file: str,  
) -> Dict[str, int]:
    """Get the best hyperparameters for a run with one cv.

    Args:
        best_params_file (str): best hyperparameters file

    Returns:
        dict: dictionary mapping hyperparameter name to best value

    """
    # dict mapping cv fold to dictionary of best parameters
    print(best_params_file)
    best_hp_dict = read_json(best_params_file)

    return list(best_hp_dict.values())[0]



