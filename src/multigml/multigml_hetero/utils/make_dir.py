# -*- coding: utf-8 -*-

"""Generate directories to store results."""

import os

from multigml.constants import RUN_RESULTS_DIR, PROJECT_DIR


def make_results_dir():
    """Make results directory."""
    HOME = os.path.expanduser('~')
    DOCUMENTS_DIR = os.path.join(HOME, 'Documents')
    MUlTIGML_DOCS_DIR = os.path.join(DOCUMENTS_DIR, 'multigml_docs')
    RESULTS_DIR = os.path.join(MUlTIGML_DOCS_DIR, 'results_multigml')
    RUN_RESULTS_DIR = os.path.join(RESULTS_DIR, 'run_results')
    DATA_SET_SPLIT_DIR = os.path.join(MUlTIGML_DOCS_DIR, 'data_set_split')
    COMPARISON_DIR = os.path.join(RESULTS_DIR, 'comparison')
    if not os.path.exists(MUlTIGML_DOCS_DIR):
        os.mkdir(MUlTIGML_DOCS_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    if not os.path.exists(RUN_RESULTS_DIR):
        os.mkdir(RUN_RESULTS_DIR)
    if not os.path.exists(DATA_SET_SPLIT_DIR):
        os.mkdir(DATA_SET_SPLIT_DIR)
    if not os.path.exists(COMPARISON_DIR):
        os.mkdir(COMPARISON_DIR)

def make_run_dir(which_graph: str):
    """Make directory for run."""
    if which_graph is None:
        which_graph = 'complete'
    # find all folder names in results directory
    repository_dir = "/".join(PROJECT_DIR.split("/")[:-2])
    run_results_dir = os.path.join(repository_dir, 'run_results')
    all_folders = os.listdir(run_results_dir)

    # if it is the first run
    if len(all_folders) == 0:
        RUN_DIR = os.path.join(run_results_dir, 'graph{}_{}'.format(which_graph, '0'))

        if not os.path.exists(RUN_DIR):
            os.mkdir(RUN_DIR)
        else:
            make_run_dir(which_graph)
    # if there are already folders existent
    else:
        # find maximum number of run
        all_num = [int(x.split('_')[1]) for x in all_folders]
        max_num = max(all_num)
        # determine new directory number
        new_dir_num = int(max_num) + 1
        RUN_DIR = os.path.join(run_results_dir, 'graph{}_{}'.format(which_graph, str(new_dir_num)))

        if not os.path.exists(RUN_DIR):
            os.mkdir(RUN_DIR)
        else:
            make_run_dir(which_graph)

    return RUN_DIR
