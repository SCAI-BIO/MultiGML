# -*- coding: utf-8 -*-

"""Functions to store and load objects used in the deployer."""

import json
import pickle
from collections import defaultdict

import dgl


def save_graph(G: dgl.DGLHeteroGraph, path: str):
    """Save the heterograph to file."""
    with open(path, 'wb') as file:
        pickle.dump(G, file)


def open_graph(path: str):
    """Read the heterograph from file."""
    with open(path, 'rb') as file:
        graph = pickle.load(file)
    return graph

def append_to_list_in_nested_defaultdict(nested_dd, key: str, original_d):
    if nested_dd[key] == list():
        nested_dd[key] = defaultdict(lambda : defaultdict(list))
    for etype, values in original_d[key].items():
        for pair, score in values.items():
                nested_dd[key][etype][pair].append(score)

def to_json(file: str, obj: object, verbose: bool = True) -> None:
    """Write file to json.

    :param file: file path
    :param obj: the object, usually dictionary, to be written to file
    """
    with open(file, 'w') as fp:
        if verbose:
            print(f"Writing dictionary object to file: {file}.")
        json.dump(obj, fp)


def read_json(file: str):
    """Read in a json file.

    :param file: path to file to be read
    :return: object of json file
    """
    # read file
    with open(file, 'r') as f:
        data = f.read()

    # parse file
    obj = json.loads(data)

    return obj
