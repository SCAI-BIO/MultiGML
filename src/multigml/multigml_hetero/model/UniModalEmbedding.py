# -*- coding: utf-8 -*-


"""This script is to create the multi-modal embedding layer."""

from typing import Dict, List

import numpy as np
import torch as th
import torch.nn as nn


class UniModalEmbeddingLayer(nn.Module):
    """Embedding Layer for multi-modal encoder consisting of two parts for one node type with one or more modalities.

    First, a dense layer transforms the feature tensor for each modality of the node type seperately into a
    lower-dimensional latent space representation. Currently, a node type with three modalities is supported.
    Secondly, a consequent dense layer merges all low-dimensional representations into a common space."""

    def __init__(
        self,
        node_type: str,
        input_sizes_dict: Dict[str, int],
        output_size: int,
        #droupout_modalities: float,
        dropout_multimodal: float,
    ):
        super(UniModalEmbeddingLayer, self).__init__()

        self.input_sizes_dict = input_sizes_dict
        self.output_size = output_size
        self.node_type = node_type
        self.num_modalities = len(input_sizes_dict)
        self.modality_names = list(input_sizes_dict.keys())
        # Dropout
        self.dropout_multimodal = dropout_multimodal
        # activation layer
        self.activation = th.nn.Tanh()

        # embedding layers
        self.embedding_layers = th.nn.ModuleDict()
        self.embedding_layers_without_batchnorm = th.nn.ModuleDict()

        # Build unimodal layers
        self.build_unimodal_layer()

        if isinstance(input_sizes_dict, dict):

            # names of modalities for every node type
            self.modality_names = list(input_sizes_dict.keys())

            # number of modalities for current node type
            self.num_modalities = len(input_sizes_dict)

        else:
            self.modality_names = None
            raise Exception("Please provide a dictionary of modality names and corresponding features.")

    def build_unimodal_layer(self):
        """Build the first unimodal layer(s) for each modality and encode the features.

        Returns:
            None
        """
        # modality name
        modality_name_1 = self.modality_names[0]
        # feature input size
        input_size_1 = self.input_sizes_dict[modality_name_1]

        # create layer
        layer_1 = self.build_layer(
            input_size=input_size_1,
            output_size=self.output_size,
        )
        batch_normalization_layer = th.nn.BatchNorm1d(num_features=int(np.floor(input_size_1/2)))

        # use a dropout ratio specific for each modality
        #tmp_dropout_ratio = self.dropout_modalities[modality_name_1]
        #dropout_layer = th.nn.Dropout(p=tmp_dropout_ratio)
        #dropout_layer = th.nn.Dropout(p=self.dropout_modalities[self.node_type][modality_name_1])
        dropout_layer = th.nn.Dropout(p=self.dropout_multimodal)
        # append layers to embedding layer modulelist
        self.embedding_layers = th.nn.Sequential(
                            layer_1,
                            dropout_layer,
                        )

        self.embedding_layers_activation = th.nn.Sequential(
            layer_1,
            dropout_layer,
            self.activation
        )

        # append layers to embedding layer modulelist
        self.embedding_layers_without_batchnorm = th.nn.Sequential(
                    layer_1,
                )

    def build_layer(
        self,
        input_size: int,
        output_size: int,
    ) -> th.nn.Linear:
        """Build the input layer for one modality.

        Args:
            input_size (int): Input size of the modality feature tensor.
            output_size (int): Output size of the modality feature tensor.

        Returns:
            (th.nn.Linear): Linear layer for one modality.
        """
        input_layer = th.nn.Linear(in_features=input_size, out_features=output_size)

        return input_layer

    def forward(
        self,
        h: th.Tensor,
    ):
        """Embedd the unimodal features into a multimodal representation.

        Args:
            h (dict): Feature tensor.

        Returns:

        """
        # ================================================================================
        # Create embedding of unimodal features
        # ================================================================================G
        if h.size()[0] == 1:
            hidden_feature = self.embedding_layers_without_batchnorm(h)
        else:
            hidden_feature = self.embedding_layers(h)

        return hidden_feature

