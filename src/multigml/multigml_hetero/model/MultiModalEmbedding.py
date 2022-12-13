# -*- coding: utf-8 -*-


"""This script is to create the multi-modal embedding layer."""

import logging
from typing import Dict, List

import numpy as np
import torch as th
import torch.nn as nn

logger = logging.getLogger(__name__)

class MultiModalEmbeddingLayer(nn.Module):
    """Embedding Layer for multi-modal encoder consisting of two parts for one node type with one or more modalities.

    First, a dense layer transforms the feature tensor for each modality of the node type seperately into a
    lower-dimensional latent space representation. Currently, a node type with three modalities is supported.
    Secondly, a consequent dense layer merges all low-dimensional representations into a common space."""


    def __init__(
        self,
        node_type: str,
        input_sizes_dict: Dict[str, int],
        output_size: int,
        dropout_multimodal: float,
    ):
        super(MultiModalEmbeddingLayer, self).__init__()

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

        # Calculate input and output size for the fusion layer
        self.get_output_size()

        # Build unimodal layers
        self.build_unimodal_layer()
        # Build multimodal layer
        self.build_multimodal_layer()

        if isinstance(input_sizes_dict, dict):

            # names of modalities for every node type
            self.modality_names = list(input_sizes_dict.keys())

            # number of modalities for current node type
            self.num_modalities = len(input_sizes_dict)

        else:
            self.modality_names = None
            raise Exception("Please provide a dictionary of modality names and corresponding features.")

    def get_output_size(self):
        """Calculate the input and output size for the fusion layer based on the modality sizes."""
        modality_names = []
        input_sizes = []
        for i in range(self.num_modalities):
            modality_name = self.modality_names[i]
            modality_names.append(modality_name)
            # feature input sizes
            input_size = self.input_sizes_dict[modality_name]
            input_sizes.append(input_size)

        # Input size is the sum of all input modality sizes by half
        fusion_input_size = 0
        for i in range(self.num_modalities):
            fusion_input_size += int(np.floor(input_sizes[i]/2))

        output_size = int(np.floor(fusion_input_size/2))

        self.fusion_input_size = fusion_input_size
        self.fusion_output_size = output_size

    def build_unimodal_layer(self):
        """Build the first unimodal layer(s) for each modality and encode the features.

        Returns:
            None
        """
        # for any number of modalities
        for i in range(self.num_modalities):
            # modality name
            tmp_modality_name = self.modality_names[i]
            # feature input size
            tmp_input_size = self.input_sizes_dict[tmp_modality_name]
            # create layer
            layer_1 = self.build_layer(
                input_size=tmp_input_size,
                output_size=int(np.floor(tmp_input_size/2)),
            )
            batch_normalization_layer = th.nn.BatchNorm1d(num_features=int(np.floor(tmp_input_size/2)))
            instance_normalization_layer = th.nn.InstanceNorm1d(num_features=int(np.floor(tmp_input_size/2)))

            dropout_layer = th.nn.Dropout(p=self.dropout_multimodal)

            # append layers to embedding layer modulelist
            self.embedding_layers.update(
                {
                    tmp_modality_name: th.nn.Sequential(
                        layer_1,
                        dropout_layer,
                        batch_normalization_layer,
                        self.activation
                    )
                }
            )
            # append layers to embedding layer modulelist
            self.embedding_layers_without_batchnorm.update(
                {
                    tmp_modality_name: th.nn.Sequential(
                        layer_1,
                        dropout_layer,
                        instance_normalization_layer,
                        self.activation
                    )
                }
            )

    def build_multimodal_layer(
        self,
        include_batchnorm: bool = True,
    ):
        """Fuse the unimodal embeddings into a multimodal embedding.

        Returns:
            None
        """
        # build fusion layer
        fusion_layer = self.build_layer(
            # multiply size with number of modalities since the features get concatenated along dimension 1
            input_size=self.fusion_input_size,#self.hidden_size * self.num_modalities,
            output_size=self.fusion_output_size,
        )
        batch_normalization_layer = th.nn.BatchNorm1d(num_features=self.fusion_output_size)
        dropout_layer = th.nn.Dropout(p=self.dropout_multimodal)

        # layer for converting fused modality representation into common space representation for all modalities
        output_layer = self.build_layer(
            # multiply size with number of modalities since the features get concatenated along dimension 1
            input_size=self.fusion_output_size,#self.hidden_size * self.num_modalities,
            output_size=self.output_size,
        )
        output_batch_normalization_layer = th.nn.BatchNorm1d(num_features=self.output_size)
        output_dropout_layer = th.nn.Dropout(p=self.dropout_multimodal)

        fusion_layer_list = []
        fusion_layer_list.append(fusion_layer)
        fusion_layer_list.append(dropout_layer)
        fusion_layer_list.append(batch_normalization_layer)
        fusion_layer_list.append(self.activation)
        fusion_layer_list.append(output_layer)
        fusion_layer_list.append(output_dropout_layer)
        fusion_layer_list.append(output_batch_normalization_layer)
        fusion_layer_list.append(self.activation)

        self.fusion_layers = th.nn.Sequential(*fusion_layer_list)

        fusion_layer_list_without_batchnorm = []
        fusion_layer_list_without_batchnorm.append(fusion_layer)
        fusion_layer_list_without_batchnorm.append(dropout_layer)
        fusion_layer_list_without_batchnorm.append(self.activation)
        fusion_layer_list_without_batchnorm.append(output_layer)
        fusion_layer_list_without_batchnorm.append(output_dropout_layer)
        fusion_layer_list_without_batchnorm.append(self.activation)

        self.fusion_layers_without_batchnorm = th.nn.Sequential(*fusion_layer_list_without_batchnorm)

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
        h: Dict[str, th.Tensor],
    ):
        """Embedd the unimodal features into a multimodal representation.

        Args:
            h (dict): Mapping from modality name to tensor.

        Returns:

        """
        # ================================================================================
        # Create embedding of unimodal features
        # ================================================================================G
        # in case of 1 modality
        if len(h) == 1:
            for modality_name, value in h.items():
                if value.size()[0] == 1:
                    hidden_feature = self.embedding_layers_without_batchnorm[modality_name](h[modality_name])
                else:
                    hidden_feature = self.embedding_layers[modality_name](h[modality_name])

            if hidden_feature.size()[0] == 1:
                output_feature = self.fusion_layers_without_batchnorm(hidden_feature)
            else:
                output_feature = self.fusion_layers(hidden_feature)

        # in case of 2 modalities
        elif len(h) == 2:
            hidden_features = []
            for modality_name, value in h.items():
                if value.size()[0] == 1:
                    hidden_feature = self.embedding_layers_without_batchnorm[modality_name](h[modality_name])
                else:
                    hidden_feature = self.embedding_layers[modality_name](h[modality_name])

                hidden_features.append(hidden_feature)

            # concatenate output representations along dimension 1
            hidden_feature = th.cat((hidden_features[0], hidden_features[1]), dim=1)

            # pass concatenated output of unimodal layer(s) to fusion layer
            if hidden_feature.size()[0] == 1:
                output_feature = self.fusion_layers_without_batchnorm(hidden_feature)
            else:
                output_feature = self.fusion_layers(hidden_feature)

        # in case of 3 modalities
        elif len(h) == 3:
            hidden_features = []
            for modality_name, value in h.items():
                if value.size()[0] == 1:
                    # handle 2D input for instance norm, it would otherwise throw an error without a 3d tensor
                    h_ = h[modality_name].reshape(1,1,-1)
                    hidden_feature = self.embedding_layers_without_batchnorm[modality_name](h_)
                    hidden_feature = hidden_feature.reshape(1,-1)
                else:
                    hidden_feature = self.embedding_layers[modality_name](h[modality_name])

                hidden_features.append(hidden_feature)

            # concatenate output representations along dimension 1

            hidden_feature = th.cat((hidden_features[0], hidden_features[1], hidden_features[2]), dim=1)

            # pass concatenated output of unimodal layer(s) to fusion layer
            if hidden_feature.size()[0] == 1:
                output_feature = self.fusion_layers_without_batchnorm(hidden_feature)
            else:
                output_feature = self.fusion_layers(hidden_feature)

        else:
            logger.info("Modalities shape: {}".format(h.shape))

            raise Exception("Currently only 1 to 3 modalities are supported, please select accordingly.")

        return output_feature


