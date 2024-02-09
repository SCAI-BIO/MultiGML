
# MultiGML: Multimodal Graph Machine Learning for Prediction of Adverse Drug Events


## Table of contents
* [General info](#general-info)
* [Installation](#installation)
* [Documentation](#documentation)
* [Input data](#input-data-formats)
* [Usage](#usage)
* [Issues](#issues)
* [Acknowledgements](#acknowledgements)
* [Disclaimer](#disclaimer)

## General info

Adverse drug events constitute a major challenge for the success of clinical trials. Experimental procedures for measuring liver-toxicity, cardio-toxicity and others are well established in pre-clinical development, but are costly and cannot fully guarantee success in later clinical studies, specifically in situations without a reliable animal model. Hence, several computational strategies have been suggested to estimate the risk associated with therapeutically targeting a specific protein, most prominently statistical methods using human genetics. While these approaches have demonstrated high utility in practice, they are at the same time limited to specific information sources and thus neglects a wealth of information that is uncovered by fusion of different data sources, including biological protein function, protein-protein interactions, gene expression, chemical compound structure, cell based imaging and others. 
In this work we propose an integrative and explainable Graph Machine Learning neural network approach (MultiGML), which fuses knowledge graphs with multiple further data modalities to predict drug related adverse events. MultiGML demonstrates excellent prediction performance compared to alternative algorithms, including various knowledge graph embedding techniques. Furthermore, MultiGML distinguishes itself from alternative techniques by providing in-depth explanations of model predictions, which point towards biological mechanisms associated with predictions of an adverse drug event.

Here is an overview of the workflow of our approach:

![Workflow](figures/workflow.png)
Overview of workflow. A) Knowledge Graph compilation. In the first step of data processing, interaction information from 14 biomedical databases was parsed with data on drug-drug interactions, drug-target interactions, protein-protein interactions, indication, drug-ADE and gene-phenotype associations. The data was harmonized across all databases and a comprehensive, heterogeneous, multi-relational knowledge graph was generated. B) Feature definition. Descriptive data modalities were selected to annotate entities in the knowledge graph. Drugs were annotated with their molecular fingerprint, the gene expression profile they cause, and the morphological profile of cells they induce. Proteins were annotated with their protein sequence embedding and a gene ontology fingerprint. Phenotypes, comprising indications and ADEs, were annotated by their clinical concept embedding. C) Proposed MultiGML approach. The heterogeneous Knowledge Graph with its feature annotations is used as the input for our graph neural network approach, the MultiGML. For each node entity, a multi-modal embedding layer learns a low dimensional representation of entity features. These embeddings are then used as input for either the RGCN or RGAT of the encoder (see section 2.2.1), which learns an embedding for each  entity in the KG. A bilinear decoder takes a source and a destination node, drug X and phenotype Y in the example here, and produces a score for the probability of their connection, considering their relation type with each other.

## Installation

We recommend starting from a clean virtual environment since many dependencies will be installed. After creating a new environment, install dependencies from requirements.txt.

The most recent code can be installed from the source on [https://github.com/SCAI-BIO/MultiGML](https://github.com/SCAI-BIO/MultiGML) with:

```
$ python3 -m pip install https://github.com/SCAI-BIO/MultiGML
```

For developers, the repository can be cloned from [https://github.com/SCAI-BIO/MultiGML](https://github.com/SCAI-BIO/MultiGML) and installed in editable mode with:

```
$ git clone https://github.com/SCAI-BIO/MultiGML
$ cd MultiGML
$ python3 -m pip install -e .
```

The file data/features/full_features/ protein_embeddings_esm.tsv.zip was too large to push to GitHub, and therefore was split into segments (protein_embeddings_esm_*). In order to run the code, you need to merge these files together:
```
$ cat protein_embeddings_esm_* > protein_embeddings_esm.tsv.zip
```



## Input data formats

### Data


### Knowledge graph

The graph format MultiGML uses is a modified version of the Edge List Format which has several columns for identifying
the source and destination node, their relation and the source database:

| Column name       | Explanation                                   | Example          |
|-------------------|-----------------------------------------------|------------------|
| source_identifier | Identifier of the source node.                | DRUGBANK:DB00001 |
| source_node_type  | Source node type.                             | drug             |
| target_identifier | Identifier of the target node.                | HGNC:3535        |
| target_node_type  | Target node type.                             | protein          |
| relation_type     | Type of the edge.                             | drug-protein     |
| source_database   | Database the edge information was taken from. | DRUGBANK         |


### Features

There are three types of nodes in the knowledge graph: drugs, phenotypes and proteins. Each of the node types have specific
node features.


### Data Repository Structure


The repository contains a main data folder ([data](data/)) containing the following folders: [graph](data/graph), [features](data/features) and [data_set_split](data/data_set_split).

The following files are listed in the [features](data/features/full_features/) directory:

* [cui2vec_disgenet.tsv](data/features/full_features/cui2vec_disgenet.tsv): phenotype embeddings for UMLS CUIs

* [drugbank_count_fp.tsv](data/features/full_features/drugbank_count_fp.tsv): compound count fingerprints for Drugbank IDs

* [lincs_drug_fc.tsv](data/features/full_features/lincs_drug_fc.tsv): fold change of protein expression due to perturbation of drugs

* [lincs_cytological_profiling_drug_features.tsv](data/features/full_features/lincs_cytological_profiling_drug_features.tsv): cytological image profiling features of drugs

* [protein_embeddings_esm.tsv](data/features/full_features/protein_embeddings_esm.tsv): protein sequence embeddings 

* [protein_go_fingerprint.tsv](data/features/full_features/protein_go_fingerprint.tsv): gene ontology fingerprint for proteins


The data files are zipped in the data folder because of large file sizes, so please unzip them before using MultiGML.

## Usage
**Note:** These are very basic commands for MultiGML, and the detailed options for each command can be found in the documentation.


1. **Hyperparameter Optimization**

    The following command applies bayesian hyperparameter optimization and selects the best hyperparameters. Set the argument --use_attention=False for using the MultiGML-RGCN variant, set it to True for using the MultiGML-RGAT variant. Set the --which_features argument to 'full' for using the multimodal features, set it to 'random' for using the basic features.

    ```
    $ python -m multigml linkpredict run-opt --which_graph='complete' --n_trials=1 --use_cuda=False --n_epochs=30 --pruner='hyperband' --max_resource=30 --min_resource=1 --reduction_factor=4 --evaluate_every=1 --n_splits=1 --data_set_split='<repository_path>/multigml/data/data_set_split/extended_trimmed_kg' --eval_edge_type='side-effect' --eval_edge_type_inv='side-effect_inverse' --which_features='full' --average='weighted' --calculate_etype_metrics=False --mlflow_experiment_name='<mlflow_experiment_name>' --optuna_study_name='<optuna_study_name>' --use_attention=False 
    ```

    Note: For using a custom graph, please set `--which_graph='custom'`. You will need to specify the graph file path via `--graph_path='your/custom/path` and the node features to use via providing the folder with node features path in the argument `--which_features='your/node/feature/folder/path`. We here assume the same type of node features as listed above, so please make sure the file names correspond. You can use the argument `--create_split=True` to do a stratified train val test split and if you want to repeat your experiment, you can use the folder path that the splits were saved to next time in the argument `--data_set_split=/your/data/set/split`

    Example:
    
    ```
    $ python -m multigml linkpredict run-opt --which_graph='custom' --graph_path='your/custom/graph/path' --n_trials=1 --use_cuda=False --n_epochs=30 --pruner='hyperband' --max_resource=30 --min_resource=1 --reduction_factor=4 --evaluate_every=1 --n_splits=1  --eval_edge_type='side-effect' --eval_edge_type_inv='side-effect_inverse' --which_features='your/custom/feature/folder/' --average='weighted' --calculate_etype_metrics=False --mlflow_experiment_name='<mlflow_experiment_name>' --optuna_study_name='<optuna_study_name>' --use_attention=False  --create_split=True
    ```

2. **Training**

    The following command trains the model with the best hyperparameters and does link prediction. Set the argument --use_attention=False for using the MultiGML-RGCN variant, set it to True for using the MultiGML-RGAT variant. Set the --which_features argument to 'full' for using the multimodal features, set it to 'random' for using the basic features.

    ```
    $ python -m multigml linkpredict run --which_graph='complete' --use_cuda=False --n_epochs=100 --evaluate_every=1 --n_splits=1 --data_set_split='<repository_path>/MultiGML/data/data_set_split/extended_trimmed_kg' --eval_edge_type='side-effect' --eval_edge_type_inv='side-effect_inverse' --which_features='full' --average='weighted' --calculate_etype_metrics=True --best_params='<best_params_path>' --mlflow_experiment_name='<mlflow_name>' --test_run=True --use_attention=False

    ```

3. **Testing**

    Use the following command to test your pretrained model. Set the argument --use_attention=False for using the MultiGML-RGCN variant, set it to True for using the MultiGML-RGAT variant. Set the --which_features argument to 'full' for using the multimodal features, set it to 'random' for using the basic features.

    ```
    $ python -m multigml linkpredict run --which_graph='complete' --neg_sample_size=1 --use_cuda=False --batch_size=100 --n_epochs=100  --evaluate_every=1 --n_splits=1 --data_set_split='<repository_path>/MultiGML/data/data_set_split/extended_trimmed_kg' --eval_edge_type='side-effect' --eval_edge_type_inv='side-effect_inverse' --which_features='full' --average='weighted' --calculate_etype_metrics=True --best_params='<best_params_path>' --mlflow_experiment_name='<mlflow_name>' --test_only_eval_etype=False --test_run=True --use_attention=False --pretrained_model='<pretrained_model_path>'
   ```

## Custom Use Case

If you have a custom data set of a knowledge graph that is tailored to a specific research question, it is possible to employ the workflow of MultiGML to your use case by A) Generating the node features for your knowledge graph and B) analysing the post-hoc explainability of your predictions. We therefore provide a software license distributed via [scapos](https://www.scapos.de/portfolio/). Please contact their team for more information on the license.

### Node Feature Generation

You will be able to create the following features:

- [Drugs](#drugs)
    - **Gene expression fingerprint**: We use a molecular transcriptomics signature that represents the effect of a drug on biological processes in a defined system of a cell culture experiment. This signature is created for each drug, measuring the gene expression fold change of selected transcripts. We chose the LINCS L1000 dataset to annotate the drugs with gene expression profile information.  
    - **Morphological fingerprint**: The effects of a drug perturbation in a cell culture experiment can not only be seen in the gene expression fold change, but also in the change in morphology of the treated cells. Therefore, we additionally annotate the drug with the Cell Painting morphological profiling assay information from the LINCS Data Portal (LDG-1192: LDS-1195).
    - **Molecular fingerprint**: The molecular structure of the drugs was also taken into account by generating the molecular fingerprints. We here took the Morgan count fingerprint with a radius = 2, generated with the RDKit.
- [Proteins](#proteins)
    - **Sequence embeddings**: We used structural information of proteins in form of protein sequence embeddings. We generated the embeddings for each protein with the ESM-1b Transformer, a recently published pre-trained deep learning model for protein sequences.
    - **Gene Ontology fingerprints**: We generated a binary Gene Ontology (GO) fingerprint for biological processes for each protein using data from the Gene Ontology Resource. A total of 12,226 human GO terms of Biological Processes were retrieved and their respective parent terms obtained. This resulted in a 1298 dimensional binary fingerprint for each protein, with each index either set to 1, if the protein was annotated with the respective GO term or 0 if not.
- [Phenotypes](#phenotypes)
    - **Medical concept embeddings**: Pre-trained embeddings of clinical concepts were used to annotate phenotypes including ADEs and indications. The so-called cui2vec embeddings were generated on the basis of clinical notes, insurance claims, and biomedical full text articles for each clinical concept. Briefly, the authors mapped ICD-9 codes in claims data to UMLS concepts and then counted co-occurrence of concept pairs. After decomposing the co-occurrence matrix via singular value decomposition, they used the popular word2vec approach to obtain concept embeddings in the Euclidean space. 

### Explainability

You will be able to explain your predictions post-hoc 
- applying the **Integrated Gradients** method to explain the attributions of your input to the predicitons
- analysing the **attention weights** from the MultiGML-RGAT model for the k-hop neighboring edges of your predictions of interest


## Issues
If you have difficulties using MultiGML, please open an issue at our [GitHub](https://github.com/SCAI-BIO/MultiGML) repository.

## Citation

When using this repository, please cite the following [publication](https://doi.org/10.1016/j.heliyon.2023.e19441):

@article{krix2023multigml,
  title={MultiGML: multimodal graph machine learning for prediction of adverse drug events},
  author={Krix, Sophia and DeLong, Lauren Nicole and Madan, Sumit and Domingo-Fern{\'a}ndez, Daniel and Ahmad, Ashar and Gul, Sheraz and Zaliani, Andrea and Fr{\"o}hlich, Holger},
  journal={Heliyon},
  volume={9},
  number={9},
  year={2023},
  publisher={Elsevier}
}

## Disclaimer
MultiGML is a scientific software that has been developed in an academic capacity, and thus comes with no warranty or guarantee of maintenance, support, or back-up of data.
