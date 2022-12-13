# -*- coding: utf-8 -*-

"""This file has all the constants used in other preprocessing modules."""

import os

# ==============================================================
# System paths
# ==============================================================

# Project data directory
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = os.path.join(os.path.abspath(os.path.join(PROJECT_DIR, os.pardir)))

# Main directories
HERE = os.path.abspath(os.path.dirname(__file__))
HOME = os.path.expanduser('~')
DESKTOP = os.path.join(HOME, 'Desktop')
DOCUMENTS_DIR = os.path.join(HOME, 'Documents')
MULTIGML_DOCS_DIR = os.path.join(DOCUMENTS_DIR, 'multigml_docs')
RESULTS_MULTIGML_DIR = os.path.join(MULTIGML_DOCS_DIR, 'results_multigml')
REPOSITORY_DIR = "/".join(PROJECT_DIR.split("/")[:-2])
MULTIGML_DATA_DIR = os.path.join(REPOSITORY_DIR, 'data')
RUN_RESULTS_DIR = os.path.join(RESULTS_MULTIGML_DIR, 'run_results')
DATA_GRAPH_DIR = os.path.join(MULTIGML_DATA_DIR, 'graph')
DATA_FEATURES_DIR = os.path.join(MULTIGML_DATA_DIR, 'features')
DATA_FULL_FEATURES_DIR = os.path.join(DATA_FEATURES_DIR, 'full_features')

# --------------------------------------------------------------
# Column names
# --------------------------------------------------------------
SOURCE = 'source'
TARGET = 'target'
RELATION = 'relation'
SOURCE_DATABASE = 'source_database'
HGNC_SYMBOL = 'hgnc_symbol'
SOURCE_IDENTIFIER = 'source_identifier'
TARGET_IDENTIFIER = 'target_identifier'
SOURCE_NAME = 'source_name'
TARGET_NAME = 'target_name'
SOURCE_NAMESPACE = 'source_namespace'
TARGET_NAMESPACE = 'target_namespace'
BEL_RELATION = 'bel_relation'
RELATION_TYPE = 'relation_type'
# --------------------------------------------------------------

# Function parameters
SEP = '\t'
RELATION = 'relation'


# ==============================

DRUG_PROTEIN = 'drug-protein'
PROTEIN_PROTEIN = 'protein-protein'
PROTEIN_DISEASE = 'protein-disease'
INDICATION = 'indication'
INDICATION_INVERSE = 'indication_inverse'
SIDE_EFFECT = 'side-effect'
SIDE_EFFECT_INVERSE = 'side-effect_inverse'
DRUG_DISEASE = 'drug-disease'
DRUG_DISEASE_INVERSE = 'drug-disease_inverse'
GENETIC_PROTEIN_CONDITION_ASSOCIATION = 'genetic protein-condition association'
GENETIC_PROTEIN_CONDITION_ASSOCIATION_INVERSE = 'genetic protein-condition association_inverse'
DRUG = 'drug'
PROTEIN = 'protein'
DISEASE = 'disease'
CONDITION = 'condition'
SIGNALLING_INTERACTION = 'signalling interaction'
GENETIC_ASSOCIATION = 'genetic association'
PHYSICAL_INTERACTION = 'physical interaction'
FUNCTIONAL_INTERACTION = 'functional interaction'
SRC_NODE_TYPE = 'src_node_type'
DST_NODE_TYPE = 'dst_node_type'
SOURCE_NODE_TYPE = 'source_node_type'
TARGET_NODE_TYPE = 'target_node_type'


# full features
FULL_DRUG_COUNT_FCFP_FILE = os.path.join(DATA_FULL_FEATURES_DIR, 'drugbank_count_fcfp.tsv')
FULL_DRUG_CYTOLOGICAL_FILE = os.path.join(DATA_FULL_FEATURES_DIR, 'lincs_cytological_profiling_drug_features.tsv')
FULL_LINCS_DRUG_FC_FILE = os.path.join(DATA_FULL_FEATURES_DIR, 'lincs_drug_fc.tsv')
FULL_ESM_PROT_EMBEDDINGS = os.path.join(DATA_FULL_FEATURES_DIR, 'protein_embeddings_esm.tsv')
FULL_PROTEIN_GENE_ONTOLOGY_FINGERPRINT = os.path.join(DATA_FULL_FEATURES_DIR, 'protein_go_fingerprint.tsv')
FULL_CUI2VEC_DISGENET_FILE = os.path.join(DATA_FULL_FEATURES_DIR, 'cui2vec_disgenet.tsv')

# random features
RANDOM_DRUG_FEATURE_FILE = os.path.join(DATA_FEATURES_DIR, 'random_drug.tsv')
RANDOM_PROTEIN_FEATURE_FILE = os.path.join(DATA_FEATURES_DIR, 'random_protein.tsv')
RANDOM_DISEASE_FEATURE_FILE = os.path.join(DATA_FEATURES_DIR, 'random_disease.tsv')

FEATURE_FILE_DRUG_FP_NAME = 'drug_fp'
FEATURE_FILE_DRUG_CYT_NAME = 'drug_cyt'
FEATURE_FILE_DRUG_FC_NAME = 'drug_fc'
FEATURE_FILE_DRUG_FCFP_NAME = 'drug_fcfp'
FEATURE_FILE_PROTEIN_FC_NAME = 'protein_fc'
FEATURE_FILE_PROTEIN_PROTVEC_NAME = 'protein_protVec'
FEATURE_FILE_DISEASE_CUI2VEC_NAME = 'disease_cui2vec'
FEATURE_FILE_PROTEIN_ESM_NAME = 'protein_esm'
FEATURE_FILE_PROTEIN_GO_NAME = 'protein_go'
FEATURE_FILE_DRUG_RANDOM_NAME = 'drug_random'
FEATURE_FILE_PROTEIN_RANDOM_NAME = 'protein_random'
FEATURE_FILE_DISEASE_RANDOM_NAME = 'disease_random'
FEATURE_FILE_DRUG_IDENTITY_NAME = 'drug_identity'
FEATURE_FILE_PROTEIN_IDENTITY_NAME = 'protein_identity'
FEATURE_FILE_CONDITION_IDENTITY_NAME = 'condition_identity'
FEATURE_FILE_LINCS_MAYAAN_NAME = 'drug_lincs'
FEATURE_FILE_GO_MAYAAN_NAME = 'drug_go'
FEATURE_FILE_MORPHOLOGY_MAYAAN_NAME = 'drug_morphology'
FEATURE_FILE_MACCS_MAYAAN_NAME = 'drug_maccs'
NODE_TYPE = 'node_type'
FILE_NAME = 'file_name'

# Complete graph
CONDITION_GRAPH_FILE = os.path.join(DATA_GRAPH_DIR, 'extended_trimmed_kg.tsv')

# ===============
# EVALUATION FILES
# ===============
EVALUATION_DIR = os.path.join(RESULTS_MULTIGML_DIR, 'evaluation')


# Comparison with other methods
COMPARISON_DIR = os.path.join(RESULTS_MULTIGML_DIR, 'comparison')

