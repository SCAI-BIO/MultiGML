#!/bin/bash -l
#SBATCH -J dlcg
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# Load Anaconda
module load Anaconda3

# Activate the virtualenv
#source activate /home/bio/groupshare/sophia/virtualenv
conda activate /home/skrix/virtualenv_nodgltorch
# Load CUDA
module load CUDA/11.0.228

# collect environment variables
 python -m torch.utils.collect_env

# Go to your repo
cd /home/bio/groupshare/sophia/mavo

# Run hyperparameter search
#python -m multigml linkpredict run-part1 --n_trials=10 --use_cuda=True --n_epochs=20 --n_crossval=1 --evaluate_every=5 --which_graph=complete_graph

# generate edgelist for deepwalk
python -m redrugnn linkpredict prepare-deepwalk --which_graph=complete_graph

# generate deepwalk embeddings
deepwalk --input /home/skrix/results_multigml/graph_edgelist.tsv  --output /home/skrix/results_multigml/deepwalk_embeddings.tsv --format edgelist

# Run with best hyperparameters and baseline comparison
python -m redrugnn linkpredict run-best-hp --use_cuda=True --n_epochs=30 --evaluate_every=5 --which_graph=complete_graph --best_params_file=/home/skrix/results_multigml/graphcomplete_graph_64/best_params.json


"runme.sh" 42L, 1580C                                                                                                       33,1       Anfang
