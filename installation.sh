# Install pytorch with your CUDA version
conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch 

# Install DGL with your CUDA version
conda install -c dglteam dgl-cuda11.3

# After moving into main folder:
cd mavo
python3 -m pip install -e .
