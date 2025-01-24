# IBCS: Learning Information Bottleneck-Constrained Denoised Causal Subgraph for Graph Classification

This is the pytorch implementation for "IBCS: Learning Information Bottleneck-Constrained Denoised Causal Subgraph for Graph Classification".

## Environment requirements

Code is tested with **Python 3.7.16**. Some principle requirements are listed as follows:

python==3.7.16  
torch==1.13.1  
torch_geometric ==2.2.0  
numpy==1.21.5  
scikit_learn==1.0.2  
networkx==2.6.3  
rdkit==2023.03.2  
scipy==1.7.3  
ogb==1.3.6  

## Download the datasets
The OOD datasets (Spurious-Motif, MNIST-75sp, Graph-SST2) are provided by [DIR](https://github.com/Wuyxin/DIR-GNN). As for TUdataset (MUTAG, IMDB-BINARY, PROTEINS, DD), they can be directly downloaded while running the code. For OGB data, the code can also automatically download them. For convenience, we have provided all datasets in the [Google Drive link](https://drive.google.com/drive/folders/1Dtfjej8BZVy16bfxMtkh1hjavF9XDbpo), where TUdatasets can be found in the './data/raw' directory. Please download the 'data.zip' file and unzip it to the 'data/' directory at the root path.

## Instructions to run the code

After installing the required environment dependencies displayed above, to reproduce our experimental results, please go into our code folder and then run the following commands. 

1) For OOD datasets, run the following command to evaluate the graph classification performance on the OOD test:  
Spurious-Motif (0.5): python main_spurious-motif_0.5.py  
Spurious-Motif(0.7): python main_spurious-motif_0.7.py  
Spurious-Motif(0.9): python main_spurious-motif_0.9.py  
MNIST-75sp: python main_mnist.py  
Graph-SST2: python main_graph-sst.py  

2) For the real-world TUDataset (MUTAG,IMDB-BINARY, PROTEINS, DD), run the following command to get the graph classification accuracy for these datasets：  
MUTAG: python main_mutag.py  
IMDB-BINARY: python main_imdb.py  
PROTEINS: python main_proteins.py  
DD: python main_DD.py

## Reference
Welcome to kindly cite our work with :  
@article{yuan2024ibcs,  
  title={IBCS: Learning Information Bottleneck-Constrained Denoised Causal Subgraph for Graph Classification},  
  author={Yuan, Ruiwen and Tang, Yongqiang and Xiao, Yanghao and Zhang, Wensheng},  
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},  
  year={2024},  
  publisher={IEEE}  
}
