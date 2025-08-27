import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import networkx as nx

# Install required packages (run in terminal):
# pip install torch torch-geometric matplotlib seaborn scikit-learn networkx# Import PyTorch Geometric for easy dataset loading
try:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.utils import to_networkx
    print("✅ PyTorch Geometric successfully imported!")
except ImportError:
    print("❌ Please install PyTorch Geometric:")
    print("pip install torch-geometric")
    exit()
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
print("🚀 Environment setup complete!")
print(f"PyTorch version: {torch.__version__}")