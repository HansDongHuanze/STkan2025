import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from module import baselines
from tools import learner
from tools import functions as fn
from tools import data_switcher
import time
from module import KAN

def choose_model(model_name, seq_l, pre_l, adj_dense, device):
    adj_dense_cuda = adj_dense.to(device)
    adj_sparse = adj_dense.to_sparse_coo().to(device)
    if model_name == "PAG":
        model = baselines.PAG(seq=seq_l, pred_len=pre_l, a_sparse=adj_sparse).to(device)  # init model
    # model = baselines.FGN()
    # model = baselines.VAR().to(device)
    # model = baselines.FCN().to(device)
    # model = baselines.GCN(seq_l, 2, adj_dense_cuda).to(device)
    # model = baselines.GAT(seq_l, 2, adj_dense_cuda).to(device)
    # model = baselines.LSTM(seq_l, 2).to(device)
    # model = baselines.TransformerModel(seq_l, 32, 16, 2, 1, 4, 32, 0.5) # input_dim, embedding_dim, hidden_dim, output_dim, n_layers, n_heads, pf_dim, dropout
    # model = baselines.STGCN(seq_l, 2, adj_dense_cuda).to(device)
    # model = baselines.LstmGcn(seq_l, 2, adj_dense_cuda).to(device)
    # model = baselines.LstmGat(seq_l, 2, adj_dense_cuda, adj_sparse).to(device)
    # model = baselines.HSTGCN(seq_l, 2, adj_dense_cuda, adj_dense_cuda).to(device)
    # model = baselines.TPA(seq_l, 2, nodes).to(device)
    # model = GAF.GATWithFourier(seq_l, 2, adj_dense_cuda).to(device)
    elif model_name == "KAN":
        model = KAN.KAN(input_dim=seq_l, output_dim=pre_l).to(device)
    return model