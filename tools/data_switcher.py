import torch
from torch.utils.data import DataLoader
from tools import functions as fn

def get_data_loaders(seq_l, pre_l, device, bs):
    # input data
    occ, prc, adj, col, dis, cap, tim, inf = fn.read_dataset()
    adj_dense = torch.Tensor(adj)

    # dataset division
    train_occupancy, valid_occupancy, test_occupancy = fn.division(occ, train_rate=0.6, valid_rate=0.2, test_rate=0.2)
    nodes = train_occupancy.shape[-1]
    train_price, valid_price, test_price = fn.division(prc, train_rate=0.6, valid_rate=0.2, test_rate=0.2)

    # data
    train_dataset = fn.CreateDataset(train_occupancy, train_price, seq_l, pre_l, device, adj_dense)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
    valid_dataset = fn.CreateDataset(valid_occupancy, valid_price, seq_l, pre_l, device, adj_dense)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_occupancy), shuffle=False)
    test_dataset = fn.CreateDataset(test_occupancy, test_price, seq_l, pre_l, device, adj_dense)
    test_loader = DataLoader(test_dataset, batch_size=len(test_occupancy), shuffle=False)
    return train_occupancy, train_price, train_loader, valid_loader, test_loader, adj_dense