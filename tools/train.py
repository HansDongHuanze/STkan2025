import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from module import models
from module import baselines
from tools import learner
from tools import functions as fn
from tools import data_switcher
import time
from module import Wave_graph_EKF_KAN

def train(args):
    # system configuration
    use_cuda = args.use_cuda
    cuda_id = "cuda:" + str(args.cuda_device)
    device = torch.device(cuda_id if use_cuda and torch.cuda.is_available() else "cpu")
    fn.set_seed(seed=args.random_seed, flag=True)
    dataset=args.dataset
    torch.cuda.empty_cache()

    # hyper params
    model_name = args.model_name
    seq_l = args.seq_len
    pre_l = args.pre_len
    bs = 512
    p_epoch = 200
    n_epoch = 1000
    law_list = np.array([-1.48, -0.74])  # price elasticities of demand for EV charging. Recommend: up to 5 elements.
    is_train = True
    mode = 'completed'  # 'simplified' or 'completed'
    is_pre_train = True

    train_occupancy, train_price, train_loader, valid_loader, test_loader, adj_dense = data_switcher.get_data_loaders(seq_l, pre_l, device, bs)

    adj_dense_cuda = adj_dense.to(device)
    adj_sparse = adj_dense.to_sparse_coo().to(device)

    # training setting
    # model = models.PAG(a_sparse=adj_sparse).to(device)  # init model
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
    model = Wave_graph_EKF_KAN.KAN(input_dim=seq_l, output_dim=pre_l).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001)

    loss_function = torch.nn.MSELoss()
    valid_loss = 100

    print(f"----Starting training {model_name} model with prediction horizen {pre_l}----")

    if is_train is True:
        model.train()
        if is_pre_train is True:
            if mode == 'simplified':  # a simplified way of physics-informed meta-learning
                model = learner.fast_learning(law_list, model, model_name, p_epoch, bs, train_occupancy, train_price, seq_l, pre_l, device, adj_dense)

            elif mode == 'completed': # the completed process
                model = learner.physics_informed_meta_learning(law_list, model, model_name, p_epoch, bs, train_occupancy, train_price, seq_l, pre_l, device, adj_dense)
            else:
                print("Mode error, skip the pre-training process.")

        for epoch in tqdm(range(n_epoch), desc='Fine-tuning'):
            for j, data in enumerate(train_loader):
                model.train()
                occupancy, price, label = data

                optimizer.zero_grad()
                predict = model(occupancy, price)
                loss = loss_function(predict, label)
                loss.backward()
                optimizer.step()

            # validation
            model.eval()
            for j, data in enumerate(valid_loader):
                model.train()
                occupancy, price, label = data
                predict = model(occupancy, price)
                loss = loss_function(predict, label)
                if loss.item() < valid_loss:
                    valid_loss = loss.item()
                    torch.save(model, './checkpoints' + '/' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + '_' + mode + '.pt')

    print(f"----Training finished!----")
    
    model = torch.load('./checkpoints' + '/' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + '_' + mode + '.pt')
    print(f"----Model was saved into folder: {'./checkpoints' + '/' + model_name + '_' + str(pre_l) + '_bs' + str(bs) + '_' + mode + '.pt'}")
    # test
    model.eval()
    result_list = []
    predict_list = []
    label_list = []

    # Initialize lists to store time and memory usage
    time_list = []
    memory_list = []

    for j, data in enumerate(test_loader):
        occupancy, price, label = data  # occupancy.shape = [batch, seq, node]
        print('occupancy:', occupancy.shape, 'price:', price.shape, 'label:', label.shape)
        with torch.no_grad():
            # Start time measurement
            start_time = time.time()
            
            # Start memory tracking
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Synchronize before measuring
                memory_before = torch.cuda.memory_allocated()

            predict = model(occupancy, price)

            # End memory tracking
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_usage = memory_after - memory_before
                memory_list.append(memory_usage / (1024 * 1024))  # Convert bytes to MB
            
            # End time measurement
            end_time = time.time()
            elapsed_time = end_time - start_time
            time_list.append(elapsed_time)
            predict_list.append(predict)
            label_list.append(label)

    predict_list = torch.cat(predict_list, dim=0).view(-1, pre_l).cpu().detach().numpy()
    label_list = torch.cat(label_list, dim=0).view(-1, pre_l).cpu().detach().numpy()

    output_no_noise = fn.metrics(test_pre=predict_list[1:, :], test_real=label_list[1:, :])
    result_list.append(output_no_noise)
    result_df = pd.DataFrame(columns=['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE', 'R2'], data=result_list)
    result_df.to_csv('./results' + '/' + model_name + '_' + str(pre_l) + 'bs' + str(bs) + '.csv', encoding='gbk')

    # Print average time and memory usage
    print(f'Average time per prediction: {np.mean(time_list)} seconds')
    print(f'Average memory usage per prediction: {np.mean(memory_list)} MB')