from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.utils.data as Data
import argparse


from tqdm import tqdm
import pickle

from sklearn.metrics import mean_absolute_error

from torch.utils.tensorboard import SummaryWriter

from models.StarNet import StarNet
from models.RRNet import RRNet


parser = argparse.ArgumentParser()

parser.add_argument(
    '--net', choices=['StarNet', 'RRNet'], default='RRNet',
    help='The models you need to use',

)
parser.add_argument(
    '--rnn_mode', choices=['raw', 'pre-RNN', 'post-RNN'], default='post-RNN',
    help='The mode of the RRN embedding',
)
parser.add_argument(
    '--data_path', type=str, default='./data/refer_set/',
    help='Path to the datasets',
)
parser.add_argument(
    '--save_dir', type=str, default='./model_log/',
    help='The path where the trained model data is saved.'
)
parser.add_argument(
    '--batch_size', type=int, default=256,
    help='The size of the batch'
)
parser.add_argument(
    '--n_epochs', type=int, default=30,
    help='Number of (pseudo) epochs to train for'
)
parser.add_argument(
    '--RRNet_inplane_list', type=list, default=[4,8,16],
    help='The size of inplane in the RRNet residual block'
)
parser.add_argument(
    '--label_list', type=list, default=['Teff[K]', 'Logg', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH',
                                         'KH', 'CaH', 'TiH', 'CrH', 'MnH', 'FeH', 'NiH', 'CuH'],
    help='Label data that needs to be learned'
)
parser.add_argument(
    '--add_training_noise', type=bool, default=True,  
    help='Whether to add Gaussian noise with a mean of 1 and a variance of 1 during training'
)


def add_noise(input_x):
    normal_noise = torch.normal(mean=torch.zeros_like(input_x), std=torch.ones_like(input_x))
    normal_prob = torch.rand_like(input_x)  
    input_x[normal_prob<0.25] += normal_noise[normal_prob<0.25]

    # noise_row = np.random.choice(input_x.shape[0], input_x.shape[0]//2, replace=False)
    # normal_noise = torch.normal(mean=torch.zeros_like(input_x[noise_row, :]), std=torch.ones_like(input_x[noise_row, :]))
    # normal_prob = torch.rand_like(input_x[noise_row, :])  
    # input_x[noise_row, :][normal_prob<0.25] += normal_noise[normal_prob<0.25]
      
    # noise_row = np.random.choice(input_x.shape[0], input_x.shape[0]//4, replace=False)
    # normal_prob = torch.rand_like(input_x[noise_row, :])     
    # input_x[noise_row, :][normal_prob<0.05] = 0

    return input_x

def train(args, dataset_info, train_label=['Teff[K]', 'Logg', 'FeH'], model_number="SP0", cuda=True):


    if args.net == "RRNet":
        model_name = "RRNet(%s)_%s"%('-'.join([str(i) for i in args.RRNet_inplane_list]), args.rnn_mode)
        
        net = RRNet(
            num_lable=len(train_label),
            mode=args.rnn_mode,
            ResBlock_inplanes_list=args.RRNet_inplane_list,
        )
        if cuda:
            net = net.to("cuda")
    elif args.net == "StarNet":
        model_name = "StarNet_%s"%args.rnn_mode

        net = StarNet(
            num_lable=len(train_label),
            mode=args.rnn_mode,
        )
        if cuda:
            net = net.to("cuda")

    if args.add_training_noise:
        model_name += "_add-noise"


    model_name += '/'+model_number
  
    print(net)
    print(model_name)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    log_dir = args.save_dir+model_name
    writer = SummaryWriter(log_dir=log_dir)

    label_index = [args.label_list.index(i) for i in train_label]

    best_loss = np.inf
    # Iterative optimization
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        net.train()
        torch.cuda.empty_cache()

        train_mae = np.zeros(len(train_label))
        train_loss = 0.0
        # Train
        for step, (batch_x, batch_y) in enumerate(dataset_info["train_loader"]):
            
            if args.add_training_noise and epoch>5:
                batch_x = add_noise(batch_x)
            
            batch_y = batch_y[:, label_index]
            if cuda:
                batch_x = batch_x.to("cuda")
                batch_y = batch_y.to("cuda")
            mu, sigma = net(batch_x)
            loss = net.get_loss(batch_y, mu, sigma)

            train_loss += loss.to("cpu").data.numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_iter = (epoch - 1) * len(dataset_info["train_loader"]) + step + 1

            mu = mu.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + dataset_info["label_mean"][label_index]
            batch_y = batch_y.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + dataset_info["label_mean"][label_index]

            mae = mean_absolute_error(batch_y, mu, multioutput='raw_values')
            train_mae += mae

            writer.add_scalar('Train/loss', loss.to("cpu").data.numpy(), n_iter)
            for i in range(len(train_label)):
                writer.add_scalar('Train/%s_MAE'%train_label[i], mae[i], n_iter)

        
        scheduler.step()
        lr = optimizer.state_dict()['param_groups'][0]['lr']

        train_loss /= (step+1)
        train_mae /= (step+1)

        torch.cuda.empty_cache()
        net.eval()

        valid_mae = np.zeros(len(label_index))
        vlaid_diff_std = np.zeros(len(label_index))
        valid_loss = 0.0
        # Valid
        for step, (batch_x, batch_y) in enumerate(dataset_info["valid_loader"]):

            with torch.no_grad():
                batch_y = batch_y[:, label_index]
                if cuda:
                    batch_x = batch_x.to("cuda")
                    batch_y = batch_y.to("cuda")
                mu, sigma = net(batch_x)
                loss = net.get_loss(batch_y, mu, sigma)

                valid_loss += loss.to("cpu").data.numpy()

                n_iter = (epoch - 1) * len(dataset_info["valid_loader"]) + step + 1

                sigma = np.sqrt(sigma.to("cpu").data.numpy()) * dataset_info["label_std"][label_index] 

                mu = mu.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + dataset_info["label_mean"][label_index]
                batch_y = batch_y.to("cpu").data.numpy() * dataset_info["label_std"][label_index] + dataset_info["label_mean"][label_index]

                diff_std = (mu-batch_y).std(axis=0)
                sigma_mean = sigma.mean(axis=0)

                vlaid_diff_std += diff_std

                mae = mean_absolute_error(batch_y, mu, multioutput='raw_values')

                valid_mae += mae

            writer.add_scalar('Valid/loss', loss.to("cpu").data.numpy(), n_iter)
            for i in range(len(train_label)):
                writer.add_scalar('Valid/%s_MAE'%train_label[i], mae[i], n_iter)
                writer.add_scalar('Valid/%s_diff_std'%train_label[i], diff_std[i], n_iter)
                writer.add_scalar('Valid/%s_sigma'%train_label[i], sigma_mean[i], n_iter)

        valid_loss /= (step+1)
        valid_mae /= (step+1)
        vlaid_diff_std /= (step+1)

        torch.save(net.state_dict(), log_dir + '/weight_temp.pkl')

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(net.state_dict(), log_dir + '/weight_best.pkl')
        
        print("EPOCH %d | lr %f | train_loss %.4f | valid_loss %.4f"%(epoch, lr, train_loss, valid_loss),
              "| valid_mae", valid_mae,
              "| valid_diff_std", vlaid_diff_std)



def get_dataset_info(args):

    label_config = pickle.load(open(args.data_path + "label_config.pkl", 'rb'))
    label_config_index = [label_config['label_list'].index(i) for i in args.label_list]

    label_mean = label_config['label_mean'][label_config_index]
    label_std = label_config['label_std'][label_config_index]
    flux_mean = label_config["flux_mean"]
    flux_std = label_config["flux_std"]

    del label_config

    X_train_torch = (pickle.load(open(args.data_path + "train_flux.pkl", 'rb')) - flux_mean) / flux_std
    X_valid_torch = (pickle.load(open(args.data_path + "valid_flux.pkl", 'rb')) - flux_mean) / flux_std
 
    X_train_torch = torch.tensor(X_train_torch, dtype=torch.float32)
    X_valid_torch = torch.tensor(X_valid_torch, dtype=torch.float32)

    y_train_torch = pd.read_csv(args.data_path + "train_label.csv")[args.label_list].values
    y_valid_torch = pd.read_csv(args.data_path + "valid_label.csv")[args.label_list].values

    y_train_torch = (y_train_torch - label_mean) / label_std
    y_valid_torch = (y_valid_torch - label_mean) / label_std

    y_train_torch = torch.tensor(y_train_torch, dtype=torch.float32)
    y_valid_torch = torch.tensor(y_valid_torch, dtype=torch.float32)

    train_dataset = Data.TensorDataset(X_train_torch, y_train_torch)
    valid_dataset = Data.TensorDataset(X_valid_torch, y_valid_torch)
    
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    valid_loader = Data.DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    dataset_info = {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "label_mean": label_mean,
        "label_std": label_std,
    }

    return dataset_info





if __name__ == "__main__":
    
    args = parser.parse_args()
    
    dataset_info = get_dataset_info(args)

    train(args, dataset_info=dataset_info, 
             train_label=['Teff[K]', 'Logg', 'FeH'], 
             model_number="SP0",
             cuda=True)
    train(args, dataset_info=dataset_info, 
             train_label=['CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH', 'MnH', 'NiH', 'CuH'], 
             model_number="CA0",
             cuda=True)

    # for i in range(1, 6):
    #     train(args, dataset_info=dataset_info, 
    #           train_label=['Teff[K]', 'Logg', 'FeH'], 
    #           model_number="SP%d"%i)
    # for i in range(1, 6):
    #     train(args, dataset_info=dataset_info, 
    #           train_label=['CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH', 'MnH', 'NiH', 'CuH'], 
    #           model_number="CA%d"%i)



    
    
    


    