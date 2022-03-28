import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.utils.data as Data
import argparse
import os


from tqdm import tqdm
import pickle

from sklearn.metrics import mean_absolute_error

from torch.utils.tensorboard import SummaryWriter

from models.StarNet import StarNet
from models.RRNet import RRNet


parser = argparse.ArgumentParser()


parser.add_argument(
    '--test_flux_path', type=str, default='./data/refer_set/test_flux.pkl',
    help='The path of the flux data to estimate',
)
parser.add_argument(
    '--test_label_path', type=str, default='./data/refer_set/test_label.pkl',
    help='The path of the label data',
)
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
    '--RRNet_inplane_list', type=list, default=[4,8,16],
    help='The size of inplane in the RRNet residual block'
)
parser.add_argument(
    '--label_list', type=list, default=['Teff[K]', 'Logg', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH',
                                        'SH', 'KH', 'CaH', 'TiH', 'CrH', 'MnH', 'FeH', 'NiH', 'CuH'],
    help='Label data that needs to be learned'
)
parser.add_argument(
    '--noise_model', type=bool, default=True,  
    help='Whether to use a model trained with noise'
)
parser.add_argument(
    '--DeepEnsemble', type=bool, default=True,
    help='Whether to use a fine-tuning model'
)


def predict(args, test_flux_path, test_label_path=None):

    def one_predict(args, test_loader, model_path):
        print(model_path)

        train_label = ['Teff[K]', 'Logg', 'FeH'] if model_path.split("/")[-1][:2] =="SP" else ['CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH', 'MnH', 'NiH', 'CuH']

        if args.net == "RRNet":            
            net = RRNet(
                num_lable=len(train_label),
                mode=args.rnn_mode,
                ResBlock_inplanes_list=args.RRNet_inplane_list,
            ).to("cuda")
        elif args.net == "StarNet":
            net = StarNet(
                num_lable=len(train_label),
                mode=args.rnn_mode,
            ).to("cuda")
        
        net.eval()
        net.load_state_dict(torch.load(model_path+"/weight_best.pkl"))
        
       
        output_label = np.zeros(shape=(len(test_loader.dataset), len(train_label)))
        output_label_err = np.zeros_like(output_label)
        for step, batch in tqdm(enumerate(test_loader)): 
            with torch.no_grad():  
                
                batch_x = batch[0]
           
                mu, sigma = net(batch_x.to("cuda"))
                mu = mu.to("cpu").data.numpy()
                sigma = np.sqrt(sigma.to("cpu").data.numpy())


            output_label[step*test_loader.batch_size :step*test_loader.batch_size + mu.shape[0]] = mu
            output_label_err[step*test_loader.batch_size :step*test_loader.batch_size + mu.shape[0]] = sigma

        return [output_label, output_label_err]


    label_config = pickle.load(open(args.data_path + "label_config.pkl", 'rb'))
    label_config_index = [label_config['label_list'].index(i) for i in args.label_list]

    label_mean = label_config['label_mean'][label_config_index]
    label_std = label_config['label_std'][label_config_index]
    flux_mean = label_config["flux_mean"]
    flux_std = label_config["flux_std"]

    del label_config

    X_test_torch = pickle.load(open(test_flux_path, 'rb'))
    # 流量上限处理
    X_test_torch[X_test_torch > 2.5] = 2.5
    X_test_torch[X_test_torch < -0.5] = -0.5

    X_test_torch = (X_test_torch - flux_mean) / flux_std
    X_test_torch = torch.tensor(X_test_torch, dtype=torch.float32)
    if test_label_path is not None:
        y_test_torch = pd.read_csv(test_label_path)[args.label_list].values
        y_test_torch = (y_test_torch - label_mean) / label_std
        y_test_torch = torch.tensor(y_test_torch, dtype=torch.float32)
        test_dataset = Data.TensorDataset(X_test_torch, y_test_torch)
    else:
        test_dataset = Data.TensorDataset(X_test_torch)

    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    
    if args.net == "RRNet":
        model_name = "RRNet(%s)_%s"%('-'.join([str(i) for i in args.RRNet_inplane_list]), args.rnn_mode)
    elif args.net == "StarNet":
        model_name = "StarNet_%s"%args.rnn_mode
    if args.noise_model:
        model_name += "_add-noise"

    model_path = args.save_dir+model_name
    model_list = os.listdir(model_path)
    output_list_SP = []
    output_list_CA = []
    if not args.DeepEnsemble:
        if "SP0" in model_list:
            output_list_SP.append(one_predict(args, test_loader, model_path=model_path+"/SP0"))
        if "CA0" in model_list:
            output_list_CA.append(one_predict(args, test_loader, model_path=model_path+"/CA0"))
    else:
        for model in model_list:
            out = one_predict(args, test_loader, model_path=model_path+"/"+model)
            if model[:2] == "SP":
                output_list_SP.append(out)
            elif  model[:2] == "CA":
                output_list_CA.append(out)
    
    mu_list = []
    sigma_list = []
    for i in range(min(len(output_list_SP), len(output_list_CA))):
        mu_list.append(np.hstack((output_list_SP[i][0], output_list_CA[i][0])))
        sigma_list.append(np.hstack((output_list_SP[i][1], output_list_CA[i][1])))
        
    del output_list_SP, output_list_CA
    mu_list = np.array(mu_list)
    sigma_list = np.array(sigma_list)


    out_mu = mu_list.mean(0)
    out_sigma = ((mu_list**2 + sigma_list**2)).mean(0) - out_mu**2
    out_sigma = np.sqrt(out_sigma)

    train_label = ['Teff[K]', 'Logg', 'FeH', 'CH', 'NH', 'OH', 'MgH', 'AlH', 'SiH', 'SH', 'KH', 'CaH', 'TiH', 'CrH', 'MnH', 'NiH', 'CuH'] 
    train_label_index = [args.label_list.index(i) for i in train_label]

    out_mu = out_mu * label_std[train_label_index] + label_mean[train_label_index]
    out_sigma *= label_std[train_label_index]

    del mu_list, sigma_list

    
    if test_label_path is not None:
        true_mu = pd.read_csv(test_label_path)[train_label].values

        diff_std = (true_mu-out_mu).std(axis=0)
        mae = mean_absolute_error(true_mu, out_mu, multioutput='raw_values')
        print(
            "mae:", mae,
            "diff_std", diff_std,
        )
        df = pd.read_csv(test_label_path)
        for i in range(len(train_label)):
            df["%s_%s"%(model_name, train_label[i])] = out_mu[:, i]
            df["%s_%s_err"%(model_name, train_label[i])] = out_sigma[:, i]
        df.to_csv(test_label_path[:-4]+"_%s_out.csv"%model_name, index=False)
    else:
        df = pd.DataFrame(data=None)
        for i in range(len(train_label)):
            df["%s"%train_label[i]] = out_mu[:, i]
            df["%s_err"%train_label[i]] = out_sigma[:, i]
        df.to_csv(test_flux_path[:-4]+"_%s_out.csv"%model_name, index=False)



if __name__ == "__main__":

    args = parser.parse_args()

    predict(args, test_flux_path=args.test_flux_path, test_label_path="./data/refer_set/test_label.csv")
    predict(args, test_flux_path='./data/refer_set/train_flux.pkl', test_label_path="./data/refer_set/train_label.csv")
    predict(args, test_flux_path='./data/refer_set/valid_flux.pkl', test_label_path="./data/refer_set/valid_label.csv")



    # all_flux_path = "./data/all_flux_data/"
    # for i in tqdm(range(85)):
    #     test_flux_path = all_flux_path+"all_not_match_valid_flux_%d.pkl"%i
    #     print(test_flux_path)
    #     predict(args, test_flux_path=test_flux_path)


    