import sys
sys.path.append("..")
import torch
import pandas as pd
from scenario_wise_rec.basic.features import DenseFeature, SparseFeature
from scenario_wise_rec.trainers import CTRTrainer
from scenario_wise_rec.utils.data import DataGenerator, reduce_mem_usage
from scenario_wise_rec.models.multi_domain import Star, MMOE, PLE, SharedBottom, AdaSparse, Sarnet, M2M, AdaptDHM, EPNet, PPNet, M3oE, HamurLarge


def get_ali_ccp_data_dict(data_path='./data/ali-ccp'):
    df_train = reduce_mem_usage(pd.read_csv(data_path + '/ali_ccp_train_sample.csv'))
    df_val = reduce_mem_usage(pd.read_csv(data_path + '/ali_ccp_val_sample.csv'))
    df_test = reduce_mem_usage(pd.read_csv(data_path + '/ali_ccp_test_sample.csv'))
    print("train : val : test = %d %d %d" % (len(df_train), len(df_val), len(df_test)))
    train_idx, val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)
    domain_map = {1: 0, 2: 1, 3: 2}
    data["domain_indicator"] = data["301"].apply(lambda fea: domain_map[fea])
    domain_num = 3
    col_names = data.columns.values.tolist()
    dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    sparse_cols = [col for col in col_names if col not in dense_cols and col not in ['click', 'purchase','domain_indicator']]
    print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))
    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in sparse_cols]

    y = data["click"]
    del data["click"]
    x = data
    x_train, y_train = x[:train_idx], y[:train_idx]
    x_val, y_val = x[train_idx:val_idx], y[train_idx:val_idx]
    x_test, y_test = x[val_idx:], y[val_idx:]
    return dense_feas, sparse_feas, x_train, y_train, x_val, y_val, x_test, y_test, domain_num


def get_ali_ccp_data_dict_adasparse(data_path='./data/ali-ccp'):
    df_train = reduce_mem_usage(pd.read_csv(data_path + '/ali_ccp_train_sample.csv'))
    df_val = reduce_mem_usage(pd.read_csv(data_path + '/ali_ccp_val_sample.csv'))
    df_test = reduce_mem_usage(pd.read_csv(data_path + '/ali_ccp_test_sample.csv'))
    print("train : val : test = %d %d %d" % (len(df_train), len(df_val), len(df_test)))

    train_idx, val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)
    domain_num = 3
    scenario_fea_num = 1
    # scenario_fea = data.pop('301')
    # data.insert(loc=0, column='301', value=scenario_fea)

    col_names = data.columns.values.tolist()
    dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    scenario_cols = ['domain_indicator']

    domain_map = {1: 0, 2: 1, 3: 2}
    data["domain_indicator"] = data["301"].apply(lambda fea: domain_map[fea])
    del data['301']
    sparse_cols = [col for col in col_names if col not in dense_cols and col not in ['click', 'purchase','domain_indicator', '301']]

    print("scenario_cols:%d sparse cols:%d dense cols:%d" % (len(scenario_cols), len(sparse_cols), len(dense_cols)))

    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in sparse_cols]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_cols]

    y = data["click"]
    del data["click"]
    x = data
    # scenario_ids = x[scenario_cols].unique()
    x_train, y_train = x[:train_idx], y[:train_idx]
    x_val, y_val = x[train_idx:val_idx], y[train_idx:val_idx]
    x_test, y_test = x[val_idx:], y[val_idx:]

    return (dense_feas, sparse_feas, scenario_feas, scenario_fea_num,
            x_train, y_train, x_val, y_val, x_test, y_test, domain_num)

def get_ali_ccp_data_dict_ppnet(data_path='./data/ali-ccp'):
    df_train = reduce_mem_usage(pd.read_csv(data_path + '/ali_ccp_train_sample.csv'))
    df_val = reduce_mem_usage(pd.read_csv(data_path + '/ali_ccp_val_sample.csv'))
    df_test = reduce_mem_usage(pd.read_csv(data_path + '/ali_ccp_test_sample.csv'))
    print("train : val : test = %d %d %d" % (len(df_train), len(df_val), len(df_test)))

    train_idx, val_idx = df_train.shape[0], df_train.shape[0] + df_val.shape[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)
    domain_num = 3
    scenario_fea_num = 1
    # scenario_fea = data.pop('301')
    # data.insert(loc=0, column='301', value=scenario_fea)

    col_names = data.columns.values.tolist()
    dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    id_cols = ['101', '205']
    scenario_cols = ['domain_indicator']
    sparse_cols = [col for col in col_names if col not in dense_cols and col not in id_cols and col not in ['click', 'purchase', 'domain_indicator', '301']]

    domain_map = {1: 0, 2: 1, 3: 2}
    data["domain_indicator"] = data["301"].apply(lambda fea: domain_map[fea])
    del data['301']

    print("scenario_cols:%d sparse cols:%d dense cols:%d" % (len(scenario_cols), len(sparse_cols), len(dense_cols)))

    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in sparse_cols]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_cols]
    id_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in id_cols]

    y = data["click"]
    del data["click"]
    x = data
    # scenario_ids = x[scenario_cols].unique()
    x_train, y_train = x[:train_idx], y[:train_idx]
    x_val, y_val = x[train_idx:val_idx], y[train_idx:val_idx]
    x_test, y_test = x[val_idx:], y[val_idx:]

    return (dense_feas, sparse_feas, scenario_feas, id_feas, scenario_fea_num,
            x_train, y_train, x_val, y_val, x_test, y_test, domain_num)


def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    torch.manual_seed(seed)
    dataset_name = "Aliccp"
    if model_name in ["adasparse", "m2m", "adaptdhm", "epnet"]:
        (dense_feas, sparse_feas, scenario_feas, scenario_fea_num, x_train, y_train, x_val, y_val,
         x_test, y_test, domain_num) = get_ali_ccp_data_dict_adasparse(dataset_path)
    elif model_name == "ppnet":
        (dense_feas, sparse_feas, scenario_feas, id_feas, scenario_fea_num, x_train, y_train, x_val,
         y_val, x_test, y_test, domain_num) = get_ali_ccp_data_dict_ppnet(dataset_path)
    else:
        dense_feas, sparse_feas, x_train, y_train, x_val, y_val, x_test, y_test, domain_num = get_ali_ccp_data_dict(
            dataset_path)

    dg = DataGenerator(x_train, y_train)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=x_val, y_val=y_val, x_test=x_test,
                                                                               y_test=y_test, batch_size=batch_size)
    if model_name == "star":
        model = Star(dense_feas + sparse_feas, domain_num, fcn_dims=[256, 128, 64, 32, 16, 8], aux_dims=[16])
    elif model_name == "Sharedbottom":
        model = SharedBottom(dense_feas + sparse_feas, domain_num, bottom_params={"dims": [512]},
                             tower_params={"dims": [256, 128, 64, 32, 16, 8]})
    elif model_name == "mmoe":
        model = MMOE(dense_feas + sparse_feas, domain_num, n_expert=domain_num,
                     expert_params={"dims": [256, 128, 64, 32, 16, 8]}, tower_params={"dims": [16]})
    elif model_name == "ple":
        model = PLE(dense_feas + sparse_feas, domain_num, n_level=1, n_expert_specific=2, n_expert_shared=1,
                    expert_params={"dims": [256, 128, 64, 32, 16, 8]}, tower_params={"dims": [16]})
    elif model_name == "adasparse":
        model = AdaSparse(sce_features=scenario_feas, agn_features=sparse_feas, form='Fusion', epsilon=1e-2, alpha=1.0,
                          delta_alpha=1e-4,
                          mlp_params={"dims": [256, 128, 64, 32, 16, 8], "dropout": 0.2, "activation": "relu"})
    elif model_name == "sarnet":
        model = Sarnet(sparse_feas, domain_num, domain_shared_expert_num=8, domain_specific_expert_num=2)
    elif model_name == "m2m":
        model = M2M(dense_feas + sparse_feas + scenario_feas, scenario_feas, domain_num, num_experts=4,
                    expert_output_size=16)
    elif model_name == "adaptdhm":
        model = AdaptDHM(features=sparse_feas + scenario_feas, fcn_dims=[256, 128, 64, 32, 16, 8], cluster_num=3, beta=0.9, device=device)
    elif model_name == "epnet":
        model = EPNet(sce_features=scenario_feas, agn_features=sparse_feas+dense_feas, fcn_dims=[256, 128, 64, 32, 16, 8])
    elif model_name == "ppnet":
        model = PPNet(id_features= id_feas, agn_features=sparse_feas+dense_feas+scenario_feas,domain_num= domain_num,fcn_dims=[256, 128, 64, 32, 16, 8])
    elif model_name == "m3oe":
        model = M3oE(features=dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[512, 256, 256, 64], expert_num=4, exp_d=1, exp_t=1, bal_d=1, bal_t=1, device=device)
    elif model_name == "hamur":
        model = HamurLarge(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128, 64, 64, 32, 16, 8], hyper_dims=[64],  k=65)
    ctr_trainer = CTRTrainer(model, dataset_name, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=5, device=device, model_path=save_dir, scheduler_params={"step_size": 4, "gamma": 0.95})
    ctr_trainer.fit(train_dataloader, val_dataloader)
    domain_logloss, domain_auc, logloss, auc = ctr_trainer.evaluate_multi_domain_loss(ctr_trainer.model,
                                                                                      test_dataloader, domain_num)
    print(f'test auc: {auc} | test logloss: {logloss}')
    for d in range(domain_num):
        print(f'test domain {d} auc: {domain_auc[d]} | test domain {d} logloss: {domain_logloss[d]}')

    import csv
    with open(model_name + "_" + dataset_name + "_" + str(seed) + '.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'seed', 'auc', 'log', 'auc0',
                         'log0', 'auc1', 'log1', 'auc2', 'log2'])
        writer.writerow([model_name, str(seed), auc, logloss,
                         domain_auc[0], domain_logloss[0],
                         domain_auc[1], domain_logloss[1],
                         domain_auc[2], domain_logloss[2]])


if __name__ == '__main__':
    import argparse
    import warnings

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data/ali-ccp")
    parser.add_argument('--model_name', default='star')
    parser.add_argument('--epoch', type=int, default=1)  # 100
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4096)  # 4096
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cpu')  # cuda:0
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay,
         args.device, args.save_dir, args.seed)
"""
python run_ali_ccp_ctr_ranking_multi_domain.py --model_name star
"""
