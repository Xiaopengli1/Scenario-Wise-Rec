import sys
sys.path.append("..")
import torch
import pandas as pd
from tqdm import tqdm
from scenario_wise_rec.basic.features import DenseFeature, SparseFeature
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scenario_wise_rec.trainers import CTRTrainer
from scenario_wise_rec.utils.data import DataGenerator
from scenario_wise_rec.models.multi_domain import Star, MMOE, PLE, SharedBottom, AdaSparse, Sarnet, M2M, AdaptDHM, EPNet, PPNet, M3oE, HamurSmall


def get_movielens_data_rank_multidomain(data_path="data/ml-1m"):
    data = pd.read_csv(data_path+"/ml-1m.csv")
    data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
    del data["genres"]

    group1 = {1, 18}
    group2 = {25}
    group3 = {35, 45, 50, 56}

    domain_num = 3

    data["domain_indicator"] = data["age"].apply(lambda x: map_group_indicator(x, [group1, group2, group3]))

    useless_features = ['title', 'timestamp']

    dense_features = ['age']
    scenario_features = ['domain_indicator']
    sparse_features = ['user_id', 'movie_id', 'gender', 'occupation', 'zip', "cate_id", "domain_indicator"]
    target = "rating"

    for feature in dense_features:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_features:
        sca = MinMaxScaler()  # scaler dense feature
        data[dense_features] = sca.fit_transform(data[dense_features])

    for feature in useless_features:
        del data[feature]
    for feature in sparse_features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1

    data[target] = data[target].apply(lambda x: convert_target(x))

    for feat in tqdm(sparse_features):  # encode sparse feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name
                   in sparse_features]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_features]
    y = data[target]
    del data[target]

    return dense_feas, sparse_feas, scenario_feas, data, y, domain_num

def get_movielens_data_rank_multidomain(data_path="data/ml-1m"):
    data = pd.read_csv(data_path+"/ml-1m.csv")
    data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
    del data["genres"]

    group1 = {1, 18}
    group2 = {25}
    group3 = {35, 45, 50, 56}

    domain_num = 3

    data["domain_indicator"] = data["age"].apply(lambda x: map_group_indicator(x, [group1, group2, group3]))

    useless_features = ['title', 'timestamp']

    dense_features = ['age']
    scenario_features = ['domain_indicator']
    sparse_features = ['user_id', 'movie_id', 'gender', 'occupation', 'zip', "cate_id"]
    target = "rating"

    for feature in dense_features:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_features:
        sca = MinMaxScaler()  # scaler dense feature
        data[dense_features] = sca.fit_transform(data[dense_features])
    for feature in useless_features:
        del data[feature]
    for feat in tqdm(sparse_features):  # encode sparse feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    data[target] = data[target].apply(lambda x: convert_target(x))

    dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name
                   in sparse_features]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_features]
    y = data[target]
    del data[target]

    return dense_feas, sparse_feas, scenario_feas, data, y, domain_num


def get_movielens_data_rank_multidomain_ppnet(data_path="data/ml-1m"):
    data = pd.read_csv(data_path + "/ml-1m.csv")
    data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
    del data["genres"]

    group1 = {1, 18}
    group2 = {25}
    group3 = {35, 45, 50, 56}

    domain_num = 3

    data["domain_indicator"] = data["age"].apply(lambda x: map_group_indicator(x, [group1, group2, group3]))

    useless_features = ['title', 'timestamp']

    dense_features = ['age']
    scenario_features = ['domain_indicator']
    sparse_features = ['gender', 'occupation', 'zip', "cate_id"]
    id_features = ['user_id', 'movie_id']
    target = "rating"

    for feature in dense_features:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_features:
        sca = MinMaxScaler()  # scaler dense feature
        data[dense_features] = sca.fit_transform(data[dense_features])
    for feature in useless_features:
        del data[feature]
    for feat in tqdm(sparse_features):  # encode sparse feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    for feat in tqdm(id_features):  # encode sparse feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    data[target] = data[target].apply(lambda x: convert_target(x))

    dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name
                   in sparse_features]
    id_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name
                   in id_features]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_features]
    y = data[target]
    del data[target]

    return dense_feas, sparse_feas, id_feas, scenario_feas, data, y, domain_num

def map_group_indicator(age, list_group):
    l = len(list(list_group))
    for i in range(l):
        if age in list_group[i]:
            return i


def convert_target(val):
    v = int(val)
    if v > 3:
        return int(1)
    else:
        return int(0)


def convert_numeric(val):
    """
    Forced conversion
    """
    return int(val)


def df_to_dict_multi_domain(data, columns):
    """
    Convert the array to a dict type input that the network can accept
    Args:
        data (array): 3D datasets of type DataFrame (Length * Domain_num * feature_num)
        columns (list): feature name list
    Returns:
        The converted dict, which can be used directly into the input network
    """

    data_dict = {}
    for i in range(len(columns)):
        data_dict[columns[i]] = data[:, :, i]
    return data_dict





def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    torch.manual_seed(seed)
    dataset_name = "Movielens"
    if model_name == "ppnet":
        dense_feas, sparse_feas, id_feas, scenario_feas, x, y, domain_num = get_movielens_data_rank_multidomain_ppnet(dataset_path)
    else:
        dense_feas, sparse_feas, scenario_feas, x, y ,domain_num= get_movielens_data_rank_multidomain(dataset_path)
    dg = DataGenerator(x, y)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.8, 0.1], batch_size=batch_size)
    if model_name == "star":
        model = Star(dense_feas+sparse_feas,domain_num,fcn_dims =[128,64,32], aux_dims= [32])
    elif model_name == "sharedbottom":
        model = SharedBottom(dense_feas+sparse_feas, domain_num, bottom_params={"dims": [128]}, tower_params={"dims": [8]})
    elif model_name == "mmoe":
        model = MMOE(dense_feas+sparse_feas, domain_num, n_expert = domain_num, expert_params={"dims": [16]}, tower_params={"dims": [8]})
    elif model_name == "ple":
        model = PLE(dense_feas+sparse_feas, domain_num, n_level=1, n_expert_specific=2, n_expert_shared=1, expert_params={"dims": [16]}, tower_params={"dims": [8]})
    elif model_name == "adasparse":
        model = AdaSparse(sce_features = scenario_feas, agn_features=sparse_feas, form='Fusion', epsilon=1e-2, alpha=1.0, delta_alpha=1e-4, mlp_params={"dims": [32, 32], "dropout": 0.2, "activation": "relu"})
    elif model_name == "sarnet":
        model = Sarnet(features = sparse_feas, domain_num = domain_num, domain_shared_expert_num=8, domain_specific_expert_num = 2)
    elif model_name == "m2m":
        model = M2M(features=sparse_feas+scenario_feas, domain_feature = scenario_feas, domain_num=domain_num, num_experts = 4, expert_output_size = 16,transformer_dims={"num_encoder_layers":2, "num_decoder_layers":2, "dim_feedforward":16})
    elif model_name == "adaptdhm":
        model = AdaptDHM(features=sparse_feas+scenario_feas, fcn_dims=[64, 64], cluster_num=3, beta=0.9, device=device)
    elif model_name == "epnet":
        model = EPNet(sce_features=scenario_feas, agn_features=sparse_feas+dense_feas, fcn_dims=[128,64,32])
    elif model_name == "ppnet":
        model = PPNet(id_features= id_feas, agn_features=sparse_feas+dense_feas+scenario_feas,domain_num= domain_num,fcn_dims=[128,64,32])
    elif model_name == "m3oe":
        model = M3oE(features = dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[128, 64, 64, 32] , expert_num=4, exp_d=1, exp_t=1, bal_d=1, bal_t=1, device=device)
    elif model_name == "hamur":
        model = HamurSmall(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128], hyper_dims=[64], k=35)
    ctr_trainer = CTRTrainer(model, dataset_name, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=4, device=device, model_path=save_dir,scheduler_params={"step_size": 2,"gamma": 0.85})
    ctr_trainer.fit(train_dataloader, val_dataloader)
    domain_logloss,domain_auc,logloss,auc = ctr_trainer.evaluate_multi_domain_loss(ctr_trainer.model, test_dataloader,domain_num)
    print(f'test auc: {auc} | test logloss: {logloss}')
    for d in range(domain_num):
        print(f'test domain {d} auc: {domain_auc[d]} | test domain {d} logloss: {domain_logloss[d]}')

    import csv
    with open(model_name+"_"+dataset_name+"_"+str(seed)+'.csv', "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'seed', 'auc', 'log', 'auc0',
                         'log0', 'auc1', 'log1', 'auc2', 'log2'])
        writer.writerow([model_name, str(seed), auc, logloss,
                         domain_auc[0], domain_logloss[0],
                         domain_auc[1], domain_logloss[1],
                         domain_auc[2], domain_logloss[2]])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data/ml-1m")
    parser.add_argument('--model_name', default='star')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4096)  #4096
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cpu')  #cuda:0
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay,
         args.device, args.save_dir, args.seed)
"""
python run_movielens_rank_multi_domain.py --model_name star
"""