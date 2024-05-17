import sys
sys.path.append("..")
import torch
import pandas as pd
from scenario_wise_rec.trainers import CTRTrainer
from scenario_wise_rec.basic.features import DenseFeature, SparseFeature
from scenario_wise_rec.utils.data import DataGenerator, reduce_mem_usage
from scenario_wise_rec.models.multi_domain import Star, SharedBottom, MMOE, PLE, AdaSparse, Sarnet, M2M, AdaptDHM


def get_example_dataset(data_path):
    # Step1: Read Data
    data = pd.read_csv(data_path)

    # Step2: Data preprocess
    #
    #

    # Step3: domain_indicator specification
    # Be noted that the domain_indicator starts from 0.
    data["domain_indicator"] = data["x"]
    domain_num = data["domain_indicator"].nunique()

    # Step4: Features type specification
    dense_cols = []
    sparse_cols = []
    scenario_cols = ['domain_indicator']

    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in sparse_cols]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_cols]

    # Step5: Target feature Specification
    y = data["y"]
    del data["y"]

    return dense_feas, sparse_feas, scenario_feas, data, y, domain_num

def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    torch.manual_seed(seed)
    dataset_name = "Your dataset name"
    dense_feas, sparse_feas, scenario_feas, x, y ,domain_num= get_example_dataset(dataset_path)
    dg = DataGenerator(x, y)
    # split training/validation/text dataset, default 8:1:1
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.8, 0.1], batch_size=batch_size)
    if model_name == "star":
        model = Star(dense_feas+sparse_feas,domain_num,fcn_dims =[128,64,32], aux_dims= [32])
    elif model_name == "SharedBottom":
        model = SharedBottom(dense_feas+sparse_feas, domain_num, bottom_params={"dims": [128]}, tower_params={"dims": [8]})
    elif model_name == "MMOE":
        model = MMOE(dense_feas+sparse_feas, domain_num, n_expert = domain_num, expert_params={"dims": [16]}, tower_params={"dims": [8]})
    elif model_name == "PLE":
        model = PLE(dense_feas+sparse_feas, domain_num, n_level=1, n_expert_specific=2, n_expert_shared=1, expert_params={"dims": [16]}, tower_params={"dims": [8]})
    elif model_name == "adasparse":
        model = AdaSparse(sce_features = scenario_feas, agn_features=sparse_feas, form='Fusion', epsilon=1e-2, alpha=1.0, delta_alpha=1e-4, mlp_params={"dims": [32, 32], "dropout": 0.2, "activation": "relu"})
    elif model_name == "sarnet":
        model = Sarnet(features = sparse_feas, domain_num = domain_num, domain_shared_expert_num=8, domain_specific_expert_num = 2)
    elif model_name == "m2m":
        model = M2M(features=sparse_feas+scenario_feas, domain_feature = scenario_feas, domain_num=domain_num, num_experts = 4, expert_output_size = 16,transformer_dims={"num_encoder_layers":2, "num_decoder_layers":2, "dim_feedforward":16})
    elif model_name == "adaptdhm":
        model = AdaptDHM(features=sparse_feas+scenario_feas, fcn_dims=[64, 64], cluster_num=3, beta=0.9, device=device)
    ctr_trainer = CTRTrainer(model, dataset_name, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=4, device=device, model_path=save_dir,scheduler_params={"step_size": 2,"gamma": 0.85})
    #scheduler_fn=torch.optim.lr_scheduler.StepLR,scheduler_params={"step_size": 2,"gamma": 0.8},
    ctr_trainer.fit(train_dataloader, val_dataloader)
    domain_logloss,domain_auc,logloss,auc = ctr_trainer.evaluate_multi_domain_loss(ctr_trainer.model, test_dataloader,domain_num)
    print(f'test auc: {auc} | test logloss: {logloss}')
    for d in range(domain_num):
        print(f'test domain {d} auc: {domain_auc[d]} | test domain {d} logloss: {domain_logloss[d]}')

    # If you want to save logs in csv, modify the following code to save csv files.
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
    import warnings
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data/xxx")
    parser.add_argument('--model_name', default='star')
    parser.add_argument('--epoch', type=int, default=1)  #100
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=100)  #4096
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cpu')  #cuda:0
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed)
"""
python run_example.py.py --model_name stra
"""