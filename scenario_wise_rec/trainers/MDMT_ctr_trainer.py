import os
import numpy as np
import pandas as pd
import torch
import tqdm, gc
from sklearn.metrics import roc_auc_score, log_loss
from ..basic.callback import EarlyStopper


class CTRTrainer(object):
    """A general trainer for single task learning.

    Args:
        model (nn.Module): any multi task learning model.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
    """

    def __init__(
            self,
            model,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=None,
            optimizer_params_darts=None,
            scheduler_fn=None,
            scheduler_params=None,
            n_epoch=10,
            earlystop_patience=10,
            device="cpu",
            gpus=None,
            model_path="./",
    ):
        self.model = model  # for uniform weights save method in one gpu or multi gpu
        if gpus is None:
            gpus = []
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.device = torch.device(device)  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        # darts_para_flag = lambda name: name.startswith('loss')
        darts_para_flag = lambda name: name.startswith('_weight_')
        self.optimizer = optimizer_fn(
            [param for name, param in self.model.named_parameters() if ~ darts_para_flag(name)],
            **optimizer_params)  # default optimizer
        self.optimizer_loss_weight = optimizer_fn(
            [param for name, param in self.model.named_parameters() if darts_para_flag(name)],
            **optimizer_params_darts)  # loss weight optimizer

        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.criterion = torch.nn.BCELoss()  # default loss cross_entropy
        self.evaluate_fn = roc_auc_score  # default evaluate function
        self.evaluate_fn_logloss = log_loss
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path

    def train_one_epoch(self, data_loader, domain_num, task_num, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)

        targets, predicts = list(), list()
        # target_list: [task_num, domain_num, [list of ground truth]]
        # predict_list: [task_num, domain_num, [list of prediction]]
        targets_list = [[[] for j in range(domain_num)] for i in range(task_num)]
        predict_list = [[[] for j in range(domain_num)] for i in range(task_num)]
        # [[[d0t0], [d1t0], [d2t0]], [[d0t1], [d1t1], [d2t1]]] T*D
        for i, (x_dict, y) in enumerate(tk0):
            x_dict, y = x_dict.transpose(0, 1).to(self.device), y.to(self.device)
            y_pred = self.model(x_dict)
            y_pred = y_pred.to(self.device)

            if not (domain_num == 3 and task_num == 2):
                loss = self.criterion(y_pred, y.float())
            else:
                # mtmd setting
                domain_mask_list = []
                domain_id = x_dict[-1, :].clone().detach()

                for d in range(domain_num):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                for t in range(task_num):
                    for d in range(domain_num):
                        # different domain&task has different size
                        targets_list[t][d] = y[domain_mask_list[d], t]
                        predict_list[t][d] = y_pred[domain_mask_list[d], t]

                if y.shape[-1] != 1:
                    targets.extend(y.tolist())
                    predicts.extend(y_pred.tolist())
                else:
                    targets.extend(y.squeeze(1).tolist())
                    predicts.extend(y_pred.squeeze(1).tolist())
                logloss_list = list()
                for t in range(task_num):
                    for d in range(domain_num):
                        logloss_list.append(self.criterion(predict_list[t][d], targets_list[t][d].float()))

                # t0d0, t0d1, t0d2, t1d0, t1d1, t1d2
                # loss = sum(self.model.loss_weight() * torch.stack(logloss_list))
                loss = sum(torch.stack(logloss_list))

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    def fit(self, train_dataloader, exp_d, exp_t, bal_d, bal_t, domain_num, task_num, val_dataloader=None):
        valid_data_iter = iter(train_dataloader)
        for epoch_i in range(self.n_epoch):
            self.train_one_epoch(train_dataloader, domain_num=domain_num, task_num=task_num)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  # update lr in epoch level by scheduler
            # update loss_weight with a batch of validation data
            try:
                _val_x, _val_y = next(valid_data_iter)
            except StopIteration:
                del valid_data_iter
                gc.collect()
                valid_data_iter = iter(train_dataloader)
                _val_x, _val_y = next(valid_data_iter)

            _val_x, _val_y = _val_x.transpose(0, 1).to(self.device), _val_y.to(self.device)
            _y_pred = self.model(_val_x).to(self.device)
            _loss = self.criterion(_y_pred, _val_y.float())
            self.model.zero_grad()
            _loss.backward()
            self.optimizer_loss_weight.step()
            # print(f'loss weight: {_loss}, {[round(i.item(), 4) for i in self.model.loss_weight()]}')

            logloss_list, auc_list, _, _ = self.evaluate_multi_domain(self.model, val_dataloader, domain_num=domain_num,
                                                                      task_num=task_num)
            auc = sum(auc_list) / (domain_num * task_num)
            log_loss = sum(logloss_list) / (domain_num * task_num)
            # auc, log_loss = self.evaluate(self.model, val_dataloader, domain_num=domain_num, task_num=task_num)
            print('epoch:', epoch_i, 'validation: auc:', auc, 'log loss:', log_loss)
            if self.early_stopper.stop_training(auc, self.model.state_dict()):
                print(f'validation: best auc: {self.early_stopper.best_auc}')
                self.model.load_state_dict(self.early_stopper.best_weights)
                break

        torch.save(self.model.state_dict(), self.model_path + "model.pth")  # save best auc model

    def evaluate(self, model, data_loader, domain_num, task_num):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict, y = x_dict.transpose(0, 1).to(self.device), y.to(self.device)
                y_pred = model(x_dict)
                if y.shape[-1] != 1:
                    targets.extend(y.tolist())
                    predicts.extend(y_pred.tolist())
                else:
                    targets.extend(y.squeeze(1).tolist())
                    predicts.extend(y_pred.squeeze(1).tolist())
        return self.evaluate_fn(targets, predicts), log_loss(targets, predicts)

    def evaluate_multi_domain(self, model, data_loader, domain_num, task_num):
        model.eval()
        targets, predicts = list(), list()
        # target_list: [task_num, domain_num, [list of ground truth]]
        # predict_list: [task_num, domain_num, [list of prediction]]
        targets_list = [[[] for j in range(domain_num)] for i in range(task_num)]
        predict_list = [[[] for j in range(domain_num)] for i in range(task_num)]
        # [[[d0t0], [d1t0], [d2t0]], [[d0t1], [d1t1], [d2t1]]] T*D
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            # gate_values = list()
            for i, (x_dict, y) in enumerate(tk0):
                domain_mask_list = []

                x_dict, y = x_dict.transpose(0, 1).to(self.device), y.to(self.device)
                domain_id = x_dict[-1, :].clone().detach()

                y_pred = model(x_dict)
                # y_pred, _gate = model(x_dict, test_flag=True)
                y_pred = y_pred.to(self.device)
                # gate_values.append(_gate)

                for d in range(domain_num):
                    domain_mask = (domain_id == d)
                    domain_mask_list.append(domain_mask)

                for t in range(task_num):
                    for d in range(domain_num):
                        y_ = y[domain_mask_list[d], t].tolist()
                        y_pred_ = y_pred[domain_mask_list[d], t].tolist()
                        targets_list[t][d].extend(y_)
                        predict_list[t][d].extend(y_pred_)

                if y.shape[-1] != 1:
                    targets.extend(y.tolist())
                    predicts.extend(y_pred.tolist())
                else:
                    targets.extend(y.squeeze(1).tolist())
                    predicts.extend(y_pred.squeeze(1).tolist())
                    # gate_values = pd.concat(gate_values)
            # gate_values.to_csv('gate.csv', index=False)
            # print(f'saved to gate.csv with shape of {gate_values.shape}.')
        logloss_list, auc_list = list(), list()
        for t in range(task_num):
            for d in range(domain_num):
                logloss_list.append(log_loss(targets_list[t][d], predict_list[t][d]))
                auc_list.append(self.evaluate_fn(targets_list[t][d], predict_list[t][d]))

        total_logloss = log_loss(targets, predicts) if predicts else None

        total_auc = self.evaluate_fn(targets, predicts) if predicts else None
        return logloss_list, auc_list, total_logloss, total_auc

    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict, y = x_dict.transpose(0, 1).to(self.device), y.to(self.device)
                y_pred = model(x_dict)
                predicts.extend(y_pred.tolist())
        return predicts

