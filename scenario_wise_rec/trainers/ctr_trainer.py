import os
import time

import torch
import tqdm
from sklearn.metrics import roc_auc_score,log_loss
from ..basic.callback import EarlyStopper


class CTRTrainer(object):
    """a general trainer for single task learning.

    Args:
        model (nn.Module): any multitask learning model.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler): torch scheduling class, e.g., `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). if the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). note only save the best weight in the validation data.
    """

    def __init__(
        self,
        model,
        data_set_type,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
        model_path="./",
    ):
        self.model = model  # for uniform weights save method in one gpu or multi gpu
        self.data_set_type = data_set_type
        if gpus is None:
            gpus = []
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.device = torch.device(device)  #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  #default optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.criterion = torch.nn.BCELoss()  #default loss cross_entropy
        self.evaluate_fn = roc_auc_score  #default evaluate function
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path

    def train_one_epoch(self, data_loader, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (x_dict, y) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  #tensor to GPU
            y = y.to(self.device)
            y_pred = self.model(x_dict)
            loss = self.criterion(y_pred, y.float())
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    def fit(self, train_dataloader, val_dataloader=None):
        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            self.train_one_epoch(train_dataloader)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  #update lr in epoch level by scheduler
            if val_dataloader:
                auc,logloss = self.evaluate(self.model, val_dataloader)
                print(f'epoch:{epoch_i} | val auc: {auc} | val logloss: {logloss}')
                if self.early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f'validation: best auc: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break
        time_now = int(round(time.time() * 1000))
        time_now = time.strftime('%m_%d_%H_%M', time.localtime(time_now / 1000))
        name = self.model.__class__.__name__ + "_" +self.data_set_type+"_"+ time_now + ".pth"
        torch.save(self.model.state_dict(), os.path.join(self.model_path, name))  #save best auc model

    def evaluate(self, model, data_loader, mode = "val"):
        model.eval()
        targets, predicts = list(), list()
        desc_str= "validation"
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc=desc_str, smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model(x_dict)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        return self.evaluate_fn(targets, predicts),log_loss(targets,predicts)

    def evaluate_multi_domain_loss(self, model, data_loader,domain_num):
        model.eval()
        targets_all_list, predicts_all_list = list(), list()
        targets_domain_specific_list, predicts_domain_specific_list= list(), list()
        for i in range(domain_num):
            targets_domain_specific_list.append(list())
            predicts_domain_specific_list.append(list())

        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                domain_id = x_dict["domain_indicator"].clone().detach()

                y = y.to(self.device)
                y_pred = model(x_dict)

                targets_all_list.extend(y.tolist())
                predicts_all_list.extend(y_pred.tolist())

                for d in range(domain_num):
                    domain_mask_d = (domain_id == d)
                    y_d = y[domain_mask_d].tolist()
                    y_pred_d = y_pred[domain_mask_d].tolist()
                    targets_domain_specific_list[d].extend(y_d)
                    predicts_domain_specific_list[d].extend(y_pred_d)

        domain_logloss_list = list()
        domain_auc_list = list()

        for d in range(domain_num):
            domain_d_logloss_val = log_loss(targets_domain_specific_list[d], predicts_domain_specific_list[d]) if targets_domain_specific_list[d] else None
            domain_d_auc_val = self.evaluate_fn(targets_domain_specific_list[d], predicts_domain_specific_list[d]) if targets_domain_specific_list[d] else None
            domain_logloss_list.append(domain_d_logloss_val)
            domain_auc_list.append(domain_d_auc_val)

        total_logloss_val = log_loss(targets_all_list, predicts_all_list) if predicts_all_list else None
        total_auc_val = self.evaluate_fn(targets_all_list, predicts_all_list) if predicts_all_list else None

        return domain_logloss_list, domain_auc_list, total_logloss_val, total_auc_val


    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model(x_dict)
                predicts.extend(y_pred.tolist())
        return predicts
