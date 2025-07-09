import os.path

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.optim as optim
import pickle
import math
import matplotlib.pyplot as plt
from collections import Counter

from intervention import SessionIntervention
from metric import metric,bucketing_metric
import configs
import backbones
import dataloader


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

gpus = [7]
default_opt = getattr(configs,'Defaultargs')().get_args()
save_dirpath = 'checkpoints/'


class ModelTrainer(pl.LightningModule):

    def __init__(self, n_node,param=None):
        super().__init__()
        self.best_res = [0, 0, 0]
        self.default_opt = default_opt
        self.model_opt = getattr(configs,self.default_opt.model+'args')().get_args()
        self.model = getattr(backbones, self.default_opt.model)(self.default_opt,self.model_opt,n_node)
        self.param = param
        self.loss_function = nn.CrossEntropyLoss()
        self.loss = nn.Parameter(torch.Tensor(1))
        print(self.default_opt)
        print(self.model_opt)
        with open('data/' + default_opt.dataset + '/Dict_item_pre_popularity.pkl', 'rb') as f:
            dict_item_topitems = pickle.load(f)
        self.transform = SessionIntervention(dict_item_topitems)
        self.bucketing_hit5 = [0] * 20
        self.bucketing_mrr5 = [0] * 20
        self.bucketing_hit10 = [0] * 20
        self.bucketing_mrr10 = [0] * 20
        self.bucketing_hit20 = [0] * 20
        self.bucketing_mrr20 = [0] * 20
        self.bucketing_count = [0] * 20
        self.bucketing_session_longer_len = [0] * 20
        self.bucketing_session_longer_count = [0] * 20
        self.all_session_longer_len = []

    def forward(self, *args):
        return self.model(*args)


    def sigmoid(self,tensor, temp=1.0):
        """ temperature controlled sigmoid
        takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        exponent = -tensor / temp
        # clamp the input tensor for stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y


    def RBOLoss(self,scores,scores_ext,sigmoid_temp):
        p = 0.8
        q = 10
        topk = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        nll = []
        for i, k in enumerate(topk):
            kth_value = torch.topk(scores, k=k, dim=1)[0][:, -1]
            diff_scores = scores - kth_value.unsqueeze(1)
            sg_scores = self.sigmoid(diff_scores, temp=sigmoid_temp)
            kth_value_ext = torch.topk(scores_ext, k=k, dim=1)[0][:, -1]
            diff_scores_ext = scores_ext - kth_value_ext.unsqueeze(1)
            sg_scores_ext = self.sigmoid(diff_scores_ext, temp=sigmoid_temp)

            inter_MM = torch.matmul(sg_scores, sg_scores.T)
            inter_MMe = torch.matmul(sg_scores, sg_scores_ext.T)
            inter_MeM = torch.matmul(sg_scores_ext, sg_scores.T)
            inter_MeMe = torch.matmul(sg_scores_ext, sg_scores_ext.T)
            intersection = torch.cat(
                (torch.cat((inter_MM, inter_MMe), dim=1), torch.cat((inter_MeM, inter_MeMe), dim=1)),
                dim=0) / k
            self_mask = torch.eye(intersection.shape[0], dtype=torch.bool, device=intersection.device)
            intersection.masked_fill_(self_mask, -9e15)
            pos_mask = self_mask.roll(shifts=intersection.shape[0] // 2, dims=0)

            nll_k = -intersection[pos_mask] + torch.logsumexp(intersection, dim=-1) # Contrastive learning Loss
            nll.append(nll_k.mean())
        ssl_loss = sum(nll) / len(nll)
        return ssl_loss

    def training_step(self, batch, batch_idx):
        beta_ListSim = 0.1
        sigmoid_temp = 10
        mul_len = int(1/4 * self.current_epoch)

        targets, session, lens = batch
        feat_session, feat_test_items = self.model(session, lens)
        scores = torch.matmul(feat_session, feat_test_items)
        if default_opt.model == 'CORE':
            scores = scores / self.model_opt.temperature
        if default_opt.model == 'AttMix':
            scores = 16 * scores


        if batch_idx == 0:
            self.bucketing_session_longer_len = [0] * 20
            self.bucketing_session_longer_count = [0] * 20
            self.all_session_longer_len = []
        session_longer, session_longer_lens, self.bucketing_session_longer_len, self.bucketing_session_longer_count, self.all_session_longer_len = self.transform(
            session, mul_len, n_node, self.bucketing_session_longer_len, self.bucketing_session_longer_count, self.all_session_longer_len)
        feat_session_longer, _ = self.model(session_longer.to(session.device), session_longer_lens.to(session.device))
        scores_session_longer = torch.matmul(feat_session_longer, feat_test_items)
        if default_opt.model == 'CORE':
            scores_session_longer = scores_session_longer / self.model_opt.temperature
        if default_opt.model == 'AttMix':
            scores_session_longer = 16 * scores_session_longer

        loss_org_longer = self.RBOLoss(scores,scores_session_longer,sigmoid_temp)

        loss_org_gt = self.loss_function(scores, targets)

        loss = loss_org_gt + beta_ListSim * loss_org_longer

        return loss

    def validation_step(self, batch, batch_idx):
        targets, session,lens = batch
        feat_session, feat_test_items = self.model(session,lens)
        scores = torch.matmul(feat_session, feat_test_items)
        if default_opt.model == 'CORE':
            scores = scores / self.model_opt.temperature
        if default_opt.model == 'AttMix':
            scores = 16 * scores
        res_list = []
        for k in [5,10,20]:
            res_list.append(torch.tensor(metric(scores,targets,k)))
        res = torch.cat(res_list,dim=1)
        return res

    def validation_epoch_end(self, validation_step_outputs):

        output = torch.cat(validation_step_outputs, dim=0)
        hit5 = torch.mean(output[:, 0]) * 100
        mrr5 = torch.mean(output[:, 1]) * 100
        ndcg5 = torch.mean(output[:, 2]) * 100
        hit10 = torch.mean(output[:, 3]) * 100
        mrr10 = torch.mean(output[:, 4]) * 100
        ndcg10 = torch.mean(output[:, 5]) * 100
        hit20 = torch.mean(output[:, 6]) * 100
        mrr20 = torch.mean(output[:, 7]) * 100
        ndcg20 = torch.mean(output[:, 8]) * 100
        if hit20 > self.best_res[0]:
            self.best_res[0] = hit20
        if mrr20 > self.best_res[1]:
            self.best_res[1] = mrr20
        if ndcg20 > self.best_res[2]:
            self.best_res[2] = ndcg20
        self.log('hit@20', self.best_res[0])
        self.log('mrr@20', self.best_res[1])
        self.log('ndcg@20', self.best_res[2])
        msg = ' \n Top-{} acc:{:.3f}, mrr:{:.3f}, ndcg:{:.3f} \n'.format(5, hit5, mrr5, ndcg5)
        msg += 'Top-{} acc:{:.3f}, mrr:{:.3f}, ndcg:{:.3f} \n'.format(10, hit10, mrr10, ndcg10)
        msg += 'Top-{} acc:{:.3f}, mrr:{:.3f}, ndcg:{:.3f} \n'.format(20, hit20, mrr20, ndcg20)
        print("EPOCH:",self.current_epoch)
        self.print(msg)
        return mrr20

    def test_step(self, batch, idx):
        targets, session,lens = batch
        feat_session, feat_test_items = self.model(session,lens)
        scores = torch.matmul(feat_session, feat_test_items)
        if default_opt.model == 'CORE':
            scores = scores / self.model_opt.temperature
        if default_opt.model == 'AttMix':
            scores = 16 * scores
        res_list = []
        for k in [5,10,20]:
            res_list.append(torch.tensor(metric(scores,targets,k)))
        res = torch.cat(res_list,dim=1)
        self.bucketing_hit5, self.bucketing_mrr5, self.bucketing_hit10, self.bucketing_mrr10,self.bucketing_hit20, self.bucketing_mrr20, self.bucketing_count = \
            bucketing_metric(scores, targets, lens, self.bucketing_hit5, self.bucketing_mrr5, self.bucketing_hit10,self.bucketing_mrr10, self.bucketing_hit20, self.bucketing_mrr20,self.bucketing_count)
        return res

    def test_epoch_end(self, test_step_outputs):
        output = torch.cat(test_step_outputs, dim=0)
        hit5 = torch.mean(output[:, 0]) * 100
        mrr5 = torch.mean(output[:, 1]) * 100
        ndcg5 = torch.mean(output[:, 2]) * 100
        hit10 = torch.mean(output[:, 3]) * 100
        mrr10 = torch.mean(output[:, 4]) * 100
        ndcg10 = torch.mean(output[:, 5]) * 100
        hit20 = torch.mean(output[:, 6]) * 100
        mrr20 = torch.mean(output[:, 7]) * 100
        ndcg20 = torch.mean(output[:, 8]) * 100
        msg = ' \n Top-{} acc:{:.3f}, mrr:{:.3f}, ndcg:{:.3f} \n'.format(5, hit5, mrr5, ndcg5)
        msg += 'Top-{} acc:{:.3f}, mrr:{:.3f}, ndcg:{:.3f} \n'.format(10, hit10, mrr10, ndcg10)
        msg += 'Top-{} acc:{:.3f}, mrr:{:.3f}, ndcg:{:.3f} \n'.format(20, hit20, mrr20, ndcg20)
        self.print(msg)

        return mrr20

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.default_opt.lr, weight_decay=self.default_opt.l2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.default_opt.lr_dc_step, gamma=self.default_opt.lr_dc)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


if __name__ == "__main__":
    pl.seed_everything(default_opt.seed)
    if default_opt.dataset == 'Diginetica':
        n_node = 42596 + 1
    elif default_opt.dataset == 'Nowplaying':
        n_node = 60416 + 1
    elif default_opt.dataset == 'Tmall':
        n_node = 40727 + 1



    early_stop_callback = EarlyStopping(
        monitor='mrr@20',
        min_delta=0.00,
        patience=default_opt.patience,
        verbose=False,
        mode='max'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='mrr@20',
        dirpath=save_dirpath,
        filename=default_opt.dataset + '-' + default_opt.model,
        save_top_k=1,
        mode='max')
    trainer = pl.Trainer(gpus=gpus, deterministic=True, max_epochs=default_opt.epoch, num_sanity_val_steps=2,
                         replace_sampler_ddp=False,
                         callbacks=[early_stop_callback, checkpoint_callback],
                         progress_bar_refresh_rate=0)
    if default_opt.opt_train:
        train_loader, valid_loader, test_loader = getattr(dataloader, default_opt.model + 'Data')(
            dataset=default_opt.dataset, batch_size=default_opt.batch_size).get_loader(opt_train=default_opt.opt_train)
        model = ModelTrainer(n_node=n_node)
        trainer.fit(model, train_loader, valid_loader)
    else:
        test_loader = getattr(dataloader, default_opt.model + 'Data')(
            dataset=default_opt.dataset, batch_size=default_opt.batch_size).get_loader(opt_train=default_opt.opt_train)
        checkpoint = save_dirpath + default_opt.dataset + '-' + default_opt.model + '.ckpt'
        print(checkpoint)
        model = ModelTrainer.load_from_checkpoint(checkpoint,n_node=n_node)
        model.eval()
        print('Testing')
        trainer.test(model, test_loader)
