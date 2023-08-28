import torch
import numpy as np
import logging
import pandas as pd
import os
from utils.global_p import *
from data_processors.DataProcessor import DataProcessor
import random
from tqdm import tqdm
import copy
from utils import utils
import datetime
from sklearn.utils import shuffle



class nonSeqCTFGenerator(DataProcessor):
    def parse_ctf_dp_args(parser):
        parser.add_argument('--ctf_num', type=int, default=1,
                            help='The number of counterfactual history for each sample')
        parser.add_argument('--ctf-type', type=str, default='R1R',
                            help='The type of counterfactual samples')
        parser.add_argument('--topk', type=int, default=50,
                            help='Append all history in the training set')
        parser.add_argument('--can_item', type=int, default=100,
                            help='The number of candidate items during generate ctf history, -1 means use all items')
        return parser
    
    def __init__(self, ctf_num, ctf_type, topk, can_item, *args, **kwargs):
        self.ctf_num = ctf_num
        self.ctf_type = ctf_type
        self.topk = topk
        self.can_item = can_item
        DataProcessor.__init__(self, *args, **kwargs)
        self.all_item = [i for i in range(1, self.data_loader.item_num)]
        
    def get_train_data(self, epoch, model, runner, processor):
        """
        convert training dataframe in dataloader into dict, shuffle every epoch
        the dict will be used for generating batches
        :param epoch: no shuffle if <0
        :param model: Model
        :return: dict
        """
        if self.train_data is None:
            logging.info('Prepare Train Data...')
            new_train_df = self.get_ctf_train_df(self.data_loader.train_df, model, runner, processor)
            self.train_data = self.format_data_dict(new_train_df, model)
            self.train_data[SAMPLE_ID] = np.arange(0, len(self.train_data[Y]))
        if epoch >= 0:
            utils.shuffle_in_unison_scary(self.train_data)
        return self.train_data
    
    def get_ctf_train_df(self, train_df, model, runner, processor):
        logging.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        file_name = os.path.join(self.data_loader.path, '{}_{}_ctftype{}_topk{}.csv'.format(self.data_loader.dataset, model.__class__.__name__, 
                                                                                                     self.ctf_type, self.topk))
        if os.path.exists(file_name):
            logging.info('load counterfactual history from ' + file_name)
            new_df = pd.read_csv(file_name, sep = '\t')
            return new_df
        users = train_df[UID].unique()
        self.user2id = dict()
        self.id2user = dict()
        for user in users:
            self.id2user[len(self.user2id)] = user
            self.user2id[user] = len(self.user2id)
        unum = len(users)
        ctf_indicator = np.zeros(unum)
        self.pre = [[] for i in range(unum)]
        new_train_df = copy.deepcopy(train_df)
        ctf_gen_model = copy.deepcopy(model)
        ctf_gen_model.model_path = os.path.join(MODEL_DIR, '%s/%s_%s.pt' % (model.__class__.__name__, processor.data_loader.dataset, model.__class__.__name__))
        val_df = shuffle(copy.deepcopy(train_df)).groupby(UID).head(1).reset_index(drop=True)
        runner._check_time(start=True)
        
        count = 0
        while np.sum(ctf_indicator) < len(users):
            count += 1
            remain_user = np.where(np.ones(unum) - ctf_indicator == 1)
#             remain_user = remain_user + np.ones(len(remain_user),dtype=int)
            ctf_df = self.generate_ctf_df(train_df, remain_user, model)
            self.data_loader.train_df = ctf_df
            logging.info("retrain the model to get ctf_data: {}/{}".format(int(np.sum(ctf_indicator)), len(users)))
            ctf_gen_model.apply(model.init_paras)
            if torch.cuda.device_count() > 0:
                # model = model.to('cuda:0')
                ctf_gen_model = ctf_gen_model.cuda()
            processor.__init__(processor.data_loader, processor.rank, processor.train_sample_n, processor.test_sample_n, processor.sample_un_p, processor.unlabel_test)
            processor.get_train_data(epoch=-1, model=model)
            processor.get_validation_data(model=model)
            processor.get_test_data(model=model)
            runner.__init__(runner.optimizer_name, runner.lr, 50, runner.batch_size, runner.eval_batch_size,
                 runner.dropout, runner.l2_weight, runner.l2s_weight, runner.l2_bias,
                 runner.grad_clip, runner.metrics_str, runner.check_epoch, runner.early_stop, runner.pre_gpu)
            runner.train(ctf_gen_model, processor)
            val_data = self.construct_val_data(train_df, users, ctf_gen_model)
            
            predictions = runner.predict(ctf_gen_model, val_data, processor)

            predictions = np.transpose(predictions.reshape(-1,len(users)))
            result = predictions - predictions[:,0][:,None]
            rank = np.where(result > 0, 1, 0).sum(axis=1)
            indicator = np.where(rank < self.topk)
            new_train_df, ctf_indicator = self.update_df(new_train_df, remain_user, indicator, ctf_indicator)
            if count >= 20:
                logging.info("Keep original history for those users without satisfied counterfactual examples at top {}".format(self.topk))
                break

        logging.info("The counterfactual data has been generated [%.1f s]" % (runner._check_time()))
        pre_lens = [len(row) for row in self.pre]
        logging.info('The largest number of samples: {}'.format(str(max(pre_lens))))
        new_train_df.to_csv(file_name, sep = '\t', index=False)
        logging.info('save the counterfactual history to ' + file_name)
        return new_train_df
    
    def construct_val_data(self, train_df, users, model):
        val_df = copy.deepcopy(train_df)
        val_df = val_df.rename(columns={self.data_loader.label: Y})
        val_df = val_df.drop(val_df[val_df[Y] <= 0].index)
        for user in users:
            uid = self.user2id[user]
            if self.ctf_type in {'R1R', 'R1N', 'D1'}:
                if self.ctf_type == 'D1':
                    iid = self.pre[uid][-1]
                    if len(train_df[train_df[UID]==user]) == 1:
                        continue
                else:
                    (iid, r_iid) = self.pre[self.user2id[user]][-1]
                val_df = val_df.loc[(val_df[UID] != user) | ((val_df[UID] == user) & (val_df[IID] == iid))].reset_index(drop=True)
            elif self.ctf_type == 'K1':
                iid = self.pre[uid][-1]
                if len(train_df[train_df[UID]==user]) == 1:
                    continue
                val_df = val_df.drop(val_df[(val_df[UID] == user) & (val_df[IID] == iid)].index)
            else:
                logging.info('No Such Heuristic Rule')
                exit()
        val_df = shuffle(val_df).groupby(UID).head(1).reset_index(drop=True)
        neg_df = DataProcessor.generate_neg_df(self,inter_df=val_df, feature_df=train_df, sample_n=self.can_item, train=False)
        df = pd.concat([val_df, neg_df], ignore_index=True, sort=False)
        val_data = DataProcessor.format_data_dict(self, df, model)
        val_data[SAMPLE_ID] = np.arange(0, len(val_data[Y]))
        return val_data
            
    def generate_ctf_df(self, train_df, remain_user, model):
        ctf_df = copy.deepcopy(train_df)
        if self.ctf_type == 'R1N':
            if model.__class__.__name__ == 'NCF':
                similarity = torch.matmul(model.gmf_iid_embeddings.weight.data,\
                                      torch.transpose(model.gmf_iid_embeddings.weight.data,0,1)).detach().cpu().numpy()
            else:
                similarity = torch.matmul(model.iid_embeddings.weight.data,\
                                      torch.transpose(model.iid_embeddings.weight.data,0,1)).detach().cpu().numpy()
            indices_all = np.argsort(similarity, axis=1)
            indices_all = np.array([idx[::-1] for idx in indices_all])
        for user, group in ctf_df.groupby(UID):
            uid = self.user2id[user]
            if uid in remain_user[0]:
                if self.ctf_type == 'R1R':
                    while True:
                        r_iid = random.randint(1, len(self.all_item))
                        iids = group[IID].values
                        index = random.randint(0, len(group) - 1)
                        if (iids[index], r_iid) not in self.pre[uid]:
                            self.pre[uid].append((iids[index], r_iid))
                            ctf_df.loc[(ctf_df[UID]==user) & (ctf_df[IID]==iids[index]), IID] = r_iid
                            break
                        if len(self.pre[uid]) > 20:
                            logging.info("No satisfiable counterfactual history for R1R at {} for user {}".format(self.topk, user))
                            break
                elif self.ctf_type == 'R1N':
                    iids = group[IID].values
                    index = random.randint(0, len(group) - 1)
                    indices = indices_all[iids[index]]
                    idx = 0
                    while True:
                        if indices[idx] in {0, iids[index]}:
                            idx += 1
                            continue
                        r_iid = indices[idx]
                        if (iids[index], r_iid) not in self.pre[uid]:
                            self.pre[uid].append((iids[index], r_iid))
                            ctf_df.loc[(ctf_df[UID]==user) & (ctf_df[IID]==iids[index]), IID] = r_iid
                            break
                        idx += 1
                        if len(self.pre[uid]) > 20:
                            logging.info("No satisfiable counterfactual history for R1N at {} for user {}".format(self.topk, user))
                            break
                elif self.ctf_type == 'D1':
                    iids = group[IID].values
                    while True:
                        index = random.randint(0, len(group) - 1)
                        d_iid = iids[index]
                        if d_iid not in self.pre[uid]:
                            self.pre[uid].append(d_iid)
                            ctf_df = ctf_df.loc[(ctf_df[UID] != user) | (ctf_df[IID] != d_iid)].reset_index(drop=True)
                            break
                        if len(self.pre[uid]) >= len(iids):
                            logging.info("No satisfiable counterfactual history for D1 at {} for user {}".format(self.topk, user))
                            break
                elif self.ctf_type == 'K1':
                    iids = group[IID].values
                    while True:
                        index = random.randint(0, len(group) - 1)
                        d_iid = iids[index]
                        if d_iid not in self.pre[uid]:
                            self.pre[uid].append(d_iid)
                            ctf_df = ctf_df.loc[(ctf_df[UID] != user) | ((ctf_df[UID] == user) & (ctf_df[IID] == d_iid))].reset_index(drop=True)
                            break
                        if len(self.pre[uid]) >= len(self.pre[uid]) + 1:
                            logging.info("No satisfiable counterfactual history for K1 at {} for user {}".format(self.topk, user))
                            break
                    
                    
        return ctf_df
    
    def update_df(self, new_train_df, remain_user, indicator, ctf_indicator):
        for i in indicator[0]:
            if i in remain_user[0]:
                ctf_indicator[i] = 1
                user = self.id2user[i]
                if self.ctf_type == 'R1R' or self.ctf_type == 'R1N':
                    (iid, r_iid) = self.pre[i][-1]
                    new_train_df.loc[(new_train_df[UID]==user) & (new_train_df[IID]==iid), IID] = r_iid
                elif self.ctf_type == 'D1':
                    d_iid = self.pre[i][-1]
                    new_train_df.loc[(new_train_df[UID] != user) | (new_train_df[IID] != d_iid)].reset_index(drop=True)
                elif self.ctf_type == 'K1':
                    k_iid = self.pre[i][-1]
                    new_train_df.loc[(new_train_df[UID] != user) | ((new_train_df[UID] == user) & (new_train_df[IID] == k_iid))].reset_index(drop=True)
        return new_train_df, ctf_indicator