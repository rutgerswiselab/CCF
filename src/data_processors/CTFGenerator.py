import torch
import numpy as np
import logging
import pandas as pd
import os
from utils.global_p import *
from data_processors.HistoryDP import HistoryDP
import random
import datetime
from tqdm import tqdm
import copy
from utils import utils


class CounterfactualGenerator(HistoryDP):
    
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
    
    def __init__(self, ctf_num, ctf_type, topk, can_item, neg_his, neg_his_neg, *args, **kwargs):
        self.ctf_num = ctf_num
        self.ctf_type = ctf_type
        self.topk = topk
        self.can_item = can_item
        self.neg_his = neg_his
        self.neg_his_neg = neg_his_neg
        HistoryDP.__init__(self, *args, **kwargs)
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
            self.train_data = self.format_train_data_dict(self.data_loader.train_df, model, runner, processor)
            self.train_data[SAMPLE_ID] = np.arange(0, len(self.train_data[Y]))
        if epoch >= 0:
            utils.shuffle_in_unison_scary(self.train_data)
        return self.train_data
    
    def generate_ctf_his(self, data, model, runner, processor):
        file_name = os.path.join(self.data_loader.path, '{}_{}_ctfnum{}_ctftype{}_topk{}.npy'.format(self.data_loader.dataset, model.__class__.__name__, 
                                                                                                     self.ctf_num, self.ctf_type, self.topk))
        if os.path.exists(file_name):
            ctf_his_list = np.load(file_name)
            logging.info('load counterfactual history from ' + file_name)
            return ctf_his_list
        data[SAMPLE_ID] = np.arange(0, len(data[Y]))
        ctf_his_list = []
        ctf_data = copy.deepcopy(data)
        logging.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        pbar = tqdm(total = len(data[Y]) * self.ctf_num)
        pre_his = [[str(row)[1:-1]] for row in ctf_data[C_HISTORY]]
        ctf_df = copy.deepcopy(self.data_loader.train_df[self.data_loader.train_df[C_HISTORY].apply(lambda x: len(x) > 0)])
        ctf_df[C_HISTORY] = [str(row)[1:-1] for row in ctf_data[C_HISTORY]]
        ctf_df = ctf_df.reset_index(drop = True)
        runner._check_time(start=True)
        
        for i in range(self.ctf_num):
            pre_len = [len(x) for x in pre_his]
            print(max(pre_len))
            
            ctf_his = copy.deepcopy(data[C_HISTORY])
            ctf_indicator = np.zeros(len(data[Y]))
            while np.sum(ctf_indicator) < len(data[Y]): 
                remain_index = np.where(np.ones(len(data[Y])) - ctf_indicator == 1)
                for key in data:
                    ctf_data[key] = copy.deepcopy(data[key][remain_index])
                ctf_data = self.sample_ctf_his(ctf_data, data[C_HISTORY][remain_index], remain_index[0], pre_his, model)
                ctf_df_cur = ctf_df.iloc[remain_index]
                neg_data = HistoryDP.generate_neg_data(self, 
                        ctf_data, ctf_df_cur, sample_n=self.can_item,
                        train=True, model=model)
                ctf_val_data = dict()
                for key in ctf_data.keys():
                    ctf_val_data[key] = np.concatenate((ctf_data[key], neg_data[key]))
                predictions = runner.predict(model, ctf_val_data, processor)
                predictions = np.transpose(predictions.reshape(-1,len(ctf_data[C_HISTORY])))
                
                result = predictions - predictions[:,0][:,None]
                rank = np.where(result > 0, 1, 0).sum(axis=1)
                indicator = np.where(rank < self.topk)
                pbar.update(len(indicator[0]))
                select_index = remain_index[0][indicator]
                ctf_his[remain_index] = ctf_data[C_HISTORY]
                ctf_indicator[select_index] = 1
                for j in remain_index[0]:
                    if str(ctf_his[j])[1:-1] == pre_his[j][0]:
                        ctf_indicator[j] = 1
                        continue
                    pre_his[j].append(str(ctf_his[j])[1:-1])
                    
            logging.info("Counterfactual history %5d have been generated. [%.1f s]" % (i + 1, runner._check_time()))
            ctf_his_list.append(ctf_his)
        pbar.close()
        pre_lens = [len(row) for row in pre_his]
        logging.info('The largest number of samples: {}'.format(str(max(pre_lens))))
        np.save(file_name, ctf_his_list)
        logging.info('save the counterfactual history to ' + file_name)
        return ctf_his_list
    
    def sample_ctf_his(self, data, history, remain_id, pre_ctf, model):
        if self.neg_his and self.neg_his_neg:
            flag = True
        else:
            flag = False
        ctf_his = copy.deepcopy(history)
        if self.ctf_type == 'R1R':
            for i in range(len(ctf_his)):
                while True:
                    row = copy.deepcopy(ctf_his[i])
                    index = random.randint(0, len(row) - 1)
                    if row[index] < 0:
                        neg_sign = True
                    r_index = random.randint(0, len(self.all_item) - 1)
                    row[index] = self.all_item[r_index]
                    if flag:
                        if neg_sign:
                            row[index] = - self.all_item[r_index]
                    
                    if str(row)[1:-1] not in pre_ctf[remain_id[i]]:
                        ctf_his[i] = row
                        break
                    if len(pre_ctf[remain_id[i]]) > 20:
                        logging.info("No satisfiable counterfactual history for R1R at {} for user {}, item {}, history {}".format(self.topk, data[UID][i], data[IID][i], history[i]))
                        break
        elif self.ctf_type == 'R1N':
            if model.__class__.__name__ == 'NLRRec':
                similarity = torch.matmul(model.feature_embeddings.weight.data,\
                                      torch.transpose(model.feature_embeddings.weight.data,0,1)).detach().cpu().numpy()
            else:
                similarity = torch.matmul(model.iid_embeddings.weight.data,\
                                      torch.transpose(model.iid_embeddings.weight.data,0,1)).detach().cpu().numpy()
            indices_all = np.argsort(similarity, axis=1)
            indices_all = np.array([idx[::-1] for idx in indices_all])

            for i in range(len(ctf_his)):
                row = copy.deepcopy(ctf_his[i])
                index = random.randint(0, len(row) - 1)
                if row[index] < 0:
                    neg_sign = True
                indices = indices_all[row[index]]
                idx = 0
                while True:
                    if indices[idx] == 0:
                        idx += 1
                        continue
                    row[index] = indices[idx]
                    if flag:
                        if neg_sign:
                            row[index] = - indices[idx]
                    if str(row)[1:-1] not in pre_ctf[remain_id[i]]:
                        break
                    idx += 1
                    if len(pre_ctf[remain_id[i]]) > 20:
                        logging.info("No satisfiable counterfactual history for R1N at {} for user {}, item {}, history {}".format(self.topk, data[UID][i], data[IID][i], history[i]))
                        break
                ctf_his[i] = row
        elif self.ctf_type == 'K1':
            for i in range(len(ctf_his)):
                while True:
                    row = copy.deepcopy(ctf_his[i])
                    index = random.randint(0, len(row) - 1)
                    row = [row[index]]
                    ctf_his[i] = row
                    if str(row)[1:-1] not in pre_ctf[remain_id[i]]:
                        ctf_his[i] = row
                        break
                    if len(pre_ctf[remain_id[i]]) > len(row) + 1:
                        logging.info("No satisfiable counterfactual history for K1 at {} for user {}, item {}, history {}".format(self.topk, data[UID][i], data[IID][i], history[i]))
                        break
                        
        elif self.ctf_type == 'D1':
            for i in range(len(ctf_his)):
                while True:
                    row = copy.deepcopy(ctf_his[i])
                    if len(row) <= 1:
                        break
                    index = random.randint(0, len(row) - 1)
                    row = row[:index] + row[index+1:]
                    ctf_his[i] = row
                    if str(row)[1:-1] not in pre_ctf[remain_id[i]]:
                        ctf_his[i] = row
                        break
                    if len(pre_ctf[remain_id[i]]) > len(row) + 1:
                        logging.info("No satisfiable counterfactual history for D1 at {} for user {}, item {}, history {}".format(self.topk, data[UID][i], data[IID][i], history[i]))
                        break
        else:
            raise NotImplementedError
        data[C_HISTORY] = ctf_his
        return data
    
    def format_train_data_dict(self, df, model, runner, processor):
        data_dict = HistoryDP.format_data_dict(self, df, model)
        column = []
        for i in range(self.ctf_num):
            column.append(C_CTF_HISTORY + '_' + str(i))
        ctf_his_list = self.generate_ctf_his(data_dict, model, runner, processor)
        
        for i in range(len(column)):
            col = column[i]
            data_dict[col] = ctf_his_list[i]
        return data_dict
    
    def get_feed_dict(self, data, batch_start, batch_size, train, neg_data=None, special_cols=None):
        """
        topn model generate a batch, if train, assign a negative sample for each positive sample, make sure first half is positive and second half is negative
        :param data: data dict，generated by self.get_*_data() and self.format_data_dict()
        :param batch_start: batch start index
        :param batch_size: batch size
        :param train: train or evaluation
        :param neg_data: data dict for negative samples，directly use it if exists
        :param special_cols: columns which need special operations
        :return: feed dict of the batch
        """
        his_cs = [C_CTF_HISTORY + '_' + str(i) for i in range(self.ctf_num)]
        his_ls = [C_CTF_HISTORY_LENGTH + '_' + str(i) for i in range(self.ctf_num)]
        
        if C_CTF_HISTORY + '_0' not in self.data_columns:
            self.data_columns += his_cs
        if C_CTF_HISTORY_LENGTH + '_0' not in self.info_columns:
            self.info_columns += his_ls
        if C_CTF_HISTORY + '_0' in data.keys():
            feed_dict = HistoryDP.get_feed_dict(
                self, data, batch_start, batch_size, train, neg_data=neg_data,
                special_cols=his_cs
                if special_cols is None else his_cs + special_cols)

            for i, c in enumerate(his_cs):
                lc, d = his_ls[i], feed_dict[c]
                if self.sparse_his == 1:  # if sparse representation
                    x, y, v = [], [], []
                    for idx, iids in enumerate(d):
                        x.extend([idx] * len(iids))
                        y.extend([abs(iid) for iid in iids])
                        v.extend([1.0 if iid > 0 else -1.0 if iid < 0 else 0 for iid in iids])
                    if len(x) <= 0:
                        i = utils.numpy_to_torch(np.array([[0], [0]]), gpu=False)
                        v = utils.numpy_to_torch(np.array([0.0], dtype=np.float32), gpu=False)
                    else:
                        i = utils.numpy_to_torch(np.array([x, y]), gpu=False)
                        v = utils.numpy_to_torch(np.array(v, dtype=np.float32), gpu=False)
                    history = torch.sparse.FloatTensor(
                        i, v, torch.Size([len(d), self.data_loader.item_num]))
                    # if torch.cuda.device_count() > 0:
                    #     history = history.cuda()
                    feed_dict[c] = history
                    feed_dict[lc] = [len(iids) for iids in d]
                    # feed_dict[lc] = utils.numpy_to_torch(np.array([len(iids) for iids in d]), gpu=False)
                else:
                    lengths = [len(iids) for iids in d]
                    max_length = max(lengths)
                    new_d = np.array([x + [0] * (max_length - len(x)) for x in d])
                    feed_dict[c] = utils.numpy_to_torch(new_d, gpu=False)
                    feed_dict[lc] = lengths
        else:
            self.data_columns = HistoryDP.data_columns
            self.info_columns = HistoryDP.info_columns
            feed_dict = HistoryDP.get_feed_dict(
                self, data, batch_start, batch_size, train, neg_data=neg_data,
                special_cols=special_cols)
        return feed_dict
    
    def generate_neg_data(self, data, feature_df, sample_n, train, model):
        """
        generate neg_data dict, use whtn prepare_batches_rk train=True
        :param data:
        :param feature_df:
        :param sample_n:
        :param train:
        :param model:
        :return:
        """
        inter_df = pd.DataFrame()
        for c in [UID, IID, Y, TIME]:
            if c in data:
                inter_df[c] = data[c]
            else:
                assert c == TIME
        his_cs = [C_CTF_HISTORY + '_' + str(i) for i in range(self.ctf_num)]
        for col in his_cs:
            inter_df[col] = np.array([str(ctf_his).replace(' ', '')[1:-1] for ctf_his in data[col]])

        
        neg_df = self.generate_neg_df(
            inter_df=inter_df, feature_df=feature_df,
            sample_n=sample_n, train=train)
        neg_data = self.format_data_dict(neg_df, model)
        neg_data[SAMPLE_ID] = np.arange(0, len(neg_data[Y])) + len(data[SAMPLE_ID])

        return neg_data
    
    def generate_neg_df(self, inter_df, feature_df, sample_n, train):
        """
        generate negative samples based on uid,iid and training or validation/test dataframe
        :param sample_n: negative samples number
        :param train: train or not
        :return:
        """
        his_cs = [C_CTF_HISTORY + '_' + str(i) for i in range(self.ctf_num)]
        other_columns = [c for c in inter_df.columns if c not in [UID, Y]]
        common_columns = [c for c in inter_df.columns if c not in [UID, Y] + his_cs]
        neg_df = self._sample_neg_from_uid_list(
            uids=inter_df[UID].tolist(), labels=inter_df[Y].tolist(), sample_n=sample_n, train=train,
            other_infos=inter_df[other_columns].to_dict('list'))

        neg_df = pd.merge(neg_df, feature_df, on=[UID] + common_columns, how='left')
        neg_df = neg_df.drop(columns=[IID])
        neg_df = neg_df.rename(columns={'iid_neg': IID})
        if C_CTF_HISTORY + '_0' not in inter_df.columns:
            neg_df = neg_df[feature_df.columns]
        neg_df[self.data_loader.label] = 0
        return neg_df
    
    def format_data_dict(self, df, model):
        """
        deal with history interaction except uid,iid,label,user、item、context features
        :param df: train、validation、test df
        :param model: Model class
        :return:
        """
        
        data_dict = HistoryDP.format_data_dict(self, df, model)

        if C_CTF_HISTORY + '_0' in df:
            his_cs = [C_CTF_HISTORY + '_' + str(i) for i in range(self.ctf_num)]
            for c in his_cs:
                his = df[c].apply(lambda x: eval('[' + x + ']'))
                data_dict[c] = his.values

        return data_dict
    