# coding=utf-8
import copy
from utils import utils
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict, Counter
from utils.global_p import *
from sklearn.utils import shuffle




class DataProcessor(object):
    data_columns = [UID, IID, X]  # keys of required feature info in data dict, need to be converted into tensor
    info_columns = [SAMPLE_ID, TIME]  # key of extra info in data dict

    @staticmethod
    def parse_dp_args(parser):
        """
        data processing command parameters
        :param parser:
        :return:
        """
        parser.add_argument('--test_sample_n', type=int, default=100,
                            help='Negative sample num for each instance in test/validation set when ranking.')
        parser.add_argument('--train_sample_n', type=int, default=1,
                            help='Negative sample num for each instance in train set when ranking.')
        parser.add_argument('--sample_un_p', type=float, default=1.0,
                            help='Sample from neg/pos with 1-p or unknown+neg/pos with p.')
        parser.add_argument('--unlabel_test', type=int, default=0,
                            help='If the label of test is unknown, do not sample neg of test set.')
        return parser

    @staticmethod
    def batch_to_gpu(batch):
        if torch.cuda.device_count() > 0:
            new_batch = {}
            for c in batch:
                if type(batch[c]) is torch.Tensor:
                    new_batch[c] = batch[c].cuda()
                else:
                    new_batch[c] = batch[c]
            return new_batch
        return batch

    def __init__(self, data_loader, rank, train_sample_n, test_sample_n, sample_un_p, unlabel_test=0):
        """
        initialization
        :param data_loader: DataLoader
        :param model: Model
        :param rank: 1=topn recommendation; 0=rating/clicking prediction
        :param test_sample_n: negative sampling ratio for evaluation when topn recommendation task. positive:negative=1:test_sample_n
        """
        self.data_loader = data_loader
        self.rank = rank
        self.train_data, self.validation_data, self.test_data = None, None, None

        self.test_sample_n = test_sample_n
        self.train_sample_n = train_sample_n
        self.sample_un_p = sample_un_p
        self.unlabel_test = unlabel_test

        if self.rank == 1:
            # generate history dict
            self.train_history_pos = defaultdict(set)
            for uid in data_loader.train_user_pos.keys():
                self.train_history_pos[uid] = set(data_loader.train_user_pos[uid])
            self.validation_history_pos = defaultdict(set)
            for uid in data_loader.validation_user_pos.keys():
                self.validation_history_pos[uid] = set(data_loader.validation_user_pos[uid])
            self.test_history_pos = defaultdict(set)
            for uid in data_loader.test_user_pos.keys():
                self.test_history_pos[uid] = set(data_loader.test_user_pos[uid])

            self.train_history_neg = defaultdict(set)
            for uid in data_loader.train_user_neg.keys():
                self.train_history_neg[uid] = set(data_loader.train_user_neg[uid])
            self.validation_history_neg = defaultdict(set)
            for uid in data_loader.validation_user_neg.keys():
                self.validation_history_neg[uid] = set(data_loader.validation_user_neg[uid])
            self.test_history_neg = defaultdict(set)
            for uid in data_loader.test_user_neg.keys():
                self.test_history_neg[uid] = set(data_loader.test_user_neg[uid])
        self.vt_batches_buffer = {}

    def get_train_data(self, epoch, model):
        """
        convert training dataframe in dataloader into dict, shuffle every epoch
        the dict will be used for generating batches
        :param epoch: no shuffle if <0
        :param model: Model
        :return: dict
        """
        if self.train_data is None:
            logging.info('Prepare Train Data...')
            self.train_data = self.format_data_dict(self.data_loader.train_df, model)
            self.train_data[SAMPLE_ID] = np.arange(0, len(self.train_data[Y]))
        if epoch >= 0:
            utils.shuffle_in_unison_scary(self.train_data)
        return self.train_data

    def get_validation_data(self, model):
        """
        convert validation dataframe in dataloader into dict
        assign test_sample_n negative samples for each positive in topn recommendation task
        the dict will be used for generating batches
        :param epoch: no shuffle if <0
        :param model: Model
        :return: dict
        """
        if self.validation_data is None:
            logging.info('Prepare Validation Data...')
            df = self.data_loader.validation_df
            if self.rank == 1:
                tmp_df = df.rename(columns={self.data_loader.label: Y})
                tmp_df = tmp_df.drop(tmp_df[tmp_df[Y] <= 0].index)
                neg_df = self.generate_neg_df(
                    inter_df=tmp_df, feature_df=df, sample_n=self.test_sample_n, train=False)
                df = pd.concat([df, neg_df], ignore_index=True)
            self.validation_data = self.format_data_dict(df, model)
            self.validation_data[SAMPLE_ID] = np.arange(0, len(self.validation_data[Y]))
        return self.validation_data

    def get_test_data(self, model):
        """
        convert test dataframe in dataloader into dict
        assign test_sample_n negative samples for each positive in topn recommendation task
        the dict will be used for generating batches
        :param epoch: no shuffle if <0
        :param model: Model
        :return: dict
        """
        if self.test_data is None:
            logging.info('Prepare Test Data...')
            df = self.data_loader.test_df
            if self.rank == 1 and self.unlabel_test == 0:
                tmp_df = df.rename(columns={self.data_loader.label: Y})
                tmp_df = tmp_df.drop(tmp_df[tmp_df[Y] <= 0].index)
                neg_df = self.generate_neg_df(
                    inter_df=tmp_df, feature_df=df, sample_n=self.test_sample_n, train=False)
                neg_df = self.drop_duplicate(neg_df, self.test_sample_n) #only keep test_sample_n negative samples for each user

                df = pd.concat([df, neg_df], ignore_index=True)
            self.test_data = self.format_data_dict(df, model)
            self.test_data[SAMPLE_ID] = np.arange(0, len(self.test_data[Y]))
        return self.test_data
    
    def drop_duplicate(self, neg_df, test_sample_n):
        neg_df = shuffle(neg_df)
        return neg_df.groupby(UID).head(test_sample_n).reset_index(drop=True)

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
        # if validation for test, negative samples are already sampled
        total_data_num = len(data[SAMPLE_ID])
        batch_end = min(len(data[self.data_columns[0]]), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        total_batch_size = real_batch_size * (self.train_sample_n + 1) if self.rank == 1 and train else real_batch_size
        feed_dict = {TRAIN: train, RANK: self.rank, REAL_BATCH_SIZE: real_batch_size,
                     TOTAL_BATCH_SIZE: total_batch_size}
        if Y in data:
            feed_dict[Y] = utils.numpy_to_torch(data[Y][batch_start:batch_start + real_batch_size], gpu=False)
        for c in self.info_columns + self.data_columns:
            if c not in data or data[c].size <= 0:
                continue
            d = data[c][batch_start: batch_start + real_batch_size]
            if self.rank == 1 and train:
                neg_d = np.concatenate(
                    [neg_data[c][total_data_num * i + batch_start: total_data_num * i + batch_start + real_batch_size]
                     for i in range(self.train_sample_n)])
                d = np.concatenate([d, neg_d])
            feed_dict[c] = d
        for c in self.data_columns:
            if c not in feed_dict:
                continue
            if special_cols is not None and c in special_cols:
                continue
            feed_dict[c] = utils.numpy_to_torch(feed_dict[c], gpu=False)
        return feed_dict

    def _check_vt_buffer(self, data, batch_size, train, model):
        buffer_key = ''
        if data is self.train_data and not train:
            buffer_key = '_'.join(['train', str(batch_size), str(model)])
        elif data is self.validation_data:
            buffer_key = '_'.join(['validation', str(batch_size), str(model)])
        elif data is self.test_data:
            buffer_key = '_'.join(['test', str(batch_size), str(model)])
        if buffer_key != '' and buffer_key in self.vt_batches_buffer:
            return self.vt_batches_buffer[buffer_key]
        return buffer_key

    def prepare_batches(self, data, batch_size, train, model):
        """
        convert data dict into batch
        :param data: dict, generate by self.get_*_data() and self.format_data_dict()
        :param batch_size: batch size 
        :param train: train or not
        :param model: Model class
        :return: list of batches
        """

        buffer_key = self._check_vt_buffer(data=data, batch_size=batch_size, train=train, model=model)
        if type(buffer_key) != str:
            return buffer_key

        if data is None:
            return None
        num_example = len(data[Y])
        total_batch = int((num_example + batch_size - 1) / batch_size)
        assert num_example > 0
        # if train, one negative sample for each positive sample
        neg_data = None
        if train and self.rank == 1:
            neg_data = self.generate_neg_data(
                data, self.data_loader.train_df, sample_n=self.train_sample_n,
                train=True, model=model)
        batches = []
        for batch in tqdm(range(total_batch), leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            batches.append(self.get_feed_dict(data=data, batch_start=batch * batch_size, batch_size=batch_size,
                                              train=train, neg_data=neg_data))

        if buffer_key != '':
            self.vt_batches_buffer[buffer_key] = batches
        return batches

    def format_data_dict(self, df, model):
        """
        deal with history interaction except uid,iid,label,user、item、context features
        :param df: train、validation、test df
        :param model: Model class
        :return:
        """

        data_loader = self.data_loader
        data = {}
        # record uid, iid
        out_columns = []
        if UID in df:
            out_columns.append(UID)
            data[UID] = df[UID].values
        if IID in df:
            out_columns.append(IID)
            data[IID] = df[IID].values
        if TIME in df:
            data[TIME] = df[TIME].values

        # record label into Y
        if data_loader.label in df.columns:
            data[Y] = np.array(df[data_loader.label], dtype=np.float32)
        else:
            logging.warning('No Labels In Data: ' + data_loader.label)
            data[Y] = np.zeros(len(df), dtype=np.float32)

        ui_id = df[out_columns]

        # concatenate user feature and item features based on uid, iid
        out_df = ui_id
        if data_loader.user_df is not None and model.include_user_features:
            out_df = pd.merge(out_df, data_loader.user_df, on=UID, how='left')
        if data_loader.item_df is not None and model.include_item_features:
            out_df = pd.merge(out_df, data_loader.item_df, on=IID, how='left')

        # whether contain context feature
        if model.include_context_features and len(data_loader.context_features) > 0:
            context = df[data_loader.context_features]
            out_df = pd.concat([out_df, context], axis=1, ignore_index=True)
        out_df = out_df.fillna(0)

        # if uid and iid are special, do not convert to multi-hot vectors as other features\
        if not model.include_id:
            out_df = out_df.drop(columns=out_columns)

        
        base = 0
        for feature in out_df.columns:
            out_df[feature] = out_df[feature].apply(lambda x: x + base)
            base += int(data_loader.column_max[feature] + 1)

        # if model requires，concatenate uid,iid before x
        data[X] = out_df.values.astype(int)
        assert len(data[X]) == len(data[Y])
        return data

    def generate_neg_data(self, data, feature_df, sample_n, train, model):
        """
        generate neg_data dict, use when prepare_batches_rk train=True
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
        other_columns = [c for c in inter_df.columns if c not in [UID, Y]]
        neg_df = self._sample_neg_from_uid_list(
            uids=inter_df[UID].tolist(), labels=inter_df[Y].tolist(), sample_n=sample_n, train=train,
            other_infos=inter_df[other_columns].to_dict('list'))
        neg_df = pd.merge(neg_df, feature_df, on=[UID] + other_columns, how='left')
        neg_df = neg_df.drop(columns=[IID])
        neg_df = neg_df.rename(columns={'iid_neg': IID})
        neg_df = neg_df[feature_df.columns]
        neg_df[self.data_loader.label] = 0
        return neg_df

    def _sample_neg_from_uid_list(self, uids, labels, sample_n, train, other_infos=None):
        """
        negative sampling based on list of user
        :param uids: uid list
        :param sample_n: number of samples for each uid
        :param train: train or not
        :param other_infos: other infomation that require copy except uid,iid,label，e.g. history
        :return: DataFrame，but require to be data dict generated by self.format_data_dict()
        """
        if other_infos is None:
            other_infos = {}
        iid_list = []

        other_info_list = {}
        for info in other_infos:
            other_info_list[info] = []

        # record iid
        item_num = self.data_loader.item_num
        for index, uid in enumerate(uids):
            if labels[index] > 0:
                # record positive samples
                train_history = self.train_history_pos
                validation_history, test_history = self.validation_history_pos, self.test_history_pos
                known_train = self.train_history_neg
            else:
                assert train
                # record negative samples
                train_history = self.train_history_neg
                validation_history, test_history = self.validation_history_neg, self.test_history_neg
                known_train = self.train_history_pos
            if train:
                # avoid getting known positive or negative for training set
                inter_iids = train_history[uid]
            else:
                # avoid getting known positive or negative for non-training set
                inter_iids = train_history[uid] | validation_history[uid] | test_history[uid]

            # check remaining number
            remain_iids_num = item_num - len(inter_iids)
            # report if no enough available items
            assert remain_iids_num >= sample_n

            # if remain_iids_num is small, list all available items and use np.choice
            remain_iids = None
            if 1.0 * remain_iids_num / item_num < 0.2:
                remain_iids = [i for i in range(1, item_num) if i not in inter_iids]

            sampled = set()
            if remain_iids is None:
                unknown_iid_list = []
                for i in range(sample_n):
                    iid = np.random.randint(1, self.data_loader.item_num)
                    while iid in inter_iids or iid in sampled:
                        iid = np.random.randint(1, self.data_loader.item_num)
                    unknown_iid_list.append(iid)
                    sampled.add(iid)
            else:
                unknown_iid_list = np.random.choice(remain_iids, sample_n, replace=False)

            # if train, sample from known negative samples or positive samples
            if train and self.sample_un_p < 1:
                known_iid_list = list(np.random.choice(
                    list(known_train[uid]), min(sample_n, len(known_train[uid])), replace=False)) \
                    if len(known_train[uid]) != 0 else []
                known_iid_list = known_iid_list + unknown_iid_list
                tmp_iid_list = []
                sampled = set()
                for i in range(sample_n):
                    p = np.random.rand()
                    if p < self.sample_un_p or len(known_iid_list) == 0:
                        iid = unknown_iid_list.pop(0)
                        while iid in sampled:
                            iid = unknown_iid_list.pop(0)
                    else:
                        iid = known_iid_list.pop(0)
                        while iid in sampled:
                            iid = known_iid_list.pop(0)
                    tmp_iid_list.append(iid)
                    sampled.add(iid)
                iid_list.append(tmp_iid_list)
            else:
                iid_list.append(unknown_iid_list)

        all_uid_list, all_iid_list = [], []
        for i in range(sample_n):
            for index, uid in enumerate(uids):
                all_uid_list.append(uid)
                all_iid_list.append(iid_list[index][i])
                # # copy other info
                for info in other_infos:
                    other_info_list[info].append(other_infos[info][index])

        neg_df = pd.DataFrame(data=list(zip(all_uid_list, all_iid_list)), columns=[UID, 'iid_neg'])
        for info in other_infos:
            neg_df[info] = other_info_list[info]
        return neg_df
