# coding=utf-8

import torch
import logging
from time import time
from utils import utils
from utils.global_p import *
from tqdm import tqdm
import gc
import numpy as np
import copy
import os



class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        """
        command
        :param parser:
        :return:
        """
        parser.add_argument('--load', type=int, default=0,
                            help='Whether load model and continue to train')
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check every epochs.')
        parser.add_argument('--early_stop', type=int, default=0,
                            help='whether to early-stop.')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=128 * 128,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2_bias', type=int, default=0,
                            help='Whether add l2 regularizer on bias.')
        parser.add_argument('--l2', type=float, default=1e-5,
                            help='Weight of l2_regularize in pytorch optimizer.')
        parser.add_argument('--l2s', type=float, default=0.0,
                            help='Weight of l2_regularize in loss.')
        parser.add_argument('--grad_clip', type=float, default=10,
                            help='clip_grad_value_ para, -1 means, no clip')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--metrics', type=str, default="RMSE",
                            help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
        parser.add_argument('--pre_gpu', type=int, default=0,
                            help='Whether put all batches to gpu before run batches. \
                            If 0, dynamically put gpu for each batch.')
        return parser

    def __init__(self, optimizer='GD', lr=0.01, epoch=100, batch_size=128, eval_batch_size=128 * 128,
                 dropout=0.2, l2=1e-5, l2s=1e-5, l2_bias=0,
                 grad_clip=10, metrics='RMSE', check_epoch=10, early_stop=1, pre_gpu=0):
        """
        initialization
        :param optimizer: optimizer name
        :param lr: learning rate
        :param epoch: number of epochs
        :param batch_size: training batch size
        :param eval_batch_size: evaluation batch size
        :param dropout: dropout value
        :param l2: l2 weight
        :param metrics: evaluation metrics, separated by comma
        :param check_epoch: print some intermediate tensor every few epochs.
        :param early_stop: early stop or not
        """
        self.optimizer_name = optimizer
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout = dropout
        self.no_dropout = 0.0
        self.l2_weight = l2
        self.l2s_weight = l2s
        self.l2_bias = l2_bias
        self.grad_clip = grad_clip
        self.pre_gpu = pre_gpu

        # convert metrics into list of string
        self.metrics_str = metrics
        self.metrics = metrics.lower().split(',')
        self.check_epoch = check_epoch
        self.early_stop = early_stop
        self.time = None

        # used for recording metrics of training, validation, testing results in each epoch
        self.train_results, self.valid_results, self.test_results = [], [], []

    def _build_optimizer(self, model):
        """
        build optimizer
        :param model: model
        :return: optimizer
        """
        weight_p, bias_p = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        if self.l2_bias == 1:
            optimize_dict = [{'params': weight_p + bias_p, 'weight_decay': self.l2_weight}]
        else:
            optimize_dict = [{'params': weight_p, 'weight_decay': self.l2_weight},
                             {'params': bias_p, 'weight_decay': 0.0}]

        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            logging.info("Optimizer: GD")
            optimizer = torch.optim.SGD(optimize_dict, lr=self.lr)
        elif optimizer_name == 'adagrad':
            logging.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(optimize_dict, lr=self.lr)
        elif optimizer_name == 'adam':
            logging.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(optimize_dict, lr=self.lr)
        else:
            logging.error("Unknown Optimizer: " + self.optimizer_name)
            assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
            optimizer = torch.optim.SGD(optimize_dict, lr=self.lr)
        return optimizer

    def _check_time(self, start=False):
        """
        for recording times, self.time store [start time, last step time]
        :param start: start timing or not
        :return: time span from last step time
        """
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def batches_add_control(self, batches, train):
        """
        add some control information into all batches like DROPOUT
        :param batches: list of all batches, generated by DataProcessor
        :param train: whether train
        :return: batch list
        """
        for batch in batches:
            batch[TRAIN] = train
            batch[DROPOUT] = self.dropout if train else self.no_dropout
        return batches

    def predict(self, model, data, data_processor):
        """
        predict, not for training
        :param model: model
        :param data: data dict, generated by self.get_*_data() and self.format_data_dict() from DataProcessors
        :param data_processor: DataProcessor
        :return: np.array of predictions
        """
        gc.collect()

        batches = data_processor.prepare_batches(data, self.eval_batch_size, train=False, model=model)
        batches = self.batches_add_control(batches, train=False)
        if self.pre_gpu == 1:
            batches = [data_processor.batch_to_gpu(b) for b in batches]

        model.eval()
        predictions = []
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            if self.pre_gpu == 0:
                batch = data_processor.batch_to_gpu(batch)
            prediction = model.predict(batch)[PREDICTION]
            predictions.append(prediction.detach().cpu().data.numpy())

        predictions = np.concatenate(predictions)
        sample_ids = np.concatenate([b[SAMPLE_ID] for b in batches])

        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data[SAMPLE_ID]])

        gc.collect()
        return predictions

    def fit(self, model, data, data_processor, epoch=-1):  # fit the results for an input set
        """
        training
        :param model: model
        :param data: data dict，generated by self.get_*_data() and self.format_data_dict() from DataProcessor
        :param data_processor: DataProcessor instant
        :param epoch: epoch number
        :return: return output of last epoch, could provide self.check to check intermediate results.
        """
        gc.collect()
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        batches = data_processor.prepare_batches(data, self.batch_size, train=True, model=model)
        batches = self.batches_add_control(batches, train=True)
        if self.pre_gpu == 1:
            batches = [data_processor.batch_to_gpu(b) for b in batches]

        batch_size = self.batch_size if data_processor.rank == 0 else self.batch_size * 2
        model.train()
        accumulate_size, prediction_list, output_dict = 0, [], None
        loss_list, loss_l2_list = [], []
        for i, batch in \
                tqdm(list(enumerate(batches)), leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100, mininterval=1):
            if self.pre_gpu == 0:
                batch = data_processor.batch_to_gpu(batch)
            accumulate_size += len(batch[Y])
            model.optimizer.zero_grad()
            output_dict = model(batch)
            l2 = output_dict[LOSS_L2]
            loss = output_dict[LOSS] + l2 * self.l2s_weight
            loss.backward()
            loss_list.append(loss.detach().cpu().data.numpy())
            loss_l2_list.append(l2.detach().cpu().data.numpy())
            prediction_list.append(output_dict[PREDICTION].detach().cpu().data.numpy()[:batch[REAL_BATCH_SIZE]])
            if self.grad_clip > 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                torch.nn.utils.clip_grad_value_(model.parameters(), self.grad_clip)
            if accumulate_size >= batch_size or i == len(batches) - 1:
                model.optimizer.step()
                accumulate_size = 0
            # model.optimizer.step()
        model.eval()
        gc.collect()

        predictions = np.concatenate(prediction_list)
        sample_ids = np.concatenate([b[SAMPLE_ID][:b[REAL_BATCH_SIZE]] for b in batches])
        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in data[SAMPLE_ID]])
        return predictions, output_dict, np.mean(loss_list), np.mean(loss_l2_list)

    def eva_termination(self, model):
        """
        check whether stop training based on validation
        :param model: model
        :return: stop or not
        """
        metric = self.metrics[0]
        valid = self.valid_results
        
        # if epoch > 20 and lower value is better and evaluation is nondecreasing for 5 epochs
        if len(valid) > 20 and metric in utils.LOWER_METRIC_LIST and utils.strictly_increasing(valid[-5:]):
            return True
        # if epoch > 20 and larger value is better and evaluation is nonincreasing for 5 epochs
        elif len(valid) > 20 and metric not in utils.LOWER_METRIC_LIST and utils.strictly_decreasing(valid[-5:]):
            return True
        # did not improve current best results for 20 epochs
        elif len(valid) - valid.index(utils.best_result(metric, valid)) > 20:
            return True
        return False

    def train(self, model, data_processor):
        """
        train the model
        :param model: model
        :param data_processor: DataProcessor
        :return:
        """

        # get data, no shuffle when epoch=-1
        train_data = data_processor.get_train_data(epoch=-1, model=model)
        validation_data = data_processor.get_validation_data(model=model)
        test_data = data_processor.get_test_data(model=model) if data_processor.unlabel_test == 0 else None
        self._check_time(start=True)  # record start time

        # evaluation before training
        init_train = self.evaluate(model, train_data, data_processor) \
            if train_data is not None else [-1.0] * len(self.metrics)
        init_valid = self.evaluate(model, validation_data, data_processor) \
            if validation_data is not None else [-1.0] * len(self.metrics)
        init_test = self.evaluate(model, test_data, data_processor) \
            if test_data is not None and data_processor.unlabel_test == 0 else [-1.0] * len(self.metrics)
        logging.info("Init: \t train= %s validation= %s test= %s [%.1f s] " % (
            utils.format_metric(init_train), utils.format_metric(init_valid), utils.format_metric(init_test),
            self._check_time()) + ','.join(self.metrics))
        # model.save_model(
        #     model_path='../model/variable_tsne_logic_epoch/variable_tsne_logic_epoch_0.pt')
        try:
            for epoch in range(self.epoch):
                self._check_time()
                # get training data for each epoch, since it involves shuffle or needs negative sampling
                epoch_train_data = data_processor.get_train_data(epoch=epoch, model=model)
                train_predictions, last_batch, mean_loss, mean_loss_l2 = \
                    self.fit(model, epoch_train_data, data_processor, epoch=epoch)

                # check intermediate results
                if self.check_epoch > 0 and (epoch == 1 or epoch % self.check_epoch == 0):
                    last_batch['mean_loss'] = mean_loss
                    last_batch['mean_loss_l2'] = mean_loss_l2
                    self.check(model, last_batch)
                training_time = self._check_time()

                # # evaluate model performance
                train_result = [mean_loss] + model.evaluate_method(train_predictions, train_data, metrics=['rmse'])
                valid_result = self.evaluate(model, validation_data, data_processor) \
                    if validation_data is not None else [-1.0] * len(self.metrics)
                test_result = self.evaluate(model, test_data, data_processor) \
                    if test_data is not None and data_processor.unlabel_test == 0 else [-1.0] * len(self.metrics)
                testing_time = self._check_time()

                self.train_results.append(train_result)
                self.valid_results.append(valid_result)
                self.test_results.append(test_result)

                # print out current performance
                logging.info("Epoch %5d [%.1f s]\t train= %s validation= %s test= %s [%.1f s] "
                             % (epoch + 1, training_time, utils.format_metric(train_result),
                                utils.format_metric(valid_result), utils.format_metric(test_result),
                                testing_time) + ','.join(self.metrics))

                # if current performance is the best, save the model, based on validation
                if utils.best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
                    model.save_model()
                # model.save_model(
                #     model_path='../model/variable_tsne_logic_epoch/variable_tsne_logic_epoch_%d.pt' % (epoch + 1))
                # check whether early stop, based on validation
                if self.eva_termination(model) and self.early_stop == 1:
                    logging.info("Early stop at %d based on validation result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                model.save_model()

        # Find the best validation result across iterations
        best_valid_score = utils.best_result(self.metrics[0], self.valid_results)
        best_epoch = self.valid_results.index(best_valid_score)
        logging.info("Best Iter(validation)= %5d\t train= %s valid= %s test= %s [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(self.train_results[best_epoch]),
                        utils.format_metric(self.valid_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))
        best_test_score = utils.best_result(self.metrics[0], self.test_results)
        best_epoch = self.test_results.index(best_test_score)
        logging.info("Best Iter(test)= %5d\t train= %s valid= %s test= %s [%.1f s] "
                     % (best_epoch + 1,
                        utils.format_metric(self.train_results[best_epoch]),
                        utils.format_metric(self.valid_results[best_epoch]),
                        utils.format_metric(self.test_results[best_epoch]),
                        self.time[1] - self.time[0]) + ','.join(self.metrics))
        model.load_model()

    def evaluate(self, model, data, data_processor, metrics=None):  # evaluate the results for an input set
        """
        evaluate model performance
        :param model: model
        :param data: data dict，generated by self.get_*_data() and self.format_data_dict() from DataProcessor
        :param data_processor: DataProcessor
        :param metrics: list of str
        :return: lists of float for each metrics
        """
        if metrics is None:
            metrics = self.metrics
        predictions = self.predict(model, data, data_processor)
        return model.evaluate_method(predictions, data, metrics=metrics)

    def check(self, model, out_dict):
        """
        check intermediate results
        :param model: model
        :param out_dict: output of one batch
        :return:
        """
        # batch = data_processor.get_feed_dict(data, 0, self.batch_size, True)
        # self.batches_add_control([batch], train=False)
        # model.eval()
        # check = model(batch)
        check = out_dict
        logging.info(os.linesep)
        for i, t in enumerate(check[CHECK]):
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]) + os.linesep)

        loss, l2 = check['mean_loss'], check['mean_loss_l2']
        logging.info('mean loss = %.4f, l2 = %.4f, %.4f' % (loss, l2 * self.l2_weight, l2 * self.l2s_weight))
        # if not (loss * 0.005 < l2 < loss * 0.1):
        #     logging.warning('l2 inappropriate: loss = %.4f, l2 = %.4f' % (loss, l2))

    def run_some_tensors(self, model, data, data_processor, dict_keys):
        """
        predict, not for training
        :param model: model
        :param data: data dict, generated by self.get_*_data() and self.format_data_dict() from DataProcessors
        :param data_processor: DataProcessor
        :return: np.array of predictions
        """
        gc.collect()

        if type(dict_keys) == str:
            dict_keys = [dict_keys]

        batches = data_processor.prepare_batches(data, self.eval_batch_size, train=False, model=model)
        batches = self.batches_add_control(batches, train=False)
        if self.pre_gpu == 1:
            batches = [data_processor.batch_to_gpu(b) for b in batches]

        result_dict = {}
        for key in dict_keys:
            result_dict[key] = []
        model.eval()
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            if self.pre_gpu == 0:
                batch = data_processor.batch_to_gpu(batch)
            out_dict = model.predict(batch)
            for key in dict_keys:
                if key in out_dict:
                    result_dict[key].append(out_dict[key].detach().cpu().data.numpy())

        sample_ids = np.concatenate([b[SAMPLE_ID] for b in batches])
        for key in dict_keys:
            try:
                result_array = np.concatenate(result_dict[key])
            except ValueError as e:
                logging.warning("run_some_tensors: %s %s" % (key, str(e)))
                result_array = np.array([d for b in result_dict[key] for d in b])
            if len(sample_ids) == len(result_array):
                reorder_dict = dict(zip(sample_ids, result_array))
                result_dict[key] = np.array([reorder_dict[i] for i in data[SAMPLE_ID]])
        gc.collect()
        return result_dict
