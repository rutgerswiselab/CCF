# coding=utf-8

import argparse
import logging
import sys
import numpy as np
import os
import torch
import datetime
import pickle
import copy

from utils import utils
from utils.global_p import *

# # import data_loaders
from data_loaders.DataLoader import DataLoader
from data_loaders.ProLogicDL import ProLogicDL

# # import models
from models.BaseModel import BaseModel
from models.RecModel import RecModel
from models.BiasedMF import BiasedMF
from models.NCF import NCF
from models.GRU4Rec import GRU4Rec
from models.STAMP import STAMP
from models.NLR import NLR
from models.NLRRec import NLRRec

from CCFmodels.CCFGRU4Rec import CCFGRU4Rec
from CCFmodels.CCFSTAMP import CCFSTAMP
from CCFmodels.CCFBiasedMF import CCFBiasedMF
from CCFmodels.CCFNCF import CCFNCF
from CCFmodels.CCFNLRRec import CCFNLRRec

from CCFmodels.ConCCFGRU4Rec import ConCCFGRU4Rec
from CCFmodels.ConCCFSTAMP import ConCCFSTAMP
from CCFmodels.ConCCFBiasedMF import ConCCFBiasedMF
from CCFmodels.ConCCFNLRRec import ConCCFNLRRec
from CCFmodels.ConCCFNCF import ConCCFNCF

# # import data processors
from data_processors.DataProcessor import DataProcessor
from data_processors.HistoryDP import HistoryDP
from data_processors.ProLogicDP import ProLogicDP
from data_processors.ProLogicRecDP import ProLogicRecDP
from data_processors.RNNLogicDP import RNNLogicDP
from data_processors.CTFGenerator import CounterfactualGenerator
from data_processors.nonSeqCTFGenerator import nonSeqCTFGenerator
from data_processors.ProLogicCTFGenerator import ProLogicCTFGenerator

# # import runners
from runners.BaseRunner import BaseRunner
from runners.ProLogicRunner import ProLogicRunner
from runners.DiscreteCTFRunner import DiscreteCTFRunner
from runners.ContinuousCTFRunner import ContinuousCTFRunner

import pdb
import pandas as pd


def build_run_environment(para_dict, dl_name, dp_name, model_name, runner_name):
    if type(para_dict) is str:
        para_dict = eval(para_dict)
    if type(dl_name) is str:
        dl_name = eval(dl_name)
    if type(dp_name) is str:
        dp_name = eval(dp_name)
    if type(model_name) is str:
        model_name = eval(model_name)
    if type(runner_name) is str:
        runner_name = eval(runner_name)

    # random seed
    torch.manual_seed(para_dict['random_seed'])
    torch.cuda.manual_seed(para_dict['random_seed'])
    np.random.seed(para_dict['random_seed'])

    # cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = para_dict['gpu']  # default '0'
    logging.info("# cuda devices: %d" % torch.cuda.device_count())

    para_dict['load_data'] = True
    dl_paras = utils.get_init_paras_dict(dl_name, para_dict)
    logging.info(str(dl_name) + ': ' + str(dl_paras))
    data_loader = dl_name(**dl_paras)

    # append_his by data_loader
    if 'all_his' in para_dict:
        data_loader.append_his(all_his=para_dict['all_his'], max_his=para_dict['max_his'],
                               neg_his=para_dict['neg_his'], neg_column=para_dict['neg_column'])

    # if topn recommendation, only keep positive samples, generate negative sampels during training process and convert label into binary value
    if para_dict['rank'] == 1:
        data_loader.label_01()
        if para_dict['drop_neg'] == 1:
            data_loader.drop_neg()

    para_dict['data_loader'] = data_loader
    dp_paras = utils.get_init_paras_dict(dp_name, para_dict)
    logging.info(str(dp_name) + ': ' + str(dp_paras))
    data_processor = dp_name(**dp_paras)

    # # prepare train,test,validation samples before generating model and training
    data_processor.get_train_data(epoch=-1, model=model_name)
    data_processor.get_validation_data(model=model_name)
    data_processor.get_test_data(model=model_name)

    # generate feature multi-hot vectors
    features, feature_dims, feature_min, feature_max = \
        data_loader.feature_info(include_id=model_name.include_id,
                                 include_item_features=model_name.include_item_features,
                                 include_user_features=model_name.include_user_features)
    para_dict['feature_num'], para_dict['feature_dims'] = len(features), feature_dims
    para_dict['user_feature_num'] = len([f for f in features if f.startswith('u_')])
    para_dict['item_feature_num'] = len([f for f in features if f.startswith('i_')])
    para_dict['context_feature_num'] = len([f for f in features if f.startswith('c_')])
    data_loader_vars = vars(data_loader)
    for key in data_loader_vars:
        if key not in para_dict:
            para_dict[key] = data_loader_vars[key]
    model_paras = utils.get_init_paras_dict(model_name, para_dict)
    logging.info(str(model_name) + ': ' + str(model_paras))
    model = model_name(**model_paras)
    model.load_model()

    # use gpu
    if torch.cuda.device_count() > 0:
        # model = model.to('cuda:0')
        model = model.cuda()

    # create runner
    runner_paras = utils.get_init_paras_dict(runner_name, para_dict)
    logging.info(str(runner_name) + ': ' + str(runner_paras))
    runner = runner_name(**runner_paras)
    return data_loader, data_processor, model, runner


def main():
    # init args
    init_parser = argparse.ArgumentParser(description='Model', add_help=False)
    init_parser.add_argument('--rank', type=int, default=1,
                             help='1=ranking, 0=rating/click')
    init_parser.add_argument('--data_loader', type=str, default='',
                             help='Choose data_loader')
    init_parser.add_argument('--model_name', type=str, default='BaseModel',
                             help='Choose model to run.')
    init_parser.add_argument('--runner_name', type=str, default='',
                             help='Choose runner')
    init_parser.add_argument('--data_processor', type=str, default='',
                             help='Choose runner')
    init_args, init_extras = init_parser.parse_known_args()

    
    # choose model
    model_name = eval(init_args.model_name)

    # choose data_loader
    if init_args.data_loader == '':
        init_args.data_loader = model_name.data_loader
    data_loader_name = eval(init_args.data_loader)

    # choose data_processor
    if init_args.data_processor == '':
        init_args.data_processor = model_name.data_processor
    data_processor_name = eval(init_args.data_processor)

    # choose runner
    if init_args.runner_name == '':
        init_args.runner_name = model_name.runner
    runner_name = eval(init_args.runner_name)
    
    

    # cmd line paras
    parser = argparse.ArgumentParser(description='')
    parser = utils.parse_global_args(parser)
    parser = data_loader_name.parse_data_args(parser)
    parser = model_name.parse_model_args(parser, model_name=init_args.model_name)
    parser = runner_name.parse_runner_args(parser)
    parser = data_processor_name.parse_dp_args(parser)

    origin_args, extras = parser.parse_known_args()
    
    if origin_args.counterfactual_constraint == 2:
        return ContinuousCCF(init_args)

    # log,model,result filename
    paras = sorted(vars(origin_args).items(), key=lambda kv: kv[0])
    log_name_exclude = ['check_epoch', 'eval_batch_size', 'gpu', 'label', 'load',
                        'log_file', 'metrics', 'model_path', 'path', 'pre_gpu', 'result_file',
                        'sep', 'seq_sep', 'train', 'unlabel_test', 'verbose',
                        'dataset', 'random_seed', 'counterfactual_constraint']
    log_file_name = [str(init_args.rank) + str(origin_args.drop_neg),
                     init_args.model_name, origin_args.dataset, str(origin_args.random_seed)] + \
                    [p[0].replace('_', '')[:3] + str(p[1]) for p in paras if p[0] not in log_name_exclude]
    log_file_name = [l.replace(' ', '-').replace('_', '-') for l in log_file_name]
    log_file_name = '_'.join(log_file_name)
    if origin_args.log_file == os.path.join(LOG_DIR, 'log.txt'):
        origin_args.log_file = os.path.join(LOG_DIR, '%s/%s.txt' % (init_args.model_name, log_file_name))
    utils.check_dir_and_mkdir(origin_args.log_file)
    if origin_args.result_file == os.path.join(RESULT_DIR, 'result.npy'):
        origin_args.result_file = os.path.join(RESULT_DIR, '%s/%s.npy' % (init_args.model_name, log_file_name))
    utils.check_dir_and_mkdir(origin_args.result_file)
    if origin_args.model_path == os.path.join(MODEL_DIR, '%s/%s.pt' % (init_args.model_name, init_args.model_name)):
        origin_args.model_path = os.path.join(MODEL_DIR, '%s/%s.pt' % (init_args.model_name, log_file_name))
    utils.check_dir_and_mkdir(origin_args.model_path)
    args = copy.deepcopy(origin_args)

    # logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(vars(init_args))
    logging.info(vars(origin_args))
    logging.info(extras)

    logging.info('DataLoader: ' + init_args.data_loader)
    logging.info('Model: ' + init_args.model_name)
    logging.info('Runner: ' + init_args.runner_name)
    logging.info('DataProcessor: ' + init_args.data_processor)

    # random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # default '0'
    logging.info("# cuda devices: %d" % torch.cuda.device_count())

    # create data_loader
    args.load_data = True
    dl_para_dict = utils.get_init_paras_dict(data_loader_name, vars(args))
    logging.info(init_args.data_loader + ': ' + str(dl_para_dict))
    data_loader = data_loader_name(**dl_para_dict)

    # append_his by data_loader
    if 'all_his' in origin_args:
        data_loader.append_his(all_his=origin_args.all_his, max_his=origin_args.max_his,
                               neg_his=origin_args.neg_his, neg_his_neg=origin_args.neg_his_neg,
                               neg_column=origin_args.neg_column)

    # if topn recommendation, only keep positive samples, generate negative sampels during training process and convert label into binary value
    if init_args.rank == 1:
        data_loader.label_01()
        if origin_args.drop_neg == 1:
            data_loader.drop_neg()
#             pdb.set_trace()

    # create data_processor
    args.data_loader, args.rank = data_loader, init_args.rank
    dp_para_dict = utils.get_init_paras_dict(data_processor_name, vars(args))
    logging.info(init_args.data_processor + ': ' + str(dp_para_dict))
    data_processor = data_processor_name(**dp_para_dict)

    # # prepare train,test,validation samples before generating model and training
    data_processor.get_train_data(epoch=-1, model=model_name)
    data_processor.get_validation_data(model=model_name)
    data_processor.get_test_data(model=model_name)
#     pdb.set_trace()

    # create model
    # generate feature multi-hot vectors，
    features, feature_dims, feature_min, feature_max = \
        data_loader.feature_info(include_id=model_name.include_id,
                                 include_item_features=model_name.include_item_features,
                                 include_user_features=model_name.include_user_features)
    args.feature_num, args.feature_dims = len(features), feature_dims
    args.user_feature_num = len([f for f in features if f.startswith('u_')])
    args.item_feature_num = len([f for f in features if f.startswith('i_')])
    args.context_feature_num = len([f for f in features if f.startswith('c_')])
    data_loader_vars = vars(data_loader)
    for key in data_loader_vars:
        if key not in args.__dict__:
            args.__dict__[key] = data_loader_vars[key]
    # print(args.__dict__.keys())

    model_para_dict = utils.get_init_paras_dict(model_name, vars(args))
    logging.info(init_args.model_name + ': ' + str(model_para_dict))
    model = model_name(**model_para_dict)

    # init model paras
    model.apply(model.init_paras)

    # use gpu
    if torch.cuda.device_count() > 0:
        # model = model.to('cuda:0')
        model = model.cuda()

    # create runner
    runner_para_dict = utils.get_init_paras_dict(runner_name, vars(args))
    logging.info(init_args.runner_name + ': ' + str(runner_para_dict))
    runner = runner_name(**runner_para_dict)

    # training/testing
    # utils.format_metric(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
    logging.info('Test Before Training: train= %s validation= %s test= %s' % (
        utils.format_metric(
            runner.evaluate(model, data_processor.get_train_data(epoch=-1, model=model), data_processor)),
        utils.format_metric(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor)),
        utils.format_metric(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
        if args.unlabel_test == 0 else '-1') + ' ' + ','.join(runner.metrics))
    # if load > 0, load model to continue training
    if args.load > 0:
        model.load_model()
    # if train > 0, need training, otherwise, test directly
    if args.train > 0:
        runner.train(model, data_processor)

    # logging.info('Test After Training: train= %s validation= %s test= %s' % (
    #     utils.format_metric(
    #         runner.evaluate(model, data_processor.get_train_data(epoch=-1, model=model), data_processor)),
    #     utils.format_metric(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor)),
    #     utils.format_metric(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
    #     if args.unlabel_test == 0 else '-1') + ' ' + ','.join(runner.metrics))

    # save test results
    train_result = runner.predict(model, data_processor.get_train_data(epoch=-1, model=model), data_processor)
    validation_result = runner.predict(model, data_processor.get_validation_data(model=model), data_processor)
    test_result = runner.predict(model, data_processor.get_test_data(model=model), data_processor)
    np.save(args.result_file.replace('.npy', '__train.npy'), train_result)
    np.save(args.result_file.replace('.npy', '__validation.npy'), validation_result)
    np.save(args.result_file.replace('.npy', '__test.npy'), test_result)
    logging.info('Save Results to ' + args.result_file)

    all_metrics = ['rmse', 'mae', 'auc', 'f1', 'accuracy', 'precision', 'recall']
    if init_args.rank == 1:
        all_metrics = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@20', 'ndcg@50', 'ndcg@100'] \
                      + ['hit@1', 'hit@5', 'hit@10', 'hit@20', 'hit@50', 'hit@100'] \
                      + ['precision@1', 'precision@5', 'precision@10', 'precision@20', 'precision@50', 'precision@100'] \
                      + ['recall@1', 'recall@5', 'recall@10', 'recall@20', 'recall@50', 'recall@100']
    results = [train_result, validation_result, test_result]
    name_map = ['Train', 'Valid', 'Test']
    datasets = [data_processor.get_train_data(epoch=-1, model=model), data_processor.get_validation_data(model=model)]
    if args.unlabel_test != 1:
        datasets.append(data_processor.get_test_data(model=model))
    for i, dataset in enumerate(datasets):
        metrics = model.evaluate_method(results[i], datasets[i], metrics=all_metrics, error_skip=True)
        log_info = 'Test After Training on %s: ' % name_map[i]
        log_metrics = ['%s=%s' % (metric, utils.format_metric(metrics[j])) for j, metric in enumerate(all_metrics)]
        log_info += ', '.join(log_metrics)
        logging.info(os.linesep + log_info + os.linesep)

    if args.verbose <= logging.DEBUG:
        if args.unlabel_test == 0:
            logging.debug(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
            logging.debug(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
        else:
            logging.debug(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor))
            logging.debug(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor))

    logging.info('# of params: %d' % model.total_parameters)
    logging.info(vars(origin_args))
    logging.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    if args.counterfactual_constraint == 1:
        DiscreteCCF(init_args, logging, data_loader, data_processor, model, runner, parser)
        
    return
    
def DiscreteCCF(init_args, logging, data_loader, data_processor, model, runner, parser):
    # Discrete counterfactual constraint
    logging.info('Given a trained model, apply counterfactual constraint on it to improve.')
    
    init_args.model_name = "CCF" + init_args.model_name
    model_name = eval(init_args.model_name)
    
    old_data_processor = data_processor
    old_model = model
    old_runner = runner
    
    
    
    init_args.data_processor = model_name.data_processor
    init_args.runner_name = model_name.runner
        
    data_processor_name = eval(init_args.data_processor)
    runner_name = eval(init_args.runner_name)
    parser = data_processor_name.parse_ctf_dp_args(parser)
    parser = runner_name.parse_ctf_runner_args(parser)
    origin_args, extras = parser.parse_known_args()

    # log,model,result filename
    paras = sorted(vars(origin_args).items(), key=lambda kv: kv[0])
    log_name_exclude = ['check_epoch', 'eval_batch_size', 'gpu', 'label', 'load', 'ctf_load', 
                        'log_file', 'metrics', 'model_path', 'path', 'pre_gpu', 'result_file',
                        'sep', 'seq_sep', 'train', 'ctf_train', 'unlabel_test', 'verbose',
                        'dataset', 'random_seed', 'counterfactual_constraint', 'early_stop']
    if init_args.model_name in {'CCFNLRRec'}:
        log_file_name = [str(init_args.rank) + str(origin_args.drop_neg) + str(origin_args.counterfactual_constraint),
                     str(origin_args.random_seed)] + \
                     [p[0].replace('_', '')[:3] + str(p[1]) for p in paras if p[0] not in log_name_exclude]
    else:
        log_file_name = [str(init_args.rank) + str(origin_args.drop_neg) + str(origin_args.counterfactual_constraint),
                     init_args.model_name, origin_args.dataset[:3], str(origin_args.random_seed)] + \
                     [p[0].replace('_', '')[:3] + str(p[1]) for p in paras if p[0] not in log_name_exclude]
    log_file_name = [l.replace(' ', '-').replace('_', '-') for l in log_file_name]
    log_file_name = '_'.join(log_file_name)
    if origin_args.log_file == os.path.join(LOG_DIR, 'log.txt'):
        origin_args.log_file = os.path.join(LOG_DIR, '%s/%s/%s/%s.txt' % (init_args.model_name, origin_args.dataset, origin_args.ctf_type, log_file_name))
    utils.check_dir_and_mkdir(origin_args.log_file)
    if origin_args.result_file == os.path.join(RESULT_DIR, 'result.npy'):
        origin_args.result_file = os.path.join(RESULT_DIR, '%s/%s/%s/%s.npy' % (init_args.model_name, origin_args.dataset, origin_args.ctf_type, log_file_name))
    utils.check_dir_and_mkdir(origin_args.result_file)
    if origin_args.model_path == os.path.join(MODEL_DIR, '%s/%s.pt' % (init_args.model_name[3:], init_args.model_name[3:])):
        origin_args.model_path = os.path.join(MODEL_DIR, '%s/%s/%s/%s.pt' % (init_args.model_name, origin_args.dataset, origin_args.ctf_type, log_file_name))
    utils.check_dir_and_mkdir(origin_args.model_path)
    args = copy.deepcopy(origin_args)
    
    args.data_loader, args.rank = data_loader, init_args.rank
        
    logging.info("Refer new logging info in file " + log_file_name)
        
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(vars(origin_args))
    logging.info(extras)
        
    
        
    logging.info('DataLoader: ' + init_args.data_loader)
    logging.info('Model: ' + init_args.model_name)
    logging.info('Runner: ' + init_args.runner_name)
    logging.info('DataProcessor: ' + init_args.data_processor)
        
        
    
    dp_para_dict = utils.get_init_paras_dict(data_processor_name, vars(args))
    logging.info(init_args.data_processor + ': ' + str(dp_para_dict))
    data_processor = data_processor_name(**dp_para_dict)
        
    data_processor.get_train_data(epoch=-1, model=old_model, runner=old_runner, processor=old_data_processor)
    data_processor.get_validation_data(model=model_name)
    data_processor.get_test_data(model=model_name)
#         pdb.set_trace()
        
    # create model
    # generate feature multi-hot
    features, feature_dims, feature_min, feature_max = \
        data_loader.feature_info(include_id=model_name.include_id,
                                 include_item_features=model_name.include_item_features,
                                 include_user_features=model_name.include_user_features)
    args.feature_num, args.feature_dims = len(features), feature_dims
    args.user_feature_num = len([f for f in features if f.startswith('u_')])
    args.item_feature_num = len([f for f in features if f.startswith('i_')])
    args.context_feature_num = len([f for f in features if f.startswith('c_')])
    data_loader_vars = vars(data_loader)
    for key in data_loader_vars:
        if key not in args.__dict__:
            args.__dict__[key] = data_loader_vars[key]
    
    model_para_dict = utils.get_init_paras_dict(model_name, vars(args))
    logging.info(init_args.model_name + ': ' + str(model_para_dict))
    model = model_name(**model_para_dict)
    if init_args.model_name in {'CCFNCF', 'CCFBiasedMF'}:
        model.upload_old(old_model)
    # init model paras
    model.apply(model.init_paras)

    # use gpu
    if torch.cuda.device_count() > 0:
        # model = model.to('cuda:0')
        model = model.cuda()

    # create runner
    runner_para_dict = utils.get_init_paras_dict(runner_name, vars(args))
    logging.info(init_args.runner_name + ': ' + str(runner_para_dict))
    runner = runner_name(**runner_para_dict)

    logging.info('Test Before Training: train= %s validation= %s test= %s' % (
        utils.format_metric(
            runner.evaluate(model, data_processor.get_train_data(epoch=-1, model=model, runner=old_runner, processor=old_data_processor), data_processor)),
        utils.format_metric(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor)),
        utils.format_metric(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
        if args.unlabel_test == 0 else '-1') + ' ' + ','.join(runner.metrics))
    # if load > 0，load model to continue training
    if args.ctf_load > 0:
        model.load_model()
    # if train > 0，need training, otherwise, test directly
    if args.ctf_train > 0:
        runner.train(model, data_processor, old_runner, old_data_processor)
        
    # save test results
    train_result = runner.predict(model, data_processor.get_train_data(epoch=-1, model=model, runner=old_runner, processor=old_data_processor), data_processor)
    validation_result = runner.predict(model, data_processor.get_validation_data(model=model), data_processor)
    test_result = runner.predict(model, data_processor.get_test_data(model=model), data_processor)
    np.save(args.result_file.replace('.npy', '__train.npy'), train_result)
    np.save(args.result_file.replace('.npy', '__validation.npy'), validation_result)
    np.save(args.result_file.replace('.npy', '__test.npy'), test_result)
    logging.info('Save Results to ' + args.result_file)

    all_metrics = ['rmse', 'mae', 'auc', 'f1', 'accuracy', 'precision', 'recall']
    if init_args.rank == 1:
        all_metrics = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@20', 'ndcg@50', 'ndcg@100'] \
                        + ['hit@1', 'hit@5', 'hit@10', 'hit@20', 'hit@50', 'hit@100'] \
                        + ['precision@1', 'precision@5', 'precision@10', 'precision@20', 'precision@50', 'precision@100'] \
                        + ['recall@1', 'recall@5', 'recall@10', 'recall@20', 'recall@50', 'recall@100']
    results = [train_result, validation_result, test_result]
    name_map = ['Train', 'Valid', 'Test']
    datasets = [data_processor.get_train_data(epoch=-1, model=model, runner=old_runner, processor=old_data_processor), data_processor.get_validation_data(model=model)]
    if args.unlabel_test != 1:
        datasets.append(data_processor.get_test_data(model=model))
    for i, dataset in enumerate(datasets):
        metrics = model.evaluate_method(results[i], datasets[i], metrics=all_metrics, error_skip=True)
        log_info = 'Test After Training on %s: ' % name_map[i]
        log_metrics = ['%s=%s' % (metric, utils.format_metric(metrics[j])) for j, metric in enumerate(all_metrics)]
        log_info += ', '.join(log_metrics)
        logging.info(os.linesep + log_info + os.linesep)

    if args.verbose <= logging.DEBUG:
        if args.unlabel_test == 0:
            logging.debug(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
            logging.debug(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
        else:
            logging.debug(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor))
            logging.debug(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor))

    logging.info('# of params: %d' % model.total_parameters)
    logging.info(vars(origin_args))
    logging.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # result_dict = runner.run_some_tensors(model, data_processor.get_train_data(epoch=-1, model=model), data_processor,
    #                                       dict_keys=['sth'])
    # pickle.dump(result_dict, open('./sth.pk', 'rb'))
    return

def ContinuousCCF(init_args):
    init_args.model_name = "ConCCF" + init_args.model_name
    model_name = eval(init_args.model_name)
    data_loader_name = eval(init_args.data_loader)
    init_args.data_processor = model_name.data_processor
    init_args.runner_name = model_name.runner
    data_processor_name = eval(init_args.data_processor)
    runner_name = eval(init_args.runner_name)
    
    
    parser = argparse.ArgumentParser(description='')
    parser = utils.parse_global_args(parser)
    parser = data_loader_name.parse_data_args(parser)
    parser = model_name.parse_model_args(parser, model_name=init_args.model_name)
    parser = runner_name.parse_ctf_runner_args(parser)
    parser = data_processor_name.parse_dp_args(parser)
        
    origin_args, extras = parser.parse_known_args()

    # log,model,result filename
    paras = sorted(vars(origin_args).items(), key=lambda kv: kv[0])
    log_name_exclude = ['check_epoch', 'eval_batch_size', 'gpu', 'label', 'load', 'ctf_load', 
                        'log_file', 'metrics', 'model_path', 'path', 'pre_gpu', 'result_file',
                        'sep', 'seq_sep', 'train', 'ctf_train', 'unlabel_test', 'verbose',
                        'dataset', 'random_seed', 'counterfactual_constraint', 'early_stop',
                        'check_ctf_loss', 'grad_clip']
    if init_args.model_name in {'CCFNLRRec'}:
        log_file_name = [str(init_args.rank) + str(origin_args.drop_neg) + str(origin_args.counterfactual_constraint),
                     str(origin_args.random_seed)] + \
                     [p[0].replace('_', '')[:3] + str(p[1]) for p in paras if p[0] not in log_name_exclude]
    else:
        log_file_name = [str(init_args.rank) + str(origin_args.drop_neg) + str(origin_args.counterfactual_constraint),
                     init_args.model_name, origin_args.dataset[:3], str(origin_args.random_seed)] + \
                     [p[0].replace('_', '')[:3] + str(p[1]) for p in paras if p[0] not in log_name_exclude]
    log_file_name = [l.replace(' ', '-').replace('_', '-') for l in log_file_name]
    log_file_name = '_'.join(log_file_name)
    if origin_args.log_file == os.path.join(LOG_DIR, 'log.txt'):
        origin_args.log_file = os.path.join(LOG_DIR, '%s/%s/%s.txt' % (init_args.model_name, origin_args.dataset, log_file_name))
    utils.check_dir_and_mkdir(origin_args.log_file)
    if origin_args.result_file == os.path.join(RESULT_DIR, 'result.npy'):
        origin_args.result_file = os.path.join(RESULT_DIR, '%s/%s/%s.npy' % (init_args.model_name, origin_args.dataset, log_file_name))
    utils.check_dir_and_mkdir(origin_args.result_file)
    if origin_args.model_path == os.path.join(MODEL_DIR, '%s/%s.pt' % (init_args.model_name, init_args.model_name)):
        origin_args.model_path = os.path.join(MODEL_DIR, '%s/%s/%s.pt' % (init_args.model_name, origin_args.dataset, log_file_name))
    utils.check_dir_and_mkdir(origin_args.model_path)
    args = copy.deepcopy(origin_args)
        
    logging.info("Continuous CCF")
        
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    
    logging.info(vars(init_args))
    logging.info(vars(origin_args))
    logging.info(extras)
    
        
    logging.info('DataLoader: ' + init_args.data_loader)
    logging.info('Model: ' + init_args.model_name)
    logging.info('Runner: ' + init_args.runner_name)
    logging.info('DataProcessor: ' + init_args.data_processor)
        
    # random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # default '0'
    logging.info("# cuda devices: %d" % torch.cuda.device_count())

    # create data_loader
    args.load_data = True
    dl_para_dict = utils.get_init_paras_dict(data_loader_name, vars(args))
    logging.info(init_args.data_loader + ': ' + str(dl_para_dict))
    data_loader = data_loader_name(**dl_para_dict)

    # # append_his by data_loader
    if 'all_his' in origin_args:
        data_loader.append_his(all_his=origin_args.all_his, max_his=origin_args.max_his,
                               neg_his=origin_args.neg_his, neg_his_neg=origin_args.neg_his_neg,
                               neg_column=origin_args.neg_column)

    # if topn recommendation, only keep positive samples, generate negative sampels during training process and convert label into binary value
    if init_args.rank == 1:
        data_loader.label_01()
        if origin_args.drop_neg == 1:
            data_loader.drop_neg()

    # create data_processor
    args.data_loader, args.rank = data_loader, init_args.rank
    dp_para_dict = utils.get_init_paras_dict(data_processor_name, vars(args))
    logging.info(init_args.data_processor + ': ' + str(dp_para_dict))
    data_processor = data_processor_name(**dp_para_dict)

    # # prepare train,test,validation samples before generating model and training
    data_processor.get_train_data(epoch=-1, model=model_name)
    data_processor.get_validation_data(model=model_name)
    data_processor.get_test_data(model=model_name)
    

        
    # create model
    # generate feature multi-hot
    features, feature_dims, feature_min, feature_max = \
        data_loader.feature_info(include_id=model_name.include_id,
                                 include_item_features=model_name.include_item_features,
                                 include_user_features=model_name.include_user_features)
    args.feature_num, args.feature_dims = len(features), feature_dims
    args.user_feature_num = len([f for f in features if f.startswith('u_')])
    args.item_feature_num = len([f for f in features if f.startswith('i_')])
    args.context_feature_num = len([f for f in features if f.startswith('c_')])
    data_loader_vars = vars(data_loader)
    for key in data_loader_vars:
        if key not in args.__dict__:
            args.__dict__[key] = data_loader_vars[key]
    # print(args.__dict__.keys())

#     pdb.set_trace()
    model_para_dict = utils.get_init_paras_dict(model_name, vars(args))
    logging.info(init_args.model_name + ': ' + str(model_para_dict))
    model = model_name(**model_para_dict)

    # init model paras
    model.apply(model.init_paras)

    # use gpu
    if torch.cuda.device_count() > 0:
        # model = model.to('cuda:0')
        model = model.cuda()

    # create runner
#     pdb.set_trace()
    runner_para_dict = utils.get_init_paras_dict(runner_name, vars(args))
    logging.info(init_args.runner_name + ': ' + str(runner_para_dict))
    runner = runner_name(**runner_para_dict)

    # training/testing
    # utils.format_metric(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
    logging.info('Test Before Training: train= %s validation= %s test= %s' % (
        utils.format_metric(
            runner.evaluate(model, data_processor.get_train_data(epoch=-1, model=model), data_processor)),
        utils.format_metric(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor)),
        utils.format_metric(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
        if args.unlabel_test == 0 else '-1') + ' ' + ','.join(runner.metrics))
    
    # if load > 0，load model to continue training
    if args.load > 0:
        model.load_model()
    # if train > 0，need training, otherwise, test directly
    if args.train > 0:
        runner.train(model, data_processor)

    # logging.info('Test After Training: train= %s validation= %s test= %s' % (
    #     utils.format_metric(
    #         runner.evaluate(model, data_processor.get_train_data(epoch=-1, model=model), data_processor)),
    #     utils.format_metric(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor)),
    #     utils.format_metric(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
    #     if args.unlabel_test == 0 else '-1') + ' ' + ','.join(runner.metrics))

    # save test results
    train_result = runner.predict(model, data_processor.get_train_data(epoch=-1, model=model), data_processor)
    validation_result = runner.predict(model, data_processor.get_validation_data(model=model), data_processor)
    test_result = runner.predict(model, data_processor.get_test_data(model=model), data_processor)
    np.save(args.result_file.replace('.npy', '__train.npy'), train_result)
    np.save(args.result_file.replace('.npy', '__validation.npy'), validation_result)
    np.save(args.result_file.replace('.npy', '__test.npy'), test_result)
    logging.info('Save Results to ' + args.result_file)

    all_metrics = ['rmse', 'mae', 'auc', 'f1', 'accuracy', 'precision', 'recall']
    if init_args.rank == 1:
        all_metrics = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@20', 'ndcg@50', 'ndcg@100'] \
                      + ['hit@1', 'hit@5', 'hit@10', 'hit@20', 'hit@50', 'hit@100'] \
                      + ['precision@1', 'precision@5', 'precision@10', 'precision@20', 'precision@50', 'precision@100'] \
                      + ['recall@1', 'recall@5', 'recall@10', 'recall@20', 'recall@50', 'recall@100']
    results = [train_result, validation_result, test_result]
    name_map = ['Train', 'Valid', 'Test']
    datasets = [data_processor.get_train_data(epoch=-1, model=model), data_processor.get_validation_data(model=model)]
    if args.unlabel_test != 1:
        datasets.append(data_processor.get_test_data(model=model))
    for i, dataset in enumerate(datasets):
        metrics = model.evaluate_method(results[i], datasets[i], metrics=all_metrics, error_skip=True)
        log_info = 'Test After Training on %s: ' % name_map[i]
        log_metrics = ['%s=%s' % (metric, utils.format_metric(metrics[j])) for j, metric in enumerate(all_metrics)]
        log_info += ', '.join(log_metrics)
        logging.info(os.linesep + log_info + os.linesep)

    if args.verbose <= logging.DEBUG:
        if args.unlabel_test == 0:
            logging.debug(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
            logging.debug(runner.evaluate(model, data_processor.get_test_data(model=model), data_processor))
        else:
            logging.debug(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor))
            logging.debug(runner.evaluate(model, data_processor.get_validation_data(model=model), data_processor))

    logging.info('# of params: %d' % model.total_parameters)
    logging.info(vars(origin_args))
    logging.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == '__main__':
    main()

    
    
    
    
