# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
from utils.global_p import *
import numpy as np

import pdb

class ConCCFGRU4Rec(RecModel):
    data_processor = 'HistoryDP'  # default data_processor
    runner = 'ContinuousCTFRunner'

    @staticmethod
    def parse_model_args(parser, model_name='ConCCFGRU4Rec'):
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Size of hidden vectors in GRU.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of GRU layers.')
        parser.add_argument('--p_layers', type=str, default='[64]',
                            help="Size of each layer.")
        parser.add_argument('--neg_emb', type=int, default=1,
                            help="Whether use negative interaction embeddings.")
        parser.add_argument('--neg_layer', type=str, default='[]',
                            help="Whether use a neg_layer to transfer negative interaction embeddings. "
                                 "[] means using -v. It is ignored when neg_emb=1")
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, neg_emb, neg_layer, hidden_size, num_layers, p_layers, *args, **kwargs):
        self.neg_emb = neg_emb
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_layers = p_layers if type(p_layers) == list else eval(p_layers)
        self.neg_layer = neg_layer if type(neg_layer) == list else eval(neg_layer)
        RecModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.l2_embeddings = ['iid_embeddings']
        if self.neg_emb == 1:
            self.iid_embeddings_neg = torch.nn.Embedding(self.item_num, self.ui_vector_size)
            self.l2_embeddings.append('iid_embeddings_neg')
        elif len(self.neg_layer) > 0:
            pre_size = self.ui_vector_size
            for i, layer_size in enumerate(self.neg_layer):
                setattr(self, 'neg_layer_%d' % i, torch.nn.Linear(pre_size, layer_size))
                pre_size = layer_size
            self.neg_layer_out = torch.nn.Linear(pre_size, self.ui_vector_size)

        self.rnn = torch.nn.GRU(
            input_size=self.ui_vector_size, hidden_size=self.hidden_size, batch_first=True,
            num_layers=self.num_layers)
        self.out = torch.nn.Linear(self.hidden_size, self.ui_vector_size, bias=False)

    def predict(self, feed_dict, ctf=100, epsilon=0.1, train=False):
        check_list, embedding_l2 = [], []
        i_ids = feed_dict[IID]
        history = feed_dict[C_HISTORY]  # B * H
        lengths = feed_dict[C_HISTORY_LENGTH]  # B

        valid_his = history.abs().gt(0).long()  # B * H
        his_pos_neg = history.ge(0).unsqueeze(-1).float()  # B * H * 1
        his_length = valid_his.sum(dim=-1)  # B

        pos_his_vectors = self.iid_embeddings(history.abs() * valid_his) * \
                          valid_his.unsqueeze(dim=-1).float()  # B * H * V
        if self.neg_emb == 1:
            neg_his_vectors = self.iid_embeddings_neg(history.abs() * valid_his) * valid_his.unsqueeze(dim=-1).float()
            his_vectors = pos_his_vectors * his_pos_neg + (-his_pos_neg + 1) * neg_his_vectors  # B * H * V
        elif len(self.neg_layer) > 1:
            pre_layer = pos_his_vectors  # B * H * V
            for i in range(0, len(self.neg_layer)):
                pre_layer = getattr(self, 'neg_layer_%d' % i)(pre_layer)
                pre_layer = F.relu(pre_layer)
            neg_his_vectors = self.neg_layer_out(pre_layer)  # B * H * V
            his_vectors = pos_his_vectors * his_pos_neg + (-his_pos_neg + 1) * neg_his_vectors  # B * H * V
            his_vectors = his_vectors * valid_his.unsqueeze(dim=-1).float()  # B * H * V
        else:
            his_vectors = (his_pos_neg - 0.5) * 2 * pos_his_vectors  # B * H * V
        embedding_l2.append(his_vectors)

        # Sort
        sort_his_lengths, sort_idx = torch.topk(his_length, k=len(lengths))  # B
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)  # B * H * V

        # Pack
        history_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_his_vectors, sort_his_lengths, batch_first=True)

        # RNN
        output, hidden = self.rnn(history_packed, None)

        # Unsort
        sort_rnn_vector = self.out(hidden[-1])  # B * V
        unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]  # B
        rnn_vector = sort_rnn_vector.index_select(dim=0, index=unsort_idx)  # B * V
#         pdb.set_trace()

        # Predict
        cf_i_vectors = self.iid_embeddings(i_ids)  # B * V
        embedding_l2.append(cf_i_vectors)

        prediction = (rnn_vector * cf_i_vectors).sum(dim=1).view([-1])
        check_list.append(('prediction', prediction))
        
        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        if ctf > 0 and train:
            
            noise = torch.normal(mean=0.0, std=epsilon, size = his_vectors.unsqueeze(3).expand(-1,-1,-1,ctf).size())
            noise = F.normalize(noise, p=2, dim=2)
            noise = noise * torch.sqrt(torch.tensor(np.random.uniform(low=0, high=epsilon)))
                
            ctf_his_vector = his_vectors.unsqueeze(3).expand(-1,-1,-1,ctf) + noise.to(his_vectors.device)
                
            ctf_his_dist = torch.norm(noise, dim=2).mean(dim=1)
            shape = ctf_his_vector.size()
            ctf_his_vector = ctf_his_vector.reshape(shape[0]*shape[3], shape[1], shape[2])
            his_length = his_length.unsqueeze(0).expand(ctf, -1).reshape(-1)
            sort_his_lengths, sort_idx = torch.topk(his_length, k=len(his_length))  # B
            sort_his_vectors = ctf_his_vector.index_select(dim=0, index=sort_idx)  # B * H * V
            
                
                # Pack
            history_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_his_vectors, sort_his_lengths, batch_first=True)
                # RNN
            output, hidden = self.rnn(history_packed, None)

                # Unsort
            sort_rnn_vector = self.out(hidden[-1])  # B * V
#             sort_rnn_vector = sort_rnn_vector.reshape(ctf_his_vector.size())
            unsort_idx = torch.topk(sort_idx, k=len(sort_idx), largest=False)[1]  # B
            rnn_vector = sort_rnn_vector.index_select(dim=0, index=unsort_idx)  # B * V

                # Predict
            i_ids = i_ids.unsqueeze(0).expand(ctf, -1).reshape(-1)
            cf_i_vectors = self.iid_embeddings(i_ids)  # B * V
            
            

            ctf_predictions = (rnn_vector * cf_i_vectors).sum(dim=1).view([-1])
            ctf_predictions = ctf_predictions.reshape(shape[3], shape[0]).transpose(1,0)
                
            out_dict[CTF_PREDICTION] = ctf_predictions
            out_dict[CTF_HIS_DIST] = ctf_his_dist
            out_dict[DIM] = his_vectors.shape[1]
#         pdb.set_trace()
        return out_dict
    
    def forward(self, feed_dict, ctf=100, epsilon=0.1):
        """
        calculate predictions and loss
        :param feed_dict: intput dict
        :ctf: number of counterfactual examples
        :epsilon: distance constraint of counterfactual examples
        :return: output，a dict，prediction represents prediction values，check represents intermediate results，loss represents loss values.
        """
        out_dict = self.predict(feed_dict, ctf, epsilon, True)
        if feed_dict[RANK] == 1:
            # ranking task
            loss = self.rank_loss(out_dict[PREDICTION], feed_dict[Y], feed_dict[REAL_BATCH_SIZE])
        else:
            # rating task
            if self.loss_sum == 1:
                loss = torch.nn.MSELoss(reduction='sum')(out_dict[PREDICTION], feed_dict[Y])
            else:
                loss = torch.nn.MSELoss(reduction='mean')(out_dict[PREDICTION], feed_dict[Y])
        out_dict[LOSS] = loss
        out_dict[LOSS_L2] = self.l2(out_dict)
        return out_dict