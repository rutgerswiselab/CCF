# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
from utils.global_p import *

import numpy as np

import pdb


class ConCCFBiasedMF(RecModel):
    data_processor = 'DataProcessor'  # default data_processor
    runner = 'ContinuousCTFRunner'
    
    def _init_weights(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1))
        self.l2_embeddings = ['uid_embeddings', 'iid_embeddings', 'user_bias', 'item_bias']
        

    def predict(self, feed_dict, ctf=100, epsilon=0.1, train=False):
#         pdb.set_trace()
        check_list, embedding_l2 = [], []
        u_ids = feed_dict[UID]
        i_ids = feed_dict[IID]

        # bias
        u_bias = self.user_bias(u_ids).view([-1])
        i_bias = self.item_bias(i_ids).view([-1])
        embedding_l2.extend([u_bias, i_bias])

        cf_u_vectors = self.uid_embeddings(u_ids)
        cf_i_vectors = self.iid_embeddings(i_ids)
        embedding_l2.extend([cf_u_vectors, cf_i_vectors])

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=1).view([-1])
        prediction = prediction + u_bias + i_bias + self.global_bias
        # prediction = prediction + self.global_bias
        check_list.append(('prediction', prediction))

        out_dict = {PREDICTION: prediction,
                    CHECK: check_list, EMBEDDING_L2: embedding_l2}
        if ctf > 0 and train:
            noise = torch.normal(mean=0.0, std=epsilon, size = cf_i_vectors.unsqueeze(2).expand(-1,-1,ctf).size())
            noise = F.normalize(noise, p=2, dim=2)
            noise = noise * torch.sqrt(torch.tensor(np.random.uniform(low=0, high=epsilon)))
            ctf_his_dist = torch.norm(noise, dim=1)
#                 pdb.set_trace()
            ctf_i_vector = cf_i_vectors.unsqueeze(2).expand(-1,-1,ctf) + noise.to(cf_i_vectors.device)
            ctf_prediction = (cf_u_vectors.unsqueeze(2).expand(-1,-1,ctf) * ctf_i_vector).sum(dim=1)
            ctf_prediction = ctf_prediction + u_bias.unsqueeze(1).expand(-1,ctf) + i_bias.unsqueeze(1).expand(-1,ctf) + self.global_bias
            out_dict[CTF_PREDICTION] = ctf_prediction
            out_dict[CTF_HIS_DIST] = ctf_his_dist
            out_dict[DIM] = 1 # cf_i_vectors.shape[1]
        return out_dict
    
    def forward(self, feed_dict, ctf=10, epsilon=0.1):
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
