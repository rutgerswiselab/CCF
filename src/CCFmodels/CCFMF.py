# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils
from utils.global_p import *

import pdb


class CCFMF(RecModel):
    data_processor = 'nonSeqCTFGenerator'  # default data_processor
    runner = 'DiscreteCTFRunner'
    
    def _init_weights(self):
        self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
        self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.user_bias = torch.nn.Embedding(self.user_num, 1)
        self.item_bias = torch.nn.Embedding(self.item_num, 1)
        self.global_bias = torch.nn.Parameter(torch.tensor(0.1))
        self.l2_embeddings = ['uid_embeddings', 'iid_embeddings', 'user_bias', 'item_bias']
        
    def upload_old(self, model):
        self.model = model

    def predict(self, feed_dict):
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
        out_dict[CTF_PREDICTION] = self.model.predict(feed_dict)[PREDICTION]
        return out_dict
