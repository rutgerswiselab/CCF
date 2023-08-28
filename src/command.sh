# train a base model and save.
# python main.py --rank 1 --model_name GRU4Rec --optimizer Adam --lr 0.001 --dataset [DATASET] --metric ndcg@10,precision@1 --l2 1e-5 --max_his 10 --sparse_his 0 --all_his 0 --neg_his 1 --neg_emb 0 --random_seed 2018 --gpu 3 --counterfactual_constraint 0


# load pretrained base model and apply discrete CCF (make sure base model exists) 
# make sure load as 1 and train as 0 to avoid retrain the base model.
# python main.py --rank 1 --model_name GRU4Rec --optimizer Adam --lr 0.001 --dataset [DATASET] --metric ndcg@10,precision@1 --l2 1e-5 --max_his 10 --sparse_his 0 --all_his 0 --neg_his 1 --neg_emb 0 --random_seed 2018 --gpu 3 --can_item 100 --load 1 --train 0 --epsilon1 0.1 --ctf_num 2 --cc_weight 0.001 --ctf-type R1N --topk 60


# apply continuous CCF
# python main.py --rank 1 --model_name GRU4Rec --optimizer Adam --lr 0.001 --dataset [DATASET] --metric ndcg@10,precision@1 --l2 1e-5 --max_his 10 --sparse_his 0 --all_his 0 --neg_his 1 --neg_emb 0 --random_seed 2018 --gpu 0 --epsilon1 0.01 --ctf_num 10 --cc_weight 0.1 --epsilon2 1 --counterfactual_constraint 2
