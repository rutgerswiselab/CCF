# CCF

## Introduction
This repository includes the implementation for Causal Collaborative Filtering

> Paper: Causal Collaborative Filtering <br>
> Paper Link: [https://dl.acm.org/doi/abs/10.1145/3578337.3605122](https://dl.acm.org/doi/abs/10.1145/3578337.3605122)

## Environment

Environment requirements can be found in `./requirement.txt`

## Datasets
- Download the row data into `./data/'
  
- **ML-100k**: The origin dataset can be found [here](https://grouplens.org/datasets/movielens/100k/). 

- **Coat Shopping**: The origin dataset can be found [here](https://www.cs.cornell.edu/~schnabts/mnar/).

- The data processing code can be found in `./src/datasets/' and store the processed data in `./dataset/'

## Example to run the codes

- Some running commands can be found in `./src/command.sh`

## Citation

```
@inproceedings{xu2023causal,
  title={Causal collaborative filtering},
  author={Xu, Shuyuan and Ge, Yingqiang and Li, Yunqi and Fu, Zuohui and Chen, Xu and Zhang, Yongfeng},
  booktitle={Proceedings of the 2023 ACM SIGIR International Conference on Theory of Information Retrieval},
  pages={235--245},
  year={2023}
}
```
