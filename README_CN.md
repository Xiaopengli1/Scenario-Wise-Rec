# Scenario-Wise-Rec: Benchmark for Multi-Scenario Recommendation 
<p align="left">
  <img src='https://img.shields.io/badge/python-3.8+-brightgreen'>
  <img src='https://img.shields.io/badge/torch-1.13+-brightgreen'>
  <img src='https://img.shields.io/badge/scikit_learn-1.2.1+-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.5.3+-brightgreen'>
  <img src="https://img.shields.io/pypi/l/torch-rechub">
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FXiaopengli1%2FScenario-Wise-Rec&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>

## Introduction
[English](README.md) | 简体中文

**Scenario-Wise-Rec**, 是我们针对多域/多场景开发的首个开源Benchmark. 包含4个公开数据集上，8种模型。具体参照以下表格。

数据集统计表格：

| Dataset   | Domain number | Interaction | User    | Item      |
|-----------|---------------|-------------|---------|-----------|
| MovieLens | Domain 0      | 210,747     | 1,325   | 3,429     |
|           | Domain 1      | 395,556     | 2,096   | 3,508     |
|           | Domain 2      | 393,906     | 2,619   | 3,595     |
| KuaiRand  | Domain 0      | 2,407,352   | 961     | 1,596,491 |
|           | Domain 1      | 7,760,237   | 991     | 2,741,383 |
|           | Domain 2      | 895,385     | 171     | 332,210   |
|           | Domain 3      | 402,366     | 832     | 547,908   |
|           | Domain 4      | 183,403     | 832     | 43,106    |
| Ali-CCP   | Domain 0      | 32,236,951  | 89,283  | 465,870   |
|           | Domain 1      | 639,897     | 2,561   | 188,610   |
|           | Domain 2      | 52,439,671  | 150,471 | 467,122   |
| Tenrec    | Domain 0      | 64,475,979  | 997,263 | 1,365,660 |
|           | Domain 1      | 54,277,815  | 989,911 | 791,826   |
|           | Domain 2      | 1,588,512   | 455,636 | 152,601   |

模型介绍：

| Model         | model-name     | Link                                              |
|---------------|----------------|---------------------------------------------------|
| Shared Bottom | SharedBottom   | [Link](https://link.springer.com/article/10.1023/A:1007379606734) |
| MMOE          | MMOE           | [Link](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-) |
| PLE           | PLE            | [Link](https://dl.acm.org/doi/10.1145/3383313.3412236) |
| SAR-Net       | sarnet         | [Link](https://arxiv.org/abs/2110.06475) |
| STAR          | star           | [Link](https://dl.acm.org/doi/abs/10.1145/3459637.3481941) | 
| M2M           | m2m            | [Link](https://dl.acm.org/doi/abs/10.1145/3488560.3498479) |
| AdaSparse     | adasparse      | [Link](https://arxiv.org/abs/2206.13108) |
| AdaptDHM      | adaptdhm       | [Link](https://arxiv.org/abs/2211.12105) |


[//]: # (Check our paper: [Scenario-Wise Rec: A Multi-Scenario Recommendation Benchmark]&#40;&#41;.)

## 安装

### 通过 `pip` 安装
用户可通过 pip 直接安装 *scenario_wise_rec*. 只需运行以下代码:
```sh
pip install -i https://test.pypi.org/simple/ scenario-wise-rec
```
因为我们的benchmark处于更新中，所以我们建议通过以下的方式安装。
### 通过GitHub源码安装 (建议)

首先, clone the repo:
```sh
git clone git clone https://github.com/Xiaopengli1/Scenario-Wise-Rec.git
```

切换目录：

```sh
cd Scenario-Wise-Rec
```

完成安装：
```sh
pip install .
```

## 使用
我们为用户提供了运行脚本方便直接运行，参考 `/examples/multi_domain_ranking/`。因为某些数据集较大，为了方便调试，我们提供采样后的 dataset samples 在目录`/examples/multi_domain_ranking/data`. 测试直接运行
```sh
python run_ali_ccp_ctr_ranking_multi_domain.py --model star
```
对于全量数据集运行，参考以下步骤
### Step 1: 数据集下载

数据集介绍
| Dataset                                                                         | Domain  Number | Users           | Items | Items |    Download     |
|:--------------------------------------------------------------------------------|:---------------|:----------------|:-----:|:-----:|:---------------:|
| [Movie-Lens](https://grouplens.org/datasets/movielens/)                         | 3              | 6k              |  4k   |  1M   | [ML_Download](https://drive.google.com/file/d/1c8yqnw0U5oTfz_Yowtd9D37UUIIAeIiM/view?usp=sharing) | 
| [KuaiRand](https://kuairand.com/)                                               | 5              | 1k              |  4M   |  11M  | [KR_Download](https://drive.google.com/file/d/1-39JNTQ-NCW1O0bFA6YtP_Rg1yl0QiSQ/view?usp=sharing) | 
| [Ali-CCP](https://tianchi.aliyun.com/dataset/408)                               | 3              | 238k            | 467k  |  85M  | [AC_Download](https://drive.google.com/drive/folders/1plgdPg_MGlgJbyFr6FAqmWnAgkL-qAxm?usp=sharing) | 
| [Tenrec](https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec\_dataset.html) | 3              | 1M              |  2M   | 120M  | [TR_Download](https://drive.google.com/file/d/1mZcUlbXoEjBLTT7y9wqJacHzZsmh0V-I/view?usp=sharing) | 

将`/examples/multi_domain_ranking/data`下的数据集替换, 同时将对应脚本中的代码做修改（见代码标注）。

### Step 2: 运行
```sh
python run_movielens_rank_multi_domain.py --model_name star --device "cuda:0" --seed 2022 
```

[//]: # (## Citation)

## 贡献
欢迎提出 PR 和 Issue，共同为 Multi-Domain/Multi-Scenario 社区做出贡献！也希望大家star我们的项目，谢谢大家！