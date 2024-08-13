<p align="center">
<img src='figures/logo.png' height="180">
</p>

# 

<p align="left">
  <img src='https://img.shields.io/badge/python-3.8+-brightgreen'>
  <img src='https://img.shields.io/badge/torch-1.13+-brightgreen'>
  <img src='https://img.shields.io/badge/scikit_learn-1.2.1+-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.5.3+-brightgreen'>
  <img src="https://img.shields.io/pypi/l/torch-rechub">
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FXiaopengli1%2FScenario-Wise-Rec&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>

## 1. Introduction

[//]: # (English | [简体中文]&#40;README_CN.md&#41;)

**Scenario-Wise Rec**, an open-sourced benchmark for multi-scenario/multi-domain recommendation.

![structures](figures/structure_new.png)

<details>

<summary>Dataset introduction</summary>


| Dataset   | Domain number | # Interaction | # User    | # Item      |
|-----------|---------------|-------------|---------|-----------|
| [MovieLens](https://grouplens.org/datasets/movielens/) | Domain 0      | 210,747     | 1,325   | 3,429     |
|           | Domain 1      | 395,556     | 2,096   | 3,508     |
|           | Domain 2      | 393,906     | 2,619   | 3,595     |
| [KuaiRand](https://kuairand.com/) | Domain 0      | 2,407,352   | 961     | 1,596,491 |
|           | Domain 1      | 7,760,237   | 991     | 2,741,383 |
|           | Domain 2      | 895,385     | 171     | 332,210   |
|           | Domain 3      | 402,366     | 832     | 547,908   |
|           | Domain 4      | 183,403     | 832     | 43,106    |
| [Ali-CCP](https://tianchi.aliyun.com/dataset/408)   | Domain 0      | 32,236,951  | 89,283  | 465,870   |
|           | Domain 1      | 639,897     | 2,561   | 188,610   |
|           | Domain 2      | 52,439,671  | 150,471 | 467,122   |
| [Amazon](https://jmcauley.ucsd.edu/data/amazon/)    | Domain 0      | 198,502     | 22,363  | 12,101    |
|           | Domain 1      | 278,677     | 39,387  | 23,033    |
|           | Domain 2      | 346,355     | 38,609  | 18,534    |
| [Douban](https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information)    | Domain 0      | 227,251     | 2,212   | 95,872    |
|           | Domain 1      | 179,847     | 1,820   | 79,878    |
|           | Domain 2      | 1,278,401   | 2,712   | 34,893    |
| [Mind](https://msnews.github.io/)      | Domain 0      | 26,057,579  | 737,687 | 8,086     |
|           | Domain 1      | 11,206,494  | 678,268 | 1,797     |
|           | Domain 2      | 10,237,589  | 696,918 | 8,284     |
|           | Domain 3      | 9,226,382   | 656,970 | 1,804     |

</details>

<details>

<summary>Model introduction</summary>


| Model         | model_name     | Link                                              |
|---------------|----------------|---------------------------------------------------|
| Shared Bottom | SharedBottom   | [Link](https://link.springer.com/article/10.1023/A:1007379606734) |
| MMOE          | MMOE           | [Link](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-) |
| PLE           | PLE            | [Link](https://dl.acm.org/doi/10.1145/3383313.3412236) |
| SAR-Net       | sarnet         | [Link](https://arxiv.org/abs/2110.06475) |
| STAR          | star           | [Link](https://dl.acm.org/doi/abs/10.1145/3459637.3481941) | 
| M2M           | m2m            | [Link](https://dl.acm.org/doi/abs/10.1145/3488560.3498479) |
| AdaSparse     | adasparse      | [Link](https://arxiv.org/abs/2206.13108) |
| AdaptDHM      | adaptdhm       | [Link](https://arxiv.org/abs/2211.12105) |
| EPNet         | ppnet          | [Link](https://arxiv.org/abs/2302.01115) |
| PPNet         | epnet          | [Link](https://arxiv.org/abs/2302.01115) |

</details>

[//]: # (Check our paper: [Scenario-Wise Rec: A Multi-Scenario Recommendation Benchmark]&#40;&#41;.)

## 2. Installation
**WARNING**: Our package is still being developed, feel free to post issues if there are any usage problems.


[//]: # (### Install via `pip`)

[//]: # (We provide a Python package *scenario_wise_rec* for users. Simply run:)

[//]: # (```sh)

[//]: # (pip install -i https://test.pypi.org/simple/ scenario-wise-rec)

[//]: # (```)

[//]: # ()
[//]: # (Note that the pip installation could be behind the recent updates. So, if you want to use the latest features or develop based on our code, you should install via source code.)

### Install via GitHub (Recommended)

First, clone the repo:
```sh
git clone https://github.com/Xiaopengli1/Scenario-Wise-Rec.git
```

Then, 

```sh
cd Scenario-Wise-Rec
```

Then use pip to install our package:

```sh
pip install .
```

## 3. Usage
We provide running scripts for users. See `/scripts`, dataset samples are provided in `/scripts/data`. You could directly test it by simply do (such as for Ali-CCP):
```sh
python run_ali_ccp_ctr_ranking_multi_domain.py --model [model_name]
```
For full-dataset download, refer to the following steps.

### Step 1: Full Datasets Download

Four multi-scenario/multi-domain datasets are provided. See the following table.

| Dataset                                                                                          | Domain  Number | Users | Items | Interaction |    Download     |
|:-------------------------------------------------------------------------------------------------|:---------------|:------|:-----:|:-----------:|:---------------:|
| [Movie-Lens](https://grouplens.org/datasets/movielens/)                                          | 3              | 6k    |  4k   |     1M      | [ML_Download](https://drive.google.com/file/d/1c8yqnw0U5oTfz_Yowtd9D37UUIIAeIiM/view?usp=sharing) | 
| [KuaiRand](https://kuairand.com/)                                                                | 5              | 1k    |  4M   |     11M     | [KR_Download](https://drive.google.com/file/d/1-39JNTQ-NCW1O0bFA6YtP_Rg1yl0QiSQ/view?usp=sharing) | 
| [Ali-CCP](https://tianchi.aliyun.com/dataset/408)                                                | 3              | 238k  | 467k  |     85M     | [AC_Download](https://drive.google.com/drive/folders/1plgdPg_MGlgJbyFr6FAqmWnAgkL-qAxm?usp=sharing) | 
| [Amazon](https://jmcauley.ucsd.edu/data/amazon/)                                                 | 3              | 85k   |  54k  |    823k     | [AZ_Download](https://drive.google.com/file/d/1-3cmIlJUTD4m_1c8c5DpWWm9Pxz3Ed6x/view?usp=drive_link) | 
| [Douban](https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information) | 3              | 2k    | 210k  |    1.7M     | [DB_Download](https://drive.google.com/file/d/1CJbbiNLlyXXGofWMMkxQ_e3tg_1VByio/view?usp=sharing) | 
| [Mind](https://msnews.github.io/)                                                                | 4              | 748k  |  20k  |     56M     | [MD_Download](https://drive.google.com/file/d/10_f9q4C9pqnetfKRdygjTCZBS_od5_7z/view?usp=drive_link) | 


Substitute the full-dataset with the sampled dataset.

### Step 2: Run the Code 
```sh
python run_movielens_rank_multi_domain.py --dataset_path [path] --model_name [model_name] --device ["cpu"/"cuda:0"] --epoch [maximum epoch] --learning_rate [1e-3/1e-5] --batch_size [2048/4096] --seed [random seed] 
```

[//]: # (## Citation)

## 4. Build Your Own Multi-scenario Dataset/Model
We offer two template files [run_example.py](https://github.com/Xiaopengli1/Scenario-Wise-Rec/blob/main/scripts/run_example.py) and [base_example.py](https://github.com/Xiaopengli1/Scenario-Wise-Rec/blob/main/scenario_wise_rec/models/multi_domain/base_example.py) for a pipeline to help you to process different multi-scenario dataset and your own multi-scenario models. 

### Instructions on Processing Your Dataset
See [run_example.py](https://github.com/Xiaopengli1/Scenario-Wise-Rec/blob/main/scripts/run_example.py).
The function `get_example_dataset(input_path)` is an example to process your dataset. Be noted the feature 
`"domain_indicator"` is the feature that indicates domains. For other implementation details, refer to the raws file.

### Instructions on Building Your Model
See [base_example.py](https://github.com/Xiaopengli1/Scenario-Wise-Rec/blob/main/scenario_wise_rec/models/multi_domain/base_example.py).
Where you could build your own multi-scenario model here. We left two spaces for users to implement **scenario-shared** 
and **scenario-specific** models. We also leave comments on how to process the final output. Please refer to 
the raws file to see more details.  

## 5. Contributing
We welcome any contribution that could help improve the benchmark, and don't forget to star 🌟 our project!


## 6. Credits
The framework is referred to [Torch-RecHub](https://github.com/datawhalechina/torch-rechub). Thanks to their contribution.
