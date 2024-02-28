# Scenario-Wise-Rec: Benchmark for Multi-Scenario Recommendation 
<p align="left">
  <img src='https://img.shields.io/badge/python-3.8+-brightgreen'>
  <img src='https://img.shields.io/badge/torch-1.13+-brightgreen'>
  <img src='https://img.shields.io/badge/scikit_learn-1.2.1+-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.5.3+-brightgreen'>
  <img src="https://img.shields.io/pypi/l/torch-rechub">
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FXiaopengli1%2FScenario-Wise-Rec&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>

## Introduction
English | [简体中文](README_CN.md)

**Scenario-Wise-Rec**, an open-sourced benchmark for Multi-Scenario/Multi-Domain Recommendation. Check our paper: [Scenario-Wise Rec: A Multi-Scenario Recommendation Benchmark]().

## Installation

### Install via `pip`
We provide a Python package *scenario_wise_rec* for users. Simply run:
```sh
pip install -i https://test.pypi.org/simple/ scenario-wise-rec
```

Note that the pip installation could be behind the recent updates. So, if you want to use the latest features or develop based on our code, you should install via source code.

### Install via GitHub (Recommend)

First, clone the repo:
```sh
git clone git clone https://github.com/Xiaopengli1/Scenario-Wise-Rec.git
```

Then, 

```sh
cd Scenario-Wise-Rec
```

To install the required packages, you can create a conda environment:

```sh
conda create --name scenario-wise-rec python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

## Usage
