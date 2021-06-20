#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 10:51:49 2021

@author: asgermunch
"""

import yaml
from src.models.model import BERT_model
from src.data.lightning_data_module import MyDataModule
from src.data.fetch_dataset import parse_datasets
from transformers import AutoModel


def test_model(config_path='config/test_config.yml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    data = MyDataModule(config)
    datasets = parse_datasets(config)

    # Initialise model
    lr = config['training']['lr']
    model = BERT_model(False, n_class=len(datasets), lr=lr)

    # Some test of the model
    data.prepare_data()
    data.setup()
    data = next(iter(data.train_dataloader()))
    seq = data[0]
    mask = data[1]
    # labels=data[2]
    outputs = model(seq, mask)
    assert outputs.shape == (
        config['training']['batch_size'], len(datasets)
    ), 'The output shape is not correct, expected shape {} but got shape {}'.format(
        [config['training']['batch_size'],
         len(datasets)], [shap for shap in outputs.shape])
