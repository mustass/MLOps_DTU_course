#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 17:40:38 2021

@author: asgermunch
"""

import yaml
from src.data.lightning_data_module import MyDataModule


def test_data(config_path='config/test_config.yml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    data = MyDataModule(config)

    # Some test of the model
    data.prepare_data()
    data.setup()
    data = next(iter(data.train_dataloader()))
    seq = data[0]
    assert seq.shape == (
        config['training']['batch_size'], config['data']['max_seq_length']
    ), 'The output shape is not correct, expected shape {} but got shape {}'.format(
        [config['training']['batch_size'], config['data']['max_seq_length']],
        [shap for shap in seq.shape])
    mask = data[1]
    assert mask.shape == (
        config['training']['batch_size'], config['data']['max_seq_length']
    ), 'The output shape is not correct, expected shape {} but got shape {}'.format(
        [config['training']['batch_size'], config['data']['max_seq_length']],
        [shap for shap in mask.shape])
    labels = data[2]
    assert len(
        labels.shape) == 1, 'The labels should only have a single dimension.'
    assert labels.shape[0] == (
        config['training']['batch_size']
    ), 'The output shape is not correct, expected shape {} but got shape {}'.format(
        [config['training']['batch_size']], [shap for shap in labels.shape])
