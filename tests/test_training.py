#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 12:10:53 2021

@author: asgermunch
"""

import yaml
from src.models.model import BERT_model
from src.data.lightning_data_module import MyDataModule
from src.data.fetch_dataset import parse_datasets
import torch

def test_training(config_path='config/config.yml'):
    # Load config file
    with open(config_path) as f:
            config = yaml.safe_load(f)
    # Load data
    data = MyDataModule(config)
    datasets = parse_datasets(config)
    
    # Initialise model
    lr = config['training']['lr']
    model = BERT_model(False, n_class=len(datasets), lr=lr)
    
    # Some test of the model
    data.prepare_data()
    data.setup()
    # Single epoch
    preloss = 0
    count = 0
    epsilon = 1e-5
    for batch in data.train_dataloader():
        count += 1
        seq, mask, labels = batch
        outputDict = model.training_step(batch,None)
        loss = outputDict['loss']
        if count > 100:
            print('Finished!')
            break
        assert loss == loss, 'Something went wrong, loss = {}'.format(loss)
        diff = torch.abs(loss-preloss).item()
        largest = loss if loss > preloss else preloss
        assert not diff <= largest*epsilon, 'The loss did not change, previous loss {} new loss {}'.format(preloss, loss)
        preloss = loss