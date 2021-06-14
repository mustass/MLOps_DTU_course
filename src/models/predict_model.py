# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 09:45:50 2021

@author: bjorn

script to load test data, BERT model from transformer package, load previously 
trained weights and do inference on test data
"""

import torch
import torch.nn as nn
# import transformers
# from sklearn.metrics import confusion_matrix
import click
from path import Path
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from src.models.model import BERT_model
from transformers import AutoModel

device = 'cpu'

from src.models.clean_data import clean_data
# to run: python src/models/clean_data.py data/raw/AmazonProductReviews.csv
# @click.command()
# @click.argument('df_name', type=click.Path(exists=True))
df_name = 'data/raw/AmazonProductReviews.csv'
# train_text, val_text, test_text, train_labels, val_labels, test_labels = clean_data(df_name)
test_seq, test_mask, test_y = clean_data(df_name)

from src.models.model import BERT_model
# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')
# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = False
model = BERT_model(bert, n_class=2)
model = model.to(device)
weights_path = 'models/BERT_weights_1.pt'
model.load_state_dict(torch.load(weights_path))
# model.load_state_dict(torch.load(weights_path, map_location=device))
print('model weights loaded successfully!')
# get predictions for test data
with torch.no_grad():
    #
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()
# show model's performance
preds = np.argmax(preds, axis=1)
print(classification_report(test_y, preds))
cm = confusion_matrix(test_y,
                      preds)  # labels=['Automotive', 'Patio_Lawn_and_Garden']
print('Confusion Matrix: \n', cm)
