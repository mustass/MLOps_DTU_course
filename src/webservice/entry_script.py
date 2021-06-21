import json
import joblib
import numpy as np
from azureml.core import Workspace, Run
from azureml.core.model import Model
import yaml
import os
from src.webservice.model import BERT_model

# Called when the service is loaded
def init():
    global model
    with open('webservice/config.yml') as f:
        config = yaml.safe_load(f)

    name = config['experiment_name']
    full = config['training']['full']
    lr = config['training']['lr']
    datasets = config['data']['used_datasets']
    classes = sum([1 for k, v in datasets.items() if v==1])

    global model_path
    model_path = 'webservice/'+name
    #print(model_path)
    #model = BERT_model(full, n_class=classes, lr=lr)
    #model = model.load_from_checkpoint(model_path)

# Called when a request is received
def run(raw_data):
    print(raw_data)
    print(model_path)
    return [0 for i in raw_data]
    