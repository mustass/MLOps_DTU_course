import json
import joblib
import numpy as np
from azureml.core import Workspace, Run
from azureml.core.model import Model
import yaml
import os
from src.models.model import BERT_model

# Called when the service is loaded
def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 
                            '/models/monki-see-monki-sleep')
    with open(os.path.join(os.getenv('AZUREML_MODEL_DIR'), 
                                '/config/config.yml')) as f:
        config = yaml.safe_load(f)

    full = config['training']['full']
    lr = config['training']['lr']
    datasets = config['data']['used_datasets']
    classes = sum([1 for k, v in datasets.items() if v==1])

    model = BERT_model(full, n_class=classes, lr=lr)
    model = model.load_from_checkpoint(model_path)

# Called when a request is received
def run(raw_data):
    
    return 0