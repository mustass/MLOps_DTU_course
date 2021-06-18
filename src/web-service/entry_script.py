import json
import joblib
import numpy as np
from azureml.core import Workspace, Run
from azureml.core.model import Model
import yaml
import os

# Called when the service is loaded
def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), '/models/')

    #global model
    #model = Model(ws, flags['experiment_name'])
    #model = BERT_model.load_from_checkpoint(model_path)
    #model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    
    return 0