import json
import joblib
import numpy as np
from azureml.core import Workspace, Run
from azureml.core.model import Model
import yaml
import os

def get_workspace(setup):
    if setup['workspace_exists']:
        return Workspace.get(name=setup['workspace_name'],
                             subscription_id=setup['subscription_id'],
                             resource_group=setup['resource_group'])
    if not setup['workspace_exists']:
        return Workspace.create(name=setup['workspace_name'],
               subscription_id=setup['subscription_id'],
               resource_group=setup['resource_group'],
               create_resource_group=True,
               location=setup['location']
               )

    raise ValueError("workspace_exists in YML file is supposed" +
                     "to be a boolean (true/false)")

# Called when the service is loaded
def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), '/models/')
    model = joblib.load(model_path+"monki-see-monki-sleep")

    #global model
    #model = Model(ws, flags['experiment_name'])
    #model = BERT_model.load_from_checkpoint(model_path)
    #model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    
    return 0