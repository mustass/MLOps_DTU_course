import yaml
import os
from src.models.model import BERT_model

# Called when the service is loaded
def init():
    global model
    with open('src/webservice/config.yml') as f:
        config = yaml.safe_load(f)

    name = config['experiment_name']
    full = config['training']['full']
    lr = config['training']['lr']
    datasets = config['data']['used_datasets']
    classes = sum([1 for k, v in datasets.items() if v==1])

    model_path = 'src/webservice/'+name
    model = BERT_model(full, n_class=classes, lr=lr)
    model = model.load_from_checkpoint(model_path)

# Called when a request is received
def run(raw_data):
    return 0
    