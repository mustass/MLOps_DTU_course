import yaml
import os
import torch
from src.models.model import BERT_model
from transformers import BertTokenizerFast
from torch.utils.data import TensorDataset
from azureml.core import Workspace

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
    global config
    with open('src/webservice/config.yml') as f:
        config = yaml.safe_load(f)

    if not config['training']['value'] and config['deploy']['value']:
        ws = get_workspace(config['compute'])
        run = ws.get_run(config['deploy']['run_id'])
        run.download_file('./models/' + config['experiment_name'], 'src/models/')

    name = config['experiment_name']
    full = config['training']['full']
    lr = config['training']['lr']
    datasets = config['data']['used_datasets']
    classes = sum([1 for k, v in datasets.items() if v==1])

    model_path = 'src/models/'+name
    model = BERT_model(full, n_class=classes, lr=lr)
    model = model.load_from_checkpoint(model_path)

# Called when a request is received
def run(raw_data):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    tokens_train = tokenizer.batch_encode_plus(raw_data,
                                               max_length=config['data']['max_seq_length'],
                                               padding=True,
                                               truncation=True)

    data = TensorDataset(torch.tensor(tokens_train['input_ids']),
                        torch.tensor(tokens_train['attention_mask']))
    
    preds = model.predict_step(data)
    return preds
    