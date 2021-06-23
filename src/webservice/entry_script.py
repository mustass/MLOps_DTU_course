import yaml
import os
import torch
from src.models.model import BERT_model
from transformers import BertTokenizerFast
from torch.utils.data import TensorDataset

# Called when the service is loaded
def init():
    global model
    global config
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
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    tokens_train = tokenizer.batch_encode_plus(raw_data,
                                               max_length=config['data']['max_seq_length'],
                                               padding=True,
                                               truncation=True)

    data = TensorDataset(torch.tensor(tokens_train['input_ids']),
                        torch.tensor(tokens_train['attention_mask']))
    
    preds = model.predict_step(data)
    return preds
    