import torch
import torch.nn as nn
import click
import yaml
import gzip
import os
import pickle
import logging
from path import Path
import numpy as np
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import classification_report, confusion_matrix
from src.models.model import BERT_model
from src.data.fetch_dataset import parse_datasets, check_and_create_data_subfolders
from transformers import AutoModel


class Trainer():
    def __init__(self, config):
        self.device = config['device']
        self.name = config['experiment_name']
        self.datasets = parse_datasets(config)
        self.batch_size = config['training']['batch_size']
        self.lr = config['training']['lr']
        self.epochs = config['training']['epochs']
        self.full = config['training']['full']
        self.model = BERT_model(AutoModel.from_pretrained('bert-base-uncased'), n_class=len(self.datasets)).to(self.device)
        
    def check_if_trained(self):
        print("lo")
        if os.path.exists('./models/'+str(self.name)+'/checkpoint.pt'):
            if click.confirm('Looks like model '+str(self.name)+' has been trained. Do you want to continue?'):
                return False
            else:
                return True
        check_and_create_data_subfolders('./models/', subfolders=[str(self.name)])
        return False


    def load_datasets(self, set_name):
        try:
            f = gzip.open('./data/processed/'+str(self.name)+'/'+str(set_name)+'.pklz', 'rb')
            return pickle.load(f, encoding="bytes")
        except Exception as ex:
            if type(ex) == FileNotFoundError:
                raise FileNotFoundError(
                "The datasets could not be found in './data/processed/"+str(self.name)+"/'.")


    def train(self):
        self.trained = self.check_if_trained()
        print(self.trained)
        train_set = self.load_datasets("train")
        #val_set = self.load_datasets("validate")
        #test_set = self.load_datasets("test")
        print(train_set)
        #train_loop(train_set, self.model, self.lr, self.batch_size, self.epochs, './models/'+str(self.name))







def train_loop(dataset, model, lr, bs, epochs, savepath):

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    trainloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=bs,
                                              shuffle=True)

    train_losses = []
    train_counter = []
    for e in range(epochs):
        for batch_idx, (image, label) in enumerate(trainloader):
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0 or e == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    e, batch_idx * len(image), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append((batch_idx * 64) +
                                     ((e - 1) * len(trainloader.dataset)))
                torch.save(model.state_dict(),
                           savepath + '/checkpoints/model.pth')
                torch.save(optimizer.state_dict(),
                           savepath + '/checkpoints/optimizer.pth')
    return train_losses, train_counter


@click.command()
@click.argument('config_path',
                type=click.Path(exists=True),
                default='./config/config.yml') 
def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    print(trainer)
    print(trainer.check_if_trained())




if __name__ == '__main__':
    #log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())
    main()

