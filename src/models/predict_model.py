import torch
import torch.nn as nn
import click
import yaml
import logging
from path import Path
import numpy as np
from dotenv import find_dotenv, load_dotenv
from sklearn.metrics import classification_report, confusion_matrix
from src.models.model import BERT_model
from src.data.fetch_dataset import parse_datasets
from transformers import AutoModel


@click.command()
@click.argument('config_path',
                type=click.Path(exists=True),
                default='./config/config.yml')
class Predictor(object):
    def __init__(self, config_path):

        with open(config_path) as f:
            yml = yaml.safe_load(f)
            self.device = yml['device']
            self.name = yml['experiment_name']
            self.datasets = parse_datasets(config_path)

        self.predicted = False
        self.model = BERT_model(AutoModel.from_pretrained('bert-base-uncased'),
                                n_class=len(self.datasets)).to(self.device)

        try:
            self.model.load_state_dict(
                torch.load('./models/' + str(self.name) + '/checkpoint.pt'))
        except Exception as ex:
            if type(ex) == FileNotFoundError:
                print("lol2")
                raise FileNotFoundError(
                    "The model checkpoint could not be found in './models/" +
                    str(self.name) +
                    "/'. Train this model or select another one.")
        print(str(self.name) + ' model weights loaded successfully!')
        self.model.eval()

    def run_predictions(self):
        """
        Run the predictions on a given dataset and save them.
        The function would have to check whether or not it has run based on model name and test set.  
        """

        # get predictions for test data
        with torch.no_grad():
            #
            preds = self.model(test_seq.to(device), test_mask.to(device))
            preds = preds.detach().cpu().numpy()
        # show model's performance
        preds = np.argmax(preds, axis=1)
        print(classification_report(test_y, preds))
        cm = confusion_matrix(
            test_y, preds)  # labels=['Automotive', 'Patio_Lawn_and_Garden']
        print('Confusion Matrix: \n', cm)

        self.predicted = True

    def create_stats(self):
        """
        Will log stuff based on predictions
        """


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    Predictor()
