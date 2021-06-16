from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.data.fetch_dataset import ensemble
import src.data.make_dataset as make_dataset
import click, logging, yaml, gzip, pickle
from typing import Optional
from dotenv import load_dotenv, find_dotenv

# This is hackidy-hacky:
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# Hackidy hack over ¯\_(ツ)_/¯


class MyDataModule(LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.train_dims = None
        self.config = config

    def prepare_data(self):
        ensemble(self.config)
        make_dataset.clean_data(self.config)

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        self.train = self.load_datasets("train")
        self.val   = self.load_datasets("validate")
        self.test  = self.load_datasets("test")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.config['training']['batch_size'], num_workers= self.config['training']['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.config['training']['batch_size'], num_workers= self.config['training']['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.config['training']['batch_size'],num_workers= self.config['training']['num_workers'])

    
    def load_datasets(self, set_name):
        try:
            f = gzip.open(
                './data/processed/' + str(self.config['experiment_name']) + '/' + str(set_name) +
                '.pklz', 'rb')
            return pickle.load(f, encoding="bytes")
        except Exception as ex:
            if type(ex) == FileNotFoundError:
                raise FileNotFoundError(
                    "The datasets could not be found in './data/processed/" +
                    str(self.config['experiment_name']) + "/'.")


@click.command()
@click.argument('config_path',
                type=click.Path(exists=True),
                default='./config/config.yml')
def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    mod = MyDataModule(config)
    mod.setup()
    print(mod.train_dataloader())


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    #load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
