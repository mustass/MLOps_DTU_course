from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.model import BERT_model
from src.data.lightning_data_module import MyDataModule
from pytorch_lightning.loggers import WandbLogger
from src.data.fetch_dataset import parse_datasets
import click, logging, yaml
from dotenv import load_dotenv, find_dotenv
from azureml.core import Run
import wandb


def train(config):
    name = config['experiment_name']
    datasets = parse_datasets(config)
    lr = config['training']['lr']
    epochs = config['training']['max_epochs']
    full = config['training']['full']
    cloud = config['compute']['cloud']

    model = BERT_model(full,n_class=len(datasets), lr=lr)

    data = MyDataModule(config)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath='./models/checkpoints',
                                          filename=name +
                                          '-{epoch:02d}-{val_loss:.2f}',
                                          save_top_k=3,
                                          mode='min')

    wandb.login(key=config['wandbkey'])
    logger = WandbLogger(name=name)
    trainer = Trainer(logger=logger,
                      max_epochs=epochs,
                      callbacks=[checkpoint_callback],
                      gpus = config['gpus'])

    trainer.fit(model, data)

    # model registration
    if cloud:
        best_model_path = checkpoint_callback.best_model_path
        run = Run.get_context()
        run.register_model(model_path=best_model_path, 
                            model_name=name)


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    train(config)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    #load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
