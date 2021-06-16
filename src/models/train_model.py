from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.model import BERT_model
from src.data.lightning_data_module import MyDataModule
from pytorch_lightning.loggers import WandbLogger
from src.data.fetch_dataset import parse_datasets
from transformers import AutoModel
import click, logging, yaml
from dotenv import load_dotenv, find_dotenv

def train(config):
    name = config['experiment_name']
    datasets = parse_datasets(config)
    lr = config['training']['lr']
    epochs = config['training']['max_epochs']
    full = config['training']['full']
    bert = AutoModel.from_pretrained('bert-base-uncased')

    if not full:
        for param in bert.parameters():
            param.requires_grad = False

    model = BERT_model(bert, n_class=len(datasets),lr=lr)

    data = MyDataModule(config)
    
    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./models/checkpoints',
    filename=name+'-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min')

    logger = WandbLogger("wandb_logs_"+name)   
    trainer = Trainer(logger =logger ,max_epochs =epochs,callbacks=[checkpoint_callback])
    trainer.fit(model, data)



@click.command()
@click.argument('config_path',
                type=click.Path(exists=True),
                default='./config/config.yml')
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
