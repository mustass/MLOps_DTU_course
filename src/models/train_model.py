from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.model import BERT_model
from transformers import AutoModel
from src.data.lightning_data_module import MyDataModule
from pytorch_lightning.loggers import WandbLogger
from src.data.fetch_dataset import parse_datasets
import click, logging, yaml
from dotenv import load_dotenv, find_dotenv
from azureml.core import Run
import wandb
import os
import joblib

from azureml.core import Workspace, Model, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

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

def launch_deployment(flags):

    setup = flags['compute']
    ws = get_workspace(setup)
    model = ws.models[flags['experiment_name']]

    env = CondaDependencies()
    packages = ["joblib", "PyYAML"]    
    for package in packages:
        env.add_conda_package(package)

    with open("./src/web-service/env_file.yml","w") as f:
        f.write(env.serialize_to_string())

    inference_config = InferenceConfig(runtime= "python",
                            source_directory='.',
                            entry_script="./src/web-service/entry_script.py",
                            conda_file="./src/web-service/env_file.yml")

    deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                                        memory_gb = 1, 
                                                        auth_enabled=False)

    service_name = flags['experiment_name']
    service = Model.deploy(ws, service_name, [model], inference_config, 
                        deployment_config, overwrite=True)
    service.wait_for_deployment(True)

    print(service.state)
    print("Endpoint: ", service.scoring_uri)


def train(config):
    name = config['experiment_name']
    datasets = parse_datasets(config)
    lr = config['training']['lr']
    epochs = config['training']['max_epochs']
    full = config['training']['full']
    cloud = config['compute']['cloud']
    deploy = config['compute']['deploy']

    bert = AutoModel.from_pretrained('bert-base-uncased')

    if not full:
        for param in bert.parameters():
            param.requires_grad = False

    model = BERT_model(bert, n_class=len(datasets), lr=lr)

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
                      gpus=config['gpus'])

    trainer.fit(model, data)

    # model registration
    if cloud:
        print("\nRegistering the model...")
        best_model_path = checkpoint_callback.best_model_path
        run = Run.get_context()
        run.upload_file(name = './models/' + name, 
                path_or_stream = best_model_path)
        run.register_model(model_path='./models/'+ name, 
                            model_name=name)
        print("Model registered successfully")

    if deploy:
        try:
            os.makedirs("./src/web-service", exist_ok=True)
        except FileExistsError:
            pass
        run.upload_file(name="./config/config.yml",
                        path_or_stream = './config/config.yml')
        launch_deployment(config)

    run.wait_for_completion(show_output=False)

    if deploy:
        try:
            os.makedirs("./src/web-service", exist_ok=True)
        except FileExistsError:
            pass
        run.upload_file(name="./config/config.yml",
                        path_or_stream = './config/config.yml')
        launch_deployment(config)

    run.wait_for_completion(show_output=False)

@click.command()
@click.argument('config_file', default="./config/config.yml")
def main(config_file):
    with open(config_file) as f:
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
