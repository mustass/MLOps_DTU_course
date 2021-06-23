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
import os
import joblib
import shutil

from azureml.core import Workspace, Model, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import AciWebservice, AksWebservice
from azureml.core.compute import AksCompute, ComputeTarget 
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
                                location=setup['location'])

    raise ValueError("workspace_exists in YML file is supposed" +
                     "to be a boolean (true/false)")


def launch_deployment(flags):

    setup = flags['compute']
    deploy_info = flags['deploy']
    ws = get_workspace(setup)
    version = flags['deploy']['version']
    if version == 0:
        model = ws.models[deploy_info['model_name']]
    elif version > 0:
        model = ws.models[deploy_info['model_name']]
    elif version == -1:
        model = ws.models[deploy_info['model_name']]

    shutil.copyfile("requirements.txt", "deployment_requirements.txt")
    with open("deployment_requirements.txt", "a") as f:
        f.write("azureml-defaults")

    env = Environment.from_pip_requirements("deploymentEnv", 
                                        "deployment_requirements.txt")

    inference_config = InferenceConfig(source_directory='./src',
                            entry_script="webservice/entry_script.py",
                            environment=env)
    
    compute_config = AksCompute.provisioning_configuration(
                                    vm_size = deploy_info['vm_instance'],
                                    agent_count = deploy_info['agent_count'],
                                    location= deploy_info['location'],
                                    cluster_purpose=deploy_info['purpose'])
    
    compute_name = deploy_info['aks_compute_name'] 
    try:
        service_cluster = ComputeTarget.create(ws, compute_name,
                                            compute_config)
        service_cluster.wait_for_completion(show_output=True)
    except:
        pass

    deploy_conf = AksWebservice.deploy_configuration(
                                compute_target_name=compute_name,
                                auth_enabled=False,
                                token_auth_enabled=False)

    service_name = deploy_info['service_name']
    service = Model.deploy(ws, service_name, [model], inference_config, 
                        deploy_conf, overwrite=True)

    service.wait_for_deployment(show_output=True)

    print(service.state)
    print("Endpoint: ", service.scoring_uri)


def run(config):
    cloud = config['compute']['cloud']
    deploy_flag = config['deploy']['value']
    train_flag = config['training']['value']

    if train_flag:

        name = config['experiment_name']
        datasets = parse_datasets(config)
        lr = config['training']['lr']
        epochs = config['training']['max_epochs']
        full = config['training']['full']

        model = BERT_model(full, n_class=len(datasets), lr=lr)
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
            best_model_path = checkpoint_callback.best_model_path
            run = Run.get_context()
            print("\nRegistering the model...\n")
            print("Run ID is: " + str(run.id) + "\n")
            run.upload_file(name = './models/' + name, 
                    path_or_stream = best_model_path)
            run.register_model(model_path='./models/'+ name, 
                                model_name=name)
            shutil.copyfile(best_model_path, "./src/webservice/"+name)
            
            print("Model registered successfully")

    if deploy_flag and cloud:
        try:
            os.makedirs("./src/webservice", exist_ok=True)
        except FileExistsError:
            pass
        #run.upload_file(name="./config/config.yml",
        #                path_or_stream = './config/config.yml')
        print("Copying static files into source directory...")
        shutil.copyfile("./config/config.yml", "./src/webservice/config.yml")
        #shutil.copyfile("./src/models/model.py", "./src/webservice/model.py")
        print("Static files copied successfully.")
        launch_deployment(config)


@click.command()
@click.argument('config_file', default="./config/config.yml")
def main(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    run(config)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    #load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
