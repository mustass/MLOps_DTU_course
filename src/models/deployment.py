from azureml.core import Workspace, Model, Environment
import yaml
import joblib
from src.models.model import BERT_model
import click, logging, yaml
from dotenv import load_dotenv, find_dotenv
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
    with open("./requirements.txt") as f:
        pass#packages = [line.strip() for line in f if not line.strip().startswith("-e")]
    
    for package in packages:
        env.add_conda_package(package)

    with open("./src/web-service/env_file.yml","w") as f:
        f.write(env.serialize_to_string())

    # Configure the scoring environment
    inference_config = InferenceConfig(runtime= "python",
                                    entry_script="./src/web-service/entry_script.py",
                                    conda_file="./src/web-service/env_file.yml")

    deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

    service_name = flags['experiment_name']

    service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)

    service.wait_for_deployment(True)
    print(service.state)
    print("Endpoint: ", service.scoring_uri)

    print(service.get_logs())

    service.delete()

"""
if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    #load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    launch_deployment()
<<<<<<< HEAD
"""
=======
"""
>>>>>>> a813e79400cde8e5a65e48b7e65a4196e031cddf
