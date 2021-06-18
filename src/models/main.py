from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
import azureml._restclient.snapshots_client

azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 1000000000000
import click
import logging
from dotenv import find_dotenv, load_dotenv
import yaml
from src.models.train_model import main as start_training

import os


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


cli = click.Group()


@click.command()
@click.argument('config_file', default="./config/config.yml")
@click.pass_context
def run(ctx, config_file):

    with open(config_file) as f:
        flags = yaml.safe_load(f)
    if not flags['compute']['cloud']:
        ctx.forward(start_training)
    elif flags['compute']['cloud']:
        setup = flags['compute']
        ws = get_workspace(setup)
        env = Environment.from_pip_requirements(setup['environment_name'],
                                                'requirements.txt')
        experiment = Experiment(workspace=ws, name=flags['experiment_name'])

        args = [config_file]
        config = ScriptRunConfig(source_directory='.',
                                script='src/models/train_model.py',
                                arguments=args,
                                compute_target=setup['compute_target'],
                                environment=env)

        run = experiment.submit(config)
        aml_url = run.get_portal_url()

        if setup['deploy']:
            deploy(ws, flags['experiment_name'])

        run.wait_for_completion()


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    run()
