from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
import click
import logging
from dotenv import find_dotenv, load_dotenv
import yaml

@click.command()
@click.argument('config_file', type=click.Path(exists=True))
def run(config_file="./config/config.yml"):

    with open(config_file) as f:
        flags = yaml.safe_load(f)
    if not flags['compute']['cloud'] :
        pass # TODO: implement this
    elif flags['compute']['cloud']:
        setup = flags['compute']
        ws = Workspace.get(name=setup['workspace_name'],
               subscription_id=setup['subscription_id'],
               resource_group=setup['resource_group']
               )
        #ws = Workspace.from_config("config/azure_conf.json")
        env = Environment.from_pip_requirements(setup['environment_name'], 'requirements.txt')
        experiment = Experiment(workspace=ws, name=setup['experiment_name'])

        args = [config_file]
        config = ScriptRunConfig(source_directory='.',
                                script='src/models/train_model.py',
                                arguments=args,
                                compute_target=setup['compute_target'],
                                environment=env)

        run = experiment.submit(config)
        aml_url = run.get_portal_url()
        #print(aml_url)
        run.wait_for_completion()


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    run()
    