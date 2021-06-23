from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
import azureml._restclient.snapshots_client

azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 1000000000000
import click
import logging
from dotenv import find_dotenv, load_dotenv
import yaml
from src.models.train_model import main as start_training
from src.models.hp_tuning import objective
import os, yaml
import optuna
from optuna.trial import TrialState


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
        
        run.wait_for_completion(show_output=True)


if __name__ == "__main__":
    config_path='config/config.yml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if config['hp']['tune']:
        print("Starting main")
        study = optuna.create_study(direction='minimize',pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
        study.optimize(objective, n_trials=3)
    
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
    
        print("Best trial:")
        trial = study.best_trial
    
        print("  Value: ", trial.value)
    
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automatically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    run()
