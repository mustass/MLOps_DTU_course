#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 10:03:35 2021

@author: asgermunch
"""

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import optuna
from optuna.trial import TrialState
import yaml
from src.models.model import BERT_model
from pytorch_lightning.loggers import WandbLogger
import wandb
from src.data.lightning_data_module import MyDataModule


def objective(trial, config_path='config/config.yml'):
    # Load config file
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup hyper paramters
    name = config['experiment_name']
    lr = trial.suggest_float('lr', config['hp']['lr_low'],
                             config['hp']['lr_high'])
    epochs = config['training']['max_epochs']
    full = config['training']['full']

    n_class = sum([
        1 for (k, v) in config['data']['used_datasets'].items() if int(v) == 1
    ])
    model = BERT_model(full, n_class=n_class, lr=lr)

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

    api = wandb.Api()
    author = config['hp']['author']
    project = config['hp']['project']
    print(f"{author}/{project}/{logger.__getstate__()['_id']}")
    run = api.run(f"{author}/{project}/{logger.__getstate__()['_id']}")

    print(run.summary)

    try:
        return run.summary['val_loss_epoch']
    except Exception as ex:
        raise ValueError(f"Somerthing went wrong: {ex}")


if __name__ == "__main__":
    print("Starting main")
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective, n_trials=3)

    pruned_trials = study.get_trials(deepcopy=False,
                                     states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False,
                                       states=[TrialState.COMPLETE])

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
