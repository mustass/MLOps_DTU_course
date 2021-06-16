# -*- coding: utf-8 -*-
from collections import namedtuple
import click
import logging
from pathlib import Path
import pickle
import gzip, warnings

from torch.utils import data
from dotenv import find_dotenv, load_dotenv
import yaml

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizerFast
import torch
from torch.utils.data import TensorDataset
from src.data.fetch_dataset import check_and_create_data_subfolders, parse_datasets


def clean_data(config):

    load_dotenv(find_dotenv())

    # Getting the rest of configs
    experiment_name = config['experiment_name']
    seed = config['seed']
    splits = config['data']['train_val_test_splits']
    max_length = config['data']['max_seq_length']
    datasets = parse_datasets(config)
    print("Using following datasets: {}".format(datasets))

    # load raw csv file for given reviews at supplied path
    df = check_and_load_raw("data/raw/" + str(experiment_name) +
                            "/AmazonProductReviews.csv")

    try:
        f = gzip.open(
            './data/processed/' + str(config['experiment_name']) +
            '/used_datasets.pklz', 'rb')
        existing_datasets = pickle.load(f, encoding="bytes")
        if existing_datasets == datasets:
            print("Datasets are allready prepared!:)")
            return
    except Exception as ex:
        print('Generating new datasets...')
        pass

    # drop any rows which have missing reviews, class or a class which is not in our class dict

    nrows = df.shape[0]
    df['review'].replace('', np.nan, inplace=True)
    df.dropna(subset=['review'], inplace=True)
    df['class'].replace('', np.nan, inplace=True)
    df.dropna(subset=['class'], inplace=True)
    print('Nr. rows dropped because containing NaN:', nrows - df.shape[0])

    nrows = df.shape[0]
    df = df[df['class'].isin(datasets)]

    print('Nr. rows dropped because class label was incorrect:',
          nrows - df.shape[0])

    # One hot encode class labels
    labelencoder = LabelEncoder()
    df['class'] = labelencoder.fit_transform(df['class'])

    # Run this if we want to see some info on string lengths
    # check_string_lengths(df)
    split1, split2 = check_splits(splits)

    # split train dataset into train, validation and test sets
    train_text, test_text, train_labels, test_labels = train_test_split(
        df['review'],
        df['class'],
        random_state=seed,
        test_size=split1,
        stratify=df['class'])

    train_text, val_text, train_labels, val_labels = train_test_split(
        train_text,
        train_labels,
        random_state=seed,
        test_size=split2,
        stratify=train_labels)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    tokens_train = tokenizer.batch_encode_plus(train_text.tolist(),
                                               max_length=max_length,
                                               padding=True,
                                               truncation=True)

    # tokenize and encode sequences in the validation set
    tokens_val = tokenizer.batch_encode_plus(val_text.tolist(),
                                             max_length=max_length,
                                             padding=True,
                                             truncation=True)

    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(test_text.tolist(),
                                              max_length=max_length,
                                              padding=True,
                                              truncation=True)

    train_data = TensorDataset(torch.tensor(tokens_train['input_ids']),
                               torch.tensor(tokens_train['attention_mask']),
                               torch.tensor(train_labels.tolist()))
    val_data = TensorDataset(torch.tensor(tokens_val['input_ids']),
                             torch.tensor(tokens_val['attention_mask']),
                             torch.tensor(val_labels.tolist()))
    test_data = TensorDataset(torch.tensor(tokens_test['input_ids']),
                              torch.tensor(tokens_test['attention_mask']),
                              torch.tensor(test_labels.tolist()))

    pickle_TensorDataset(train_data, experiment_name, 'train')
    pickle_TensorDataset(val_data, experiment_name, 'validate')
    pickle_TensorDataset(test_data, experiment_name, 'test')

    f = gzip.open(
        './data/processed/' + str(experiment_name) + '/used_datasets.pklz',
        'wb')
    pickle.dump(datasets, f)
    f.close()


def check_and_load_raw(file):

    try:
        df = pd.read_csv(file,
                         error_bad_lines=False,
                         names=['review', 'class'])
        return df
    except Exception as ex:
        if type(ex) == FileNotFoundError:
            raise FileNotFoundError(
                "The ./data/raw/" + str(file) +
                "file does not exists. Fetch the dataset before contiunuing")


def check_string_lengths(df):
    # get length of all the messages in the train set
    seq_len = [len(i.split()) for i in df['review']]

    plot = pd.Series(seq_len).hist(bins=30)
    plot.figure.savefig('./reports/figures/hist_of_string_lengths.pdf')
    print("Mean seq-len:", np.mean(seq_len))
    print("Median seq-len:", np.median(seq_len))


def check_splits(splits):
    assert int(np.sum(splits)) == 1, 'Splits must sum to one'
    first = splits[2]
    second = splits[1] / (1 - splits[2])
    return first, second


def pickle_TensorDataset(dataset, experiment_name, dataset_name):
    check_and_create_data_subfolders('./data/processed/',
                                     subfolders=[str(experiment_name)])
    f = gzip.open(
        './data/processed/' + str(experiment_name) + '/' + str(dataset_name) +
        '.pklz', 'wb')
    pickle.dump(dataset, f)
    f.close()


@click.command()
@click.argument('config_path',
                type=click.Path(exists=True),
                default='./config/config.yml')
def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    clean_data(config)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    #load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
