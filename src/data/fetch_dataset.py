import requests
import zlib
import json
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
import random
import logging
from dotenv import find_dotenv, load_dotenv
import click
import yaml


def download_dataset(dataset_name, chunk_size=8192):
    endpoint = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_'
    endpoint += dataset_name + '_5.json.gz'

    print("Downloading dataset " + dataset_name + "...")
    r = requests.get(endpoint, allow_redirects=True, stream=True)
    progr_bar = tqdm(total=int(r.headers.get('content-length', 0)),
                     unit='iB',
                     unit_scale=True)
    if r.status_code == 200:
        with open("data/external/" + dataset_name + ".bin", "wb") as extfile:
            for chunk in r.iter_content(chunk_size=chunk_size):
                progr_bar.update(len(chunk))
                extfile.write(chunk)
    elif r.status_code == 404:
        raise ValueError("Requested dataset does not exists on server.")


def fetch_raw_dataset(dataset_name):
    try:
        with open("data/external/" + dataset_name + ".bin", "rb") as extfile:
            data = zlib.decompress(extfile.read(),
                                   zlib.MAX_WBITS | 32).decode("utf-8")
            data = data.split("\n")

            with open("data/interim/" + dataset_name + ".csv", 'w') as outfile:
                for review in data:
                    try:
                        obj = json.loads(review)
                        try:
                            outfile.write('"' + obj["textReview"] + '"' + "," +
                                          dataset_name)
                        except KeyError:
                            outfile.write('"' + obj["reviewText"] + '"' + "," +
                                          dataset_name)
                        outfile.write("\n")
                    except:
                        pass  #warnings.warn("A record in dataset "+dataset_name+" has been skipped as it was corrupted.")
    except FileNotFoundError:
        download_dataset(dataset_name)
        fetch_raw_dataset(dataset_name)


def download_if_not_existing(datasets):
    try:
        available_datasets = [
            f[:-4] for f in listdir("data/external/")
            if isfile(join("data/external", f)) and f[:-4] in datasets
        ]
        to_download = [
            item for item in datasets if item not in available_datasets
        ]

        for dataset in to_download:
            fetch_raw_dataset(dataset)
    except Exception as ex:
        if type(ex) == FileNotFoundError:
            raise FileNotFoundError(
                "The ./data/ directory does not exists. Create it before moving on."
            )


def check_and_create_data_subfolders(
        root='./data/',
        subfolders=['raw', 'interim', 'processed', 'external']):
    for folder in subfolders:
        if not os.path.exists(root + folder):
            os.makedirs(root + folder)


def ensemble(config):
    check_and_create_data_subfolders()
    datasets = parse_datasets(config)

    name = config['experiment_name']

    download_if_not_existing(datasets)
    check_and_create_data_subfolders("data/raw/", subfolders=[str(name)])

    f = open("data/raw/" + str(name) + "/AmazonProductReviews.csv", "w")
    for filename in datasets:
        fetch_raw_dataset(filename)
        with open("data/interim/" + filename + ".csv") as subfile:
            f.write(subfile.read())

        os.remove("data/interim/" + filename + ".csv")

    with open("data/raw/" + str(name) + '/datasets.txt', 'w') as f:
        f.write(f'Used datasets: {datasets}')


def parse_datasets(config):

    flags = config['data']['used_datasets']
    try:
        datasets = [k for (k, v) in flags.items() if int(v) == 1]
    except ValueError:
        raise ValueError(
            "Insert only 0 (not wanted) or 1 (wanted) in the config file")

    return datasets


# unused
def shuffle_final_dataset():
    with open("data/raw/AmazonProductReviews.csv") as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open("data/raw/AmazonProductReviews.csv", "w") as f:
        f.writelines(lines)


@click.command()
@click.argument('config_path',
                type=click.Path(exists=True),
                default='./config/config.yml')
def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    ensemble(config)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
