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
from path import Path
import click


def download_dataset(dataset_name, chunk_size=8192):
    endpoint = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_'
    endpoint += dataset_name+'_5.json.gz'

    if not os.path.exists('data/external'):
        os.makedirs('data/external')

    print("Downloading dataset "+dataset_name+"...")
    r = requests.get(endpoint, allow_redirects=True, stream=True)
    progr_bar = tqdm(total=int(r.headers.get('content-length', 0)), unit='iB', unit_scale=True)
    if r.status_code == 200:
        with open("data/external/"+dataset_name+".bin", "wb") as extfile:
            for chunk in r.iter_content(chunk_size=chunk_size):
                progr_bar.update(len(chunk))
                extfile.write(chunk)
    elif r.status_code == 404:
        raise ValueError("Requested dataset does not exists on server.")


def fetch_raw_dataset(dataset_name):
    try:
        with open("data/external/"+dataset_name+".bin", "rb") as extfile:
            data = zlib.decompress(extfile.read(), zlib.MAX_WBITS|32).decode("utf-8")
            data = data.split("\n")

            if not os.path.exists('data/interim'):
                os.makedirs('data/interim')

            with open("data/interim/"+dataset_name+".csv", 'w') as outfile:
                for review in data:
                    try:
                        obj = json.loads(review)
                        try:
                            outfile.write('"'+obj["textReview"]+'"'+","+dataset_name)
                        except KeyError:
                            outfile.write('"'+obj["reviewText"]+'"'+","+dataset_name)
                        outfile.write("\n")
                    except:
                        pass
    except FileNotFoundError:
        download_dataset(dataset_name)
        fetch_raw_dataset(dataset_name)

def download_if_not_existing(datasets):
    try:
        all_datasets = [f for f in listdir("data/interim/") if isfile(join("data/interim", f)) and f in datasets]
        if len(all_datasets) == 0:
            for dataset in datasets:
                fetch_raw_dataset(dataset)
    except:
        pass

@click.command()
@click.argument('config_datasets_path', type=click.Path(exists=True))
def ensemble(config_datasets_path):
    datasets = parse_datasets(config_datasets_path)
    download_if_not_existing(datasets)
    
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')

    f = open("data/raw/AmazonProductReviews.csv", "w")
    for filename in datasets:
        with open ("data/interim/"+filename+".csv") as subfile:
            f.write(subfile.read())

        os.remove("data/interim/"+filename+".csv")

def parse_datasets(config_datasets_path):
    datasets_list = []
    with open(config_datasets_path) as f:
        flags = json.load(f)
    return [k for (k, v) in flags.items() if v==1]

# unused
def shuffle_final_dataset():
    with open("data/raw/AmazonProductReviews.csv") as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open("data/raw/AmazonProductReviews.csv", "w") as f:
        f.writelines(lines)
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    ensemble()
