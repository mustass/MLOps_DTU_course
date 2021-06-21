import pytest
from src.data.fetch_dataset import ensemble
import yaml, os, pickle, gzip

@pytest.mark.parametrize(
    "dataset_name", ["CD and Vinyl", "Video Games", "video_games", "baby"])
def test_dataset(dataset_name,config_path='config/config.yml'):
    with pytest.raises(ValueError,
                       match=r"Requested dataset does not exists on server."):
        with open(config_path) as f:
            config_file = yaml.safe_load(f)
        ensemble(config_file)
        
        # Check train data
        data_path = 'data/processed/'+str(config_file['experiment_name'])+'/'
        file_path = 'train.pklz'
        check_data(data_path,file_path)
        
        # Check test data
        file_path = 'test.pklz'
        check_data(data_path,file_path)
        
        # Check validation data
        file_path = 'validate.pklz'
        check_data(data_path,file_path)
        
def check_data(data_path,file_path):
    assert os.path.isfile(data_path + file_path), f'directory {data_path+file_path} does not exist'
    dataset = []
    f = gzip.open(data_path + file_path, 'rb')
    dataset = pickle.load(f, encoding="bytes")
    assert len(dataset) != 0, f'Dataset not loaded correctly from file {data_path+file_path} or dataset is empty'