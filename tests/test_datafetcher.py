import pytest
from src.data.fetch_dataset import ensemble, parse_datasets, download_if_not_existing, fetch_raw_dataset, download_dataset


@pytest.mark.parametrize("dataset_name", ["CD and Vinyl", "Video Games", "video_games", "baby"])
def test(dataset_name):
    with pytest.raises(ValueError, match=r"Requested dataset does not exists on server."):
        download_dataset(dataset_name)
