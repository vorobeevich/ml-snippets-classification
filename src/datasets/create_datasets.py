from copy import deepcopy

import src.datasets
from src.utils import init_object


def create_datasets(dataset):
    train_dataset, val_dataset, test_dataset = deepcopy(
        dataset), deepcopy(
        dataset), deepcopy(
        dataset)

    train_dataset["kwargs"]["dataset_type"], val_dataset["kwargs"]["dataset_type"], test_dataset["kwargs"]["dataset_type"] = "train", "val", "test"

    train_dataset = init_object(src.datasets, train_dataset)
    val_dataset = init_object(src.datasets, val_dataset)
    test_dataset = init_object(src.datasets, test_dataset)

    return train_dataset, val_dataset, test_dataset
