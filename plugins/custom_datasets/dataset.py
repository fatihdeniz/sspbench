"""
Dataset Module
==============
This module provides the Dataset class for managing and manipulating collections
of Instance objects with common dataset operations.
"""

import csv
import logging
import json
import os.path
import random
from typing import List, Union, Callable, Any

import torch
import datasets as hf_datasets
load_dataset = hf_datasets.load_dataset

from custom_datasets.instance import Instance

__all__ = ["Dataset"]

class Dataset(torch.utils.data.Dataset):
    """
    A generic Dataset class for handling collections of Instance objects.
    Provides functionality for loading, saving, shuffling, and manipulating datasets.
    """

    def __init__(
        self,
        data: Union[List[Instance], str],
        shuffle: bool = False,
        local_file_type: str = 'jsonl'
    ):
        """
        Initializes the Dataset with either a list of Instance objects or a dataset path/name.

        :param Union[List[Instance], str] data: A list of Instance objects or path/name of dataset.
        :param bool shuffle: Whether to shuffle the dataset upon initialization. Defaults to False.
        :param str local_file_type: Type of local file if data is a file path ('json' or 'jsonl').
        """

        if isinstance(data, list):
            self._dataset = data
        elif isinstance(data, str):
            if os.path.exists(data):
                logging.info(f'Loading dataset from local file: {data}')
                raw_dataset = load_dataset(local_file_type, data_files=data)
                instance_list = [Instance(**sample) for sample in raw_dataset['train']]
                self._dataset = instance_list
            else:
                logging.error(f'File not found: {data}')
                raise FileNotFoundError(f'Dataset file not found: {data}')
        else:
            logging.error('The data parameter must be a list of Instance objects or a file path')
            raise ValueError(f'Invalid data parameter: {data}')

        self.shuffled = shuffle

        if shuffle:
            random.shuffle(self._dataset)

    @staticmethod
    def load_csv(path: str = 'data.csv', headers: List[str] = None):
        """
        Loads a CSV file into the dataset.

        :param str path: Path to the CSV file.
        :param List[str] headers: Optional list of column names to use as headers.
        :return Dataset: A new Dataset instance.
        """
        dataset = Dataset([])
        with open(path, mode='r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dataset.add(Instance(**dict(row)))
        return dataset

    @staticmethod
    def load_jsonl(path: str = 'data.jsonl'):
        """
        Loads a JSONL file into the dataset.

        :param str path: Path to the JSONL file.
        :return Dataset: A new Dataset instance.
        """
        import jsonlines

        dataset = Dataset([])
        with jsonlines.open(path) as reader:
            for obj in reader:
                dataset.add(Instance(**obj))
        return dataset

    @staticmethod
    def load_json(path: str = 'data.json'):
        """
        Loads a JSON file into the dataset.

        :param str path: Path to the JSON file.
        :return Dataset: A new Dataset instance.
        """
        dataset = Dataset([])
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for obj in data:
                    dataset.add(Instance(**obj))
            else:
                dataset.add(Instance(**data))
        return dataset

    def shuffle(self):
        """
        Shuffles the dataset in place.
        """
        random.shuffle(self._dataset)
        self.shuffled = True

    def __getitem__(self, key):
        """
        Retrieves an item or subset from the dataset.

        :param key: Index or slice for retrieval.
        :return Instance | Dataset: Single instance or subset dataset.
        """
        if isinstance(key, slice):
            return Dataset(self._dataset[key])
        else:
            return self._dataset[key]

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        :return int: Size of the dataset.
        """
        return len(self._dataset)

    def __iter__(self):
        """
        Provides an iterator over the dataset.

        :return iterator: Iterator for the dataset.
        """
        return iter(self._dataset)

    def add(self, instance: Instance):
        """
        Adds a new Instance to the dataset.

        :param Instance instance: The Instance to add.
        """
        self._dataset.append(instance)

    @classmethod
    def merge(cls, dataset_list: List['Dataset']):
        """
        Merges multiple Dataset instances into a single dataset.

        :param List[Dataset] dataset_list: List of Dataset instances to merge.
        :return Dataset: A new merged Dataset instance.
        """
        merged_data = []
        for dataset in dataset_list:
            merged_data.extend(dataset._dataset)
        return cls(merged_data)

    def save_to_jsonl(self, path: str = 'data.jsonl', keys: List[str] = None):
        """
        Saves the dataset to a JSONL file.

        :param str path: Path where the file will be saved.
        :param List[str] keys: Optional list of specific keys to save. If None, saves all.
        """
        import jsonlines

        with jsonlines.open(path, mode='w') as writer:
            for instance in self._dataset:
                if keys:
                    data = {key: getattr(instance, key, None) for key in keys}
                else:
                    data = instance.to_dict()
                writer.write(data)

    def save_to_json(self, path: str = 'data.json', keys: List[str] = None):
        """
        Saves the dataset to a JSON file.

        :param str path: Path where the file will be saved.
        :param List[str] keys: Optional list of specific keys to save. If None, saves all.
        """
        data_list = []
        for instance in self._dataset:
            if keys:
                data = {key: getattr(instance, key, None) for key in keys}
            else:
                data = instance.to_dict()
            data_list.append(data)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)

    def save_to_csv(self, path: str = 'data.csv', keys: List[str] = None):
        """
        Saves the dataset to a CSV file.

        :param str path: Path where the file will be saved.
        :param List[str] keys: Optional list of specific keys to save.
        """
        if not self._dataset:
            logging.warning("Dataset is empty, nothing to save.")
            return

        # Determine fields to save
        if keys:
            fieldnames = keys
        else:
            fieldnames = list(self._dataset[0].keys())

        data_list = []
        for instance in self._dataset:
            row = {}
            for key in fieldnames:
                value = getattr(instance, key, None)
                # Convert lists and dicts to strings for CSV compatibility
                if isinstance(value, (list, dict)):
                    row[key] = str(value)
                else:
                    row[key] = value
            data_list.append(row)

        with open(path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for data in data_list:
                writer.writerow(data)

    def filter(self, predicate: Callable[[Instance], bool]) -> 'Dataset':
        """
        Filters the dataset based on a predicate function.

        :param Callable predicate: Function that takes an Instance and returns bool.
        :return Dataset: A new Dataset with filtered instances.
        """
        filtered_data = [instance for instance in self._dataset if predicate(instance)]
        return Dataset(filtered_data)

    def map(self, transform: Callable[[Instance], Instance]) -> 'Dataset':
        """
        Applies a transformation function to each instance.

        :param Callable transform: Function that transforms an Instance.
        :return Dataset: A new Dataset with transformed instances.
        """
        transformed_data = [transform(instance) for instance in self._dataset]
        return Dataset(transformed_data)

    def group_by(self, key: Callable[[Instance], Any]) -> List[List[Instance]]:
        """
        Groups instances based on a key function.

        :param Callable key: Function that takes an Instance and returns a hashable key.
        :return List[List[Instance]]: List of grouped instances.
        """
        groups = {}
        for instance in self:
            instance_key = key(instance)
            if instance_key not in groups:
                groups[instance_key] = []
            groups[instance_key].append(instance)
        return list(groups.values())

    def group_by_parents(self) -> List[List[Instance]]:
        """
        Groups instances based on their parent instances.

        :return List[List[Instance]]: List of grouped instances by parents.
        """
        return self.group_by(
            lambda x: tuple(sorted(list(set(id(parent) for parent in x.parents))))
        )

    def chunks(self, size: int):
        """
        Yields chunks of the dataset of specified size.

        :param int size: Size of each chunk.
        :yield Dataset: Dataset chunks.
        """
        for i in range(0, len(self), size):
            yield self[i:i + size]

    def split(self, ratio: float) -> tuple['Dataset', 'Dataset']:
        """
        Splits the dataset into two parts based on a ratio.

        :param float ratio: Ratio for the first split (0 < ratio < 1).
        :return tuple: Two Dataset instances (train, test).
        """
        if not 0 < ratio < 1:
            raise ValueError("Ratio must be between 0 and 1")
        
        split_point = int(len(self._dataset) * ratio)
        return (
            Dataset(self._dataset[:split_point]),
            Dataset(self._dataset[split_point:])
        )


if __name__ == '__main__':
    # Example usage
    dataset = Dataset.load_csv(path='data.csv')
    dataset.save_to_jsonl(path='data.jsonl')
    
    # Split into train/test
    train_data, test_data = dataset.split(0.8)
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")