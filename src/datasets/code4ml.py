import torch
import pandas as pd
import numpy as np
from src.datasets import BaseDataset
from sklearn.model_selection import train_test_split

class code4ml(BaseDataset):
    def __init__(self, tokenizer, dataset_type: str, num_classes: int, marks: list[int], seed: float):
        """_summary_

        Args:
            tokenizer (_type_): _description_
            dataset_type (str): _description_
            marks (list[int]): _description_
            seed (float): _description_
        """
        super().__init__(tokenizer)
        self.dataset_type = dataset_type
        self.seed = seed
        self.num_classes = num_classes
        self.load_data("data/code4ml/markup_data.csv", marks)
        self.split_data()
        
    def load_data(self, path: str, marks: list[int]) -> pd.DataFrame:
        """Loads dataset.

        Args:
            path, sep: Default parameters for read_csv method.
            marks: A list of those confidence indicators that remain in the dataset.

        Returns:
            Result dataset.
        """
        data = pd.read_csv(path, sep=",")
        data = data.dropna()
        data = data[data['marks'].isin(marks)]
        # non defined classes
        data = data[(data['graph_vertex_id'] != 53) &
                    (data['graph_vertex_id'] != 84)]
        all_classes = list(sorted(data['graph_vertex_id'].unique()))
        assert len(all_classes) == self.num_classes, f"Set num_classes in dataset to len(all_classes)"
        self.snippets, self.labels = data['code_block'].values, data['graph_vertex_id'].apply(lambda label: all_classes.index(label)).values

    
    def split_data(self):
        snippets_train, snippets_test, labels_train, labels_test = train_test_split(self.snippets, self.labels, test_size=0.4, random_state=self.seed)
        snippets_train, snippets_val, labels_train, labels_val = train_test_split(snippets_train, labels_train, test_size=0.33, random_state=self.seed)
        if self.dataset_type == "train":
            self.snippets, self.labels = snippets_train, labels_train 
        elif self.dataset_type == "val":
            self.snippets, self.labels = snippets_val, labels_val 
        else:
            self.snippets, self.labels = snippets_test, labels_test
