from src.datasets import BaseDataset, load_data
from sklearn.model_selection import train_test_split

class code4ml(BaseDataset):
    def __init__(self, tokenizer, path: str, dataset_type: str, num_classes: int, marks: list[int], seed: float):
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
        self.path = path
        data = load_data(path, num_classes, marks)
        self.snippets, self.labels =  data['code_block'].values, data['graph_vertex_id'].values
        if self.dataset_type == "all":
            return
        self.split_data()
    
    def split_data(self) -> None:
        snippets_train, snippets_test, labels_train, labels_test = train_test_split(self.snippets, self.labels, test_size=0.4, random_state=self.seed)
        snippets_train, snippets_val, labels_train, labels_val = train_test_split(snippets_train, labels_train, test_size=0.33, random_state=self.seed)
        if self.dataset_type == "train":
            self.snippets, self.labels = snippets_train, labels_train 
        elif self.dataset_type == "val":
            self.snippets, self.labels = snippets_val, labels_val 
        elif self.dataset_type == "test":
            self.snippets, self.labels = snippets_test, labels_test
