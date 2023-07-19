import torch
import pandas as pd
    
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Returns the number of snippets in the dataset.

        Returns:
            int: dataset len
        """
        return len(self.snippets)

    def __getitem__(self, ind: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Returns a snippet (after tokenizer) from the dataset by its number.
        Also, the class label is returned.

        Args:
            idx (int): index of snippet
        Returns:
            tuple[torch.Tensor, torch.Tensor, float]:
                tokens, attention_mask, label
        """
        snippet = self.tokenizer.batch_encode_plus([self.snippets[ind]], add_special_tokens=True, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
        input_ids, attention_mask = snippet['input_ids'].squeeze(0), snippet['attention_mask'].squeeze(0)
        label = self.labels[ind]
        return input_ids, attention_mask, label
