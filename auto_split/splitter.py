from typing import List
from itertools import combinations
import numpy as np


class Splitter:
    """A class that implements the required functionality for automatic splitting of snippets.
    Predictions are made by a user-specified model (only BERT arhitechture).
    """

    TOO_MANY_LINES = 3
    IMPOSSIBLE_TO_SPLIT = -1
    NO_SPLIT = []

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def processing(self, snippet: str):
        """Converts the snippet into a format that can be fed into the model with the BERT architecture."""
        sentence = [snippet]
        rew = self.tokenizer.batch_encode_plus(
            sentence,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors='pt')
        return rew['input_ids'].to(
            self.device), rew['attention_mask'].to(
            self.device)

    def predict_subsnippet_proba(self, snippet: str) -> float:
        """Returns the probability of the snippet belonging to the predicted class."""
        return float(
            self.model(
                *
                self.processing(snippet))['logits'].softmax(
                dim=1).max(
                dim=1).values)

    def predict_subsnippet_class(self, snippet: str) -> int:
        """Returns the class of the snippet predicted by the model."""
        return int(self.model(*self.processing(snippet))
                   ['logits'].argmax(dim=1).view((-1, 1)) + 1)

    def check_splitted_substr(self, substr: str) -> bool:
        """Checks that the string is a valid set of code without unclosed parentheses."""
        brackets = {'(': 0, ')': 0, '[': 0, ']': 0, '{': 0, '}': 0}
        for bracket in brackets:
            brackets[bracket] = substr.count(bracket)
        if brackets['('] == brackets[')'] and brackets[
                '['] == brackets[']'] and brackets['{'] == brackets['}']:
            return False
        return True

    def split_by_indexes(self, snippet: str, split: List[int]) -> List[str]:
        """Returs a set of substrings over a set of intermediate indices.

        Args:
            snippet: Input str.
            split: Index list. Output substrings must be between these indices not including them.

        Returns:
            List of substrings in order from left to right.
        """

        indexes = [-1] + list(split) + [len(snippet)]
        sub_snippets = [snippet[indexes[i] + 1:indexes[i + 1]]
                        for i in range(len(indexes) - 1)]
        return sub_snippets

    def min_prob_of_split(self, snippet: str, split: List[int]) -> float:
        """Calculates the minimum probability of a subsnippet among subsnippets.

        Args:
            snippet: Input str.
            split: Index list. Subsnippets must be between these indices not including them.

        Returns:
            The minimum probability among all subsnippets - and 0 if at least one of the subsnippets is incorrect.
        """

        sub_snippets = self.split_by_indexes(snippet, split)
        check_snippets = [self.check_splitted_substr(
            sub_snippet) for sub_snippet in sub_snippets]
        if np.sum(check_snippets) > 0:
            return 0
        return np.min([self.predict_subsnippet_proba(sub_snippet)
                      for sub_snippet in sub_snippets])

    def mean_prob_of_split(self, snippet: str, split: List[int]) -> float:
        """Calculates the mean probability of a subsnippet among subsnippets.

        Args:
            snippet: Input str.
            split: Index list. Subsnippets must be between these indices not including them.

        Returns:
            The mean probability among all subsnippets - and 0 if at least one of the subsnippets is incorrect.
        """
        sub_snippets = self.split_by_indexes(snippet, split)
        check_snippets = [self.check_splitted_substr(
            sub_snippet) for sub_snippet in sub_snippets]
        if np.sum(check_snippets) > 0:
            return 0
        return np.mean([self.predict_subsnippet_proba(sub_snippet)
                       for sub_snippet in sub_snippets])

    def find_best_split(self, snippet: str,
                        all_splits: List[List[int]]) -> List[int]:
        """Finds the best subsnippets split among the given set.

        Args:
            snippet: Input str.
            split: Splits list.

        Returns:
            Best split. The one whose minimum probability is maximum is searched
            for - and if the minimum probability coincides, the average probability is compared.
        """
        res_split = all_splits[0]
        for split in all_splits:
            mi = self.min_prob_of_split(snippet, split)
            mea = self.mean_prob_of_split(snippet, split)
            if mi > self.min_prob_of_split(
                snippet,
                res_split) or (
                mi == self.min_prob_of_split(
                    snippet,
                    res_split) and mea > self.mean_prob_of_split(
                    snippet,
                    res_split)):
                res_split = split
        return res_split

    def split_by_n(self, snippet: str, n_splits: int) -> List[int]:
        """Finds the best split among all splits with n_splits intermediate indexes.

        Args:
            snippet: Input str.
            n_splits: Number of intermediate indexes. All intermediate elements must be equal to the \n character.

        Returns:
            Best split. The one whose minimum probability is maximum is searched
            for - and if the minimum probability coincides, the average probability is compared.
        """
        indexes = [i for i in range(len(snippet)) if snippet[i] == '\n']
        indexes = sorted(list(set(indexes) - set((0, len(snippet) - 1))))
        if len(indexes) < n_splits - 1:
            return Splitter.NO_SPLIT
        all_splits = list(map(list, combinations(indexes, n_splits - 1)))
        return self.find_best_split(snippet, all_splits)

    def split_snippet(self, snippet: str) -> List[int]:
        """Finds the best split among all splits of snippet.

        Args:
            snippet: Input str. Snippet can be split any number of snippets, splitting occurs on the \n character.

        Returns:
            Best split. The one whose minimum probability is maximum is searched
            for - and if the minimum probability coincides, the average probability is compared.
        """
        all_n_splits = list(range(1, snippet.count('\n') + 2))

        if len(all_n_splits) > Splitter.TOO_MANY_LINES or len(
                all_n_splits) == 0:
            return Splitter.NO_SPLIT

        best_splits = [self.split_by_n(snippet, n_splits)
                       for n_splits in all_n_splits]
        return self.find_best_split(snippet, best_splits)
