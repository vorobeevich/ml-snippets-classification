import pandas as pd
from typing import List


def load_data(
        path: str,
        marks: List[int] = [5],
        sep: str = ',') -> pd.DataFrame:
    """Loads dataset.

        Args:
            path, sep: Default parameters for read_csv method.
            marks: A list of those confidence indicators that remain in the dataset.

        Returns:
            Result dataset.
        """
    data = pd.read_csv(path, sep)
    data = data.dropna()
    data = data[data['marks'].isin(marks)]
    data = data[(data['graph_vertex_id'] != 53) &
                (data['graph_vertex_id'] != 84)]
    return data
