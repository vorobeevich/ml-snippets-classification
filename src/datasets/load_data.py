import pandas as pd

def load_data(path: str, num_classes: int, marks: list[int]) -> pd.DataFrame:
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
    # classes in code4ml are in [1, 88]
    assert data['graph_vertex_id'].max() <= num_classes, f"Set num_classes in dataset to {data['graph_vertex_id'].max()}"
    data['graph_vertex_id'] = data['graph_vertex_id'] - 1
    return data