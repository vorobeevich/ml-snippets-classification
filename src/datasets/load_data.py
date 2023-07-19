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
    all_classes = list(sorted(data['graph_vertex_id'].unique()))
    assert len(all_classes) <= num_classes, f"Set num_classes in dataset to {len(all_classes)}"
    data['graph_vertex_id'] = data['graph_vertex_id'].apply(lambda label: all_classes.index(label))
    return data