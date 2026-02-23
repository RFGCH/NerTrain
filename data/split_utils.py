from datasets import Dataset
from typing import Tuple

def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Divide un dataset en train / validation / test.

    test_ratio se infiere automáticamente.
    """
    assert train_ratio + val_ratio < 1.0, "Ratios inválidos"

    test_ratio = 1.0 - train_ratio - val_ratio

    # first split: train vs temp
    train_dataset, temp_dataset = dataset.train_test_split(
        test_size=(1.0 - train_ratio),
        seed=seed
    ).values()

    # second split: val vs test
    val_dataset, test_dataset = temp_dataset.train_test_split(
        test_size=test_ratio / (val_ratio + test_ratio),
        seed=seed
    ).values()

    return train_dataset, val_dataset, test_dataset