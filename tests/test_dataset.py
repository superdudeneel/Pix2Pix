import numpy as np
from PIL import Image

from src.pix2pix.dataset import MapDataset


def test_dataset_splits_a_side_by_side_pair_and_returns_normalized_tensors(tmp_path):
    left = np.full((100, 600, 3), 20, dtype=np.uint8)
    right = np.full((100, 600, 3), 230, dtype=np.uint8)
    Image.fromarray(np.concatenate([left, right], axis=1)).save(tmp_path / "pair.png")

    source, target = MapDataset(str(tmp_path))[0]

    assert source.shape == (3, 256, 256)
    assert target.shape == (3, 256, 256)
    assert source.dtype.is_floating_point
    assert target.dtype.is_floating_point
