import torch

from src.pix2pix.discriminator_model import Discriminator
from src.pix2pix.generator_model import Generator


def test_generator_preserves_image_shape_and_output_range():
    model = Generator(features=8).eval()
    image = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        output = model(image)

    assert output.shape == image.shape
    assert torch.all(output >= -1)
    assert torch.all(output <= 1)


def test_discriminator_returns_one_patch_score_per_image_pair():
    model = Discriminator(features=[8, 16]).eval()
    source = torch.randn(1, 3, 256, 256)
    target = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        scores = model(source, target)

    assert scores.shape[0] == 1
    assert scores.shape[1] == 1
    assert scores.shape[2] > 0
    assert scores.shape[3] > 0
