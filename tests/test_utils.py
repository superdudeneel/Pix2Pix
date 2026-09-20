import torch

from src.pix2pix import config
from src.pix2pix.utils import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_restores_weights_optimizer_and_learning_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEVICE", "cpu")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    expected_weight = model.weight.detach().clone()
    checkpoint = tmp_path / "generator.pth.tar"

    save_checkpoint(model, optimizer, str(checkpoint))
    with torch.no_grad():
        model.weight.zero_()
    optimizer.param_groups[0]["lr"] = 0.9

    load_checkpoint(str(checkpoint), model, optimizer, lr=0.01)

    assert torch.equal(model.weight, expected_weight)
    assert optimizer.param_groups[0]["lr"] == 0.01
