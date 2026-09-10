import torch

from tsrf.train.losses import DEFAULT_SCALES, MultiScaleTextureLoss, local_std


def test_loss_is_zero_for_an_exact_match():
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    assert MultiScaleTextureLoss()(x, x).item() < 1e-6


def test_loss_penalises_over_smoothing():
    torch.manual_seed(1)
    hr = torch.rand(2, 1, 64, 64)
    crit = MultiScaleTextureLoss()

    blurred = torch.nn.functional.avg_pool2d(hr, 5, stride=1, padding=2)
    assert crit(blurred, hr).item() > crit(hr, hr).item() + 0.1


def test_loss_penalises_fabricated_texture_of_the_wrong_amount():
    torch.manual_seed(2)
    hr = torch.rand(2, 1, 64, 64)
    crit = MultiScaleTextureLoss()
    assert crit(hr * 3.0, hr).item() > 1.0          # three times the variation


def test_loss_is_blind_to_correspondence():
    """The property the paper tests for, pinned as a unit test.

    Two independent draws from the same distribution have the same ladder, so
    the loss cannot tell a reconstruction from a fabrication. Only M7 can.
    """
    torch.manual_seed(3)
    hr = torch.rand(1, 1, 128, 128)
    independent = torch.rand(1, 1, 128, 128)
    assert MultiScaleTextureLoss()(independent, hr).item() < 0.02


def test_gradients_flow_to_the_prediction():
    torch.manual_seed(4)
    sr = torch.rand(1, 1, 32, 32, requires_grad=True)
    hr = torch.rand(1, 1, 32, 32)
    MultiScaleTextureLoss()(sr, hr).backward()
    assert sr.grad is not None and torch.isfinite(sr.grad).all()
    assert sr.grad.abs().sum() > 0


def test_local_std_agrees_with_the_numpy_metric():
    import numpy as np
    from tsrf.metrics.fractal import local_std as np_local_std

    torch.manual_seed(5)
    x = torch.rand(1, 1, 48, 48, dtype=torch.float64)
    for w in DEFAULT_SCALES:
        got = local_std(x, w)[0, 0].numpy()
        want = np_local_std(x[0, 0].numpy(), w)
        # Interiors must agree; the two pad differently at the border.
        b = w // 2
        assert np.allclose(got[b:-b, b:-b], want[b:-b, b:-b], atol=1e-6)
