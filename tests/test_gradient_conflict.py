# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.utils.gradient_conflict import bootstrap_ci, flatten_gradients, safe_cosine_similarity
from ultralytics.utils.training_gradient_probe import probe_forward_state


def test_flatten_gradients_skips_none_and_preserves_values():
    grads = [torch.tensor([1.0, 2.0]), None, torch.tensor([[3.0], [4.0]])]
    flat = flatten_gradients(grads)
    assert torch.equal(flat, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_safe_cosine_similarity_basic_cases():
    vec = torch.tensor([1.0, 0.0, 0.0])
    assert safe_cosine_similarity(vec, vec) == 1.0
    assert safe_cosine_similarity(vec, -vec) == -1.0
    assert safe_cosine_similarity(vec, torch.tensor([0.0, 1.0, 0.0])) == 0.0


def test_safe_cosine_similarity_returns_none_for_empty_vectors():
    empty = torch.zeros(0)
    assert safe_cosine_similarity(empty, torch.tensor([1.0])) is None
    assert safe_cosine_similarity(torch.tensor([1.0]), empty) is None


def test_bootstrap_ci_single_value_degenerates_to_point_interval():
    assert bootstrap_ci([0.25]) == (0.25, 0.25)


def test_probe_forward_state_restores_training_mode_and_batch_norm_state():
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(2), torch.nn.Conv2d(2, 2, 1))
    model.train()
    batch_norm = model[0]
    mean_before = batch_norm.running_mean.clone()
    var_before = batch_norm.running_var.clone()

    with probe_forward_state(model):
        assert not model.training
        model(torch.randn(2, 2, 4, 4))

    assert model.training
    assert torch.equal(batch_norm.running_mean, mean_before)
    assert torch.equal(batch_norm.running_var, var_before)
