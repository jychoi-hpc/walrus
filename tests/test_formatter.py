import pytest
import torch

from walrus.data.well_to_multi_transformer import (
    ChannelsFirstWithTimeFormatter,
    consecutive_deltas,
)


@pytest.mark.parametrize("n_head", [1, 2, 3])
@pytest.mark.parametrize("n_tail", [1, 2, 3])
def test_consecutive_deltas_matches_diff_of_cat(n_head, n_tail):
    head = torch.randn(2, n_head, 5, 6, 3)
    tail = torch.randn(2, n_tail, 5, 6, 3)
    expected = torch.diff(torch.cat([head, tail], dim=1), dim=1)
    assert torch.equal(consecutive_deltas(head, tail), expected)


@pytest.mark.parametrize("causal_in_time", [False, True])
@pytest.mark.parametrize("t_in, t_out", [(1, 1), (3, 1), (2, 3)])
def test_process_input_delta_target_unchanged(causal_in_time, t_in, t_out):
    data = {
        "input_fields": torch.randn(2, t_in, 8, 8, 8, 4),
        "output_fields": torch.randn(2, t_out, 8, 8, 8, 4),
        "field_indices": torch.arange(4),
        "boundary_conditions": torch.zeros(2, 3, 2),
    }
    x, y = data["input_fields"], data["output_fields"]
    if causal_in_time:
        seq = torch.cat([x, y], dim=1)
    else:
        seq = torch.cat([x[:, -1:], y], dim=1)
    expected = seq[:, 1:] - seq[:, :-1]

    _, target = ChannelsFirstWithTimeFormatter().process_input(
        data, causal_in_time=causal_in_time, predict_delta=True, train=True
    )
    assert torch.equal(target, expected)
