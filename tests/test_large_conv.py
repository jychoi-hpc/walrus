"""Chunked convolutions (tracker task t406) give the plain convolution's
outputs and gradients; chunking is forced with a small element limit."""

import pytest
import torch
import torch.nn.functional as F

from walrus.models.shared_utils.large_conv import chunked_conv


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("length", [64, 70])  # 70: points left over after the last window
def test_chunked_conv_matches_plain(transpose, length):
    torch.manual_seed(0)
    conv = F.conv_transpose3d if transpose else F.conv3d
    x0 = torch.randn(1, 5, length if not transpose else length // 8, 9, 8, dtype=torch.float64)
    shape = (5, 6, 8, 3, 2) if transpose else (6, 5, 8, 3, 2)  # x: kernel = stride 8
    w0 = torch.randn(shape, dtype=torch.float64)
    b0 = torch.randn(6, dtype=torch.float64)
    stride, padding = (8, 3, 2), (0, 0, 0)
    results = []
    for limit in (2**31 - 1, 900):  # plain, then chunked
        x, w, b = (t.clone().requires_grad_() for t in (x0, w0, b0))
        y = chunked_conv(conv, x, w, b, stride, padding, transpose=transpose, max_elements=limit)
        g = torch.randn(y.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        (y * g).sum().backward()
        results.append((y.detach(), x.grad, w.grad, b.grad))
    for plain, chunked in zip(*results):
        torch.testing.assert_close(chunked, plain, rtol=1e-12, atol=1e-12)


def test_no_split_axis_falls_back_to_plain():
    # kernel 3 > stride 2 everywhere: no axis to split; must still be correct
    x = torch.randn(1, 2, 20, 6, 6, dtype=torch.float64)
    w = torch.randn(3, 2, 3, 3, 3, dtype=torch.float64)
    y = chunked_conv(F.conv3d, x, w, None, (2, 2, 2), (0, 0, 0), max_elements=100)
    torch.testing.assert_close(y, F.conv3d(x, w, None, (2, 2, 2)))
