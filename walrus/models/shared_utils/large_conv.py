"""Convolutions on tensors with more than 2^31 elements (tracker task t406).

Above that size cuDNN's 32-bit indexing no longer applies and PyTorch falls
back to a much slower implementation: on a 768 x 410 x 1024 slab (7 input
channels) the encoder's first convolution took 291 s backward instead of
~0.3 s per half. With batch size 1 the batch cannot be split instead.

chunked_conv splits such inputs along a spatial axis where kernel == stride
(windows do not overlap) into pieces below the limit, convolves each and
concatenates the outputs: every output value uses the same inputs as
before. Smaller tensors take the plain path.
"""

from typing import Callable, Optional, Sequence

import torch

MAX_ELEMENTS = 2**31 - 1


def _chunk_axis(x, weight, stride, padding, transpose: bool) -> Optional[int]:
    """First spatial axis (tensor dim) with kernel == stride, no padding and
    more than one window: chunks along it do not interact."""
    n_spatial = x.dim() - 2
    for i in range(n_spatial):
        k, s = weight.shape[2 + i], stride[i]
        windows = x.shape[2 + i] if transpose else (x.shape[2 + i] - k) // s + 1
        if k == s and padding[i] == 0 and windows > 1:
            return 2 + i
    return None


def chunked_conv(conv_func: Callable, x: torch.Tensor, weight: torch.Tensor,
                 bias: Optional[torch.Tensor], stride: Sequence[int],
                 padding: Sequence[int], transpose: bool = False,
                 max_elements: int = MAX_ELEMENTS) -> torch.Tensor:
    """conv_func(x, weight, bias, stride, padding) (a convolution, or with
    transpose=True a transposed convolution), computed in chunks along a
    non-overlapping spatial axis when input or output exceeds max_elements."""
    stride, padding = tuple(stride), tuple(padding)
    # Output size along every axis, without materializing it
    out_numel = x.shape[0] * weight.shape[1 if transpose else 0]
    for i in range(x.dim() - 2):
        k, s, p, n = weight.shape[2 + i], stride[i], padding[i], x.shape[2 + i]
        out_numel *= (n - 1) * s + k - 2 * p if transpose else (n + 2 * p - k) // s + 1
    if max(x.numel(), out_numel) <= max_elements:
        return conv_func(x, weight, bias, stride, padding)
    axis = _chunk_axis(x, weight, stride, padding, transpose)
    if axis is None:  # no axis to split without overlap: plain (slow) path
        return conv_func(x, weight, bias, stride, padding)
    s = stride[axis - 2]
    n = x.shape[axis]
    windows = n if transpose else (n - s) // s + 1
    # Elements per window along the axis, input and output side
    per_window = max(x.numel() // n * (1 if transpose else s),
                     out_numel // (n * s if transpose else windows) * (s if transpose else 1))
    per_chunk = max(1, max_elements // per_window)
    step = per_chunk if transpose else per_chunk * s
    sizes = [step] * (n // step) + ([n % step] if n % step else [])
    # Chunk edges fall on multiples of the stride, so no window straddles two
    # chunks; a last piece shorter than a window holds points no window reads
    outs = [conv_func(piece, weight, bias, stride, padding)
            for piece in x.split(sizes, dim=axis)
            if piece.shape[axis] >= (1 if transpose else s)]
    return torch.cat(outs, dim=axis)
