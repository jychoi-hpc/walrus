from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from einops import rearrange, repeat


def consecutive_deltas(head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """Same as torch.diff(torch.cat([head, tail], dim=1), dim=1), without
    materializing the concatenation: one output buffer instead of the
    concatenated copy plus the difference."""
    n_head = head.shape[1]
    out = tail.new_empty((tail.shape[0], n_head + tail.shape[1] - 1, *tail.shape[2:]))
    if n_head > 1:
        torch.sub(head[:, 1:], head[:, :-1], out=out[:, : n_head - 1])
    torch.sub(tail[:, :1], head[:, -1:], out=out[:, n_head - 1 : n_head])
    if tail.shape[1] > 1:
        torch.sub(tail[:, 1:], tail[:, :-1], out=out[:, n_head:])
    return out


class AbstractFormatter(ABC):
    """
    Default preprocessor for Well to MPP data.
    """

    @abstractmethod
    def process_input(
        self, data: Dict, causal_in_time: bool = False, predict_delta: bool = False
    ) -> Tuple:
        pass

    @abstractmethod
    def process_output(self, output, metadata) -> torch.Tensor:
        pass


class ChannelsFirstWithTimeFormatter(AbstractFormatter):
    """
    Default preprocessor for data in channels first format.
    """

    def process_input(
        self,
        data: Dict,
        causal_in_time: bool = False,
        predict_delta: bool = False,
        train: bool = True,
    ):
        """Convert data from Well format to model format.

        Model format for MPPX is channels first (t, b, c, ...) where t is the time dimension, b is the batch dimension, and c is the channel dimension.

        During training, y is the loss target. During validation, it is always the raw ground truth.
        """
        x = data["input_fields"]
        x = rearrange(x, "b t ... c -> t b c ...")
        if "constant_fields" in data:
            flat_constants = repeat(
                data["constant_fields"],
                "b ... c -> (repeat) b c ...",
                repeat=x.shape[0],
            )
            x = torch.cat(
                [
                    x,
                    flat_constants,
                ],
                dim=2,  # Different from the well due to time
            )
        y = data["output_fields"]
        if train:
            if causal_in_time:
                if predict_delta:
                    y = consecutive_deltas(data["input_fields"], y)
                else:
                    y = torch.cat([data["input_fields"][:, 1:, ...], y], dim=1)
            else:
                # For non-causal predict delta, we only need to append the last step
                # For the sizes we're doing, this could be merged with above, but could
                # be unnecessarily expensive at higher res/content lengths
                if predict_delta:
                    y = consecutive_deltas(data["input_fields"][:, -1:, ...], y)
        # TODO - Add warning to output if nan has to be replaced
        # in some cases (staircase), its ok. In others, it's not.
        return (
            x,
            data["field_indices"],
            data["boundary_conditions"],
        ), y

    def process_output(self, output, metadata):
        return rearrange(output, "t b c ... -> b t ... c")
