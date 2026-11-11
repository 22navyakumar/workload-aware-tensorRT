"""Compile the model with Torch-TensorRT under a given setting."""
import torch
import torch_tensorrt


def compile_trt(model, example_inputs, min_block_size=5, **kwargs):
    return torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=list(example_inputs),
        enabled_precisions={torch.float16},
        min_block_size=min_block_size,
        **kwargs,
    )
