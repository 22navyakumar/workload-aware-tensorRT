"""Compile one named config from workloads.yaml and describe the resulting partition.

Kept separate from harness/compile.py so the original helper stays untouched for E1-E3.
FP16 comes from the model's own dtype (load_model casts the weights), not a compile flag.
"""
import operator
from collections import Counter
from contextlib import contextmanager

import torch

TRT_PREFIX = "_run_on_acc"    # submodule name Torch-TensorRT gives a TensorRT block
TORCH_PREFIX = "_run_on_gpu"  # submodule name it gives a PyTorch fallback block


class EncoderWrapper(torch.nn.Module):
    """Positional inputs in, one tensor out, so export and timing see a plain function."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def compile_config(model, example_inputs, settings=None, dryrun=False):
    import torch_tensorrt

    kwargs = dict(settings or {})
    if "torch_executed_ops" in kwargs:
        kwargs["torch_executed_ops"] = set(kwargs["torch_executed_ops"])
    return torch_tensorrt.compile(
        model, ir="dynamo", inputs=list(example_inputs), dryrun=dryrun, **kwargs
    )


@contextmanager
def runnable(module, cudagraphs=False):
    """Yield the callable to time; wraps it in CUDA Graphs when the config asks for it."""
    if not cudagraphs:
        yield module
        return
    import torch_tensorrt

    with torch_tensorrt.runtime.enable_cudagraphs(module) as wrapped:
        yield wrapped


def _is_glue(node):
    return node.target is operator.getitem  # tuple unpacking between blocks, not real work


def _op_targets(child):
    graph = getattr(child, "graph", None)
    if graph is None:  # a built TensorRT engine has no fx graph
        return None
    return Counter(str(n.target) for n in graph.nodes if n.op == "call_function" and not _is_glue(n))


def partition_summary(module):
    """Walk the partitioned graph in execution order and list its TensorRT and PyTorch blocks.

    The fast partitioner wraps fallback ops in `_run_on_gpu_N` submodules; the global partitioner
    leaves them inline in the top-level graph. Both count as PyTorch blocks here: a block is a
    maximal run of PyTorch ops between TensorRT engines.
    """
    children = dict(module.named_children())
    blocks, inline = [], None
    for node in module.graph.nodes:
        target = str(node.target)
        if node.op == "call_module" and target.startswith((TRT_PREFIX, TORCH_PREFIX)):
            inline = None
            ops = _op_targets(children.get(target))
            blocks.append({"name": target, "kind": "trt" if target.startswith(TRT_PREFIX) else "torch",
                           "n_ops": None if ops is None else sum(ops.values()),
                           "ops": None if ops is None else dict(ops)})
        elif node.op in ("call_function", "call_method", "call_module") and not _is_glue(node):
            if inline is None:
                inline = {"name": f"inline_{len(blocks)}", "kind": "torch", "n_ops": 0, "ops": {}}
                blocks.append(inline)
            inline["n_ops"] += 1
            inline["ops"][target] = inline["ops"].get(target, 0) + 1
    return {
        "n_trt_blocks": sum(b["kind"] == "trt" for b in blocks),
        "n_torch_blocks": sum(b["kind"] == "torch" for b in blocks),
        "blocks": blocks,
    }


def first_tensor(out):
    """Compiled modules may return a tensor, a tuple, or a dict; take the first tensor."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        out = list(out.values())
    return first_tensor(out[0])
