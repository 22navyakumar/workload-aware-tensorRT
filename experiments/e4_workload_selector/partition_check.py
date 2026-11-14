"""Step 2 of the plan: how does each config partition the model? No engines are built.

    python experiments/e4_workload_selector/partition_check.py

Prints, for every TensorRT config, the number of TensorRT and PyTorch blocks and the ops in
each PyTorch block, plus the op names present in the lowered graph (needed to check the
torch_executed_ops entries in configs/workloads.yaml). Also records the software versions.
Writes results/e4_workload_selector/partition_check.json.
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from harness import configs as cfgmod  # noqa: E402
from harness.model import load_model, make_inputs  # noqa: E402
from harness.trt import EncoderWrapper, compile_config, partition_summary  # noqa: E402


def environment():
    import tensorrt
    import torch_tensorrt
    import transformers

    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "torch_tensorrt": torch_tensorrt.__version__,
        "tensorrt": tensorrt.__version__,
        "transformers": transformers.__version__,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--verbose", action="store_true", help="also print Torch-TensorRT's own dry-run report")
    ap.add_argument("--out", type=Path,
                    default=cfgmod.REPO_ROOT / "results/e4_workload_selector/partition_check.json")
    args = ap.parse_args()

    if not args.verbose:
        logging.getLogger("torch_tensorrt").setLevel(logging.ERROR)
    env = environment()
    print("environment:", json.dumps(env, indent=2))

    spec = cfgmod.load_spec()
    eager = EncoderWrapper(load_model(spec["model"])).eval()
    hf_cfg = eager.model.config
    print(f"model {spec['model']}: max_position_embeddings={hf_cfg.max_position_embeddings}")
    inputs = make_inputs(args.batch, args.seq_len, vocab_size=hf_cfg.vocab_size)

    report = {"environment": env, "model": spec["model"],
              "max_position_embeddings": hf_cfg.max_position_embeddings, "configs": {}}
    seen_signature = {}
    for cfg in cfgmod.configs(spec):
        if cfg["backend"] != "trt" or cfg.get("cudagraphs"):
            continue  # eager has no partition; cudagraphs does not change it
        print(f"\n=== {cfg['id']}  {cfg.get('settings')}")
        try:
            part = partition_summary(compile_config(eager, inputs, cfg.get("settings"), dryrun=True))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            report["configs"][cfg["id"]] = {"error": f"{type(e).__name__}: {e}"}
            continue
        report["configs"][cfg["id"]] = part
        print(f"  {part['n_trt_blocks']} TensorRT blocks, {part['n_torch_blocks']} PyTorch blocks")
        for b in part["blocks"]:
            detail = "" if b["kind"] == "trt" or not b["ops"] else f"  {b['ops']}"
            print(f"    {b['name']:<18} {b['kind']:<5} ops={b['n_ops']}{detail}")
        signature = json.dumps([(b["kind"], b["n_ops"]) for b in part["blocks"]])
        if signature in seen_signature:
            print(f"  same partition as {seen_signature[signature]}")
        seen_signature.setdefault(signature, cfg["id"])

        all_ops = Counter()
        for b in part["blocks"]:
            all_ops.update(b["ops"] or {})
        requested = cfg.get("settings", {}).get("torch_executed_ops", [])
        for op in requested:
            short = op.replace("torch.ops.", "")
            if not any(short in k for b in part["blocks"] if b["kind"] == "torch" for k in (b["ops"] or {})):
                print(f"  WARNING: {op} was requested in torch_executed_ops but is in no PyTorch block")
        if cfg.get("default") and cfg["scenario"] == "natural":
            report["lowered_ops"] = dict(all_ops.most_common())
            print("  ops in the lowered graph:")
            for name, n in all_ops.most_common():
                print(f"    {n:>4}  {name}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
