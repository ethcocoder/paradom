"""Experiment 016: development-only interface-boundary diagnosis for Experiment 015.

This script intentionally loads only WikiText train and validation splits. It must not load,
score, or inspect the test split.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import experiment_015_adaptive_interface as e15

SEED = int(os.environ.get("EXPERIMENT_SEED", "20260841"))
OUT = Path("research/experiment_016_interface_boundary") / f"seed_{SEED}"
ALPHAS = (0.00, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
REFERENCE_ALPHA = 0.80
SHORT_LENGTH = 24
EPS = 1e-8


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def rel_l2(x: torch.Tensor, y: torch.Tensor) -> float:
    return scalar((x - y).pow(2).sum().sqrt() / y.pow(2).sum().sqrt().clamp(min=EPS))


def cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    return scalar(F.cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)).mean())


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean().sqrt()


def representation_metrics(mamba: torch.Tensor, attention: torch.Tensor) -> dict[str, float]:
    feature_var_m = mamba.var(dim=(0, 1), correction=0)
    feature_var_a = attention.var(dim=(0, 1), correction=0)
    gram_m = torch.matmul(mamba[0], mamba[0].t()) / mamba.shape[-1]
    gram_a = torch.matmul(attention[0], attention[0].t()) / attention.shape[-1]
    return {
        "branch_relative_l2": rel_l2(mamba, attention),
        "branch_cosine": cosine(mamba, attention),
        "branch_mean_shift_abs": scalar((mamba.mean() - attention.mean()).abs()),
        "branch_rms_ratio": scalar(rms(mamba) / rms(attention).clamp(min=EPS)),
        "branch_feature_log_variance_ratio_abs": scalar(torch.log((feature_var_m + EPS) / (feature_var_a + EPS)).abs().mean()),
        "branch_token_gram_relative_l2": rel_l2(gram_m, gram_a),
    }


def mean_dict(records: list[dict[str, float]]) -> dict[str, float]:
    keys = records[0].keys()
    return {key: sum(record[key] for record in records) / len(records) for key in keys}


def hook_output(store: dict, name: str):
    def capture(_module, _inputs, output):
        store[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
    return capture


def forward_with_capture(branch, teacher, ids: torch.Tensor, captures: dict) -> dict[str, torch.Tensor]:
    captures.clear()
    with torch.no_grad():
        hybrid_logits = branch(input_ids=ids, use_cache=False).logits.detach()
        teacher_logits = teacher(input_ids=ids, use_cache=False).logits.detach()
        gate = branch.model.layers[1].self_attn
        source = gate.last_input.detach()
        attention = gate.last_attention.detach()
        mamba = gate.mixer(source).detach()
    if "post_layer1" not in captures or "post_layer2" not in captures:
        raise RuntimeError("Layer hooks did not capture the residual stream")
    return {"hybrid_logits": hybrid_logits, "teacher_logits": teacher_logits, "mamba": mamba, "attention": attention, "post_layer1": captures["post_layer1"], "post_layer2": captures["post_layer2"]}


def score_only(branch, teacher, ids: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        h = branch(input_ids=ids, use_cache=False).logits[:, :-1]
        t = teacher(input_ids=ids, use_cache=False).logits[:, :-1]
        ce = F.cross_entropy(h.reshape(-1, h.shape[-1]), ids[:, 1:].reshape(-1))
        kl = F.kl_div(F.log_softmax(h / e15.TEMPERATURE, dim=-1), F.softmax(t / e15.TEMPERATURE, dim=-1), reduction="batchmean") * e15.TEMPERATURE**2
    return {"ce": scalar(ce), "logit_kl": scalar(kl)}


def reconstruct_adaptive(source_config, teacher, train, dev):
    """Reproduce E015 adaptive training without loading a test partition."""
    e15.set_seed(SEED)
    branch, audit = e15.build_branch(source_config)
    stages = {}
    for layer in e15.LAYERS:
        before = e15.local_diagnostics(branch, layer, dev)
        local_trace = e15.local_train(branch, "adaptive_interface", layer, train)
        after = e15.local_diagnostics(branch, layer, dev)
        gate_trace, gate_result = e15.adaptive_gate_train(branch, teacher, layer, train, dev)
        stages[f"layer_{layer}"] = {"before": before, "after": after, "local_trace": local_trace, "gate_trace": gate_trace, "gate_result": gate_result}
        print(f"reconstructed L{layer}: final alpha={gate_result.get('final_alpha', 1.0):.2f}")
    for parameter in branch.parameters(): parameter.requires_grad_(False)
    branch.eval()
    return branch, audit, stages


def measure_sweep(branch, teacher, dev):
    captures = {}
    handles = [
        branch.model.layers[1].register_forward_hook(hook_output(captures, "post_layer1")),
        branch.model.layers[2].register_forward_hook(hook_output(captures, "post_layer2")),
    ]
    try:
        gate = branch.model.layers[1].self_attn
        reference = []
        gate.set_alpha(REFERENCE_ALPHA)
        for ids in dev:
            reference.append(forward_with_capture(branch, teacher, ids[None].to(e15.DEVICE), captures))

        per_alpha = {}
        for alpha in ALPHAS:
            gate.set_alpha(alpha)
            full_records, short_records = [], []
            for index, ids in enumerate(dev):
                x = ids[None].to(e15.DEVICE)
                current = forward_with_capture(branch, teacher, x, captures)
                ref = reference[index]
                logits = current["hybrid_logits"][:, :-1]
                teacher_logits = current["teacher_logits"][:, :-1]
                full = score_only(branch, teacher, x)
                full.update(representation_metrics(current["mamba"], current["attention"]))
                full.update({
                    "logit_relative_l2_from_080": rel_l2(logits, ref["hybrid_logits"][:, :-1]),
                    "post_layer1_relative_l2_from_080": rel_l2(current["post_layer1"], ref["post_layer1"]),
                    "post_layer1_cosine_from_080": cosine(current["post_layer1"], ref["post_layer1"]),
                    "post_layer1_rms_ratio_from_080": scalar(rms(current["post_layer1"]) / rms(ref["post_layer1"]).clamp(min=EPS)),
                    "post_layer2_relative_l2_from_080": rel_l2(current["post_layer2"], ref["post_layer2"]),
                    "post_layer2_cosine_from_080": cosine(current["post_layer2"], ref["post_layer2"]),
                    "post_layer2_rms_ratio_from_080": scalar(rms(current["post_layer2"]) / rms(ref["post_layer2"]).clamp(min=EPS)),
                })
                full_records.append(full)
                short_ids = x[:, :SHORT_LENGTH]
                short_records.append(score_only(branch, teacher, short_ids))
            per_alpha[f"{alpha:.2f}"] = {"full_48": mean_dict(full_records), "short_24": mean_dict(short_records)}
            print(f"alpha={alpha:.2f} dev-CE={per_alpha[f'{alpha:.2f}']['full_48']['ce']:.4f} KL={per_alpha[f'{alpha:.2f}']['full_48']['logit_kl']:.4f}")
        return per_alpha
    finally:
        for handle in handles: handle.remove()


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    e15.set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"EXPERIMENT 016 seed={SEED}: development-only alpha boundary diagnosis; test split prohibited")
    # Load individual approved partitions only; the frozen test partition is never requested.
    train_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    validation_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    tokenizer = AutoTokenizer.from_pretrained(e15.SOURCE_ID)
    train = e15.collect(tokenizer, train_split, e15.TRAIN_SEQUENCES)
    dev = e15.collect(tokenizer, validation_split, e15.DEV_SEQUENCES)
    # Deliberately no test split request or reference is permitted below this point.
    source_config = AutoConfig.from_pretrained(e15.SOURCE_ID)
    teacher = AutoModelForCausalLM.from_pretrained(e15.SOURCE_ID, torch_dtype=torch.float32).to(e15.DEVICE).eval()
    branch, audit, stages = reconstruct_adaptive(source_config, teacher, train, dev)
    sweep = measure_sweep(branch, teacher, dev)
    output = {
        "experiment": "experiment_016_interface_boundary",
        "seed": SEED,
        "data_guardrail": {"test_split_loaded": False, "development_sequences": len(dev), "calibration_sequences": len(train), "short_context_length": SHORT_LENGTH, "full_context_length": e15.MAX_LENGTH},
        "source": {"model_id": e15.SOURCE_ID, "teacher_frozen": True, "reconstructed_condition": "experiment_015_adaptive_interface"},
        "audit": audit,
        "reconstructed_stages": stages,
        "alpha_reference": REFERENCE_ALPHA,
        "alpha_sweep": sweep,
    }
    (OUT / "results.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
