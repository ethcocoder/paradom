"""Experiment 013: independent-seed replication on an untouched WikiText test set.

Layer 0 is fitted and gated in first. Layer 1 is then fitted on the partially
converted hybrid's real trajectories and gated in second. Direct source-derived
and fresh-random Mamba initializations receive identical training budgets.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, MambaConfig
from transformers.models.mamba.modeling_mamba import MambaMixer


SOURCE_ID = "HuggingFaceTB/SmolLM-135M"
SEED = int(os.environ.get("EXPERIMENT_SEED", "20260821"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT = Path("research/experiment_013_replication") / f"seed_{SEED}"
MAX_LENGTH = 48
TRAIN_SEQUENCES = 256
EVAL_SEQUENCES = 16
LOCAL_LR = 2e-4
GATE_LR = 5e-5
LAYER_BUDGETS = {
    0: {"state_size": 64, "local_steps": 180, "gate_steps": 60},
    1: {"state_size": 96, "local_steps": 360, "gate_steps": 120},
}
TEMPERATURE = 2.0
LAYERS = (0, 1)
PROMPTS = [
    "The purpose of scientific research is to",
    "Once upon a time, a young explorer discovered",
    "The capital of France is",
    "Artificial intelligence can help people by",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resize_1d(tensor: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(tensor.detach().float().reshape(1, 1, -1), size=size, mode="linear", align_corners=False).reshape(-1)


def resize_2d(tensor: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(tensor.detach().float().reshape(1, 1, *tensor.shape), size=shape, mode="bilinear", align_corners=False).reshape(shape)


def mamba_config(source_config, state_size: int) -> MambaConfig:
    return MambaConfig(
        vocab_size=source_config.vocab_size,
        hidden_size=source_config.hidden_size,
        num_hidden_layers=2,
        state_size=state_size,
        expand=2,
        conv_kernel=4,
        use_bias=False,
        use_conv_bias=True,
        use_cache=False,
        use_mambapy=False,
        use_associative_scan=False,
    )


def copy_(destination: torch.Tensor, source: torch.Tensor) -> None:
    destination.copy_(source.detach().to(device=destination.device, dtype=destination.dtype))


def map_attention_to_mamba(source_model, mixer: MambaMixer, layer_index: int) -> dict[str, int]:
    state = source_model.state_dict()
    prefix = f"model.layers.{layer_index}"
    q = state[f"{prefix}.self_attn.q_proj.weight"]
    k = state[f"{prefix}.self_attn.k_proj.weight"]
    v = state[f"{prefix}.self_attn.v_proj.weight"]
    o = state[f"{prefix}.self_attn.o_proj.weight"]
    gate = state[f"{prefix}.mlp.gate_proj.weight"]
    up = state[f"{prefix}.mlp.up_proj.weight"]
    down = state[f"{prefix}.mlp.down_proj.weight"]
    hidden, inner, state_size, rank = mixer.hidden_size, mixer.intermediate_size, mixer.ssm_state_size, mixer.time_step_rank
    with torch.no_grad():
        copy_(mixer.in_proj.weight, torch.cat([resize_2d(v, (inner, hidden)), resize_2d(torch.cat([q, k]), (inner, hidden))], dim=0))
        copy_(mixer.out_proj.weight, resize_2d(o, (hidden, inner)))
        energy = resize_1d(q.norm(dim=1), inner)
        scale = (energy / energy.mean().clamp(min=1e-6)).clamp(0.5, 2.0)
        copy_(mixer.A_log, scale[:, None] * torch.log(torch.arange(1, state_size + 1, dtype=torch.float32))[None, :])
        d = resize_1d(v.norm(dim=1), inner)
        copy_(mixer.D, d / d.mean().clamp(min=1e-6))
        conv_scale = d / d.mean().clamp(min=1e-6)
        copy_(mixer.conv1d.weight, conv_scale[:, None, None] * torch.full((inner, 1, mixer.conv_kernel_size), 1.0 / mixer.conv_kernel_size))
        copy_(mixer.conv1d.bias, torch.zeros(inner))
        copy_(mixer.x_proj.weight, 0.5 * (resize_2d(gate, (rank + 2 * state_size, inner)) + resize_2d(up, (rank + 2 * state_size, inner))))
        copy_(mixer.dt_proj.weight, resize_2d(down, (inner, rank)))
    return {"mapped_mixer_tensors": 8, "fresh_time_step_bias": 1}


class GatedAttentionReplacement(nn.Module):
    """Original attention plus a Mamba replacement branch with observable local targets."""

    def __init__(self, attention: nn.Module, mixer: MambaMixer):
        super().__init__()
        self.attention = attention
        self.mixer = mixer
        self.register_buffer("alpha", torch.tensor(0.0))
        self.last_input: torch.Tensor | None = None
        self.last_attention: torch.Tensor | None = None
        for parameter in self.attention.parameters():
            parameter.requires_grad_(False)

    def set_alpha(self, alpha: float) -> None:
        self.alpha.fill_(alpha)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        attention_output, attention_weights = self.attention(hidden_states, **kwargs)
        self.last_input = hidden_states
        self.last_attention = attention_output
        mamba_output = self.mixer(hidden_states)
        return attention_output + self.alpha * (mamba_output - attention_output), attention_weights


def collect_sequences(tokenizer, split, limit: int) -> list[torch.Tensor]:
    sequences = []
    for row in split:
        text = row["text"].strip()
        if len(text) < 80:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).input_ids.squeeze(0)
        if ids.numel() >= 16:
            sequences.append(ids)
        if len(sequences) == limit:
            break
    if len(sequences) != limit:
        raise RuntimeError(f"Requested {limit} usable WikiText sequences, found {len(sequences)}")
    return sequences


def nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target) / target.pow(2).mean().clamp(min=1e-8)


def cosine(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(prediction.reshape(1, -1), target.reshape(1, -1)).mean()


def local_loss(prediction: torch.Tensor, target: torch.Tensor):
    e = nmse(prediction, target)
    c = cosine(prediction, target)
    s = torch.log(prediction.pow(2).mean().sqrt().clamp(min=1e-8) / target.pow(2).mean().sqrt().clamp(min=1e-8)).pow(2)
    return e + 0.5 * (1 - c) + 0.2 * s, e, c


def held_out_loss(model, sequences: list[torch.Tensor]) -> dict[str, float]:
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for ids in sequences:
            inputs = ids[None, :].to(DEVICE)
            logits = model(input_ids=inputs, use_cache=False).logits
            total += float(F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), inputs[:, 1:].reshape(-1), reduction="sum").item())
            tokens += inputs.shape[1] - 1
    loss = total / tokens
    return {"loss": loss, "perplexity": float(torch.exp(torch.tensor(loss)).item()), "tokens": tokens}


def generate(model, tokenizer) -> dict[str, str]:
    output = {}
    model.eval()
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            for _ in range(8):
                next_token = model(input_ids=ids, use_cache=False).logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, next_token], dim=-1)
            output[prompt] = tokenizer.decode(ids[0], skip_special_tokens=True)
    return output


def set_trainable(branch, active_layer: int) -> None:
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
    for parameter in branch.model.layers[active_layer].self_attn.mixer.parameters():
        parameter.requires_grad_(True)


def trajectory_from_branch(branch, layer_index: int, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        logits = branch(input_ids=ids, use_cache=False).logits
    gate = branch.model.layers[layer_index].self_attn
    if gate.last_input is None or gate.last_attention is None:
        raise RuntimeError("Layer trajectory was not captured")
    return gate.last_input.detach(), gate.last_attention.detach(), logits.detach()


def local_metrics(branch, layer_index: int, sequences: list[torch.Tensor]) -> dict[str, float]:
    branch.eval()
    gate = branch.model.layers[layer_index].self_attn
    mixer = gate.mixer
    mixer.eval()
    values = []
    with torch.no_grad():
        for ids in sequences:
            inputs = ids[None, :].to(DEVICE)
            source_input, target, _ = trajectory_from_branch(branch, layer_index, inputs)
            pred = mixer(source_input)
            values.append((float(nmse(pred, target)), float(cosine(pred, target)), float(pred.pow(2).mean().sqrt()), float(pred.abs().max())))
    return {
        "normalized_mse": sum(v[0] for v in values) / len(values),
        "cosine_similarity": sum(v[1] for v in values) / len(values),
        "mamba_output_rms": sum(v[2] for v in values) / len(values),
        "mamba_output_absmax": max(v[3] for v in values),
    }


def train_local(branch, layer_index: int, sequences: list[torch.Tensor]) -> list[dict[str, float]]:
    gate = branch.model.layers[layer_index].self_attn
    mixer = gate.mixer
    steps = LAYER_BUDGETS[layer_index]["local_steps"]
    set_trainable(branch, layer_index)
    mixer.train()
    optimizer = torch.optim.AdamW(mixer.parameters(), lr=LOCAL_LR, weight_decay=0.01)
    trace = []
    for step in range(steps):
        ids = sequences[step % len(sequences)][None, :].to(DEVICE)
        source_input, target, _ = trajectory_from_branch(branch, layer_index, ids)
        pred = mixer(source_input)
        loss, e, c = local_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mixer.parameters(), 1.0)
        optimizer.step()
        trace.append({"step": step + 1, "loss": float(loss.detach()), "nmse": float(e.detach()), "cosine": float(c.detach())})
        if (step + 1) % 60 == 0:
            print(f"    L{layer_index} local {step + 1:>3}/{steps}: NMSE={e.item():.4f}, cosine={c.item():.4f}")
    return trace


def train_gate(branch, teacher, layer_index: int, sequences: list[torch.Tensor], eval_sequences: list[torch.Tensor]):
    gate = branch.model.layers[layer_index].self_attn
    mixer = gate.mixer
    steps = LAYER_BUDGETS[layer_index]["gate_steps"]
    warmup = max(10, steps // 6)
    set_trainable(branch, layer_index)
    mixer.train()
    optimizer = torch.optim.AdamW(mixer.parameters(), lr=GATE_LR, weight_decay=0.01)
    trace, curve = [], []
    checkpoints = {warmup: 0.0, warmup + (steps - warmup) // 3: 0.25, warmup + 2 * (steps - warmup) // 3: 0.5, steps: 1.0}
    for step in range(steps):
        alpha = 0.0 if step < warmup else min(1.0, (step - warmup + 1) / (steps - warmup))
        gate.set_alpha(alpha)
        ids = sequences[step % len(sequences)][None, :].to(DEVICE)
        with torch.no_grad():
            teacher_logits = teacher(input_ids=ids, use_cache=False).logits
        student_logits = branch(input_ids=ids, use_cache=False).logits
        source_input = gate.last_input.detach()
        target = gate.last_attention.detach()
        local, e, c = local_loss(mixer(source_input), target)
        kl = F.kl_div(F.log_softmax(student_logits[:, :-1] / TEMPERATURE, dim=-1), F.softmax(teacher_logits[:, :-1] / TEMPERATURE, dim=-1), reduction="batchmean") * TEMPERATURE**2
        ce = F.cross_entropy(student_logits[:, :-1].reshape(-1, student_logits.shape[-1]), ids[:, 1:].reshape(-1))
        loss = 0.70 * kl + 0.20 * ce + 0.10 * local
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mixer.parameters(), 1.0)
        optimizer.step()
        trace.append({"step": step + 1, "alpha": alpha, "loss": float(loss.detach()), "kl": float(kl.detach()), "ce": float(ce.detach()), "local_nmse": float(e.detach()), "local_cosine": float(c.detach())})
        if (step + 1) in checkpoints:
            gate.set_alpha(checkpoints[step + 1])
            curve.append({"alpha": checkpoints[step + 1], "metrics": held_out_loss(branch, eval_sequences)})
            mixer.train()
        if (step + 1) % max(20, steps // 3) == 0:
            print(f"    L{layer_index} gate {step + 1:>3}/{steps}: alpha={alpha:.2f}, NMSE={e.item():.4f}, cosine={c.item():.4f}")
    gate.set_alpha(1.0)
    return trace, curve


def build_branch(source_config, teacher, mode: str):
    branch = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE)
    audit = {}
    for layer_index in LAYERS:
        config = mamba_config(source_config, LAYER_BUDGETS[layer_index]["state_size"])
        mixer = MambaMixer(config, layer_index).to(DEVICE)
        audit[f"layer_{layer_index}"] = {"mapped_mixer_tensors": 0, "fresh_time_step_bias": 1}
        if mode == "direct":
            audit[f"layer_{layer_index}"] = map_attention_to_mamba(teacher, mixer, layer_index)
        branch.model.layers[layer_index].self_attn = GatedAttentionReplacement(branch.model.layers[layer_index].self_attn, mixer).to(DEVICE)
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
    return branch, audit


def run_condition(mode, source_config, teacher, tokenizer, train_sequences, dev_sequences, test_sequences):
    print(f"\n  Condition: {mode}")
    set_seed(SEED + (1 if mode == "direct" else 2))
    branch, audit = build_branch(source_config, teacher, mode)
    zero = held_out_loss(branch, test_sequences)
    stages = {}
    for layer_index in LAYERS:
        print(f"\n  [Layer {layer_index}] fitting target trajectories")
        before = local_metrics(branch, layer_index, dev_sequences)
        local_trace = train_local(branch, layer_index, train_sequences)
        after = local_metrics(branch, layer_index, dev_sequences)
        gate = branch.model.layers[layer_index].self_attn
        gate.set_alpha(1.0)
        abrupt = held_out_loss(branch, dev_sequences)
        gate.set_alpha(0.0)
        gate_trace, curve = train_gate(branch, teacher, layer_index, train_sequences, dev_sequences)
        final = held_out_loss(branch, dev_sequences)
        print(f"  L{layer_index}: local NMSE {before['normalized_mse']:.4f}->{after['normalized_mse']:.4f}; abrupt={abrupt['loss']:.4f}; gated={final['loss']:.4f}")
        stages[f"layer_{layer_index}"] = {"local_before": before, "local_after": after, "abrupt": abrupt, "gate_curve": curve, "final_after_stage": final, "local_trace": local_trace, "gate_trace": gate_trace}
    result = {"audit": audit, "alpha_zero_test": zero, "stages_development": stages, "final_test": held_out_loss(branch, test_sequences), "generation": generate(branch, tokenizer)}
    del branch
    gc.collect()
    return result


def main() -> None:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_seed(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("EXPERIMENT 013 — INDEPENDENT-SEED TWO-LAYER REPLICATION")
    print(f"Seed={SEED}; train is calibration, validation is development, test is untouched final evaluation.")
    print("=" * 80)
    corpus = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_ID)
    train_sequences = collect_sequences(tokenizer, corpus["train"], TRAIN_SEQUENCES)
    dev_sequences = collect_sequences(tokenizer, corpus["validation"], EVAL_SEQUENCES)
    test_sequences = collect_sequences(tokenizer, corpus["test"], EVAL_SEQUENCES)
    source_config = AutoConfig.from_pretrained(SOURCE_ID)
    teacher = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE).eval()
    teacher_metrics = held_out_loss(teacher, test_sequences)
    print(f"Teacher TEST loss={teacher_metrics['loss']:.4f}; calibration sequences={len(train_sequences)}")
    direct = run_condition("direct", source_config, teacher, tokenizer, train_sequences, dev_sequences, test_sequences)
    random_branch = run_condition("random", source_config, teacher, tokenizer, train_sequences, dev_sequences, test_sequences)
    result = {
        "experiment": "experiment_013_independent_seed_replication",
        "source": {"model_id": SOURCE_ID, "replaced_layers": list(LAYERS), "teacher_frozen": True},
        "budget": {"train_sequences": TRAIN_SEQUENCES, "development_sequences": EVAL_SEQUENCES, "untouched_test_sequences": EVAL_SEQUENCES, "layer_budgets": LAYER_BUDGETS, "seed": SEED},
        "teacher": teacher_metrics,
        "direct": direct,
        "random": random_branch,
        "interpretation": {
            "alpha_zero_preserves_teacher_on_test": abs(direct["alpha_zero_test"]["loss"] - teacher_metrics["loss"]) < 1e-5,
            "direct_final_beats_random_on_test": direct["final_test"]["loss"] < random_branch["final_test"]["loss"],
            "two_layer_hybrid_generates_english_requires_review": True,
            "full_pure_mamba_conversion": False,
        },
    }
    with (OUTPUT / "results.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["interpretation"], indent=2))
    print(f"Saved results to {OUTPUT / 'results.json'}")


if __name__ == "__main__":
    main()
