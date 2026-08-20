"""Experiment 014: causal control for two-layer attention-to-Mamba transfer.

This benchmark compares two identically initialized Mamba hybrid branches. Functional
Distillation receives frozen teacher trajectories and logits; CE-only receives the same
calibration texts and optimization budget but no teacher-derived optimizer target.
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
SEED = int(os.environ.get("EXPERIMENT_SEED", "20260831"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT = Path("research/experiment_014_causal_control") / f"seed_{SEED}"
MAX_LENGTH = 48
TRAIN_SEQUENCES = 1024
DEV_SEQUENCES = 64
TEST_SEQUENCES = 128
LOCAL_LR = 2e-4
GATE_LR = 5e-5
TEMPERATURE = 2.0
LAYERS = (0, 1)
LAYER_BUDGETS = {
    0: {"state_size": 64, "local_steps": 180, "gate_steps": 60},
    1: {"state_size": 96, "local_steps": 360, "gate_steps": 120},
}
PROMPTS = (
    "The purpose of scientific research is to",
    "Once upon a time, a young explorer discovered",
    "The capital of France is",
    "Artificial intelligence can help people by",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class GatedAttentionReplacement(nn.Module):
    """Frozen source attention plus a trainable Mamba branch with a continuous gate."""

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
    sequences: list[torch.Tensor] = []
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
        raise RuntimeError(f"Requested {limit} usable sequences, found {len(sequences)}")
    return sequences


def nmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target) / target.pow(2).mean().clamp(min=1e-8)


def cosine(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(prediction.reshape(1, -1), target.reshape(1, -1)).mean()


def local_loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e = nmse(prediction, target)
    c = cosine(prediction, target)
    scale = torch.log(prediction.pow(2).mean().sqrt().clamp(min=1e-8) / target.pow(2).mean().sqrt().clamp(min=1e-8)).pow(2)
    return e + 0.5 * (1 - c) + 0.2 * scale, e, c


def token_ce(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


def held_out_loss(model, sequences: list[torch.Tensor]) -> dict[str, float]:
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for ids in sequences:
            inputs = ids[None, :].to(DEVICE)
            logits = model(input_ids=inputs, use_cache=False).logits
            total += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), inputs[:, 1:].reshape(-1), reduction="sum"))
            tokens += inputs.shape[1] - 1
    loss = total / tokens
    return {"loss": loss, "perplexity": float(torch.exp(torch.tensor(loss))), "tokens": tokens}


def generate(model, tokenizer) -> dict[str, str]:
    model.eval()
    output: dict[str, str] = {}
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            for _ in range(16):
                next_token = model(input_ids=ids, use_cache=False).logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat((ids, next_token), dim=-1)
            output[prompt] = tokenizer.decode(ids[0], skip_special_tokens=True)
    return output


def set_trainable(branch, active_layer: int) -> None:
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
    for parameter in branch.model.layers[active_layer].self_attn.mixer.parameters():
        parameter.requires_grad_(True)


def trajectory_from_branch(branch, layer_index: int, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        branch(input_ids=ids, use_cache=False)
    gate = branch.model.layers[layer_index].self_attn
    if gate.last_input is None or gate.last_attention is None:
        raise RuntimeError("Failed to capture frozen attention trajectory")
    return gate.last_input.detach(), gate.last_attention.detach()


def local_metrics(branch, layer_index: int, sequences: list[torch.Tensor]) -> dict[str, float]:
    """Post-hoc diagnostic only; no values are supplied to the optimizer in CE-only."""
    branch.eval()
    gate = branch.model.layers[layer_index].self_attn
    mixer = gate.mixer
    mixer.eval()
    values = []
    with torch.no_grad():
        for ids in sequences:
            inputs = ids[None, :].to(DEVICE)
            source_input, target = trajectory_from_branch(branch, layer_index, inputs)
            prediction = mixer(source_input)
            values.append((float(nmse(prediction, target)), float(cosine(prediction, target))))
    return {
        "normalized_mse": sum(value[0] for value in values) / len(values),
        "cosine_similarity": sum(value[1] for value in values) / len(values),
    }


def train_local_functional(branch, layer_index: int, sequences: list[torch.Tensor]) -> list[dict[str, float]]:
    gate = branch.model.layers[layer_index].self_attn
    mixer = gate.mixer
    steps = LAYER_BUDGETS[layer_index]["local_steps"]
    gate.set_alpha(0.0)
    set_trainable(branch, layer_index)
    mixer.train()
    optimizer = torch.optim.AdamW(mixer.parameters(), lr=LOCAL_LR, weight_decay=0.01)
    trace = []
    for step in range(steps):
        ids = sequences[step % len(sequences)][None, :].to(DEVICE)
        source_input, target = trajectory_from_branch(branch, layer_index, ids)
        prediction = mixer(source_input)
        loss, error, similarity = local_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mixer.parameters(), 1.0)
        optimizer.step()
        trace.append({"step": step + 1, "objective": float(loss.detach()), "ce": None, "local_nmse": float(error.detach()), "local_cosine": float(similarity.detach())})
        if (step + 1) % 60 == 0:
            print(f"    L{layer_index} functional local {step + 1:>3}/{steps}: NMSE={error.item():.4f}, cosine={similarity.item():.4f}")
    return trace


def train_local_ce_only(branch, layer_index: int, sequences: list[torch.Tensor]) -> list[dict[str, float]]:
    gate = branch.model.layers[layer_index].self_attn
    mixer = gate.mixer
    steps = LAYER_BUDGETS[layer_index]["local_steps"]
    gate.set_alpha(1.0)
    set_trainable(branch, layer_index)
    mixer.train()
    optimizer = torch.optim.AdamW(mixer.parameters(), lr=LOCAL_LR, weight_decay=0.01)
    trace = []
    for step in range(steps):
        ids = sequences[step % len(sequences)][None, :].to(DEVICE)
        logits = branch(input_ids=ids, use_cache=False).logits
        loss = token_ce(logits, ids)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mixer.parameters(), 1.0)
        optimizer.step()
        trace.append({"step": step + 1, "objective": float(loss.detach()), "ce": float(loss.detach()), "local_nmse": None, "local_cosine": None})
        if (step + 1) % 60 == 0:
            print(f"    L{layer_index} CE-only local {step + 1:>3}/{steps}: CE={loss.item():.4f}")
    return trace


def gate_alpha(step: int, steps: int) -> float:
    warmup = max(10, steps // 6)
    floor = 0.05
    if step < warmup:
        return floor
    return floor + (1.0 - floor) * (step - warmup + 1) / (steps - warmup)


def train_gate_functional(branch, teacher, layer_index: int, sequences: list[torch.Tensor], dev_sequences: list[torch.Tensor]) -> tuple[list[dict[str, float]], list[dict[str, object]]]:
    gate = branch.model.layers[layer_index].self_attn
    mixer = gate.mixer
    steps = LAYER_BUDGETS[layer_index]["gate_steps"]
    set_trainable(branch, layer_index)
    mixer.train()
    optimizer = torch.optim.AdamW(mixer.parameters(), lr=GATE_LR, weight_decay=0.01)
    trace, curve = [], []
    checkpoints = {max(10, steps // 6): 0.05, max(10, steps // 6) + (steps - max(10, steps // 6)) // 3: 0.35, max(10, steps // 6) + 2 * (steps - max(10, steps // 6)) // 3: 0.65, steps: 1.0}
    for step in range(steps):
        alpha = gate_alpha(step, steps)
        gate.set_alpha(alpha)
        ids = sequences[step % len(sequences)][None, :].to(DEVICE)
        with torch.no_grad():
            teacher_logits = teacher(input_ids=ids, use_cache=False).logits
        student_logits = branch(input_ids=ids, use_cache=False).logits
        source_input = gate.last_input.detach()
        target = gate.last_attention.detach()
        local, error, similarity = local_loss(mixer(source_input), target)
        kl = F.kl_div(F.log_softmax(student_logits[:, :-1] / TEMPERATURE, dim=-1), F.softmax(teacher_logits[:, :-1] / TEMPERATURE, dim=-1), reduction="batchmean") * TEMPERATURE**2
        ce = token_ce(student_logits, ids)
        loss = 0.70 * kl + 0.20 * ce + 0.10 * local
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mixer.parameters(), 1.0)
        optimizer.step()
        trace.append({"step": step + 1, "alpha": alpha, "objective": float(loss.detach()), "kl": float(kl.detach()), "ce": float(ce.detach()), "local_nmse": float(error.detach()), "local_cosine": float(similarity.detach())})
        if (step + 1) in checkpoints:
            gate.set_alpha(checkpoints[step + 1])
            curve.append({"alpha": checkpoints[step + 1], "development": held_out_loss(branch, dev_sequences)})
            mixer.train()
        if (step + 1) % max(20, steps // 3) == 0:
            print(f"    L{layer_index} functional gate {step + 1:>3}/{steps}: alpha={alpha:.2f}, KL={kl.item():.4f}, CE={ce.item():.4f}")
    gate.set_alpha(1.0)
    return trace, curve


def train_gate_ce_only(branch, layer_index: int, sequences: list[torch.Tensor], dev_sequences: list[torch.Tensor]) -> tuple[list[dict[str, float]], list[dict[str, object]]]:
    gate = branch.model.layers[layer_index].self_attn
    mixer = gate.mixer
    steps = LAYER_BUDGETS[layer_index]["gate_steps"]
    set_trainable(branch, layer_index)
    mixer.train()
    optimizer = torch.optim.AdamW(mixer.parameters(), lr=GATE_LR, weight_decay=0.01)
    trace, curve = [], []
    checkpoints = {max(10, steps // 6): 0.05, max(10, steps // 6) + (steps - max(10, steps // 6)) // 3: 0.35, max(10, steps // 6) + 2 * (steps - max(10, steps // 6)) // 3: 0.65, steps: 1.0}
    for step in range(steps):
        alpha = gate_alpha(step, steps)
        gate.set_alpha(alpha)
        ids = sequences[step % len(sequences)][None, :].to(DEVICE)
        logits = branch(input_ids=ids, use_cache=False).logits
        ce = token_ce(logits, ids)
        optimizer.zero_grad(set_to_none=True)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(mixer.parameters(), 1.0)
        optimizer.step()
        trace.append({"step": step + 1, "alpha": alpha, "objective": float(ce.detach()), "kl": None, "ce": float(ce.detach()), "local_nmse": None, "local_cosine": None})
        if (step + 1) in checkpoints:
            gate.set_alpha(checkpoints[step + 1])
            curve.append({"alpha": checkpoints[step + 1], "development": held_out_loss(branch, dev_sequences)})
            mixer.train()
        if (step + 1) % max(20, steps // 3) == 0:
            print(f"    L{layer_index} CE-only gate {step + 1:>3}/{steps}: alpha={alpha:.2f}, CE={ce.item():.4f}")
    gate.set_alpha(1.0)
    return trace, curve


def build_branch(source_config):
    branch = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE)
    audit: dict[str, object] = {}
    for layer_index in LAYERS:
        mixer = MambaMixer(mamba_config(source_config, LAYER_BUDGETS[layer_index]["state_size"]), layer_index).to(DEVICE)
        branch.model.layers[layer_index].self_attn = GatedAttentionReplacement(branch.model.layers[layer_index].self_attn, mixer).to(DEVICE)
        audit[f"layer_{layer_index}"] = {"initialization": "fresh_random_mamba", "state_size": LAYER_BUDGETS[layer_index]["state_size"]}
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
    return branch, audit


def run_condition(condition: str, source_config, teacher, tokenizer, train_sequences, dev_sequences, test_sequences) -> dict[str, object]:
    print(f"\n  Condition: {condition}")
    set_seed(SEED)  # Identical Mamba weights in both paired conditions.
    branch, audit = build_branch(source_config)
    alpha_zero = held_out_loss(branch, test_sequences)
    stages: dict[str, object] = {}

    for layer_index in LAYERS:
        print(f"\n  [Layer {layer_index}] matched {condition} fitting")
        pre_local_diagnostic = local_metrics(branch, layer_index, dev_sequences)
        if condition == "functional":
            local_trace = train_local_functional(branch, layer_index, train_sequences)
        elif condition == "ce_only":
            local_trace = train_local_ce_only(branch, layer_index, train_sequences)
        else:
            raise ValueError(condition)
        post_local_diagnostic = local_metrics(branch, layer_index, dev_sequences)
        gate = branch.model.layers[layer_index].self_attn
        gate.set_alpha(1.0)
        abrupt_development = held_out_loss(branch, dev_sequences)
        if condition == "functional":
            gate_trace, curve = train_gate_functional(branch, teacher, layer_index, train_sequences, dev_sequences)
        else:
            gate_trace, curve = train_gate_ce_only(branch, layer_index, train_sequences, dev_sequences)
        final_development = held_out_loss(branch, dev_sequences)
        print(f"  L{layer_index}: local NMSE {pre_local_diagnostic['normalized_mse']:.4f}->{post_local_diagnostic['normalized_mse']:.4f}; abrupt-dev={abrupt_development['loss']:.4f}; gated-dev={final_development['loss']:.4f}")
        stages[f"layer_{layer_index}"] = {
            "local_diagnostic_pre": pre_local_diagnostic,
            "local_diagnostic_post": post_local_diagnostic,
            "abrupt_development": abrupt_development,
            "gate_curve_development": curve,
            "final_after_stage_development": final_development,
            "local_trace": local_trace,
            "gate_trace": gate_trace,
        }

    final_test = held_out_loss(branch, test_sequences)
    result = {
        "audit": audit,
        "alpha_zero_test": alpha_zero,
        "stages_development": stages,
        "final_test": final_test,
        "generation": generate(branch, tokenizer),
    }
    del branch
    gc.collect()
    return result


def main() -> None:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_seed(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("EXPERIMENT 014 — CAUSAL CONTROL FOR FUNCTIONAL DISTILLATION")
    print(f"Seed={SEED}; train=1,024 calibration, validation=64 diagnostics, test=128 frozen final evaluation.")
    print("=" * 80)
    corpus = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_ID)
    train_sequences = collect_sequences(tokenizer, corpus["train"], TRAIN_SEQUENCES)
    dev_sequences = collect_sequences(tokenizer, corpus["validation"], DEV_SEQUENCES)
    test_sequences = collect_sequences(tokenizer, corpus["test"], TEST_SEQUENCES)
    source_config = AutoConfig.from_pretrained(SOURCE_ID)
    teacher = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE).eval()
    teacher_test = held_out_loss(teacher, test_sequences)
    print(f"Teacher TEST loss={teacher_test['loss']:.4f}; test tokens={teacher_test['tokens']}")
    functional = run_condition("functional", source_config, teacher, tokenizer, train_sequences, dev_sequences, test_sequences)
    ce_only = run_condition("ce_only", source_config, teacher, tokenizer, train_sequences, dev_sequences, test_sequences)
    functional_loss = functional["final_test"]["loss"]
    ce_only_loss = ce_only["final_test"]["loss"]
    result = {
        "experiment": "experiment_014_causal_control",
        "source": {"model_id": SOURCE_ID, "replaced_layers": list(LAYERS), "teacher_frozen": True, "full_pure_mamba_conversion": False},
        "budget": {
            "calibration_sequences": TRAIN_SEQUENCES,
            "development_sequences": DEV_SEQUENCES,
            "untouched_test_sequences": TEST_SEQUENCES,
            "max_length": MAX_LENGTH,
            "layer_budgets": LAYER_BUDGETS,
            "seed": SEED,
            "matched_initialization": True,
        },
        "teacher": teacher_test,
        "functional": functional,
        "ce_only": ce_only,
        "interpretation": {
            "alpha_zero_preserves_teacher_functional": abs(functional["alpha_zero_test"]["loss"] - teacher_test["loss"]) < 1e-5,
            "alpha_zero_preserves_teacher_ce_only": abs(ce_only["alpha_zero_test"]["loss"] - teacher_test["loss"]) < 1e-5,
            "functional_minus_ce_only_test_loss": functional_loss - ce_only_loss,
            "functional_beats_ce_only_on_test": functional_loss < ce_only_loss,
            "finite_final_losses": math.isfinite(functional_loss) and math.isfinite(ce_only_loss),
            "generation_requires_human_review": True,
        },
    }
    with (OUTPUT / "results.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["interpretation"], indent=2))
    print(f"Saved results to {OUTPUT / 'results.json'}")


if __name__ == "__main__":
    main()
