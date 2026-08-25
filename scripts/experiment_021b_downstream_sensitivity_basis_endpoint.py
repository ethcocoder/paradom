from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba.modeling_mamba import MambaMixer

import experiment_015_adaptive_interface as base

SEED = int(os.environ.get("EXPERIMENT_SEED", "20260911"))
OUT = Path("research/experiment_021b_downstream_sensitivity_basis_endpoint") / f"seed_{SEED}"
SOURCE_ID, DEVICE = base.SOURCE_ID, base.DEVICE
MAX_LENGTH = 48
SENSITIVITY_SEQUENCES, TRAIN_SEQUENCES = 128, 1024
DEV_SEQUENCES, DEV_SKIP = 64, 64
FRESH_TEST_SEQUENCES, FRESH_TEST_SKIP = 128, 640
LAYERS, LAYER_STEPS, STATE_SIZES = (0, 1), {0: 300, 1: 420}, {0: 64, 1: 96}
TRANSPORT_RANK, LEARNING_RATE, WEIGHT_DECAY, WARMUP_STEPS = 16, 1e-4, 0.01, 10
GAIN_MIN, GAIN_MAX = 0.50, 2.00
EPS = 1e-12
CONDITIONS = {"unit_conditional_ce_transport": "unit", "correct_sensitivity_basis": "correct", "permuted_sensitivity_basis": "permuted"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect(split, tokenizer, limit: int, skip: int = 0) -> list[torch.Tensor]:
    items, eligible = [], 0
    for row in split:
        text = row["text"].strip()
        if len(text) < 80:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).input_ids.squeeze(0)
        if ids.numel() < 16:
            continue
        if eligible >= skip:
            items.append(ids)
            if len(items) == limit:
                break
        eligible += 1
    if len(items) != limit:
        raise RuntimeError(f"Expected {limit} eligible sequences after skip={skip}, got {len(items)}")
    return items


def ce(logits: torch.Tensor, ids: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction=reduction)


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)


def alpha_for_step(step: int, total: int) -> float:
    return 0.05 if step < WARMUP_STEPS else 0.05 + 0.95 * (step - WARMUP_STEPS + 1) / (total - WARMUP_STEPS)


def avg(values: list[float]) -> float:
    if not values:
        raise RuntimeError("empty metric list")
    return sum(values) / len(values)


def gradient_rms(params) -> float:
    total, count = 0.0, 0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
            count += p.grad.numel()
    return math.sqrt(total / count) if count else 0.0


def effective_rank(x: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(x.detach().reshape(-1, x.shape[-1]).float())
    mass = singular / singular.sum().clamp(min=EPS)
    return float((-(mass * mass.clamp(min=EPS).log()).sum()).exp())


def relative_drift(student: torch.Tensor, teacher: torch.Tensor) -> float:
    return float((student - teacher).pow(2).mean().sqrt() / teacher.pow(2).mean().sqrt().clamp(min=EPS))


class InputCapture:
    def __init__(self) -> None:
        self.hidden: torch.Tensor | None = None

    def attach(self, module: nn.Module) -> None:
        module.register_forward_pre_hook(lambda _m, args: setattr(self, "hidden", args[0]))


class AttentionOutputCapture:
    def __init__(self) -> None:
        self.tensor: torch.Tensor | None = None

    def attach(self, module: nn.Module) -> None:
        def hook(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            self.tensor = tensor
            if tensor.requires_grad:
                tensor.retain_grad()
        module.register_forward_hook(hook)


class SensitivityTransport(nn.Module):
    def __init__(self, mixer: MambaMixer, hidden_size: int, gain: torch.Tensor):
        super().__init__()
        self.mixer = mixer
        self.down = nn.Linear(3 * hidden_size, TRANSPORT_RANK, bias=False)
        self.up = nn.Linear(TRANSPORT_RANK, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        self.register_buffer("gain", gain.detach().clone())
        self.last_mamba: torch.Tensor | None = None
        self.last_correction: torch.Tensor | None = None

    def forward(self, attention_input: torch.Tensor) -> torch.Tensor:
        mamba = self.mixer(attention_input)
        normalized_input, normalized_mamba = rms_norm(attention_input), rms_norm(mamba)
        features = torch.cat((normalized_input, normalized_mamba, normalized_input - normalized_mamba), dim=-1)
        correction = self.gain * self.up(F.silu(self.down(features)))
        self.last_mamba, self.last_correction = mamba, correction
        return mamba + correction


class Replacement(nn.Module):
    def __init__(self, attention: nn.Module, transport: SensitivityTransport):
        super().__init__()
        self.attention, self.mixer = attention, transport
        self.register_buffer("alpha", torch.tensor(0.0))
        self.last_attention: torch.Tensor | None = None
        self.last_transport: torch.Tensor | None = None
        for parameter in attention.parameters():
            parameter.requires_grad_(False)

    def set_alpha(self, value: float) -> None:
        self.alpha.fill_(float(value))

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        attention_out, weights = self.attention(hidden_states, **kwargs)
        transport_out = self.mixer(hidden_states)
        self.last_attention, self.last_transport = attention_out, transport_out
        return attention_out + self.alpha * (transport_out - attention_out), weights


def set_all_alpha(branch: nn.Module, value: float) -> None:
    for layer in LAYERS:
        branch.model.layers[layer].self_attn.set_alpha(value)


def build_branch(config, gains: dict[int, torch.Tensor]) -> tuple[nn.Module, dict[int, InputCapture]]:
    branch = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE)
    for layer in LAYERS:
        wrapped = branch.model.layers[layer]
        mixer = MambaMixer(base.make_config(config, STATE_SIZES[layer]), layer).to(DEVICE)
        transport = SensitivityTransport(mixer, config.hidden_size, gains[layer].to(DEVICE)).to(DEVICE)
        wrapped.self_attn = Replacement(wrapped.self_attn, transport).to(DEVICE)
    captures = {1: InputCapture(), 2: InputCapture()}
    for layer, capture in captures.items():
        capture.attach(branch.model.layers[layer])
    for p in branch.parameters():
        p.requires_grad_(False)
    return branch, captures


def attach_teacher_input_captures(teacher: nn.Module) -> dict[int, InputCapture]:
    captures = {1: InputCapture(), 2: InputCapture()}
    for layer, capture in captures.items():
        capture.attach(teacher.model.layers[layer])
    return captures


def derive_sensitivity(teacher: nn.Module, sequences: list[torch.Tensor], hidden_size: int) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    captures = {layer: AttentionOutputCapture() for layer in LAYERS}
    for layer, capture in captures.items():
        capture.attach(teacher.model.layers[layer].self_attn)
    accumulation = {layer: torch.zeros(hidden_size, device=DEVICE) for layer in LAYERS}
    total_positions = 0
    teacher.eval()
    for sequence in sequences:
        ids = sequence[None].to(DEVICE)
        embeds = teacher.model.embed_tokens(ids).detach().requires_grad_(True)
        teacher.zero_grad(set_to_none=True)
        logits = teacher(inputs_embeds=embeds, use_cache=False).logits
        loss = ce(logits, ids)
        loss.backward()
        for layer, capture in captures.items():
            if capture.tensor is None or capture.tensor.grad is None:
                raise RuntimeError(f"Missing attention-output gradient for sensitivity layer {layer}")
            accumulation[layer] += capture.tensor.grad.detach().pow(2).sum(dim=(0, 1))
        total_positions += ids.shape[1]
    gains, fishers, summary = {}, {}, {}
    for layer in LAYERS:
        fisher = accumulation[layer] / total_positions
        raw = torch.sqrt(fisher / fisher.mean().clamp(min=EPS))
        gain = raw.clamp(GAIN_MIN, GAIN_MAX)
        gain = gain / gain.mean().clamp(min=EPS)
        gains[layer] = gain.detach()
        fishers[layer] = fisher.detach()
        summary[str(layer)] = {"fisher_mean": float(fisher.mean()), "fisher_min": float(fisher.min()), "fisher_max": float(fisher.max()), "gain_mean": float(gain.mean()), "gain_sd": float(gain.std()), "gain_min": float(gain.min()), "gain_max": float(gain.max()), "cyclic_shift": hidden_size // 2}
    return gains, fishers, summary


def make_condition_gains(correct: dict[int, torch.Tensor], kind: str) -> dict[int, torch.Tensor]:
    if kind == "unit":
        return {layer: torch.ones_like(correct[layer]) for layer in LAYERS}
    if kind == "correct":
        return {layer: correct[layer].clone() for layer in LAYERS}
    if kind == "permuted":
        return {layer: torch.roll(correct[layer], shifts=correct[layer].numel() // 2, dims=0) for layer in LAYERS}
    raise ValueError(kind)


def set_active_trainable(branch: nn.Module, layer: int) -> None:
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
    for parameter in branch.model.layers[layer].self_attn.mixer.parameters():
        parameter.requires_grad_(True)


def train_layer(branch: nn.Module, layer: int, train: list[torch.Tensor]) -> list[dict[str, float]]:
    set_active_trainable(branch, layer)
    branch.train()
    wrapper: Replacement = branch.model.layers[layer].self_attn
    optimizer = torch.optim.AdamW(wrapper.mixer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    trace = []
    for step in range(LAYER_STEPS[layer]):
        alpha = alpha_for_step(step, LAYER_STEPS[layer])
        wrapper.set_alpha(alpha)
        ids = train[step % len(train)][None].to(DEVICE)
        objective = ce(branch(input_ids=ids, use_cache=False).logits, ids)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        mamba_gradient = gradient_rms(wrapper.mixer.mixer.parameters())
        transport_gradient = gradient_rms(list(wrapper.mixer.down.parameters()) + list(wrapper.mixer.up.parameters()))
        torch.nn.utils.clip_grad_norm_(wrapper.mixer.parameters(), 1.0)
        optimizer.step()
        trace.append({"step": float(step + 1), "alpha": alpha, "ce": float(objective.detach()), "mamba_gradient_rms": mamba_gradient, "transport_gradient_rms": transport_gradient})
    wrapper.set_alpha(1.0)
    return trace


def dev_loss(model: nn.Module, sequences: list[torch.Tensor]) -> float:
    model.eval()
    total_loss, tokens = 0.0, 0
    with torch.no_grad():
        for sequence in sequences:
            ids = sequence[None].to(DEVICE)
            logits = model(input_ids=ids, use_cache=False).logits
            total_loss += float(ce(logits, ids, reduction="sum"))
            tokens += ids.shape[1] - 1
    return total_loss / tokens


def alpha_zero_integrity(branch: nn.Module, teacher: nn.Module, dev: list[torch.Tensor]) -> dict[str, float]:
    set_all_alpha(branch, 0.0)
    branch_loss = dev_loss(branch, dev)
    teacher_loss = dev_loss(teacher, dev)
    set_all_alpha(branch, 1.0)
    return {"teacher_development_ce": teacher_loss, "alpha_zero_development_ce": branch_loss, "absolute_loss_deviation": abs(branch_loss - teacher_loss)}


def development_diagnostics(branch: nn.Module, teacher: nn.Module, teacher_captures: dict[int, InputCapture], branch_captures: dict[int, InputCapture], fisher: dict[int, torch.Tensor], dev: list[torch.Tensor]) -> dict[str, Any]:
    branch.eval(); teacher.eval()
    ce_values, weighted_errors, unweighted_errors, l1, l2 = [], [], [], [], []
    ratios, ranks = {layer: [] for layer in LAYERS}, {layer: [] for layer in LAYERS}
    with torch.no_grad():
        for sequence in dev:
            ids = sequence[None].to(DEVICE)
            teacher_logits = teacher(input_ids=ids, use_cache=False).logits
            branch_logits = branch(input_ids=ids, use_cache=False).logits
            ce_values.append(float(ce(branch_logits, ids)))
            for layer in LAYERS:
                wrapper: Replacement = branch.model.layers[layer].self_attn
                if wrapper.last_attention is None or wrapper.last_transport is None:
                    raise RuntimeError("Transport output diagnostic unavailable")
                error = (wrapper.last_transport - wrapper.last_attention).pow(2)
                weighted_errors.append(float((error * fisher[layer].view(1, 1, -1)).mean() / fisher[layer].mean().clamp(min=EPS)))
                unweighted_errors.append(float(error.mean()))
                transport = wrapper.mixer
                if transport.last_mamba is None or transport.last_correction is None:
                    raise RuntimeError("Correction diagnostic unavailable")
                ratios[layer].append(float(transport.last_correction.pow(2).mean().sqrt() / transport.last_mamba.pow(2).mean().sqrt().clamp(min=EPS)))
                ranks[layer].append(effective_rank(transport.last_correction))
            if teacher_captures[1].hidden is None or teacher_captures[2].hidden is None or branch_captures[1].hidden is None or branch_captures[2].hidden is None:
                raise RuntimeError("Residual captures unavailable")
            l1.append(relative_drift(branch_captures[1].hidden, teacher_captures[1].hidden))
            l2.append(relative_drift(branch_captures[2].hidden, teacher_captures[2].hidden))
    return {"development_ce": avg(ce_values), "sensitivity_weighted_attention_output_discrepancy": avg(weighted_errors), "unweighted_attention_output_mse": avg(unweighted_errors), "post_layer_1_relative_drift": avg(l1), "post_layer_2_relative_drift": avg(l2), "transport_correction_rms_ratio": {str(layer): avg(values) for layer, values in ratios.items()}, "transport_correction_effective_rank": {str(layer): avg(values) for layer, values in ranks.items()}}


def run_condition(name: str, kind: str, config, correct_gains: dict[int, torch.Tensor], teacher: nn.Module, teacher_captures: dict[int, InputCapture], fisher: dict[int, torch.Tensor], train: list[torch.Tensor], dev: list[torch.Tensor]) -> dict[str, Any]:
    set_seed(SEED)
    gains = make_condition_gains(correct_gains, kind)
    branch, branch_captures = build_branch(config, gains)
    stages = {}
    for layer in LAYERS:
        for prior in LAYERS:
            if prior < layer:
                branch.model.layers[prior].self_attn.set_alpha(1.0)
        stages[f"layer_{layer}"] = {"updates": LAYER_STEPS[layer], "training_trace": train_layer(branch, layer, train), "endpoint_alpha": float(branch.model.layers[layer].self_attn.alpha)}
    set_all_alpha(branch, 1.0)
    endpoint = development_diagnostics(branch, teacher, teacher_captures, branch_captures, fisher, dev)
    integrity = alpha_zero_integrity(branch, teacher, dev)
    print(f"  {name}: dev CE={endpoint['development_ce']:.4f}; sensitivity-weighted discrepancy={endpoint['sensitivity_weighted_attention_output_discrepancy']:.6f}")
    result = {"basis_type": kind, "gain_summary": {str(layer): {"mean": float(gains[layer].mean()), "sd": float(gains[layer].std()), "min": float(gains[layer].min()), "max": float(gains[layer].max())} for layer in LAYERS}, "stages": stages, "development_endpoint": endpoint, "alpha_zero_integrity": integrity, "endpoint_alpha_one": all(float(branch.model.layers[layer].self_attn.alpha) > 0.999 for layer in LAYERS)}
    del branch
    return result


def main() -> None:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"EXPERIMENT 021A seed={SEED}: development-only sensitivity-basis pilot; test split prohibited")
    train_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    validation_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_ID)
    sensitivity_sequences = collect(train_split, tokenizer, SENSITIVITY_SEQUENCES)
    train = collect(train_split, tokenizer, TRAIN_SEQUENCES)
    dev = collect(validation_split, tokenizer, DEV_SEQUENCES, skip=DEV_SKIP)
    config = AutoConfig.from_pretrained(SOURCE_ID)
    teacher = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE).eval()
    teacher_captures = attach_teacher_input_captures(teacher)
    correct_gains, fisher, basis_summary = derive_sensitivity(teacher, sensitivity_sequences, config.hidden_size)
    results = {name: run_condition(name, kind, config, correct_gains, teacher, teacher_captures, fisher, train, dev) for name, kind in CONDITIONS.items()}
    unit_ce = results["unit_conditional_ce_transport"]["development_endpoint"]["development_ce"]
    unit_weighted = results["unit_conditional_ce_transport"]["development_endpoint"]["sensitivity_weighted_attention_output_discrepancy"]
    correct = results["correct_sensitivity_basis"]["development_endpoint"]
    permuted = results["permuted_sensitivity_basis"]["development_endpoint"]
    output = {"experiment": "experiment_021a_downstream_sensitivity_basis_pilot", "seed": SEED, "data": {"sensitivity_split": "wikitext-2-raw-v1/train", "sensitivity_sequences": SENSITIVITY_SEQUENCES, "calibration_split": "wikitext-2-raw-v1/train", "calibration_sequences": TRAIN_SEQUENCES, "development_split": "wikitext-2-raw-v1/validation", "development_eligible_offset": DEV_SKIP, "development_sequences": DEV_SEQUENCES, "test_split_requested": False}, "budget": {"layer_updates": LAYER_STEPS, "total_updates_per_condition": sum(LAYER_STEPS.values()), "transport_rank": TRANSPORT_RANK, "gain_bounds": [GAIN_MIN, GAIN_MAX], "gain_formula": "renorm_mean_1(clip(sqrt(fisher/mean_fisher), 0.50, 2.00))"}, "sensitivity_basis": basis_summary, **results, "interpretation": {"alpha_zero_exact_all": all(results[name]["alpha_zero_integrity"]["absolute_loss_deviation"] <= 1e-5 for name in CONDITIONS), "all_alpha_one_endpoints": all(results[name]["endpoint_alpha_one"] for name in CONDITIONS), "correct_development_ce_minus_unit": correct["development_ce"] - unit_ce, "correct_weighted_discrepancy_minus_unit": correct["sensitivity_weighted_attention_output_discrepancy"] - unit_weighted, "correct_weighted_discrepancy_minus_permuted": correct["sensitivity_weighted_attention_output_discrepancy"] - permuted["sensitivity_weighted_attention_output_discrepancy"]}}
    (OUT / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["interpretation"], indent=2))


def fresh_metrics(model: nn.Module, sequences: list[torch.Tensor]) -> dict[str, float]:
    model.eval()
    total_loss, total_entropy, tokens = 0.0, 0.0, 0
    bins = [{"count": 0, "confidence": 0.0, "correct": 0.0} for _ in range(20)]
    with torch.no_grad():
        for sequence in sequences:
            ids = sequence[None].to(DEVICE)
            logits = model(input_ids=ids, use_cache=False).logits[:, :-1]
            labels = ids[:, 1:]
            flat_logits, flat_labels = logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            total_loss += float(F.cross_entropy(flat_logits, flat_labels, reduction="sum"))
            probabilities = F.softmax(flat_logits, dim=-1)
            total_entropy += float((-(probabilities * probabilities.clamp(min=EPS).log()).sum(dim=-1)).sum())
            confidence, prediction = probabilities.max(dim=-1)
            for conf, correct in zip(confidence.tolist(), prediction.eq(flat_labels).tolist()):
                index = min(19, int(conf * 20))
                bins[index]["count"] += 1
                bins[index]["confidence"] += conf
                bins[index]["correct"] += float(correct)
            tokens += flat_labels.numel()
    loss = total_loss / tokens
    ece = sum(bucket["count"] / tokens * abs(bucket["confidence"] / bucket["count"] - bucket["correct"] / bucket["count"]) for bucket in bins if bucket["count"])
    return {"loss": loss, "perplexity": float(torch.exp(torch.tensor(loss))), "tokens": tokens, "token_entropy": total_entropy / tokens, "top_token_ece_20bin": ece}


def alpha_zero_fresh(branch: nn.Module, teacher: nn.Module, fresh: list[torch.Tensor]) -> dict[str, Any]:
    set_all_alpha(branch, 0.0)
    branch_metrics = fresh_metrics(branch, fresh)
    teacher_metrics = fresh_metrics(teacher, fresh)
    set_all_alpha(branch, 1.0)
    return {"teacher_fresh_test": teacher_metrics, "alpha_zero_fresh_test": branch_metrics, "absolute_loss_deviation": abs(branch_metrics["loss"] - teacher_metrics["loss"])}


def greedy_continuations(branch: nn.Module, tokenizer) -> dict[str, str]:
    prompts = ("The purpose of scientific research is to", "Once upon a time, a young explorer discovered", "Artificial intelligence can help people by")
    branch.eval()
    output = {}
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            for _ in range(16):
                token = branch(input_ids=ids, use_cache=False).logits[:, -1].argmax(dim=-1, keepdim=True)
                ids = torch.cat((ids, token), dim=-1)
            output[prompt] = tokenizer.decode(ids[0], skip_special_tokens=True)
    return output


def run_endpoint_condition(name: str, kind: str, config, correct_gains: dict[int, torch.Tensor], teacher: nn.Module, teacher_captures: dict[int, InputCapture], fisher: dict[int, torch.Tensor], train: list[torch.Tensor], dev: list[torch.Tensor]) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(SEED)
    gains = make_condition_gains(correct_gains, kind)
    branch, branch_captures = build_branch(config, gains)
    stages = {}
    for layer in LAYERS:
        for prior in LAYERS:
            if prior < layer:
                branch.model.layers[prior].self_attn.set_alpha(1.0)
        stages[f"layer_{layer}"] = {"updates": LAYER_STEPS[layer], "training_trace": train_layer(branch, layer, train), "endpoint_alpha": float(branch.model.layers[layer].self_attn.alpha)}
    set_all_alpha(branch, 1.0)
    endpoint = development_diagnostics(branch, teacher, teacher_captures, branch_captures, fisher, dev)
    print(f"  {name}: dev CE={endpoint['development_ce']:.4f}; sensitivity-weighted discrepancy={endpoint['sensitivity_weighted_attention_output_discrepancy']:.6f}")
    result = {"basis_type": kind, "gain_summary": {str(layer): {"mean": float(gains[layer].mean()), "sd": float(gains[layer].std()), "min": float(gains[layer].min()), "max": float(gains[layer].max())} for layer in LAYERS}, "stages": stages, "development_endpoint": endpoint, "endpoint_alpha_one": all(float(branch.model.layers[layer].self_attn.alpha) > 0.999 for layer in LAYERS)}
    return branch, result


def main_endpoint() -> None:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"EXPERIMENT 021B seed={SEED}: fixed sensitivity-basis endpoint; final test reserved until all conditions train")
    train_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    validation_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_ID)
    sensitivity_sequences = collect(train_split, tokenizer, SENSITIVITY_SEQUENCES)
    train = collect(train_split, tokenizer, TRAIN_SEQUENCES)
    dev = collect(validation_split, tokenizer, DEV_SEQUENCES, skip=DEV_SKIP)
    config = AutoConfig.from_pretrained(SOURCE_ID)
    teacher = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE).eval()
    teacher_captures = attach_teacher_input_captures(teacher)
    correct_gains, fisher, basis_summary = derive_sensitivity(teacher, sensitivity_sequences, config.hidden_size)
    trained = {name: run_endpoint_condition(name, kind, config, correct_gains, teacher, teacher_captures, fisher, train, dev) for name, kind in CONDITIONS.items()}
    test_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    fresh = collect(test_split, tokenizer, FRESH_TEST_SEQUENCES, skip=FRESH_TEST_SKIP)
    teacher_fresh = fresh_metrics(teacher, fresh)
    results = {}
    for name, (branch, result) in trained.items():
        integrity = alpha_zero_fresh(branch, teacher, fresh)
        results[name] = {**result, "alpha_zero_fresh_test": integrity["alpha_zero_fresh_test"], "alpha_zero_loss_deviation": integrity["absolute_loss_deviation"], "final_fresh_test": fresh_metrics(branch, fresh), "generation": greedy_continuations(branch, tokenizer)}
    unit_loss = results["unit_conditional_ce_transport"]["final_fresh_test"]["loss"]
    correct_loss = results["correct_sensitivity_basis"]["final_fresh_test"]["loss"]
    permuted_loss = results["permuted_sensitivity_basis"]["final_fresh_test"]["loss"]
    output = {"experiment": "experiment_021b_downstream_sensitivity_basis_endpoint", "seed": SEED, "data": {"sensitivity_split": "wikitext-2-raw-v1/train", "sensitivity_sequences": SENSITIVITY_SEQUENCES, "calibration_split": "wikitext-2-raw-v1/train", "calibration_sequences": TRAIN_SEQUENCES, "development_split": "wikitext-2-raw-v1/validation", "development_eligible_offset": DEV_SKIP, "development_sequences": DEV_SEQUENCES, "fresh_final_split": "wikitext-2-raw-v1/test", "fresh_final_eligible_offset": FRESH_TEST_SKIP, "fresh_final_sequences": FRESH_TEST_SEQUENCES, "fresh_test_loaded_after_all_conditions": True}, "budget": {"layer_updates": LAYER_STEPS, "total_updates_per_condition": sum(LAYER_STEPS.values()), "transport_rank": TRANSPORT_RANK, "gain_bounds": [GAIN_MIN, GAIN_MAX], "gain_formula": "renorm_mean_1(clip(sqrt(fisher/mean_fisher), 0.50, 2.00))", "permuted_target_shift_rule": "floor(hidden_size/2)"}, "sensitivity_basis": basis_summary, "teacher_fresh_test": teacher_fresh, **results, "interpretation": {"alpha_zero_exact_all": all(results[name]["alpha_zero_loss_deviation"] <= 1e-5 for name in CONDITIONS), "all_alpha_one_endpoints": all(results[name]["endpoint_alpha_one"] for name in CONDITIONS), "correct_minus_unit_fresh_test_loss": correct_loss - unit_loss, "correct_minus_permuted_fresh_test_loss": correct_loss - permuted_loss}}
    (OUT / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["interpretation"], indent=2))


if __name__ == "__main__":
    main_endpoint()
