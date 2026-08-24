from __future__ import annotations

import gc
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

SEED = int(os.environ.get("EXPERIMENT_SEED", "20260891"))
OUT = Path("research/experiment_020b_relational_geometry_endpoint") / f"seed_{SEED}"
SOURCE_ID = base.SOURCE_ID
DEVICE = base.DEVICE
MAX_LENGTH = 48
TRAIN_SEQUENCES = 1024
DEV_SEQUENCES, DEV_SKIP = 64, 64
FRESH_TEST_SEQUENCES, FRESH_TEST_SKIP = 128, 512
LAYERS = (0, 1)
LAYER_STEPS = {0: 300, 1: 420}
STATE_SIZES = {0: 64, 1: 96}
TRANSPORT_RANK = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 10
RELATION_TEMPERATURE = 0.20
RELATION_COEFFICIENT = 0.20
CONDITIONS = {
    "conditional_ce_transport": {"coefficient": 0.0, "permuted_target": False},
    "correct_relational_geometry": {"coefficient": RELATION_COEFFICIENT, "permuted_target": False},
    "permuted_relational_control": {"coefficient": RELATION_COEFFICIENT, "permuted_target": True},
}
PROMPTS = (
    "The purpose of scientific research is to",
    "Once upon a time, a young explorer discovered",
    "The capital of France is",
    "Artificial intelligence can help people by",
)
EPS = 1e-8


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect(split, tokenizer, limit: int, skip: int = 0) -> list[torch.Tensor]:
    sequences: list[torch.Tensor] = []
    eligible = 0
    for row in split:
        text = row["text"].strip()
        if len(text) < 80:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).input_ids.squeeze(0)
        if ids.numel() < 16:
            continue
        if eligible >= skip:
            sequences.append(ids)
            if len(sequences) == limit:
                break
        eligible += 1
    if len(sequences) != limit:
        raise RuntimeError(f"Requested {limit} eligible sequences after skip={skip}; found {len(sequences)}")
    return sequences


def token_ce(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


def token_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    student = student_logits[:, :-1] / base.TEMPERATURE
    teacher = teacher_logits[:, :-1] / base.TEMPERATURE
    positions = student.shape[0] * student.shape[1]
    return F.kl_div(F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1), reduction="sum") * base.TEMPERATURE**2 / positions


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)


def alpha_for_step(step: int, steps: int) -> float:
    if step < WARMUP_STEPS:
        return 0.05
    return 0.05 + 0.95 * (step - WARMUP_STEPS + 1) / (steps - WARMUP_STEPS)


def average(values: list[float]) -> float:
    if not values:
        raise RuntimeError("Cannot average an empty metric list")
    return sum(values) / len(values)


def rms_of_grads(parameters) -> float:
    total_sq, count = 0.0, 0
    for parameter in parameters:
        if parameter.grad is not None:
            total_sq += float(parameter.grad.detach().pow(2).sum())
            count += parameter.grad.numel()
    return math.sqrt(total_sq / count) if count else 0.0


def relative_drift(student: torch.Tensor, teacher: torch.Tensor) -> float:
    return float((student - teacher).pow(2).mean().sqrt() / teacher.pow(2).mean().sqrt().clamp(min=EPS))


def effective_rank(x: torch.Tensor) -> float:
    matrix = x.detach().reshape(-1, x.shape[-1]).float()
    values = torch.linalg.svdvals(matrix)
    mass = values / values.sum().clamp(min=EPS)
    return float((-(mass * mass.clamp(min=EPS).log()).sum()).exp())


class InputCapture:
    def __init__(self) -> None:
        self.hidden: torch.Tensor | None = None

    def attach(self, module: nn.Module) -> None:
        def capture(_module, args):
            if not args:
                raise RuntimeError("Decoder-layer pre-hook did not receive hidden states")
            self.hidden = args[0]

        module.register_forward_pre_hook(capture)


class ResidualInterfaceTransport(nn.Module):
    """Conditional rank-16 transport with an exact-zero deployed correction at initialization."""

    def __init__(self, mixer: MambaMixer, hidden_size: int):
        super().__init__()
        self.mixer = mixer
        self.down = nn.Linear(3 * hidden_size, TRANSPORT_RANK, bias=False)
        self.up = nn.Linear(TRANSPORT_RANK, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        self.last_mamba: torch.Tensor | None = None
        self.last_correction: torch.Tensor | None = None

    def forward(self, attention_input: torch.Tensor) -> torch.Tensor:
        mamba_output = self.mixer(attention_input)
        norm_mamba, norm_input = rms_norm(mamba_output), rms_norm(attention_input)
        features = torch.cat((norm_input, norm_mamba, norm_input - norm_mamba), dim=-1)
        correction = self.up(F.silu(self.down(features)))
        self.last_mamba, self.last_correction = mamba_output, correction
        return mamba_output + correction


class TransportAttentionReplacement(nn.Module):
    def __init__(self, attention: nn.Module, transport: ResidualInterfaceTransport):
        super().__init__()
        self.attention = attention
        self.mixer = transport
        self.register_buffer("alpha", torch.tensor(0.0))
        for parameter in attention.parameters():
            parameter.requires_grad_(False)

    def set_alpha(self, alpha: float) -> None:
        self.alpha.fill_(float(alpha))

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        attention_output, attention_weights = self.attention(hidden_states, **kwargs)
        transport_output = self.mixer(hidden_states)
        return attention_output + self.alpha * (transport_output - attention_output), attention_weights


def build_branch(source_config) -> tuple[nn.Module, dict[str, Any], dict[int, InputCapture]]:
    branch = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE)
    audit: dict[str, Any] = {}
    for layer_index in LAYERS:
        layer_module = branch.model.layers[layer_index]
        raw_mamba = MambaMixer(base.make_config(source_config, STATE_SIZES[layer_index]), layer_index).to(DEVICE)
        transport = ResidualInterfaceTransport(raw_mamba, source_config.hidden_size).to(DEVICE)
        layer_module.self_attn = TransportAttentionReplacement(layer_module.self_attn, transport).to(DEVICE)
        audit[f"layer_{layer_index}"] = {
            "mamba_initialization": "fresh_random",
            "transport_rank": TRANSPORT_RANK,
            "conditional": True,
            "transport_output_correction_initialization": "exact_zero",
            "attention_frozen": True,
        }
    captures = {1: InputCapture(), 2: InputCapture()}
    for layer_index, capture in captures.items():
        capture.attach(branch.model.layers[layer_index])
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
    return branch, audit, captures


def attach_teacher_captures(teacher: nn.Module) -> dict[int, InputCapture]:
    captures = {1: InputCapture(), 2: InputCapture()}
    for layer_index, capture in captures.items():
        capture.attach(teacher.model.layers[layer_index])
    return captures


def set_all_alpha(branch: nn.Module, alpha: float) -> None:
    for layer_index in LAYERS:
        branch.model.layers[layer_index].self_attn.set_alpha(alpha)


def set_active_trainable(branch: nn.Module, layer_index: int) -> None:
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
    for parameter in branch.model.layers[layer_index].self_attn.mixer.parameters():
        parameter.requires_grad_(True)


def relation_kl(teacher_hidden: torch.Tensor, student_hidden: torch.Tensor, permuted_target: bool) -> torch.Tensor:
    teacher_features = rms_norm(teacher_hidden[:, :-1]).detach()
    student_features = rms_norm(student_hidden[:, :-1])
    d = teacher_features.shape[-1]
    teacher_similarity = teacher_features @ teacher_features.transpose(-2, -1) / d
    student_similarity = student_features @ student_features.transpose(-2, -1) / d
    target = F.softmax(teacher_similarity / RELATION_TEMPERATURE, dim=-1)
    if permuted_target:
        target = torch.roll(target, shifts=target.shape[-1] // 2, dims=-1)
    return F.kl_div(F.log_softmax(student_similarity / RELATION_TEMPERATURE, dim=-1), target, reduction="batchmean") / target.shape[-2]


def held_out_metrics(model: nn.Module, sequences: list[torch.Tensor]) -> dict[str, float]:
    model.eval()
    loss_sum, entropy_sum, ece_bins, tokens = 0.0, 0.0, [{"count": 0, "confidence": 0.0, "correct": 0.0} for _ in range(20)], 0
    with torch.no_grad():
        for sequence in sequences:
            ids = sequence[None].to(DEVICE)
            logits = model(input_ids=ids, use_cache=False).logits[:, :-1]
            labels = ids[:, 1:]
            flat_logits, flat_labels = logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            loss_sum += float(F.cross_entropy(flat_logits, flat_labels, reduction="sum"))
            probabilities = F.softmax(flat_logits, dim=-1)
            entropy_sum += float((-(probabilities * probabilities.clamp(min=EPS).log()).sum(dim=-1)).sum())
            confidence, prediction = probabilities.max(dim=-1)
            correct = prediction.eq(flat_labels)
            for conf, is_correct in zip(confidence.tolist(), correct.tolist()):
                index = min(19, int(conf * 20))
                ece_bins[index]["count"] += 1
                ece_bins[index]["confidence"] += conf
                ece_bins[index]["correct"] += float(is_correct)
            tokens += flat_labels.numel()
    ece = sum(
        bucket["count"] / tokens * abs(bucket["confidence"] / bucket["count"] - bucket["correct"] / bucket["count"])
        for bucket in ece_bins if bucket["count"]
    )
    loss = loss_sum / tokens
    return {"loss": loss, "perplexity": float(torch.exp(torch.tensor(loss))), "tokens": tokens, "token_entropy": entropy_sum / tokens, "top_token_ece_20bin": ece}


def development_diagnostics(branch: nn.Module, teacher: nn.Module, dev: list[torch.Tensor], teacher_captures: dict[int, InputCapture], branch_captures: dict[int, InputCapture]) -> dict[str, Any]:
    branch.eval()
    teacher.eval()
    ce_values, logit_kl_values, correct_relation_values, permuted_relation_values, l1_drift, l2_drift = [], [], [], [], [], []
    correction_ratios = {layer: [] for layer in LAYERS}
    correction_ranks = {layer: [] for layer in LAYERS}
    with torch.no_grad():
        for sequence in dev:
            ids = sequence[None].to(DEVICE)
            teacher_logits = teacher(input_ids=ids, use_cache=False).logits
            branch_logits = branch(input_ids=ids, use_cache=False).logits
            teacher_l1, teacher_l2 = teacher_captures[1].hidden, teacher_captures[2].hidden
            branch_l1, branch_l2 = branch_captures[1].hidden, branch_captures[2].hidden
            if any(value is None for value in (teacher_l1, teacher_l2, branch_l1, branch_l2)):
                raise RuntimeError("Required decoder-layer captures are unavailable")
            ce_values.append(float(token_ce(branch_logits, ids)))
            logit_kl_values.append(float(token_kl(branch_logits, teacher_logits)))
            correct_relation_values.append(float(relation_kl(teacher_l2, branch_l2, permuted_target=False)))
            permuted_relation_values.append(float(relation_kl(teacher_l2, branch_l2, permuted_target=True)))
            l1_drift.append(relative_drift(branch_l1, teacher_l1))
            l2_drift.append(relative_drift(branch_l2, teacher_l2))
            for layer in LAYERS:
                transport = branch.model.layers[layer].self_attn.mixer
                if transport.last_mamba is None or transport.last_correction is None:
                    raise RuntimeError("Transport diagnostics unavailable")
                ratio = transport.last_correction.pow(2).mean().sqrt() / transport.last_mamba.pow(2).mean().sqrt().clamp(min=EPS)
                correction_ratios[layer].append(float(ratio))
                correction_ranks[layer].append(effective_rank(transport.last_correction))
    return {
        "development_ce": average(ce_values),
        "development_token_normalized_kl": average(logit_kl_values),
        "development_correct_relation_kl": average(correct_relation_values),
        "development_permuted_relation_kl": average(permuted_relation_values),
        "post_layer_1_relative_drift": average(l1_drift),
        "post_layer_2_relative_drift": average(l2_drift),
        "transport_correction_rms_ratio": {str(layer): average(values) for layer, values in correction_ratios.items()},
        "transport_correction_effective_rank": {str(layer): average(values) for layer, values in correction_ranks.items()},
    }


def train_active_layer(branch: nn.Module, teacher: nn.Module, teacher_captures: dict[int, InputCapture], branch_captures: dict[int, InputCapture], layer_index: int, coefficient: float, permuted_target: bool, train: list[torch.Tensor]) -> list[dict[str, float | None]]:
    gate: TransportAttentionReplacement = branch.model.layers[layer_index].self_attn
    set_active_trainable(branch, layer_index)
    branch.train()
    teacher.eval()
    optimizer = torch.optim.AdamW(gate.mixer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    trace: list[dict[str, float | None]] = []
    for step in range(LAYER_STEPS[layer_index]):
        alpha = alpha_for_step(step, LAYER_STEPS[layer_index])
        gate.set_alpha(alpha)
        ids = train[step % len(train)][None].to(DEVICE)
        relation: torch.Tensor | None = None
        if coefficient:
            with torch.no_grad():
                teacher(input_ids=ids, use_cache=False)
        logits = branch(input_ids=ids, use_cache=False).logits
        ce = token_ce(logits, ids)
        if coefficient:
            teacher_hidden, student_hidden = teacher_captures[2].hidden, branch_captures[2].hidden
            if teacher_hidden is None or student_hidden is None:
                raise RuntimeError("Layer-2 input capture unavailable for relational objective")
            relation = relation_kl(teacher_hidden, student_hidden, permuted_target)
        objective = ce if relation is None else ce + coefficient * relation
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        mamba_grad = rms_of_grads(gate.mixer.mixer.parameters())
        transport_grad = rms_of_grads(list(gate.mixer.down.parameters()) + list(gate.mixer.up.parameters()))
        torch.nn.utils.clip_grad_norm_(gate.mixer.parameters(), 1.0)
        optimizer.step()
        trace.append({"step": float(step + 1), "alpha": alpha, "objective": float(objective.detach()), "ce": float(ce.detach()), "relation_kl": None if relation is None else float(relation.detach()), "mamba_gradient_rms": mamba_grad, "transport_gradient_rms": transport_grad})
    gate.set_alpha(1.0)
    return trace


def alpha_zero_integrity(branch: nn.Module, teacher: nn.Module, fresh: list[torch.Tensor]) -> dict[str, Any]:
    set_all_alpha(branch, 0.0)
    alpha_zero = held_out_metrics(branch, fresh)
    teacher_metrics = held_out_metrics(teacher, fresh)
    set_all_alpha(branch, 1.0)
    return {"teacher_fresh_test": teacher_metrics, "alpha_zero_fresh_test": alpha_zero, "absolute_loss_deviation": abs(alpha_zero["loss"] - teacher_metrics["loss"])}


def generations(branch: nn.Module, tokenizer) -> dict[str, str]:
    branch.eval()
    outputs = {}
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            for _ in range(16):
                token = branch(input_ids=ids, use_cache=False).logits[:, -1].argmax(dim=-1, keepdim=True)
                ids = torch.cat((ids, token), dim=-1)
            outputs[prompt] = tokenizer.decode(ids[0], skip_special_tokens=True)
    return outputs


def run_condition(condition: str, specification: dict[str, Any], source_config, teacher: nn.Module, teacher_captures: dict[int, InputCapture], train: list[torch.Tensor], dev: list[torch.Tensor]) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(SEED)
    branch, audit, branch_captures = build_branch(source_config)
    stages: dict[str, Any] = {}
    for layer in LAYERS:
        for previous in LAYERS:
            if previous < layer:
                branch.model.layers[previous].self_attn.set_alpha(1.0)
        trace = train_active_layer(branch, teacher, teacher_captures, branch_captures, layer, specification["coefficient"], specification["permuted_target"], train)
        stages[f"layer_{layer}"] = {"updates": LAYER_STEPS[layer], "training_trace": trace, "endpoint_alpha": float(branch.model.layers[layer].self_attn.alpha)}
    set_all_alpha(branch, 1.0)
    diagnostics = development_diagnostics(branch, teacher, dev, teacher_captures, branch_captures)
    print(f"  {condition}: dev CE={diagnostics['development_ce']:.4f}; correct-relation={diagnostics['development_correct_relation_kl']:.6f}; permuted-relation={diagnostics['development_permuted_relation_kl']:.6f}")
    return branch, {"audit": audit, "relation_coefficient": specification["coefficient"], "permuted_target": specification["permuted_target"], "stages": stages, "development_endpoint": diagnostics, "endpoint_alpha_one": all(float(branch.model.layers[i].self_attn.alpha) > 0.999 for i in LAYERS)}


def main() -> None:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"EXPERIMENT 020B seed={SEED}: relational geometry endpoint; final test reserved until all conditions train")
    train_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    validation_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_ID)
    train = collect(train_split, tokenizer, TRAIN_SEQUENCES)
    dev = collect(validation_split, tokenizer, DEV_SEQUENCES, skip=DEV_SKIP)
    source_config = AutoConfig.from_pretrained(SOURCE_ID)
    teacher = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE).eval()
    teacher_captures = attach_teacher_captures(teacher)

    trained: dict[str, tuple[nn.Module, dict[str, Any]]] = {}
    for condition, specification in CONDITIONS.items():
        trained[condition] = run_condition(condition, specification, source_config, teacher, teacher_captures, train, dev)

    test_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    fresh = collect(test_split, tokenizer, FRESH_TEST_SEQUENCES, skip=FRESH_TEST_SKIP)
    teacher_fresh = held_out_metrics(teacher, fresh)
    results: dict[str, Any] = {}
    for condition, (branch, result) in trained.items():
        integrity = alpha_zero_integrity(branch, teacher, fresh)
        final = held_out_metrics(branch, fresh)
        results[condition] = {**result, "alpha_zero_fresh_test": integrity["alpha_zero_fresh_test"], "final_fresh_test": final, "alpha_zero_loss_deviation": integrity["absolute_loss_deviation"], "generation": generations(branch, tokenizer)}
        del branch
    gc.collect()

    ce_loss = results["conditional_ce_transport"]["final_fresh_test"]["loss"]
    correct_loss = results["correct_relational_geometry"]["final_fresh_test"]["loss"]
    permuted_loss = results["permuted_relational_control"]["final_fresh_test"]["loss"]
    output = {
        "experiment": "experiment_020b_relational_geometry_endpoint",
        "seed": SEED,
        "source": {"model_id": SOURCE_ID, "teacher_frozen": True, "replaced_layers": list(LAYERS)},
        "data": {"calibration_split": "wikitext-2-raw-v1/train", "calibration_sequences": TRAIN_SEQUENCES, "development_split": "wikitext-2-raw-v1/validation", "development_eligible_offset": DEV_SKIP, "development_sequences": DEV_SEQUENCES, "fresh_final_split": "wikitext-2-raw-v1/test", "fresh_final_eligible_offset": FRESH_TEST_SKIP, "fresh_final_sequences": FRESH_TEST_SEQUENCES, "fresh_test_loaded_after_all_conditions": True},
        "budget": {"layer_updates": LAYER_STEPS, "total_updates_per_condition": sum(LAYER_STEPS.values()), "transport_rank": TRANSPORT_RANK, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "relation_temperature": RELATION_TEMPERATURE, "relation_coefficient": RELATION_COEFFICIENT, "permuted_target_shift_rule": "floor(nonterminal_positions/2)"},
        "teacher_fresh_test": teacher_fresh,
        **results,
        "interpretation": {"alpha_zero_exact_all": all(results[c]["alpha_zero_loss_deviation"] <= 1e-5 for c in CONDITIONS), "all_alpha_one_endpoints": all(results[c]["endpoint_alpha_one"] for c in CONDITIONS), "correct_minus_ce_fresh_test_loss": correct_loss - ce_loss, "correct_minus_permuted_fresh_test_loss": correct_loss - permuted_loss},
    }
    (OUT / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["interpretation"], indent=2))


if __name__ == "__main__":
    main()
