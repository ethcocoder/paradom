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

SEED = int(os.environ.get("EXPERIMENT_SEED", "20260881"))
OUT = Path("research/experiment_020a_relational_geometry_pilot") / f"seed_{SEED}"
SOURCE_ID = base.SOURCE_ID
DEVICE = base.DEVICE
MAX_LENGTH = 48
TRAIN_SEQUENCES = 1024
DEV_SEQUENCES, DEV_SKIP = 64, 64
LAYERS = (0, 1)
LAYER_STEPS = {0: 300, 1: 420}
STATE_SIZES = {0: 64, 1: 96}
TRANSPORT_RANK = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 10
RELATION_TEMPERATURE = 0.20
CONDITIONS: dict[str, float] = {
    "conditional_ce_transport": 0.00,
    "relational_geometry_lambda_005": 0.05,
    "relational_geometry_lambda_010": 0.10,
    "relational_geometry_lambda_020": 0.20,
}
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
    numerator = (student - teacher).pow(2).mean().sqrt()
    denominator = teacher.pow(2).mean().sqrt().clamp(min=EPS)
    return float(numerator / denominator)


def effective_rank(x: torch.Tensor) -> float:
    matrix = x.detach().reshape(-1, x.shape[-1]).float()
    values = torch.linalg.svdvals(matrix)
    mass = values / values.sum().clamp(min=EPS)
    entropy = -(mass * mass.clamp(min=EPS).log()).sum()
    return float(entropy.exp())


class InputCapture:
    """Stores a module's latest positional hidden-state input."""

    def __init__(self) -> None:
        self.hidden: torch.Tensor | None = None

    def attach(self, module: nn.Module) -> None:
        def capture(_module, args):
            if not args:
                raise RuntimeError("Decoder-layer pre-hook did not receive hidden states")
            self.hidden = args[0]

        module.register_forward_pre_hook(capture)


class ResidualInterfaceTransport(nn.Module):
    """Experiment-018 conditional transport with exact-zero correction initialization."""

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
        normalized_mamba = rms_norm(mamba_output)
        normalized_input = rms_norm(attention_input)
        transport_input = torch.cat((normalized_input, normalized_mamba, normalized_input - normalized_mamba), dim=-1)
        correction = self.up(F.silu(self.down(transport_input)))
        self.last_mamba = mamba_output
        self.last_correction = correction
        return mamba_output + correction


class TransportAttentionReplacement(nn.Module):
    """Frozen source attention combined with a fully deployable Mamba transport branch."""

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


def token_relation_kl(teacher_hidden: torch.Tensor, student_hidden: torch.Tensor) -> torch.Tensor:
    """KL between row-wise position relation distributions at a common post-deployment location."""
    teacher_features = rms_norm(teacher_hidden[:, :-1]).detach()
    student_features = rms_norm(student_hidden[:, :-1])
    d = teacher_features.shape[-1]
    teacher_similarity = torch.matmul(teacher_features, teacher_features.transpose(-2, -1)) / d
    student_similarity = torch.matmul(student_features, student_features.transpose(-2, -1)) / d
    target = F.softmax(teacher_similarity / RELATION_TEMPERATURE, dim=-1)
    return F.kl_div(F.log_softmax(student_similarity / RELATION_TEMPERATURE, dim=-1), target, reduction="batchmean") / target.shape[-2]


def forward_teacher_and_branch(
    branch: nn.Module,
    teacher: nn.Module,
    ids: torch.Tensor,
    teacher_captures: dict[int, InputCapture],
    branch_captures: dict[int, InputCapture],
    require_branch_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        teacher_logits = teacher(input_ids=ids, use_cache=False).logits
    if require_branch_grad:
        branch_logits = branch(input_ids=ids, use_cache=False).logits
    else:
        with torch.no_grad():
            branch_logits = branch(input_ids=ids, use_cache=False).logits
    teacher_hidden = teacher_captures[2].hidden
    student_hidden = branch_captures[2].hidden
    if teacher_hidden is None or student_hidden is None:
        raise RuntimeError("Layer-2 input capture unavailable for relational loss")
    return teacher_logits, branch_logits, token_relation_kl(teacher_hidden, student_hidden)


def development_diagnostics(
    branch: nn.Module,
    teacher: nn.Module,
    dev: list[torch.Tensor],
    teacher_captures: dict[int, InputCapture],
    branch_captures: dict[int, InputCapture],
) -> dict[str, Any]:
    branch.eval()
    teacher.eval()
    ce_values, kl_values, relation_values = [], [], []
    l1_drifts, l2_drifts = [], []
    correction_ratios = {layer_index: [] for layer_index in LAYERS}
    correction_effective_ranks = {layer_index: [] for layer_index in LAYERS}
    with torch.no_grad():
        for sequence in dev:
            ids = sequence[None].to(DEVICE)
            teacher_logits = teacher(input_ids=ids, use_cache=False).logits
            branch_logits = branch(input_ids=ids, use_cache=False).logits
            teacher_l1, student_l1 = teacher_captures[1].hidden, branch_captures[1].hidden
            teacher_l2, student_l2 = teacher_captures[2].hidden, branch_captures[2].hidden
            if any(item is None for item in (teacher_l1, student_l1, teacher_l2, student_l2)):
                raise RuntimeError("Layer-input captures unavailable during development diagnostics")
            ce_values.append(float(token_ce(branch_logits, ids)))
            kl_values.append(float(token_kl(branch_logits, teacher_logits)))
            relation_values.append(float(token_relation_kl(teacher_l2, student_l2)))
            l1_drifts.append(relative_drift(student_l1, teacher_l1))
            l2_drifts.append(relative_drift(student_l2, teacher_l2))
            for layer_index in LAYERS:
                transport = branch.model.layers[layer_index].self_attn.mixer
                if transport.last_mamba is None or transport.last_correction is None:
                    raise RuntimeError("Transport diagnostics unavailable")
                ratio = transport.last_correction.pow(2).mean().sqrt() / transport.last_mamba.pow(2).mean().sqrt().clamp(min=EPS)
                correction_ratios[layer_index].append(float(ratio))
                correction_effective_ranks[layer_index].append(effective_rank(transport.last_correction))
    return {
        "development_ce": average(ce_values),
        "development_token_normalized_kl": average(kl_values),
        "post_layer_1_relative_drift": average(l1_drifts),
        "post_layer_2_relative_drift": average(l2_drifts),
        "post_block_relation_kl": average(relation_values),
        "transport_correction_rms_ratio": {str(layer): average(values) for layer, values in correction_ratios.items()},
        "transport_correction_effective_rank": {str(layer): average(values) for layer, values in correction_effective_ranks.items()},
    }


def validation_loss(branch: nn.Module, sequences: list[torch.Tensor]) -> dict[str, float]:
    branch.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for sequence in sequences:
            ids = sequence[None].to(DEVICE)
            logits = branch(input_ids=ids, use_cache=False).logits
            total += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum"))
            tokens += ids.shape[1] - 1
    loss = total / tokens
    return {"loss": loss, "perplexity": float(torch.exp(torch.tensor(loss))), "tokens": tokens}


def train_active_layer(
    branch: nn.Module,
    teacher: nn.Module,
    teacher_captures: dict[int, InputCapture],
    branch_captures: dict[int, InputCapture],
    layer_index: int,
    coefficient: float,
    train: list[torch.Tensor],
) -> list[dict[str, float]]:
    gate: TransportAttentionReplacement = branch.model.layers[layer_index].self_attn
    set_active_trainable(branch, layer_index)
    branch.train()
    teacher.eval()
    optimizer = torch.optim.AdamW(gate.mixer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    trace: list[dict[str, float]] = []
    steps = LAYER_STEPS[layer_index]
    for step in range(steps):
        alpha = alpha_for_step(step, steps)
        gate.set_alpha(alpha)
        ids = train[step % len(train)][None].to(DEVICE)
        with torch.no_grad():
            teacher(input_ids=ids, use_cache=False)
        logits = branch(input_ids=ids, use_cache=False).logits
        teacher_hidden, student_hidden = teacher_captures[2].hidden, branch_captures[2].hidden
        if teacher_hidden is None or student_hidden is None:
            raise RuntimeError("Layer-2 input capture unavailable for relational optimization")
        ce = token_ce(logits, ids)
        relation = token_relation_kl(teacher_hidden, student_hidden)
        objective = ce + coefficient * relation
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        raw_mamba_gradient_rms = rms_of_grads(gate.mixer.mixer.parameters())
        transport_gradient_rms = rms_of_grads(list(gate.mixer.down.parameters()) + list(gate.mixer.up.parameters()))
        torch.nn.utils.clip_grad_norm_(gate.mixer.parameters(), 1.0)
        optimizer.step()
        trace.append({
            "step": float(step + 1),
            "alpha": alpha,
            "objective": float(objective.detach()),
            "ce": float(ce.detach()),
            "relation_kl": float(relation.detach()),
            "mamba_gradient_rms": raw_mamba_gradient_rms,
            "transport_gradient_rms": transport_gradient_rms,
        })
    gate.set_alpha(1.0)
    return trace


def alpha_zero_integrity(branch: nn.Module, teacher: nn.Module, dev: list[torch.Tensor]) -> dict[str, Any]:
    set_all_alpha(branch, 0.0)
    alpha_zero = validation_loss(branch, dev)
    teacher_loss = validation_loss(teacher, dev)
    set_all_alpha(branch, 1.0)
    return {
        "teacher_validation_loss": teacher_loss,
        "alpha_zero_validation_loss": alpha_zero,
        "absolute_loss_deviation": abs(alpha_zero["loss"] - teacher_loss["loss"]),
    }


def run_condition(
    condition: str,
    coefficient: float,
    source_config,
    teacher: nn.Module,
    teacher_captures: dict[int, InputCapture],
    train: list[torch.Tensor],
    dev: list[torch.Tensor],
) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(SEED)
    branch, audit, branch_captures = build_branch(source_config)
    stages: dict[str, Any] = {}
    for layer_index in LAYERS:
        for previous_layer in LAYERS:
            if previous_layer < layer_index:
                branch.model.layers[previous_layer].self_attn.set_alpha(1.0)
        trace = train_active_layer(branch, teacher, teacher_captures, branch_captures, layer_index, coefficient, train)
        stages[f"layer_{layer_index}"] = {
            "updates": LAYER_STEPS[layer_index],
            "training_trace": trace,
            "endpoint_alpha": float(branch.model.layers[layer_index].self_attn.alpha),
        }
    set_all_alpha(branch, 1.0)
    diagnostics = development_diagnostics(branch, teacher, dev, teacher_captures, branch_captures)
    integrity = alpha_zero_integrity(branch, teacher, dev)
    print(
        f"  {condition}: dev CE={diagnostics['development_ce']:.4f}; "
        f"relation={diagnostics['post_block_relation_kl']:.6f}; "
        f"drift L1/L2={diagnostics['post_layer_1_relative_drift']:.4f}/{diagnostics['post_layer_2_relative_drift']:.4f}; "
        f"alpha0 dev deviation={integrity['absolute_loss_deviation']:.2e}"
    )
    return branch, {
        "audit": audit,
        "relation_coefficient": coefficient,
        "stages": stages,
        "development_endpoint": diagnostics,
        "alpha_zero_integrity": integrity,
        "endpoint_alpha_one": all(float(branch.model.layers[i].self_attn.alpha) > 0.999 for i in LAYERS),
    }


def main() -> None:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"EXPERIMENT 020A seed={SEED}: development-only relational geometry coefficient pilot; test split prohibited")

    train_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    validation_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_ID)
    train = collect(train_split, tokenizer, TRAIN_SEQUENCES)
    dev = collect(validation_split, tokenizer, DEV_SEQUENCES, skip=DEV_SKIP)
    source_config = AutoConfig.from_pretrained(SOURCE_ID)
    teacher = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE).eval()
    teacher_captures = attach_teacher_captures(teacher)

    results: dict[str, Any] = {}
    for condition, coefficient in CONDITIONS.items():
        branch, result = run_condition(condition, coefficient, source_config, teacher, teacher_captures, train, dev)
        results[condition] = result
        del branch
        gc.collect()

    ce = results["conditional_ce_transport"]["development_endpoint"]
    interpretation: dict[str, Any] = {}
    for condition, coefficient in CONDITIONS.items():
        endpoint = results[condition]["development_endpoint"]
        integrity = results[condition]["alpha_zero_integrity"]
        interpretation[condition] = {
            "relation_coefficient": coefficient,
            "development_ce_minus_ce_control": endpoint["development_ce"] - ce["development_ce"],
            "relation_kl_minus_ce_control": endpoint["post_block_relation_kl"] - ce["post_block_relation_kl"],
            "all_endpoint_metrics_finite": all(math.isfinite(float(value)) for key, value in endpoint.items() if isinstance(value, float)),
            "endpoint_alpha_one": results[condition]["endpoint_alpha_one"],
            "alpha_zero_deviation_at_most_1e_5": integrity["absolute_loss_deviation"] <= 1e-5,
        }
    output = {
        "experiment": "experiment_020a_relational_geometry_pilot",
        "seed": SEED,
        "status": "development_only_no_test_access",
        "source": {"model_id": SOURCE_ID, "teacher_frozen": True, "replaced_layers": list(LAYERS)},
        "data": {
            "calibration_split": "wikitext-2-raw-v1/train",
            "calibration_sequences": TRAIN_SEQUENCES,
            "development_split": "wikitext-2-raw-v1/validation",
            "development_eligible_offset": DEV_SKIP,
            "development_sequences": DEV_SEQUENCES,
            "test_split_requested": False,
        },
        "budget": {
            "layer_updates": LAYER_STEPS,
            "total_updates_per_condition": sum(LAYER_STEPS.values()),
            "transport_rank": TRANSPORT_RANK,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "relation_temperature": RELATION_TEMPERATURE,
            "coefficients": CONDITIONS,
        },
        **results,
        "interpretation": interpretation,
    }
    (OUT / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(interpretation, indent=2))


if __name__ == "__main__":
    main()
