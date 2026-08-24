from __future__ import annotations

import gc
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba.modeling_mamba import MambaMixer

import experiment_015_adaptive_interface as base

SEED = int(os.environ.get("EXPERIMENT_SEED", "20260861"))
OUT = Path("research/experiment_018_residual_interface_transport") / f"seed_{SEED}"
SOURCE_ID = base.SOURCE_ID
DEVICE = base.DEVICE
MAX_LENGTH = 48
TRAIN_SEQUENCES, DEV_SEQUENCES, FRESH_TEST_SEQUENCES, FRESH_TEST_SKIP = 1024, 64, 128, 256
LAYERS = (0, 1)
LAYER_STEPS = {0: 300, 1: 420}
STATE_SIZES = {0: 64, 1: 96}
TRANSPORT_RANK = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
ANCHOR_WEIGHT = 0.05
WARMUP_STEPS = 10
CONDITIONS = ("unconditional_ce_transport", "conditional_ce_transport", "conditional_downstream_anchor")
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


class ResidualInterfaceTransport(nn.Module):
    """A deployed Mamba-output transport map with identity (zero-correction) initialization."""

    def __init__(self, mixer: MambaMixer, hidden_size: int, conditional: bool):
        super().__init__()
        self.mixer = mixer
        self.conditional = conditional
        self.down = nn.Linear(3 * hidden_size, TRANSPORT_RANK, bias=False)
        self.up = nn.Linear(TRANSPORT_RANK, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        self.last_mamba: torch.Tensor | None = None
        self.last_correction: torch.Tensor | None = None
        self.last_transport: torch.Tensor | None = None

    def forward(self, attention_input: torch.Tensor) -> torch.Tensor:
        mamba_output = self.mixer(attention_input)
        normalized_mamba = rms_norm(mamba_output)
        if self.conditional:
            normalized_input = rms_norm(attention_input)
            transport_input = torch.cat((normalized_input, normalized_mamba, normalized_input - normalized_mamba), dim=-1)
        else:
            transport_input = torch.cat((normalized_mamba, normalized_mamba, normalized_mamba), dim=-1)
        correction = self.up(F.silu(self.down(transport_input)))
        transported = mamba_output + correction
        self.last_mamba = mamba_output
        self.last_correction = correction
        self.last_transport = transported
        return transported


class TransportAttentionReplacement(nn.Module):
    """Frozen source attention plus a gated, deployable Mamba transport branch."""

    def __init__(self, attention: nn.Module, transport: ResidualInterfaceTransport):
        super().__init__()
        self.attention = attention
        self.mixer = transport
        self.register_buffer("alpha", torch.tensor(0.0))
        self.last_attention_input: torch.Tensor | None = None
        self.last_attention: torch.Tensor | None = None
        self.last_mixed_output: torch.Tensor | None = None
        self.last_block_input: torch.Tensor | None = None
        for parameter in attention.parameters():
            parameter.requires_grad_(False)

    def set_alpha(self, alpha: float) -> None:
        self.alpha.fill_(float(alpha))

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        attention_output, attention_weights = self.attention(hidden_states, **kwargs)
        transport_output = self.mixer(hidden_states)
        mixed_output = attention_output + self.alpha * (transport_output - attention_output)
        self.last_attention_input = hidden_states
        self.last_attention = attention_output
        self.last_mixed_output = mixed_output
        return mixed_output, attention_weights


def attach_block_input_capture(layer_module: nn.Module, gate: TransportAttentionReplacement) -> None:
    def capture_block_input(_module, args):
        if not args:
            raise RuntimeError("Decoder-layer pre-hook did not receive positional hidden states")
        gate.last_block_input = args[0]

    layer_module.register_forward_pre_hook(capture_block_input)


def build_branch(source_config, conditional: bool):
    branch = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE)
    audit: dict[str, object] = {}
    for layer_index in LAYERS:
        layer_module = branch.model.layers[layer_index]
        raw_mamba = MambaMixer(base.make_config(source_config, STATE_SIZES[layer_index]), layer_index).to(DEVICE)
        transport = ResidualInterfaceTransport(raw_mamba, source_config.hidden_size, conditional=conditional).to(DEVICE)
        gate = TransportAttentionReplacement(layer_module.self_attn, transport).to(DEVICE)
        layer_module.self_attn = gate
        attach_block_input_capture(layer_module, gate)
        audit[f"layer_{layer_index}"] = {
            "mamba_initialization": "fresh_random",
            "transport_rank": TRANSPORT_RANK,
            "conditional": conditional,
            "transport_output_correction_initialization": "exact_zero",
            "attention_frozen": True,
        }
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
    return branch, audit


def set_active_trainable(branch, layer_index: int) -> None:
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
    for parameter in branch.model.layers[layer_index].self_attn.mixer.parameters():
        parameter.requires_grad_(True)


def downstream_anchor(branch, layer_index: int) -> torch.Tensor:
    """Compute an in-deployment frozen-block anchor after the attention-output intervention."""
    layer_module = branch.model.layers[layer_index]
    gate: TransportAttentionReplacement = layer_module.self_attn
    if gate.last_block_input is None or gate.last_attention is None or gate.last_mixed_output is None:
        raise RuntimeError("Downstream anchor requested before capture")

    raw_residual = gate.last_block_input
    hybrid_residual = raw_residual + gate.last_mixed_output
    hybrid_block_output = hybrid_residual + layer_module.mlp(layer_module.post_attention_layernorm(hybrid_residual))

    with torch.no_grad():
        source_residual = raw_residual.detach() + gate.last_attention.detach()
        source_block_output = source_residual + layer_module.mlp(layer_module.post_attention_layernorm(source_residual))
        source_block_output = source_block_output.detach()

    hybrid_normalized = rms_norm(hybrid_block_output)
    source_normalized = rms_norm(source_block_output)
    cosine = 1.0 - F.cosine_similarity(hybrid_normalized, source_normalized, dim=-1).mean()
    mse = F.mse_loss(hybrid_normalized, source_normalized)
    return cosine + 0.25 * mse


def held_out_loss(model, sequences: list[torch.Tensor]) -> dict[str, float]:
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for ids in sequences:
            x = ids[None].to(DEVICE)
            logits = model(input_ids=x, use_cache=False).logits
            total += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum"))
            tokens += x.shape[1] - 1
    loss = total / tokens
    return {"loss": loss, "perplexity": float(torch.exp(torch.tensor(loss))), "tokens": tokens}


def development_diagnostics(branch, teacher, active_layer: int, dev: list[torch.Tensor], include_anchor: bool) -> dict[str, float | None]:
    branch.eval()
    teacher.eval()
    ce_values, kl_values, correction_ratios, anchors = [], [], [], []
    with torch.no_grad():
        for ids in dev:
            x = ids[None].to(DEVICE)
            branch_logits = branch(input_ids=x, use_cache=False).logits
            teacher_logits = teacher(input_ids=x, use_cache=False).logits
            ce_values.append(float(token_ce(branch_logits, x)))
            kl_values.append(float(token_kl(branch_logits, teacher_logits)))
            gate: TransportAttentionReplacement = branch.model.layers[active_layer].self_attn
            if gate.mixer.last_correction is None or gate.mixer.last_mamba is None:
                raise RuntimeError("Transport diagnostics unavailable")
            ratio = gate.mixer.last_correction.pow(2).mean().sqrt() / gate.mixer.last_mamba.pow(2).mean().sqrt().clamp(min=EPS)
            correction_ratios.append(float(ratio))
    if include_anchor:
        # Anchor requires gradients through the hybrid transport only while training. Post-training diagnostic is recomputed below with graph enabled.
        for ids in dev:
            x = ids[None].to(DEVICE)
            with torch.enable_grad():
                branch(input_ids=x, use_cache=False)
                anchors.append(float(downstream_anchor(branch, active_layer).detach()))
    return {
        "development_ce": sum(ce_values) / len(ce_values),
        "development_token_normalized_kl": sum(kl_values) / len(kl_values),
        "transport_correction_rms_ratio": sum(correction_ratios) / len(correction_ratios),
        "downstream_anchor": None if not include_anchor else sum(anchors) / len(anchors),
    }


def train_active_layer(branch, condition: str, layer_index: int, train: list[torch.Tensor]) -> list[dict[str, float | None]]:
    gate: TransportAttentionReplacement = branch.model.layers[layer_index].self_attn
    set_active_trainable(branch, layer_index)
    branch.train()
    optimizer = torch.optim.AdamW(gate.mixer.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    trace: list[dict[str, float | None]] = []
    steps = LAYER_STEPS[layer_index]
    for step in range(steps):
        alpha = alpha_for_step(step, steps)
        gate.set_alpha(alpha)
        ids = train[step % len(train)][None].to(DEVICE)
        logits = branch(input_ids=ids, use_cache=False).logits
        ce = token_ce(logits, ids)
        anchor = downstream_anchor(branch, layer_index) if condition == "conditional_downstream_anchor" else None
        objective = ce if anchor is None else ce + ANCHOR_WEIGHT * anchor
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(gate.mixer.parameters(), 1.0)
        optimizer.step()
        trace.append({
            "step": step + 1,
            "alpha": alpha,
            "objective": float(objective.detach()),
            "ce": float(ce.detach()),
            "anchor": None if anchor is None else float(anchor.detach()),
        })
    gate.set_alpha(1.0)
    return trace


def evaluate_final(branch, tokenizer, fresh_test: list[torch.Tensor]) -> dict[str, object]:
    for layer_index in LAYERS:
        branch.model.layers[layer_index].self_attn.set_alpha(0.0)
    alpha_zero = held_out_loss(branch, fresh_test)
    for layer_index in LAYERS:
        branch.model.layers[layer_index].self_attn.set_alpha(1.0)
    final = held_out_loss(branch, fresh_test)
    generations: dict[str, str] = {}
    branch.eval()
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            for _ in range(16):
                next_id = branch(input_ids=ids, use_cache=False).logits[:, -1].argmax(dim=-1, keepdim=True)
                ids = torch.cat((ids, next_id), dim=-1)
            generations[prompt] = tokenizer.decode(ids[0], skip_special_tokens=True)
    return {"alpha_zero_fresh_test": alpha_zero, "final_fresh_test": final, "generation": generations}


def run_condition(condition: str, source_config, teacher, train: list[torch.Tensor], dev: list[torch.Tensor]):
    set_seed(SEED)
    conditional = condition != "unconditional_ce_transport"
    branch, audit = build_branch(source_config, conditional=conditional)
    stages: dict[str, object] = {}
    for layer_index in LAYERS:
        for previous_layer in LAYERS:
            if previous_layer < layer_index:
                branch.model.layers[previous_layer].self_attn.set_alpha(1.0)
        pre = development_diagnostics(branch, teacher, layer_index, dev, include_anchor=condition == "conditional_downstream_anchor")
        trace = train_active_layer(branch, condition, layer_index, train)
        post = development_diagnostics(branch, teacher, layer_index, dev, include_anchor=condition == "conditional_downstream_anchor")
        stages[f"layer_{layer_index}"] = {
            "updates": LAYER_STEPS[layer_index],
            "development_before": pre,
            "training_trace": trace,
            "development_after": post,
            "endpoint_alpha": float(branch.model.layers[layer_index].self_attn.alpha),
        }
        print(
            f"  {condition} L{layer_index}: CE {pre['development_ce']:.4f}->{post['development_ce']:.4f}; "
            f"KL {pre['development_token_normalized_kl']:.4f}->{post['development_token_normalized_kl']:.4f}; "
            f"correction-ratio={post['transport_correction_rms_ratio']:.4f}"
        )
    return branch, {"audit": audit, "stages_development": stages, "endpoint_alpha_one": all(float(branch.model.layers[i].self_attn.alpha) > 0.999 for i in LAYERS)}


def main() -> None:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"EXPERIMENT 018 seed={SEED}: residual interface transport; fresh test reserved until all conditions train")

    train_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    validation_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_ID)
    train = collect(train_split, tokenizer, TRAIN_SEQUENCES)
    dev = collect(validation_split, tokenizer, DEV_SEQUENCES)
    source_config = AutoConfig.from_pretrained(SOURCE_ID)
    teacher = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE).eval()

    trained: dict[str, tuple[object, dict[str, object]]] = {}
    for condition in CONDITIONS:
        branch, result = run_condition(condition, source_config, teacher, train, dev)
        trained[condition] = (branch, result)

    # The frozen final split is requested only after all three conditions complete every update.
    test_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    fresh_test = collect(test_split, tokenizer, FRESH_TEST_SEQUENCES, skip=FRESH_TEST_SKIP)
    teacher_fresh_test = held_out_loss(teacher, fresh_test)

    results: dict[str, object] = {}
    for condition, (branch, result) in trained.items():
        results[condition] = {**result, **evaluate_final(branch, tokenizer, fresh_test)}
        del branch
    gc.collect()

    alpha_zero_exact = all(
        abs(results[condition]["alpha_zero_fresh_test"]["loss"] - teacher_fresh_test["loss"]) < 1e-5
        for condition in CONDITIONS
    )
    unconditional_loss = results["unconditional_ce_transport"]["final_fresh_test"]["loss"]
    conditional_loss = results["conditional_ce_transport"]["final_fresh_test"]["loss"]
    anchored_loss = results["conditional_downstream_anchor"]["final_fresh_test"]["loss"]
    interpretation = {
        "alpha_zero_exact_all": alpha_zero_exact,
        "all_alpha_one_endpoints": all(results[condition]["endpoint_alpha_one"] for condition in CONDITIONS),
        "conditional_minus_unconditional_fresh_test_loss": conditional_loss - unconditional_loss,
        "anchored_minus_conditional_fresh_test_loss": anchored_loss - conditional_loss,
        "anchored_minus_unconditional_fresh_test_loss": anchored_loss - unconditional_loss,
    }
    output = {
        "experiment": "experiment_018_residual_interface_transport",
        "seed": SEED,
        "source": {"model_id": SOURCE_ID, "teacher_frozen": True, "replaced_layers": list(LAYERS)},
        "data": {
            "calibration_split": "wikitext-2-raw-v1/train",
            "calibration_sequences": TRAIN_SEQUENCES,
            "development_split": "wikitext-2-raw-v1/validation",
            "development_sequences": DEV_SEQUENCES,
            "fresh_final_split": "wikitext-2-raw-v1/test",
            "fresh_final_eligible_offset": FRESH_TEST_SKIP,
            "fresh_final_sequences": FRESH_TEST_SEQUENCES,
            "fresh_test_loaded_after_all_conditions": True,
        },
        "budget": {
            "layer_updates": LAYER_STEPS,
            "total_updates_per_condition": sum(LAYER_STEPS.values()),
            "transport_rank": TRANSPORT_RANK,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "anchor_weight": ANCHOR_WEIGHT,
        },
        "teacher_fresh_test": teacher_fresh_test,
        **results,
        "interpretation": interpretation,
    }
    (OUT / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(interpretation, indent=2))


if __name__ == "__main__":
    main()
