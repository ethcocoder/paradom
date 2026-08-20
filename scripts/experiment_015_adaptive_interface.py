"""Experiment 015: adaptive interface matching at a fixed two-layer Mamba endpoint."""
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba.modeling_mamba import MambaMixer

SOURCE_ID = "HuggingFaceTB/SmolLM-135M"
SEED = int(os.environ.get("EXPERIMENT_SEED", "20260841"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT = Path("research/experiment_015_adaptive_interface") / f"seed_{SEED}"
MAX_LENGTH, TRAIN_SEQUENCES, DEV_SEQUENCES, TEST_SEQUENCES = 48, 1024, 64, 128
LOCAL_LR, GATE_LR, TEMPERATURE, ADAPTER_RANK = 2e-4, 5e-5, 2.0, 8
LAYERS = (0, 1)
LAYER_BUDGETS = {0: {"state_size": 64, "local_steps": 180, "gate_steps": 60}, 1: {"state_size": 96, "local_steps": 360, "gate_steps": 120}}
PROMPTS = ("The purpose of scientific research is to", "Once upon a time, a young explorer discovered", "The capital of France is", "Artificial intelligence can help people by")


def set_seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def collect(tokenizer, split, limit: int) -> list[torch.Tensor]:
    out = []
    for row in split:
        text = row["text"].strip()
        if len(text) < 80: continue
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).input_ids.squeeze(0)
        if ids.numel() >= 16: out.append(ids)
        if len(out) == limit: break
    if len(out) != limit: raise RuntimeError(f"Requested {limit} sequences, found {len(out)}")
    return out


def token_ce(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


def held_out_loss(model, seqs: list[torch.Tensor]) -> dict[str, float]:
    model.eval(); total = 0.0; tokens = 0
    with torch.no_grad():
        for ids in seqs:
            x = ids[None].to(DEVICE); logits = model(input_ids=x, use_cache=False).logits
            total += float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum"))
            tokens += x.shape[1] - 1
    loss = total / tokens
    return {"loss": loss, "perplexity": float(torch.exp(torch.tensor(loss))), "tokens": tokens}


def model_kl(student, teacher, seqs: list[torch.Tensor]) -> float:
    student.eval(); teacher.eval(); values = []
    with torch.no_grad():
        for ids in seqs:
            x = ids[None].to(DEVICE)
            s = student(input_ids=x, use_cache=False).logits[:, :-1]
            t = teacher(input_ids=x, use_cache=False).logits[:, :-1]
            values.append(float(F.kl_div(F.log_softmax(s / TEMPERATURE, dim=-1), F.softmax(t / TEMPERATURE, dim=-1), reduction="batchmean") * TEMPERATURE**2))
    return sum(values) / len(values)


def generate(model, tokenizer) -> dict[str, str]:
    model.eval(); result = {}
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            for _ in range(16):
                nxt = model(input_ids=ids, use_cache=False).logits[:, -1].argmax(dim=-1, keepdim=True)
                ids = torch.cat((ids, nxt), dim=-1)
            result[prompt] = tokenizer.decode(ids[0], skip_special_tokens=True)
    return result


class LowRankResidual(nn.Module):
    def __init__(self, hidden: int, rank: int):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(hidden, rank))
        self.v = nn.Parameter(torch.zeros(rank, hidden))
        nn.init.normal_(self.v, std=1e-3)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (x @ self.v.t()) @ self.u.t()


class AdaptedMamba(nn.Module):
    def __init__(self, mixer: MambaMixer, hidden: int):
        super().__init__(); self.pre = LowRankResidual(hidden, ADAPTER_RANK); self.mixer = mixer; self.post = LowRankResidual(hidden, ADAPTER_RANK)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post(self.mixer(self.pre(x)))


class GatedAttentionReplacement(nn.Module):
    def __init__(self, attention: nn.Module, mixer: AdaptedMamba):
        super().__init__(); self.attention = attention; self.mixer = mixer; self.register_buffer("alpha", torch.tensor(0.0)); self.last_input = None; self.last_attention = None; self.last_kwargs = None
        for parameter in attention.parameters(): parameter.requires_grad_(False)
    def set_alpha(self, alpha: float) -> None: self.alpha.fill_(alpha)
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        attention_output, attention_weights = self.attention(hidden_states, **kwargs)
        self.last_input, self.last_attention, self.last_kwargs = hidden_states, attention_output, kwargs
        mamba_output = self.mixer(hidden_states)
        return attention_output + self.alpha * (mamba_output - attention_output), attention_weights


def make_config(source_config, state_size: int):
    from transformers import MambaConfig
    return MambaConfig(vocab_size=source_config.vocab_size, hidden_size=source_config.hidden_size, num_hidden_layers=2, state_size=state_size, expand=2, conv_kernel=4, use_bias=False, use_conv_bias=True, use_cache=False, use_mambapy=False, use_associative_scan=False)


def build_branch(source_config):
    branch = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE)
    audit = {}
    for layer in LAYERS:
        raw = MambaMixer(make_config(source_config, LAYER_BUDGETS[layer]["state_size"]), layer).to(DEVICE)
        adapted = AdaptedMamba(raw, source_config.hidden_size).to(DEVICE)
        branch.model.layers[layer].self_attn = GatedAttentionReplacement(branch.model.layers[layer].self_attn, adapted).to(DEVICE)
        audit[f"layer_{layer}"] = {"mamba_initialization": "fresh_random", "adapter_rank": ADAPTER_RANK, "adapter_identity_initialized": True}
    for p in branch.parameters(): p.requires_grad_(False)
    return branch, audit


def trainable(branch, layer: int):
    for p in branch.parameters(): p.requires_grad_(False)
    for p in branch.model.layers[layer].self_attn.mixer.parameters(): p.requires_grad_(True)


def capture(branch, layer: int, ids: torch.Tensor):
    with torch.no_grad(): branch(input_ids=ids, use_cache=False)
    gate = branch.model.layers[layer].self_attn
    if gate.last_input is None or gate.last_attention is None or gate.last_kwargs is None: raise RuntimeError("No layer capture")
    return gate, gate.last_input.detach(), gate.last_attention.detach(), gate.last_kwargs


def value_loss(pred, target):
    nmse = F.mse_loss(pred, target) / target.pow(2).mean().clamp(min=1e-8)
    cosine = F.cosine_similarity(pred.reshape(1, -1), target.reshape(1, -1)).mean()
    scale = torch.log(pred.pow(2).mean().sqrt().clamp(min=1e-8) / target.pow(2).mean().sqrt().clamp(min=1e-8)).pow(2)
    return nmse + 0.5 * (1 - cosine) + 0.2 * scale, nmse, cosine


def tangent_loss(gate, source, kwargs, layer: int, step: int):
    generator = torch.Generator(device=source.device); generator.manual_seed(SEED * 1000003 + layer * 1009 + step)
    direction = torch.randn(source.shape, generator=generator, device=source.device, dtype=source.dtype)
    direction = direction / direction.pow(2).mean().sqrt().clamp(min=1e-8)
    epsilon = 0.01 * source.pow(2).mean().sqrt().detach()
    perturbed = source + epsilon * direction
    with torch.no_grad(): target_perturbed = gate.attention(perturbed, **kwargs)[0]
    prediction = gate.mixer(source); prediction_perturbed = gate.mixer(perturbed)
    target_delta, prediction_delta = target_perturbed - gate.last_attention.detach(), prediction_perturbed - prediction
    return F.mse_loss(prediction_delta, target_delta) / target_delta.pow(2).mean().clamp(min=1e-8)


def local_diagnostics(branch, layer: int, seqs: list[torch.Tensor]) -> dict[str, float]:
    branch.eval(); values = []; tangents = []
    with torch.no_grad():
        for index, ids in enumerate(seqs):
            gate, source, target, kwargs = capture(branch, layer, ids[None].to(DEVICE))
            pred = gate.mixer(source); _, nmse, cosine = value_loss(pred, target); values.append((float(nmse), float(cosine)))
            # Diagnostic tangent is recomputed outside an optimizer graph.
            direction = torch.randn_like(source); direction = direction / direction.pow(2).mean().sqrt().clamp(min=1e-8); eps = 0.01 * source.pow(2).mean().sqrt()
            with torch.no_grad(): target_p = gate.attention(source + eps * direction, **kwargs)[0]; pred_p = gate.mixer(source + eps * direction)
            tangents.append(float(F.mse_loss(pred_p - pred, target_p - target) / (target_p - target).pow(2).mean().clamp(min=1e-8)))
    return {"value_nmse": sum(x[0] for x in values) / len(values), "value_cosine": sum(x[1] for x in values) / len(values), "tangent_nmse": sum(tangents) / len(tangents)}


def fixed_alpha(step: int, steps: int) -> float:
    warmup = 10
    return 0.05 if step < warmup else 0.05 + 0.95 * (step - warmup + 1) / (steps - warmup)


def local_train(branch, condition: str, layer: int, seqs: list[torch.Tensor]):
    gate = branch.model.layers[layer].self_attn; steps = LAYER_BUDGETS[layer]["local_steps"]; trainable(branch, layer); optimizer = torch.optim.AdamW(gate.mixer.parameters(), lr=LOCAL_LR, weight_decay=0.01); trace = []
    gate.set_alpha(1.0 if condition == "ce_adapter" else 0.0); gate.mixer.train()
    for step in range(steps):
        ids = seqs[step % len(seqs)][None].to(DEVICE)
        if condition == "ce_adapter":
            logits = branch(input_ids=ids, use_cache=False).logits; loss = token_ce(logits, ids); fields = {"ce": float(loss.detach()), "value": None, "tangent": None}
        else:
            gate, source, target, kwargs = capture(branch, layer, ids); pred = gate.mixer(source); value, nmse, cosine = value_loss(pred, target)
            tangent = tangent_loss(gate, source, kwargs, layer, step) if condition == "adaptive_interface" else None
            loss = 0.7 * value + 0.3 * tangent if tangent is not None else value
            fields = {"ce": None, "value": float(value.detach()), "tangent": None if tangent is None else float(tangent.detach()), "value_nmse": float(nmse.detach()), "value_cosine": float(cosine.detach())}
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(gate.mixer.parameters(), 1.0); optimizer.step()
        trace.append({"step": step + 1, "objective": float(loss.detach()), **fields})
    return trace


def fixed_gate_train(branch, teacher, condition: str, layer: int, seqs: list[torch.Tensor], dev: list[torch.Tensor]):
    gate = branch.model.layers[layer].self_attn; steps = LAYER_BUDGETS[layer]["gate_steps"]; trainable(branch, layer); optimizer = torch.optim.AdamW(gate.mixer.parameters(), lr=GATE_LR, weight_decay=0.01); trace = []
    for step in range(steps):
        alpha = fixed_alpha(step, steps); gate.set_alpha(alpha); ids = seqs[step % len(seqs)][None].to(DEVICE); logits = branch(input_ids=ids, use_cache=False).logits
        if condition == "ce_adapter": loss = token_ce(logits, ids); fields = {"ce": float(loss.detach()), "kl": None, "value": None}
        else:
            with torch.no_grad(): teacher_logits = teacher(input_ids=ids, use_cache=False).logits
            gate, source, target, _ = capture(branch, layer, ids); value, _, _ = value_loss(gate.mixer(source), target)
            kl = F.kl_div(F.log_softmax(logits[:, :-1] / TEMPERATURE, dim=-1), F.softmax(teacher_logits[:, :-1] / TEMPERATURE, dim=-1), reduction="batchmean") * TEMPERATURE**2
            ce = token_ce(logits, ids); loss = 0.70 * kl + 0.20 * ce + 0.10 * value; fields = {"ce": float(ce.detach()), "kl": float(kl.detach()), "value": float(value.detach())}
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(gate.mixer.parameters(), 1.0); optimizer.step(); trace.append({"step": step + 1, "alpha": alpha, "objective": float(loss.detach()), **fields})
    gate.set_alpha(1.0); return trace, {"reached_alpha_one": True, "final_development": held_out_loss(branch, dev)}


def adaptive_gate_train(branch, teacher, layer: int, seqs: list[torch.Tensor], dev: list[torch.Tensor]):
    gate = branch.model.layers[layer].self_attn; steps = LAYER_BUDGETS[layer]["gate_steps"]; trainable(branch, layer); optimizer = torch.optim.AdamW(gate.mixer.parameters(), lr=GATE_LR, weight_decay=0.01); trace = []; controller = []
    targets = [round(0.1 * i, 1) for i in range(1, 11)]; updates_per_target = steps // len(targets); extra = steps % len(targets); accepted = 0.05; last_dev = held_out_loss(branch, dev)["loss"]
    for target_index, target in enumerate(targets):
        candidate = target if target > accepted else accepted; count = updates_per_target + (1 if target_index < extra else 0)
        for inner in range(count):
            step = len(trace); gate.set_alpha(candidate); ids = seqs[step % len(seqs)][None].to(DEVICE)
            with torch.no_grad(): teacher_logits = teacher(input_ids=ids, use_cache=False).logits
            logits = branch(input_ids=ids, use_cache=False).logits; gate, source, attention_target, kwargs = capture(branch, layer, ids)
            value, _, _ = value_loss(gate.mixer(source), attention_target); tangent = tangent_loss(gate, source, kwargs, layer, step); kl = F.kl_div(F.log_softmax(logits[:, :-1] / TEMPERATURE, dim=-1), F.softmax(teacher_logits[:, :-1] / TEMPERATURE, dim=-1), reduction="batchmean") * TEMPERATURE**2; ce = token_ce(logits, ids)
            loss = 0.50 * kl + 0.20 * ce + 0.15 * value + 0.15 * tangent
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(gate.mixer.parameters(), 1.0); optimizer.step(); trace.append({"step": step + 1, "alpha": candidate, "objective": float(loss.detach()), "ce": float(ce.detach()), "kl": float(kl.detach()), "value": float(value.detach()), "tangent": float(tangent.detach())})
        dev_loss = held_out_loss(branch, dev)["loss"]; dev_kl = model_kl(branch, teacher, dev); accept = dev_loss - last_dev <= 0.05 and dev_kl <= 8.0
        if accept: accepted, last_dev = candidate, dev_loss
        else: gate.set_alpha(accepted)
        controller.append({"target": target, "candidate": candidate, "accepted": accept, "accepted_alpha": accepted, "development_loss": dev_loss, "development_kl": dev_kl})
    gate.set_alpha(accepted)
    return trace, {"reached_alpha_one": accepted >= 0.999, "final_alpha": accepted, "controller": controller, "final_development": held_out_loss(branch, dev)}


def run_condition(condition: str, source_config, teacher, tokenizer, train, dev, test):
    set_seed(SEED); branch, audit = build_branch(source_config); alpha_zero = held_out_loss(branch, test); stages = {}
    for layer in LAYERS:
        before = local_diagnostics(branch, layer, dev); local_trace = local_train(branch, condition, layer, train); after = local_diagnostics(branch, layer, dev)
        gate = branch.model.layers[layer].self_attn; gate.set_alpha(1.0); abrupt = held_out_loss(branch, dev)
        if condition == "adaptive_interface": gate_trace, gate_result = adaptive_gate_train(branch, teacher, layer, train, dev)
        else: gate_trace, gate_result = fixed_gate_train(branch, teacher, condition, layer, train, dev)
        stages[f"layer_{layer}"] = {"diagnostic_before": before, "diagnostic_after": after, "abrupt_development": abrupt, "local_trace": local_trace, "gate_trace": gate_trace, "gate_result": gate_result}
        print(f"  {condition} L{layer}: value-NMSE {before['value_nmse']:.4f}->{after['value_nmse']:.4f}; tangent={after['tangent_nmse']:.4f}; alpha={gate_result.get('final_alpha', 1.0):.2f}")
    endpoint = all(stages[f"layer_{layer}"]["gate_result"].get("reached_alpha_one", True) for layer in LAYERS)
    result = {"audit": audit, "alpha_zero_test": alpha_zero, "stages_development": stages, "endpoint_alpha_one": endpoint, "final_test": held_out_loss(branch, test) if endpoint else None, "generation": generate(branch, tokenizer) if endpoint else None}
    del branch; gc.collect(); return result


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 1)); set_seed(SEED); OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"EXPERIMENT 015 seed={SEED}: adapter CE vs value/logit vs adaptive interface")
    corpus = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1"); tokenizer = AutoTokenizer.from_pretrained(SOURCE_ID)
    train, dev, test = collect(tokenizer, corpus["train"], TRAIN_SEQUENCES), collect(tokenizer, corpus["validation"], DEV_SEQUENCES), collect(tokenizer, corpus["test"], TEST_SEQUENCES)
    config = AutoConfig.from_pretrained(SOURCE_ID); teacher = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE).eval(); teacher_test = held_out_loss(teacher, test); print(f"Teacher test={teacher_test['loss']:.4f}, tokens={teacher_test['tokens']}")
    ce = run_condition("ce_adapter", config, teacher, tokenizer, train, dev, test); value = run_condition("value_functional", config, teacher, tokenizer, train, dev, test); adaptive = run_condition("adaptive_interface", config, teacher, tokenizer, train, dev, test)
    result = {"experiment": "experiment_015_adaptive_interface", "source": {"model_id": SOURCE_ID, "replaced_layers": list(LAYERS), "teacher_frozen": True}, "budget": {"seed": SEED, "calibration_sequences": TRAIN_SEQUENCES, "development_sequences": DEV_SEQUENCES, "untouched_test_sequences": TEST_SEQUENCES, "adapter_rank": ADAPTER_RANK, "layer_budgets": LAYER_BUDGETS}, "teacher": teacher_test, "ce_adapter": ce, "value_functional": value, "adaptive_interface": adaptive}
    if adaptive["final_test"] and ce["final_test"] and value["final_test"]:
        result["interpretation"] = {"alpha_zero_exact": all(abs(c["alpha_zero_test"]["loss"] - teacher_test["loss"]) < 1e-5 for c in (ce, value, adaptive)), "adaptive_minus_ce_test_loss": adaptive["final_test"]["loss"] - ce["final_test"]["loss"], "adaptive_minus_value_test_loss": adaptive["final_test"]["loss"] - value["final_test"]["loss"], "adaptive_endpoint_reached": adaptive["endpoint_alpha_one"]}
    else: result["interpretation"] = {"adaptive_endpoint_reached": adaptive["endpoint_alpha_one"], "endpoint_failure_recorded": True}
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["interpretation"], indent=2))


if __name__ == "__main__": main()
