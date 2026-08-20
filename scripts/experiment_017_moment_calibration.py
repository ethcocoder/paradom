"""Experiment 017: featurewise Mamba-output moment calibration with causal controls."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba.modeling_mamba import MambaMixer

import experiment_015_adaptive_interface as base

SEED = int(os.environ.get("EXPERIMENT_SEED", "20260851"))
OUT = Path("research/experiment_017_moment_calibration") / f"seed_{SEED}"
TRAIN_SEQUENCES, DEV_SEQUENCES, TEST_SEQUENCES, TEST_SKIP = 1024, 64, 128, 128
CONDITIONS = ("ce_calibrator", "value_functional", "moment_functional")
PROMPTS = ("The purpose of scientific research is to", "Once upon a time, a young explorer discovered", "The capital of France is", "Artificial intelligence can help people by")
EPS = 1e-8


def collect(split, tokenizer, limit: int, skip: int = 0) -> list[torch.Tensor]:
    sequences, eligible = [], 0
    for row in split:
        text = row["text"].strip()
        if len(text) < 80:
            continue
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=base.MAX_LENGTH).input_ids.squeeze(0)
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


def token_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    student = student_logits[:, :-1] / base.TEMPERATURE
    teacher = teacher_logits[:, :-1] / base.TEMPERATURE
    positions = student.shape[0] * student.shape[1]
    return F.kl_div(F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1), reduction="sum") * base.TEMPERATURE**2 / positions


def moment_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_mean, target_mean = pred.mean(dim=(0, 1)), target.mean(dim=(0, 1))
    pred_var, target_var = pred.var(dim=(0, 1), correction=0), target.var(dim=(0, 1), correction=0)
    mean_term = ((pred_mean - target_mean).pow(2) / (target_var + EPS)).mean()
    variance_term = (torch.log(pred_var + EPS) - torch.log(target_var + EPS)).pow(2).mean()
    return mean_term + variance_term, mean_term, variance_term


class CalibratedMamba(nn.Module):
    def __init__(self, raw: MambaMixer, hidden: int):
        super().__init__()
        self.pre = base.LowRankResidual(hidden, base.ADAPTER_RANK)
        self.mixer = raw
        self.post = base.LowRankResidual(hidden, base.ADAPTER_RANK)
        self.gamma = nn.Parameter(torch.ones(hidden))
        self.beta = nn.Parameter(torch.zeros(hidden))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.post(self.mixer(self.pre(x)))
        return output * self.gamma + self.beta


def build_branch(source_config):
    branch = AutoModelForCausalLM.from_pretrained(base.SOURCE_ID, torch_dtype=torch.float32).to(base.DEVICE)
    audit = {}
    for layer in base.LAYERS:
        raw = MambaMixer(base.make_config(source_config, base.LAYER_BUDGETS[layer]["state_size"]), layer).to(base.DEVICE)
        calibrated = CalibratedMamba(raw, source_config.hidden_size).to(base.DEVICE)
        branch.model.layers[layer].self_attn = base.GatedAttentionReplacement(branch.model.layers[layer].self_attn, calibrated).to(base.DEVICE)
        audit[f"layer_{layer}"] = {"mamba_initialization": "fresh_random", "adapter_rank": base.ADAPTER_RANK, "calibrator": "featurewise_gamma_beta", "gamma_initialization": 1.0, "beta_initialization": 0.0}
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
    return branch, audit


def set_trainable(branch, layer: int):
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
    for parameter in branch.model.layers[layer].self_attn.mixer.parameters():
        parameter.requires_grad_(True)


def local_diagnostics(branch, layer: int, seqs: list[torch.Tensor]) -> dict[str, float]:
    values, means, variances = [], [], []
    branch.eval()
    with torch.no_grad():
        for ids in seqs:
            gate, source, target, _ = base.capture(branch, layer, ids[None].to(base.DEVICE))
            pred = gate.mixer(source)
            _, nmse, cosine = base.value_loss(pred, target)
            _, mean_term, variance_term = moment_loss(pred, target)
            values.append((float(nmse), float(cosine))); means.append(float(mean_term)); variances.append(float(variance_term))
    return {"value_nmse": sum(v[0] for v in values) / len(values), "value_cosine": sum(v[1] for v in values) / len(values), "moment_mean": sum(means) / len(means), "moment_log_variance": sum(variances) / len(variances)}


def local_train(branch, condition: str, layer: int, train: list[torch.Tensor]):
    gate = branch.model.layers[layer].self_attn
    set_trainable(branch, layer)
    optimizer = torch.optim.AdamW(gate.mixer.parameters(), lr=base.LOCAL_LR, weight_decay=0.01)
    trace = []
    gate.set_alpha(1.0 if condition == "ce_calibrator" else 0.0)
    for step in range(base.LAYER_BUDGETS[layer]["local_steps"]):
        ids = train[step % len(train)][None].to(base.DEVICE)
        if condition == "ce_calibrator":
            logits = branch(input_ids=ids, use_cache=False).logits
            loss = base.token_ce(logits, ids)
            fields = {"ce": float(loss.detach()), "value": None, "moment": None}
        else:
            gate, source, target, _ = base.capture(branch, layer, ids)
            pred = gate.mixer(source)
            value, _, _ = base.value_loss(pred, target)
            moment, _, _ = moment_loss(pred, target)
            loss = value if condition == "value_functional" else 0.65 * value + 0.35 * moment
            fields = {"ce": None, "value": float(value.detach()), "moment": float(moment.detach())}
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(gate.mixer.parameters(), 1.0); optimizer.step()
        trace.append({"step": step + 1, "objective": float(loss.detach()), **fields})
    return trace


def fixed_gate_train(branch, teacher, condition: str, layer: int, train: list[torch.Tensor]):
    gate = branch.model.layers[layer].self_attn
    set_trainable(branch, layer)
    optimizer = torch.optim.AdamW(gate.mixer.parameters(), lr=base.GATE_LR, weight_decay=0.01)
    trace = []
    steps = base.LAYER_BUDGETS[layer]["gate_steps"]
    for step in range(steps):
        alpha = base.fixed_alpha(step, steps); gate.set_alpha(alpha)
        ids = train[step % len(train)][None].to(base.DEVICE)
        logits = branch(input_ids=ids, use_cache=False).logits
        if condition == "ce_calibrator":
            loss = base.token_ce(logits, ids)
            fields = {"ce": float(loss.detach()), "kl_token": None, "value": None, "moment": None}
        else:
            with torch.no_grad(): teacher_logits = teacher(input_ids=ids, use_cache=False).logits
            gate, source, target, _ = base.capture(branch, layer, ids)
            pred = gate.mixer(source)
            value, _, _ = base.value_loss(pred, target)
            moment, _, _ = moment_loss(pred, target)
            kl = token_kl(logits, teacher_logits); ce = base.token_ce(logits, ids)
            if condition == "value_functional": loss = 0.70 * kl + 0.20 * ce + 0.10 * value
            else: loss = 0.55 * kl + 0.20 * ce + 0.10 * value + 0.15 * moment
            fields = {"ce": float(ce.detach()), "kl_token": float(kl.detach()), "value": float(value.detach()), "moment": float(moment.detach())}
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(gate.mixer.parameters(), 1.0); optimizer.step()
        trace.append({"step": step + 1, "alpha": alpha, "objective": float(loss.detach()), **fields})
    gate.set_alpha(1.0)
    return trace


def token_kl_on_sequences(branch, teacher, seqs: list[torch.Tensor]) -> float:
    values = []
    with torch.no_grad():
        for ids in seqs:
            x = ids[None].to(base.DEVICE)
            values.append(float(token_kl(branch(input_ids=x, use_cache=False).logits, teacher(input_ids=x, use_cache=False).logits)))
    return sum(values) / len(values)


def safety_trace(branch, teacher, layer: int, dev: list[torch.Tensor]):
    gate = branch.model.layers[layer].self_attn
    trace = []
    for alpha in [round(.1 * item, 1) for item in range(1, 11)]:
        gate.set_alpha(alpha)
        development = base.held_out_loss(branch, dev)
        kl = token_kl_on_sequences(branch, teacher, dev)
        trace.append({"alpha": alpha, "development_loss": development["loss"], "token_normalized_kl": kl, "kl_above_0_22_flag": kl > .22})
    gate.set_alpha(1.0)
    return trace


def evaluate_final(branch, teacher, tokenizer, fresh_test: list[torch.Tensor]):
    # This function is called only after all training phases have completed.
    for layer in base.LAYERS: branch.model.layers[layer].self_attn.set_alpha(0.0)
    alpha_zero = base.held_out_loss(branch, fresh_test)
    for layer in base.LAYERS: branch.model.layers[layer].self_attn.set_alpha(1.0)
    final = base.held_out_loss(branch, fresh_test)
    generations = {}
    branch.eval()
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(base.DEVICE)
            for _ in range(16):
                ids = torch.cat((ids, branch(input_ids=ids, use_cache=False).logits[:, -1].argmax(dim=-1, keepdim=True)), dim=-1)
            generations[prompt] = tokenizer.decode(ids[0], skip_special_tokens=True)
    return {"alpha_zero_fresh_test": alpha_zero, "final_fresh_test": final, "generation": generations}


def run_condition(condition, source_config, teacher, train, dev):
    base.set_seed(SEED)
    branch, audit = build_branch(source_config)
    stages = {}
    for layer in base.LAYERS:
        before = local_diagnostics(branch, layer, dev)
        local_trace = local_train(branch, condition, layer, train)
        after = local_diagnostics(branch, layer, dev)
        gate_trace = fixed_gate_train(branch, teacher, condition, layer, train)
        stages[f"layer_{layer}"] = {"diagnostic_before": before, "diagnostic_after": after, "local_trace": local_trace, "gate_trace": gate_trace, "post_gate_development": base.held_out_loss(branch, dev), "post_gate_token_kl": token_kl_on_sequences(branch, teacher, dev), "safety_trace": safety_trace(branch, teacher, layer, dev)}
        print(f"  {condition} L{layer}: value-NMSE {before['value_nmse']:.4f}->{after['value_nmse']:.4f}; log-var {before['moment_log_variance']:.4f}->{after['moment_log_variance']:.4f}")
    result = {"audit": audit, "stages_development": stages, "endpoint_alpha_one": True}
    return branch, result


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 1)); base.set_seed(SEED); OUT.mkdir(parents=True, exist_ok=True)
    print(f"EXPERIMENT 017 seed={SEED}: fresh test reserved until post-training evaluation")
    train_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    validation_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    tokenizer = AutoTokenizer.from_pretrained(base.SOURCE_ID)
    train = collect(train_split, tokenizer, TRAIN_SEQUENCES)
    dev = collect(validation_split, tokenizer, DEV_SEQUENCES)
    config = AutoConfig.from_pretrained(base.SOURCE_ID)
    teacher = AutoModelForCausalLM.from_pretrained(base.SOURCE_ID, torch_dtype=torch.float32).to(base.DEVICE).eval()
    trained = {}
    for condition in CONDITIONS:
        branch, result = run_condition(condition, config, teacher, train, dev)
        trained[condition] = (branch, result)
    # The fresh test partition is requested only after every condition has completed all optimization.
    test_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    fresh_test = collect(test_split, tokenizer, TEST_SEQUENCES, skip=TEST_SKIP)
    teacher_fresh = base.held_out_loss(teacher, fresh_test)
    results = {}
    for condition, (branch, result) in trained.items():
        results[condition] = {**result, **evaluate_final(branch, teacher, tokenizer, fresh_test)}
        del branch
    gc.collect()
    interpretation = {"alpha_zero_exact_all": all(abs(results[c]["alpha_zero_fresh_test"]["loss"] - teacher_fresh["loss"]) < 1e-5 for c in CONDITIONS), "moment_minus_ce_test_loss": results["moment_functional"]["final_fresh_test"]["loss"] - results["ce_calibrator"]["final_fresh_test"]["loss"], "moment_minus_value_test_loss": results["moment_functional"]["final_fresh_test"]["loss"] - results["value_functional"]["final_fresh_test"]["loss"]}
    output = {"experiment": "experiment_017_moment_calibration", "seed": SEED, "source": {"model_id": base.SOURCE_ID, "teacher_frozen": True, "replaced_layers": list(base.LAYERS)}, "data": {"calibration_sequences": TRAIN_SEQUENCES, "development_sequences": DEV_SEQUENCES, "fresh_test_sequences": TEST_SEQUENCES, "fresh_test_eligible_offset": TEST_SKIP}, "teacher_fresh_test": teacher_fresh, **results, "interpretation": interpretation}
    (OUT / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(interpretation, indent=2))


if __name__ == "__main__": main()
