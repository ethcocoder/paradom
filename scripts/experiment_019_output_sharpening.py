from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import experiment_018_residual_interface_transport as transport

SEED = int(os.environ.get("EXPERIMENT_SEED", "20260871"))
OUT = Path("research/experiment_019_output_sharpening") / f"seed_{SEED}"
SOURCE_ID = transport.SOURCE_ID
DEVICE = transport.DEVICE
TRAIN_SEQUENCES, DEV_SEQUENCES, DEV_SKIP = 1024, 64, 64
FRESH_TEST_SEQUENCES, FRESH_TEST_SKIP = 128, 384
PRE_STAGE_UPDATES = {0: 300, 1: 420}
OUTPUT_STAGE_UPDATES = 180
OUTPUT_STAGE_LR = 5e-5
WEIGHT_DECAY = 0.01
ENTROPY_WEIGHT = 0.05
TEACHER_KL_WEIGHT = 0.20
TEACHER_TEMPERATURE = 0.7
CONDITIONS = ("ce_continuation", "self_entropy_sharpening", "teacher_temperature_sharpening")
PROMPTS = transport.PROMPTS
EPS = 1e-8


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    values = logits[:, :-1]
    probabilities = F.softmax(values, dim=-1)
    log_probabilities = F.log_softmax(values, dim=-1)
    return -(probabilities * log_probabilities).sum(dim=-1).mean()


def teacher_temperature_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    student = student_logits[:, :-1]
    teacher = teacher_logits[:, :-1] / TEACHER_TEMPERATURE
    positions = student.shape[0] * student.shape[1]
    return F.kl_div(F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1), reduction="sum") / positions


def set_all_transport_trainable(branch) -> None:
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
    for layer_index in transport.LAYERS:
        for parameter in branch.model.layers[layer_index].self_attn.mixer.parameters():
            parameter.requires_grad_(True)


def quality_metrics(model, sequences: list[torch.Tensor]) -> dict[str, float]:
    """Token-weighted NLL plus non-optimized sharpness and calibration diagnostics."""
    model.eval()
    nll_sum, entropy_sum, tokens = 0.0, 0.0, 0
    confidences, correctness = [], []
    with torch.no_grad():
        for ids in sequences:
            x = ids[None].to(DEVICE)
            logits = model(input_ids=x, use_cache=False).logits[:, :-1]
            targets = x[:, 1:]
            nll_sum += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"))
            probabilities = F.softmax(logits, dim=-1)
            log_probabilities = F.log_softmax(logits, dim=-1)
            entropy_sum += float((-(probabilities * log_probabilities).sum(dim=-1)).sum())
            top_confidence, top_index = probabilities.max(dim=-1)
            confidences.extend(top_confidence.reshape(-1).cpu().tolist())
            correctness.extend((top_index == targets).reshape(-1).float().cpu().tolist())
            tokens += targets.numel()
    loss = nll_sum / tokens
    bins = 20
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, confidence in enumerate(confidences) if low <= confidence < high or (index == bins - 1 and confidence == high)]
        if members:
            average_confidence = sum(confidences[i] for i in members) / len(members)
            average_accuracy = sum(correctness[i] for i in members) / len(members)
            ece += (len(members) / len(confidences)) * abs(average_confidence - average_accuracy)
    return {
        "loss": loss,
        "perplexity": float(math.exp(loss)),
        "tokens": tokens,
        "token_entropy": entropy_sum / tokens,
        "top_token_ece_20bin": ece,
    }


def development_diagnostics(branch, teacher, dev: list[torch.Tensor]) -> dict[str, float]:
    branch.eval()
    teacher.eval()
    ce_values, entropy_values, ece_values, kl_values = [], [], [], []
    with torch.no_grad():
        for ids in dev:
            x = ids[None].to(DEVICE)
            student_logits = branch(input_ids=x, use_cache=False).logits
            teacher_logits = teacher(input_ids=x, use_cache=False).logits
            ce_values.append(float(transport.token_ce(student_logits, x)))
            entropy_values.append(float(token_entropy(student_logits)))
            kl_values.append(float(transport.token_kl(student_logits, teacher_logits)))
    aggregate_quality = quality_metrics(branch, dev)
    return {
        "development_ce": sum(ce_values) / len(ce_values),
        "development_token_entropy": sum(entropy_values) / len(entropy_values),
        "development_top_token_ece_20bin": aggregate_quality["top_token_ece_20bin"],
        "development_teacher_kl": sum(kl_values) / len(kl_values),
    }


def run_pre_stage(branch, train: list[torch.Tensor]) -> dict[str, object]:
    """The identical conditional-CE two-layer deployment pre-stage for every condition."""
    stages: dict[str, object] = {}
    for layer_index in transport.LAYERS:
        for prior_layer in transport.LAYERS:
            if prior_layer < layer_index:
                branch.model.layers[prior_layer].self_attn.set_alpha(1.0)
        trace = transport.train_active_layer(branch, "conditional_ce_transport", layer_index, train)
        stages[f"layer_{layer_index}"] = {
            "updates": PRE_STAGE_UPDATES[layer_index],
            "training_trace": trace,
            "endpoint_alpha": float(branch.model.layers[layer_index].self_attn.alpha),
        }
    return stages


def run_output_stage(branch, teacher, condition: str, train: list[torch.Tensor]) -> list[dict[str, float | None]]:
    for layer_index in transport.LAYERS:
        branch.model.layers[layer_index].self_attn.set_alpha(1.0)
    set_all_transport_trainable(branch)
    branch.train()
    parameters = [parameter for parameter in branch.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=OUTPUT_STAGE_LR, weight_decay=WEIGHT_DECAY)
    trace: list[dict[str, float | None]] = []
    for step in range(OUTPUT_STAGE_UPDATES):
        ids = train[step % len(train)][None].to(DEVICE)
        student_logits = branch(input_ids=ids, use_cache=False).logits
        ce = transport.token_ce(student_logits, ids)
        entropy = token_entropy(student_logits)
        if condition == "ce_continuation":
            teacher_kl = None
            objective = ce
        elif condition == "self_entropy_sharpening":
            teacher_kl = None
            objective = ce + ENTROPY_WEIGHT * entropy
        elif condition == "teacher_temperature_sharpening":
            with torch.no_grad():
                teacher_logits = teacher(input_ids=ids, use_cache=False).logits
            teacher_kl = teacher_temperature_kl(student_logits, teacher_logits)
            objective = ce + TEACHER_KL_WEIGHT * teacher_kl
        else:
            raise ValueError(f"Unknown condition: {condition}")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        trace.append({
            "step": step + 1,
            "objective": float(objective.detach()),
            "ce": float(ce.detach()),
            "entropy": float(entropy.detach()),
            "teacher_temperature_kl": None if teacher_kl is None else float(teacher_kl.detach()),
        })
    return trace


def generate(branch, tokenizer) -> dict[str, str]:
    generated: dict[str, str] = {}
    branch.eval()
    with torch.no_grad():
        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            for _ in range(16):
                next_id = branch(input_ids=ids, use_cache=False).logits[:, -1].argmax(dim=-1, keepdim=True)
                ids = torch.cat((ids, next_id), dim=-1)
            generated[prompt] = tokenizer.decode(ids[0], skip_special_tokens=True)
    return generated


def evaluate_final(branch, tokenizer, fresh_test: list[torch.Tensor]) -> dict[str, object]:
    for layer_index in transport.LAYERS:
        branch.model.layers[layer_index].self_attn.set_alpha(0.0)
    alpha_zero = quality_metrics(branch, fresh_test)
    for layer_index in transport.LAYERS:
        branch.model.layers[layer_index].self_attn.set_alpha(1.0)
    final = quality_metrics(branch, fresh_test)
    return {"alpha_zero_fresh_test": alpha_zero, "final_fresh_test": final, "generation": generate(branch, tokenizer)}


def run_condition(condition: str, source_config, teacher, train: list[torch.Tensor], dev: list[torch.Tensor]):
    transport.set_seed(SEED)
    branch, audit = transport.build_branch(source_config, conditional=True)
    pre_stage = run_pre_stage(branch, train)
    before_output_stage = development_diagnostics(branch, teacher, dev)
    output_trace = run_output_stage(branch, teacher, condition, train)
    after_output_stage = development_diagnostics(branch, teacher, dev)
    endpoint_alpha_one = all(float(branch.model.layers[layer_index].self_attn.alpha) > 0.999 for layer_index in transport.LAYERS)
    print(
        f"  {condition}: dev CE {before_output_stage['development_ce']:.4f}->{after_output_stage['development_ce']:.4f}; "
        f"entropy {before_output_stage['development_token_entropy']:.4f}->{after_output_stage['development_token_entropy']:.4f}; "
        f"ECE {before_output_stage['development_top_token_ece_20bin']:.4f}->{after_output_stage['development_top_token_ece_20bin']:.4f}"
    )
    return branch, {
        "audit": audit,
        "pre_stage": pre_stage,
        "development_before_output_stage": before_output_stage,
        "output_stage": {"updates": OUTPUT_STAGE_UPDATES, "training_trace": output_trace, "development_after": after_output_stage},
        "endpoint_alpha_one": endpoint_alpha_one,
    }


def main() -> None:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    transport.set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"EXPERIMENT 019 seed={SEED}: output sharpening; fresh final slice reserved until all conditions finish")

    train_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    validation_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_ID)
    train = transport.collect(train_split, tokenizer, TRAIN_SEQUENCES)
    dev = transport.collect(validation_split, tokenizer, DEV_SEQUENCES, skip=DEV_SKIP)
    source_config = AutoConfig.from_pretrained(SOURCE_ID)
    teacher = AutoModelForCausalLM.from_pretrained(SOURCE_ID, torch_dtype=torch.float32).to(DEVICE).eval()

    trained: dict[str, tuple[object, dict[str, object]]] = {}
    for condition in CONDITIONS:
        branch, result = run_condition(condition, source_config, teacher, train, dev)
        trained[condition] = (branch, result)

    # The fresh final partition is intentionally not requested until every condition completed both stages.
    test_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    fresh_test = transport.collect(test_split, tokenizer, FRESH_TEST_SEQUENCES, skip=FRESH_TEST_SKIP)
    teacher_fresh_test = quality_metrics(teacher, fresh_test)

    results: dict[str, object] = {}
    for condition, (branch, result) in trained.items():
        results[condition] = {**result, **evaluate_final(branch, tokenizer, fresh_test)}
        del branch
    gc.collect()

    alpha_zero_exact = all(abs(results[condition]["alpha_zero_fresh_test"]["loss"] - teacher_fresh_test["loss"]) < 1e-5 for condition in CONDITIONS)
    ce_loss = results["ce_continuation"]["final_fresh_test"]["loss"]
    self_loss = results["self_entropy_sharpening"]["final_fresh_test"]["loss"]
    teacher_loss = results["teacher_temperature_sharpening"]["final_fresh_test"]["loss"]
    interpretation = {
        "alpha_zero_exact_all": alpha_zero_exact,
        "all_alpha_one_endpoints": all(results[condition]["endpoint_alpha_one"] for condition in CONDITIONS),
        "self_minus_ce_fresh_test_loss": self_loss - ce_loss,
        "teacher_minus_ce_fresh_test_loss": teacher_loss - ce_loss,
        "teacher_minus_self_fresh_test_loss": teacher_loss - self_loss,
        "self_entropy_minus_ce": results["self_entropy_sharpening"]["final_fresh_test"]["token_entropy"] - results["ce_continuation"]["final_fresh_test"]["token_entropy"],
        "teacher_entropy_minus_ce": results["teacher_temperature_sharpening"]["final_fresh_test"]["token_entropy"] - results["ce_continuation"]["final_fresh_test"]["token_entropy"],
    }
    output = {
        "experiment": "experiment_019_output_sharpening",
        "seed": SEED,
        "source": {"model_id": SOURCE_ID, "teacher_frozen": True, "replaced_layers": list(transport.LAYERS)},
        "data": {
            "calibration_split": "wikitext-2-raw-v1/train",
            "calibration_sequences": TRAIN_SEQUENCES,
            "development_split": "wikitext-2-raw-v1/validation",
            "development_eligible_offset": DEV_SKIP,
            "development_sequences": DEV_SEQUENCES,
            "fresh_final_split": "wikitext-2-raw-v1/test",
            "fresh_final_eligible_offset": FRESH_TEST_SKIP,
            "fresh_final_sequences": FRESH_TEST_SEQUENCES,
            "fresh_test_loaded_after_all_conditions": True,
        },
        "budget": {
            "pre_stage_updates": PRE_STAGE_UPDATES,
            "output_stage_updates": OUTPUT_STAGE_UPDATES,
            "total_updates_per_condition": sum(PRE_STAGE_UPDATES.values()) + OUTPUT_STAGE_UPDATES,
            "transport_rank": transport.TRANSPORT_RANK,
            "pre_stage_learning_rate": transport.LEARNING_RATE,
            "output_stage_learning_rate": OUTPUT_STAGE_LR,
            "weight_decay": WEIGHT_DECAY,
            "entropy_weight": ENTROPY_WEIGHT,
            "teacher_temperature": TEACHER_TEMPERATURE,
            "teacher_kl_weight": TEACHER_KL_WEIGHT,
        },
        "teacher_fresh_test": teacher_fresh_test,
        **results,
        "interpretation": interpretation,
    }
    (OUT / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(interpretation, indent=2))


if __name__ == "__main__":
    main()
