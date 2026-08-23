"""Interactive chat with the TinyLlama 1.1B Adam LoRA adapter.

The adapter directory should contain adapter_config.json, adapter_model.safetensors,
and tokenizer files. The TinyLlama base model is downloaded separately from Hugging Face.
"""

import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "/content/paradom/adam-tinyllama-qlora")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "160"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "1536"))


def load_adam():
    if not os.path.isdir(ADAPTER_PATH):
        raise FileNotFoundError(
            f"Adapter directory not found: {ADAPTER_PATH}\n"
            "Set ADAPTER_PATH to the folder containing adapter_config.json."
        )

    adapter_config = os.path.join(ADAPTER_PATH, "adapter_config.json")
    if not os.path.isfile(adapter_config):
        raise FileNotFoundError(
            f"Missing adapter_config.json in {ADAPTER_PATH}. "
            "This folder may not be the completed LoRA output."
        )

    use_cuda = torch.cuda.is_available()
    compute_dtype = (
        torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported()
        else torch.float16 if use_cuda
        else torch.float32
    )
    print(f"Device mode: {'CUDA' if use_cuda else 'CPU'} full-precision LoRA")

    print(f"Loading tokenizer from {ADAPTER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading full-precision base model {BASE_MODEL}...")
    model_kwargs = {"torch_dtype": compute_dtype}
    if use_cuda:
        model_kwargs["device_map"] = "auto"
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    if not use_cuda:
        base = base.to("cpu")
    print("Applying Adam LoRA adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return tokenizer, model


def build_prompt(history, user_text):
    pieces = []
    for user_message, assistant_message in history:
        pieces.append(f"### Instruction:\n{user_message}\n\n### Response:\n{assistant_message}")
    pieces.append(f"### Instruction:\n{user_text}\n\n### Response:\n")
    return "\n\n".join(pieces)


def generate_reply(tokenizer, model, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_TOKENS,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return reply or "I was unable to generate a response."


def main():
    tokenizer, model = load_adam()
    history = []
    print("\nAdam is ready. Type /reset to clear history or /exit to quit.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            print("Goodbye.")
            break
        if user_text.lower() == "/reset":
            history.clear()
            print("Conversation history cleared.")
            continue

        prompt = build_prompt(history, user_text)
        reply = generate_reply(tokenizer, model, prompt)
        print(f"Adam: {reply}\n")
        history.append((user_text, reply))


if __name__ == "__main__":
    main()
