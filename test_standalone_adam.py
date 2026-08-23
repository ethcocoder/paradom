import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "paradoxquantum/adam")
THREADS = int(os.getenv("CPU_THREADS", str(os.cpu_count() or 1)))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "48"))


def main():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(max(1, min(4, THREADS)))
    print(f"Loading standalone model: {MODEL_ID}")
    print(f"Using CPU with {THREADS} PyTorch threads")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
    )
    model.eval()

    questions = ["What is your name?", "Who are you?"]
    for question in questions:
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        print(f"Q: {question}\nA: {answer}\n")


if __name__ == "__main__":
    main()
