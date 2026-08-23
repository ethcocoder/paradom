import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "paradoxquantum/adam")
MAX_INPUT_TOKENS = 1536
MAX_NEW_TOKENS = 160

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

use_cuda = torch.cuda.is_available()
model_kwargs = {"torch_dtype": torch.float16 if use_cuda else torch.float32}
if use_cuda:
    model_kwargs["device_map"] = "auto"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
if not use_cuda:
    model = model.to("cpu")
model.eval()


def make_prompt(message, history):
    parts = []
    for item in history or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_message, assistant_message = item
            parts.append(
                f"### Instruction:\n{user_message}\n\n### Response:\n{assistant_message}"
            )
    parts.append(f"### Instruction:\n{message}\n\n### Response:\n")
    return "\n\n".join(parts)


def respond(message, history):
    prompt = make_prompt(message, history)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


app = gr.ChatInterface(
    fn=respond,
    title="Adam",
    description="Chat with the Adam finetuned model.",
    textbox=gr.Textbox(placeholder="Ask Adam something...", label="Message"),
)

if __name__ == "__main__":
    app.launch()
