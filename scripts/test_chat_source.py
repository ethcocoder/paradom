import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
import os

def test_source_chat():
    model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading model structure...")
    # Load model shell without weights
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
    
    local_weights_path = "models/nova_source/model.safetensors"
    if os.path.exists(local_weights_path):
        print(f"Loading local weights from {local_weights_path}...")
        state_dict = load_file(local_weights_path)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: {local_weights_path} not found. Using default weights.")

    print("\n--- CHAT WITH SOURCE MODEL ---")
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            break
        
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Model: {response}")

if __name__ == "__main__":
    test_source_chat()
