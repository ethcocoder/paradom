import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./adam-finetuned"

def test_model(prompt):
    print(f"\nTesting prompt: {prompt}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")

if __name__ == "__main__":
    # Test 1: Direct name question
    test_model("### Instruction:\nWhat is your name?\n\n### Response:\n")
    
    # Test 2: Who are you?
    test_model("### Instruction:\nWho are you?\n\n### Response:\n")
