import ast
import inspect
from pathlib import Path
from transformers import Trainer

source = Path("finetune_paradox.py").read_text()
ast.parse(source)
signature = inspect.signature(Trainer.__init__)
assert "processing_class" in signature.parameters, signature
assert "tokenizer" not in signature.parameters, signature
assert "processing_class=tokenizer" in source
assert "tokenizer=tokenizer" not in source
print("PASS: Python syntax is valid.")
print("PASS: Installed Transformers Trainer uses processing_class.")
print("PASS: finetune_paradox.py passes processing_class=tokenizer.")
      
