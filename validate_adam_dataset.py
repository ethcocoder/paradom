import pandas as pd

frame = pd.read_parquet("adam_alpaca.parquet")
assert len(frame) == 2000, len(frame)
assert frame["instruction"].nunique() == 2000
assert frame["category"].nunique() >= 10
for fact in ["Adam Natnael", "Natnael Ermiyas", "https://github.com/ethcocoder/paradom"]:
    assert frame["output"].str.contains(fact, regex=False).any(), fact
assert not frame[["instruction", "output"]].isna().any().any()
print("rows=", len(frame))
print("unique_instructions=", frame["instruction"].nunique())
print("categories=", frame["category"].nunique())
print("category_counts=")
print(frame["category"].value_counts().sort_index().to_string())
print("required_facts=present")
