# ===================================================
# Python / PyTorch code associated with Module-1 of
# online Coursera course:
#   Introduction to Neural Networks and PyTorch
# ===================================================
import torch

from torch.utils.data import DataLoader

# Hugging Face datasets.  Take simple one to practice on.
from datasets import load_dataset
ds = load_dataset("boolq")

# This shows the "split".
# In this case a "train" dataset of 120000 entries
#              a "test" dataset of 7600 entries
print(ds)

# Check the labels and type (train & test will be same)
print(ds["train"].features) # Features

train_ds = ds["train"]

# make DataLoader return dicts with torch tensors where possible
train_ds.set_format(type="torch")

loader = DataLoader(train_ds, batch_size=16, shuffle=True)

"""
for batch in loader:
    # batch is a dict
    questions = batch["question"]   # list[str]
    passages  = batch["passage"]    # list[str]
    answers   = batch["answer"]     # tensor([True/False], shape [16])

    print(questions)
    print(passages)
    print(answers)
    break
"""