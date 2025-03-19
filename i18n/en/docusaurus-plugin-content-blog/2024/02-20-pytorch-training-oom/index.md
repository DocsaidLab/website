---
slug: pytorch-training-out-of-memory
title: The PyTorch List Trap
authors: Z. Yuan
tags: [PyTorch, OOM]
image: /en/img/2024/0220.webp
description: Discovering and solving PyTorch OOM issues.
---

As a professional PyTorch user, you should already be familiar with how to train models, tune hyperparameters, optimize performance, and more.

How could you possibly write a program that encounters OOM (Out of Memory) issues?

<!-- truncate -->

:::tip
This is referring to system memory, not GPU memory.
:::

## Problem Description

There are many potential causes for OOM errors, but this time, I will focus on a specific issue that even professional workers often encounter:

- You might be using a List structure!

Based on my recent experience training models, I identified the exact scenario when the memory leak occurs.

Consider the following code example:

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class DataIter(Dataset):

    def __init__(self):
        self.data_np = np.array([x for x in range(10000000)])
        self.data = [x for x in range(10000000)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = np.array([data], dtype=np.int64)
        return torch.tensor(data)


train_data = DataIter()
train_loader = DataLoader(train_data, batch_size=300, num_workers=18)

for i, item in enumerate(train_loader):
    if i % 1000 == 0:
        print(i)
```

---

After reviewing this example, let's cut to the chase:

- **Do you see the `self.data` List? This List will cause the OOM problem.**

I did some research and found that this doesn't seem to be a PyTorch issue, but rather a Python issue.

In any case, don't use List; use Numpy or Tensor to store data instead. This way, you won’t encounter OOM issues.

At least in this example, doing so was effective.

## What About Me?

You might say: "I wrote it the same way, and nothing happened!"

---

The world is great until you encounter a large dataset.

Based on my own test results, when the dataset is small, using a List does not cause memory leaks.

More specifically:

- Using around 10,000 data points? No problem!
- Using over a million data points? Boom, it crashes!

So, if your dataset is small, you may never encounter this issue.

As for the boundary of the dataset size, I’m not sure... My guess is that it’s some anomaly that occurs at a specific moment in the interaction between Python and PyTorch.