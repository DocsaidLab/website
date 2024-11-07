---
slug: pytorch-training-out-of-memory
title: The Pitfall of Lists in PyTorch
authors: Zephyr
tags: [PyTorch, OOM]
image: /en/img/2024/0220.webp
description: Resolving PyTorch OOM Issues.
---

As a seasoned PyTorch user, you're likely well-versed in training models, hyperparameter tuning, and optimization techniques.

How could you possibly write code that runs out of memory (OOM)?

<!-- truncate -->

:::tip
We're talking about system memory here, not GPU memory.
:::

## Problem Description

With OOM issues stemming from various causes, this time we'll focus on one commonly encountered by professionals:

- You might be using a List structure!

After investigation, we've pinpointed the exact scenario where leaks occur.

Consider the following code snippet:

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

Cutting to the chase after examining this example:

See the `self.data` List? That's what leads to the OOM problem.

We attempted to find related information and it seems this isn't a PyTorch issue but rather a Python one.

In essence, refrain from using Lists; use NumPy or Tensors to store data, and you won't encounter OOM problems.

At least, that's effective in this example.

## What About Me?

You might be wondering: I've written code like this, why haven't I encountered any issues?

---

The world is a beautiful place until you encounter a large dataset.

Based on my own testing, when the dataset is small, using Lists doesn't trigger memory leaks.

More specifically:

- When we use over 10,000 data points, no issues arise!
- When we use over 1.2 million data points, it blows up!

So, if your dataset isn't large, you might never encounter this problem.

As for the threshold of data volume, we're unsure...

We speculate this anomaly arises at a certain point during Python or PyTorch interaction.
