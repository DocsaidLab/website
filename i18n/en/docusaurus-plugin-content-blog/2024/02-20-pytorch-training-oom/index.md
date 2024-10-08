---
slug: pytorch-training-out-of-memory
title: The Pitfall of Lists in PyTorch
authors: Zephyr
tags: [PyTorch, OOM]
image: /en/img/2024/0220.webp
description: Resolving PyTorch OOM Issues.
---

<figure>
![title](/img/2024/0220.webp)
<figcaption>Cover Image: Automatically generated by GPT-4 after reading this article</figcaption>
</figure>

---

As a seasoned PyTorch user, you should be well-versed in techniques for model training, parameter tuning, and optimization. How could you possibly still end up with an OOM (Out of Memory) error?

<!-- truncate -->

---

- Please note, we're talking about system memory here, not GPU memory.
- Of course, GPU memory is also a common concern, but we're not discussing that here.

---

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

---

## I've written code like this, why haven't I experienced any issues?

The world is a beautiful place until you encounter a large dataset.

Based on my own testing, when the dataset is small, using Lists doesn't trigger memory leaks.

More specifically, with just over ten thousand data points, there's no issue; but with over a million, it blows up.

So, if your dataset isn't large, you might never encounter this problem.

As for the threshold of data size, I'm unsure. I speculate it's a moment of interaction between Python or PyTorch that triggers the anomaly.
