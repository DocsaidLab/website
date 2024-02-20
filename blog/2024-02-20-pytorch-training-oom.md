---
slug: pytorch-training-out-of-memory
title: 啊！PyTorch 怎麼把記憶體弄爆了？
authors: Zephyr
tags: [PyTorch, OOM]
---

身為一個專業的 PyTorch 使用者，你應該早已經熟悉了如何訓練模型，如何調參，如何優化等技巧。

怎麼可能還會寫出 OOM（Out of Memory）的程式呢？但怎麼就是爆了？

<!--truncate-->

---

- 請注意，這裡我們講的是系統的記憶體，不是 GPU 的記憶體。
- 當然，GPU 的記憶體也是一個常見的問題，但這裡我們不談這個。

---

由於 OOM 的成因太多，這次我們就只講一個專業工作者也常會遇到的問題：

- 你可能用了 List 結構啦！

經過調查，我們找到了洩漏發生時的確切場景。

考慮下面的程式碼範例：

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

看完這個例子，不廢話，直接講結論：

看到 `self.data` 這個 List 了嗎？這個 List 會導致 OOM 問題。

我們試著翻找了一下相關資料，發現這似乎不是 Pytorch 的問題，而是 Python 的問題。

總之，不要用 List，改用 Numpy 或者 Tensor 來存儲數據，這樣就不會出現 OOM 問題了。

至少在這個例子中，這樣做是有效的。
