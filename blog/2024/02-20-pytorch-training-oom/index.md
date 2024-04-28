---
slug: pytorch-training-out-of-memory
title: PyTorch 的 List 陷阱
authors: Zephyr
tags: [PyTorch, OOM]
image: /img/2024/0220.webp
description: 發現與解決 PyTorch OOM 問題。
---

<figure>
![title](/img/2024/0220.webp)
<figcaption>封面圖片：由 GPT-4 閱讀本文之後自動生成</figcaption>
</figure>

---

身為一個專業的 PyTorch 使用者，你應該早已經熟悉了如何訓練模型，如何調參，如何優化等技巧。

怎麼可能還會寫出 OOM（Out of Memory）的程式呢？

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

---

## 我也是這樣寫，為什麼什麼事情都沒有發生？

世界很美好，直到你遇到了一個大數據集。

就我自己的測試的結果來看，當數據量小的時候，用 List 不會出現記憶體洩漏的問題。

更具體來說，我使用一萬多筆數據的時候，沒問題；使用一百多萬筆數據的時候就爆了。

所以，如果你的數據量不大，你可能永遠不會遇到這個問題。

至於數據多寡的分界點，我也不知道，我推估這是 Python 或是 PyTorch 在交互的某個時刻才會出現的異常。
