---
slug: pytorch-training-out-of-memory
title: PyTorchのListによる罠
authors: Z. Yuan
tags: [PyTorch, OOM]
image: /ja/img/2024/0220.webp
description: PyTorchでのOOM問題の発見と解決方法。
---

プロのPyTorchユーザーとして、あなたはすでにモデルのトレーニング方法、ハイパーパラメータの調整、最適化の技術について熟知しているはずです。

そんなあなたが、OOM（Out of Memory）のプログラムを書くわけがありませんよね？

<!-- truncate -->

:::tip
ここで言うのは、システムのメモリのことです。GPUメモリのことではありません。
:::

## 問題の説明

OOMの原因は多岐に渡りますが、今回はプロフェッショナルなユーザーでもよく遭遇する問題の1つについて話します：

- あなたがリスト（List）を使っているかもしれません！

私が最近モデルをトレーニングしていた経験から、メモリリークが発生する正確なシナリオを見つけました。

次のコード例を見てみましょう：

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

この例を見て、無駄な話はせず、結論から言います：

- **`self.data`というリストが見えましたか？このリストがOOM問題を引き起こします。**

関連する資料を調べた結果、これはPyTorchの問題ではなく、Pythonの問題であることが分かりました。

とにかく、リストは使わず、NumpyやTensorを使ってデータを保存することで、OOM問題は発生しません。

少なくとも、この例ではそれが有効でした。

## では、私は？

あなたが言うかもしれません：「私もこう書いているけど、何も問題は起きていないよ？」

---

世界は素晴らしい、しかし大規模なデータセットに遭遇すると事態は一変します。

私のテスト結果によると、データ量が小さい場合、リストを使ってもメモリリークの問題は発生しません。

具体的に言うと：

- 約1万件のデータでは問題なし！
- 約100万件のデータでは、メモリが爆発します！

したがって、データ量が少ない場合、あなたはおそらくこの問題に直面することはないでしょう。

データ量の境界線については、私も分かりません……おそらく、PythonとPyTorchが相互作用するタイミングで異常が発生するのでしょう。