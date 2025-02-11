---
slug: pytorch-training-out-of-memory
title: PyTorch の List トラップ
authors: Z. Yuan
tags: [PyTorch, OOM]
image: /ja/img/2024/0220.webp
description: PyTorch OOM 問題の発見と解決。
---

プロフェッショナルな PyTorch ユーザーであれば、モデルのトレーニング方法、ハイパーパラメータの調整方法、最適化の技術に精通しているはずです。

それなのに、どうして OOM（Out of Memory）のコードを書いてしまうのでしょうか？

<!-- truncate -->

:::tip
ここで言うメモリはシステムメモリのことです。GPU メモリではありません。
:::

## 問題の説明

OOM の原因は多岐にわたりますが、今回はプロフェッショナルでも陥りやすい問題に絞って解説します：

- **List 構造を使用していませんか？**

調査を進めた結果、メモリリークが発生する正確な状況を特定しました。

以下のコード例を考えてみてください：

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

この例を見たら、さっそく結論に行きましょう：

`self.data` という List を見つけましたか？この List が OOM 問題を引き起こします。

関連する資料を調べたところ、これは PyTorch の問題ではなく、Python 自体の問題である可能性が高いことが分かりました。

結論として、**List を使わずに Numpy または Tensor を使ってデータを保存してください**。これにより、OOM 問題を回避できます。

少なくともこの例では、この方法で問題が解消されます。

## あなたの場合は？

「私も同じように書いているけど、問題が起きないのはなぜ？」

---

問題は世界が平和である間は発生しません。しかし、ある日、巨大なデータセットに出会った時に問題が顕在化します。

私自身のテスト結果によると、データ量が小さい場合には List を使ってもメモリリークは発生しません。

具体的には：

- データ量が数万件の場合、問題ありません！
- データ量が数百万件に達すると、OOM が発生します！

したがって、もしあなたのデータ量が小規模であれば、この問題に直面することはないかもしれません。

データ量の境界線については、私たちも正確には分かりませんが、おそらく Python と PyTorch が特定のタイミングで交互に処理する際に異常が発生しているのではないかと推測されます。
