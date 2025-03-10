---
slug: torch-layernorm-mismatch
title: 手算的 LayerNorm 數值不對？
authors: Z. Yuan
image: /img/2024/0820.webp
tags: [PyTorch, LayerNorm]
description: 閒來無事，動手算算。恩，怎麼不對啊？
---

今天突然想算一下 LayerNorm 的數值。

<!-- truncate -->

我們都知道，LayerNorm 的公式如下：

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\text{Var[}x\text{]} + \epsilon}} \times \gamma + \beta
$$

其中，$\mu$ 是 $x$ 的均值，$\text{Var}$ 是 $x$ 的變異數。

有了上面的資訊，我們直接來動手算一下，忽略 $\gamma$ 和 $\beta$：

```python
import torch

x = torch.rand(16, 768)
mu = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
eps = 1e-5
y = (x - mu) / (var + eps).sqrt()
```

得到以下數值：

```python
# tensor([[ 0.1219, -0.0222, -1.4742,  ...,  0.1738, -0.6124, -0.3001],
#         [-1.6009, -1.5814,  1.5357,  ...,  0.1917,  1.3787, -0.2772],
#         [ 0.3738,  1.0520,  0.4403,  ...,  1.1353, -0.7488, -0.9137],
#         ...,
#         [ 0.8823, -1.5427,  0.4725,  ..., -1.2544, -1.5354, -0.4305],
#         [ 1.4548,  0.3059, -0.6732,  ..., -0.7109,  0.4908, -1.2447],
#         [-0.4067,  0.5974, -0.9113,  ..., -0.2511, -0.2279, -0.9675]])
```

接著把這個數值和 PyTorch 的 `torch.nn.LayerNorm` 進行比較：

```python
layer_norm = torch.nn.LayerNorm(768, elementwise_affine=False, bias=False)

y_ln = layer_norm(x)
```

得到數值：

```python
# tensor([[ 0.1220, -0.0222, -1.4752,  ...,  0.1739, -0.6128, -0.3003],
#         [-1.6020, -1.5824,  1.5367,  ...,  0.1918,  1.3796, -0.2774],
#         [ 0.3741,  1.0527,  0.4406,  ...,  1.1360, -0.7493, -0.9143],
#         ...,
#         [ 0.8829, -1.5437,  0.4728,  ..., -1.2552, -1.5364, -0.4308],
#         [ 1.4557,  0.3061, -0.6736,  ..., -0.7113,  0.4911, -1.2455],
#         [-0.4069,  0.5978, -0.9119,  ..., -0.2513, -0.2281, -0.9681]])
```

上下比對一下，發現怎麼不一樣？

## 無偏估計

快速查詢一下相關資料，原來是 `torch.var` 在使用上有一個參數 `correction`，預設是 `1`，即使用無偏估計。

意思是這裡會除以 `N-1` 而不是 `N`，而 `torch.nn.LayerNorm` 使用的是 `N`。

所以我們修改一下 `torch.var` 的參數，設定 `correction=0`：

```python
var = x.var(dim=-1, correction=0, keepdim=True)
```

:::tip
`correction` 是 `unbiased` 的別名，在 PyTorch 2.0.0 版本中被引入。

比較早期的版本中，設定方式改為 `unbiased=False`：

```python
var = x.var(dim=-1, unbiased=False, keepdim=True)
```

:::

再次比對：

```python
# tensor([[ 0.1220, -0.0222, -1.4752,  ...,  0.1739, -0.6128, -0.3003],
#         [-1.6020, -1.5824,  1.5367,  ...,  0.1918,  1.3796, -0.2774],
#         [ 0.3741,  1.0527,  0.4406,  ...,  1.1360, -0.7493, -0.9143],
#         ...,
#         [ 0.8829, -1.5437,  0.4728,  ..., -1.2552, -1.5364, -0.4308],
#         [ 1.4557,  0.3061, -0.6736,  ..., -0.7113,  0.4911, -1.2455],
#         [-0.4069,  0.5978, -0.9119,  ..., -0.2513, -0.2281, -0.9681]])
```

這次數值就對啦！

## 所以為什麼 LayerNorm 不是無偏估計？

大概總結一下，就是為了穩定性和計算簡化。

如果你對這個問題感興趣，可以繼續看下去：

- **小批量計算的穩定性**

  LayerNorm 通常應用在單一樣本的特徵維度（例如每個神經元或每個特徵）上，而不是在整個批次上。每個樣本的特徵數量通常遠大於樣本的數量。因此，與樣本標準差相比，母體標準差可以提供更穩定和更準確的估計，特別是在小樣本量時。

- **無偏估計的重要性降低**

  樣本標準差的無偏性（即以 n-1 而非 n 作為分母）在需要對整體樣本估計總體參數時更為重要。這在統計學中用於避免偏差。然而，在深度學習中的正則化和歸一化操作中，尤其是像 LayerNorm 這樣的場景，偏差的影響相對較小，因為這些計算只是用於標準化啟動值，而不是估計總體的統計量。因此，使用母體標準差可以簡化計算，同時對於訓練效果的影響很小。

- **梯度計算的穩定性**

  在反向傳播中，穩定的梯度非常重要。使用母體標準差使得梯度計算更為平滑和穩定，避免了由於樣本數量較少時引入的額外噪聲，從而有助於網路的收斂性和訓練效果。

- **計算的簡化**

  從計算角度來看，母體標準差的計算比樣本標準差稍微簡單一些，因為它少了一個減法操作（即分母為 n 而不是 n-1），這在計算效率上有細微的優勢。雖然這並不是決定性的因素，但也是設計時的一個考量。

## 小結

其實就只是突然想到這個問題，所以寫了這篇文章。

希望這個問題的解答能對你有所幫助。
