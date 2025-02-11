---
slug: impl-normalized-levenshtein-similarity
title: 實作 ANLS
authors: Z. Yuan
image: /img/2024/0516.webp
tags: [pytorch, anls]
description: Average Normalized Levenshtein Similarity
---

Average Normalized Levenshtein Similarity，簡稱 ANLS，是一種用於計算兩個字串之間相似性的指標。

<!-- truncate -->

Levenshtein Similarity，以下我們簡稱為 LS。

在自然語言處理（NLP）中，我們經常需要比較兩個字串的相似性。LS 是一種常見的度量方法，它衡量了兩個字串之間的「**編輯距離**」，即通過多少次插入、刪除或替換操作可以將一個字串轉換為另一個字串。

只是 LS 本身並不直觀，因為它取決於字串的長度。為了解決這個問題，我們可以將 LS 標準化為 [0, 1] 區間，這樣我們就可以更容易地理解和比較不同字串之間的相似性，稱為 Normalized Levenshtein Similarity（NLS）。

由於 NLS 指的是一組字串之間的相似性，我們可以將其進一步擴展為 ANLS，它計算了多組字串之間的平均相似性，藉此來橫量模型的性能。

然後......

我們總是找不到喜歡的實作，最後決定自己寫一個。

## 參考資料

- [**torchmetrics.text.EditDistance**](https://lightning.ai/docs/torchmetrics/stable/text/edit.html)

## 導入必要的庫

首先，我們需要導入一些必要的庫，特別是由 `torchmetrics` 實作的 `EditDistance`：

```python
from typing import Any, Literal, Optional, Sequence, Union

import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.text import EditDistance
from torchmetrics.utilities.data import dim_zero_cat
```

由於 `EditDistance` 已經可以計算 Levenshtein 距離，我們可以直接使用它來計算兩個字串之間的編輯距離。然而，`EditDistance` 並沒有提供標準化的功能，所以我們需要自己實現這一部分。

## 實作標準化功能

在這裡，我們繼承 `torchmetrics.metric.Metric` 的介面，所以我們需要實作 `update` 和 `compute` 方法：

```python
class NormalizedLevenshteinSimilarity(Metric):

    def __init__(
        self,
        substitution_cost: int = 1,
        reduction: Optional[Literal["mean", "sum", "none"]] = "mean",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.edit_distance = EditDistance(
            substitution_cost=substitution_cost,
            reduction=None  # Set to None to get distances for all string pairs
        )

        # ...
```

這裡有幾個要點：

1. 確保輸入的 `preds` 和 `target` 是字串列表，否則函數就會計算到「字元」的部分。
2. 計算每個字串的最大長度，這樣才能進行標準化。

```python
def update(self, preds: Union[str, Sequence[str]], target: Union[str, Sequence[str]]) -> None:
    """Update state with predictions and targets."""

    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]

    distances = self.edit_distance(preds, target)
    max_lengths = torch.tensor([
        max(len(p), len(t))
        for p, t in zip(preds, target)
    ], dtype=torch.float)

    ratio = torch.where(
        max_lengths == 0,
        torch.zeros_like(distances).float(),
        distances.float() / max_lengths
    )

    nls_values = 1 - ratio

    # ...
```

## 實作 `reduction` 參數

我們還需要保留 `reduction` 參數的發揮空間，如果我們指定 `mean`，那就是常見的 ANLS 分數。

除了一般的 `mean`，我們也可以使用 `sum` 或 `none`，來完成不同的需求。

```python
def _compute(
    self,
    nls_score: Tensor,
    num_elements: Union[Tensor, int],
) -> Tensor:
    """Compute the ANLS over state."""
    if nls_score.numel() == 0:
        return torch.tensor(0, dtype=torch.int32)
    if self.reduction == "mean":
        return nls_score.sum() / num_elements
    if self.reduction == "sum":
        return nls_score.sum()
    if self.reduction is None or self.reduction == "none":
        return nls_score

def compute(self) -> torch.Tensor:
    """Compute the NLS over state."""
    if self.reduction == "none" or self.reduction is None:
        return self._compute(dim_zero_cat(self.nls_values_list), 1)
    return self._compute(self.nls_score, self.num_elements)
```

這裡需要注意的部分是當我們指定 `reduction` 為 `none` 時，我們需要將所有的 NLS 值返回，而不是計算平均值。這邊我參考了 `torchmetrics.text.EditDistance` 的實現方式，使用了 `dim_zero_cat` 來將列表中的值拼接在一起，確保回傳的是一個 `Tensor`。

## 程式碼

完整的實作如下：

```python
from typing import Any, Literal, Optional, Sequence, Union

import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.text import EditDistance
from torchmetrics.utilities.data import dim_zero_cat


class NormalizedLevenshteinSimilarity(Metric):
    """
    Normalized Levenshtein Similarity (NLS) is a metric that computes the
    normalized Levenshtein similarity between two sequences.
    This metric is calculated as 1 - (levenshtein_distance / max_length),
    where `levenshtein_distance` is the Levenshtein distance between the two
    sequences and `max_length` is the maximum length of the two sequences.

    NLS aims to provide a similarity measure for character sequences
    (such as text), making it useful in areas like text similarity analysis,
    Optical Character Recognition (OCR), and Natural Language Processing (NLP).

    This class inherits from `Metric` and uses the `EditDistance` class to
    compute the Levenshtein distance.

    Inputs to the ``update`` and ``compute`` methods are as follows:

    - ``preds`` (:class:`~Union[str, Sequence[str]]`):
        Predicted text sequences or a collection of sequences.
    - ``target`` (:class:`~Union[str, Sequence[str]]`):
        Target text sequences or a collection of sequences.

    Output from the ``compute`` method is as follows:

    - ``nls`` (:class:`~torch.Tensor`): A tensor containing the NLS value.
        Returns 0.0 when there are no samples; otherwise, it returns the NLS.

    Args:
        substitution_cost:
            The cost of substituting one character for another. Default is 1.
        reduction:
            Method to aggregate metric scores.
            Default is 'mean', options are 'sum' or None.

            - ``'mean'``: takes the mean over samples, which is ANLS.
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: returns the score per sample

        kwargs: Additional keyword arguments.

    Example::
        Multiple strings example:

        >>> metric = NormalizedLevenshteinSimilarity(reduction=None)
        >>> preds = ["rain", "lnaguaeg"]
        >>> target = ["shine", "language"]
        >>> metric(preds, target)
        tensor([0.4000, 0.5000])
        >>> metric = NormalizedLevenshteinSimilarity(reduction="mean")
        >>> metric(preds, target)
        tensor(0.4500)
    """

    def __init__(
        self,
        substitution_cost: int = 1,
        reduction: Optional[Literal["mean", "sum", "none"]] = "mean",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.edit_distance = EditDistance(
            substitution_cost=substitution_cost,
            reduction=None  # Set to None to get distances for all string pairs
        )

        allowed_reduction = (None, "mean", "sum", "none")
        if reduction not in allowed_reduction:
            raise ValueError(
                f"Expected argument `reduction` to be one of {allowed_reduction}, but got {reduction}")
        self.reduction = reduction

        if self.reduction == "none" or self.reduction is None:
            self.add_state(
                "nls_values_list",
                default=[],
                dist_reduce_fx="cat"
            )
        else:
            self.add_state(
                "nls_score",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum"
            )
            self.add_state(
                "num_elements",
                default=torch.tensor(0),
                dist_reduce_fx="sum"
            )

    def update(self, preds: Union[str, Sequence[str]], target: Union[str, Sequence[str]]) -> None:
        """Update state with predictions and targets."""
        if isinstance(preds, str):
            preds = [preds]
        if isinstance(target, str):
            target = [target]

        distances = self.edit_distance(preds, target)
        max_lengths = torch.tensor([
            max(len(p), len(t))
            for p, t in zip(preds, target)
        ], dtype=torch.float)

        ratio = torch.where(
            max_lengths == 0,
            torch.zeros_like(distances).float(),
            distances.float() / max_lengths
        )

        nls_values = 1 - ratio

        if self.reduction == "none" or self.reduction is None:
            self.nls_values_list.append(nls_values)
        else:
            self.nls_score += nls_values.sum()
            self.num_elements += nls_values.shape[0]

    def _compute(
        self,
        nls_score: Tensor,
        num_elements: Union[Tensor, int],
    ) -> Tensor:
        """Compute the ANLS over state."""
        if nls_score.numel() == 0:
            return torch.tensor(0, dtype=torch.int32)
        if self.reduction == "mean":
            return nls_score.sum() / num_elements
        if self.reduction == "sum":
            return nls_score.sum()
        if self.reduction is None or self.reduction == "none":
            return nls_score

    def compute(self) -> torch.Tensor:
        """Compute the NLS over state."""
        if self.reduction == "none" or self.reduction is None:
            return self._compute(dim_zero_cat(self.nls_values_list), 1)
        return self._compute(self.nls_score, self.num_elements)


if __name__ == "__main__":
    anls = NormalizedLevenshteinSimilarity(reduction='mean')
    preds = ["rain", "lnaguaeg"]
    target = ["shine", "language"]
    print(anls(preds, target))
```

## 最後

我們可以保證這個實作是正確的嗎？

答案是不行，如果你發現了任何問題，請告訴我們，非常感謝！
