---
slug: impl-normalized-levenshtein-similarity
title: ANLS の実装
authors: Zephyr
image: /ja/img/2024/0516.webp
tags: [pytorch, anls]
description: Average Normalized Levenshtein Similarity
---

**Average Normalized Levenshtein Similarity**（略して ANLS）は、2 つの文字列間の類似度を測定する指標です。

<!-- truncate -->

**Levenshtein Similarity**（以下、LS と呼びます）。

自然言語処理（NLP）の分野では、2 つの文字列の類似度を比較することが頻繁に求められます。LS は一般的な測定方法であり、1 つの文字列を他の文字列に変換するために必要な「**編集距離**」を評価します。編集距離とは、挿入、削除、置換の操作回数を指します。

しかし、LS 自体は直感的ではなく、文字列の長さに依存します。この問題を解決するために、LS を [0, 1] の範囲に標準化することができます。この標準化したものが **Normalized Levenshtein Similarity**（NLS）と呼ばれ、異なる文字列間の類似度を理解しやすく比較可能になります。

さらに、NLS は複数の文字列ペア間の類似度を扱うことができます。この拡張版が **ANLS** であり、複数の文字列ペア間の平均類似度を計算して、モデルの性能を評価する指標となります。

それで……

既存の実装が満足できるものではなかったため、自分で実装することにしました。

## 参考資料

- [**torchmetrics.text.EditDistance**](https://lightning.ai/docs/torchmetrics/stable/text/edit.html)

## 必要なライブラリの導入

まず、必要なライブラリをインポートします。特に、`torchmetrics` の `EditDistance` を使用します：

```python
from typing import Any, Literal, Optional, Sequence, Union

import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.text import EditDistance
from torchmetrics.utilities.data import dim_zero_cat
```

`EditDistance` はすでに Levenshtein 距離を計算できます。そのため、2 つの文字列間の編集距離を直接求めることができます。ただし、`EditDistance` 自体には標準化機能がないため、この部分は自分で実装する必要があります。

## 標準化機能の実装

ここでは、`torchmetrics.metric.Metric` のインターフェースを継承します。そのため、`update` メソッドと `compute` メソッドを実装する必要があります：

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
            reduction=None  # すべての文字列ペアに対する距離を取得する設定
        )

        # ...
```

いくつかのポイントがあります：

1. 入力された `preds` と `target` が文字列のリストであることを確認します。そうしないと、関数が「文字」単位で計算を行う可能性があります。
2. 各文字列の最大長を計算して、標準化に使用します。

```python
def update(self, preds: Union[str, Sequence[str]], target: Union[str, Sequence[str]]) -> None:
    """予測値とターゲットで状態を更新します。"""

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

## `reduction` パラメータの実装

`reduction` パラメータを柔軟に活用できるようにする必要があります。例えば、`mean` を指定した場合、それは一般的な ANLS スコアを意味します。

また、通常の `mean` 以外にも、`sum` や `none` を使用して、異なるニーズを満たすことができます。

```python
def _compute(
    self,
    nls_score: Tensor,
    num_elements: Union[Tensor, int],
) -> Tensor:
    """状態に基づいて ANLS を計算します。"""
    if nls_score.numel() == 0:
        return torch.tensor(0, dtype=torch.int32)
    if self.reduction == "mean":
        return nls_score.sum() / num_elements
    if self.reduction == "sum":
        return nls_score.sum()
    if self.reduction is None or self.reduction == "none":
        return nls_score

def compute(self) -> torch.Tensor:
    """状態に基づいて NLS を計算します。"""
    if self.reduction == "none" or self.reduction is None:
        return self._compute(dim_zero_cat(self.nls_values_list), 1)
    return self._compute(self.nls_score, self.num_elements)
```

ここで注意が必要なのは、`reduction` に `none` を指定した場合、NLS の値すべてを返し、平均値を計算しない点です。この部分では、`torchmetrics.text.EditDistance` の実装方法を参考にしており、`dim_zero_cat` を利用してリスト内の値を連結し、返されるのが `Tensor` 形式になるようにしています。

## 完全な実装

以下は完全な実装コードです：

```python
from typing import Any, Literal, Optional, Sequence, Union

import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.text import EditDistance
from torchmetrics.utilities.data import dim_zero_cat


class NormalizedLevenshteinSimilarity(Metric):
    """
    Normalized Levenshtein Similarity (NLS) は、2つのシーケンス間の正規化された
    Levenshtein 類似度を計算する指標です。
    この指標は次のように計算されます：
        1 - (levenshtein_distance / max_length)
    ここで、`levenshtein_distance` は2つのシーケンス間の Levenshtein 距離を表し、
    `max_length` は2つのシーケンスのうちの最大長です。

    NLS は文字列の類似性を測定するために設計されており、特にテキスト類似性分析、
    光学式文字認識（OCR）、自然言語処理（NLP）などの分野で有用です。

    ``update`` および ``compute`` メソッドの入力形式は以下の通りです：

    - ``preds`` (:class:`~Union[str, Sequence[str]]`):
        予測されたテキストシーケンスまたはそのコレクション。
    - ``target`` (:class:`~Union[str, Sequence[str]]`):
        ターゲットとなるテキストシーケンスまたはそのコレクション。

    ``compute`` メソッドの出力形式：

    - ``nls`` (:class:`~torch.Tensor`): NLS 値を含むテンソル。
        サンプルがない場合は 0.0 を返し、サンプルがある場合は NLS を返します。

    Args:
        substitution_cost:
            1文字の置換にかかるコスト。デフォルトは1です。
        reduction:
            メトリックスコアを集約する方法。
            デフォルトは 'mean' で、選択肢は 'sum' または None。

            - ``'mean'``: サンプル全体の平均（ANLS）を取る。
            - ``'sum'``: サンプル全体の合計を取る。
            - ``None`` または ``'none'``: サンプルごとのスコアを返す。

        kwargs: その他のキーワード引数。

    Example::
        複数の文字列を扱う例：

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
            reduction=None  # 全ての文字列ペアの距離を取得する設定
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
        """予測値とターゲットで状態を更新します。"""
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
        """状態に基づいて ANLS を計算します。"""
        if nls_score.numel() == 0:
            return torch.tensor(0, dtype=torch.int32)
        if self.reduction == "mean":
            return nls_score.sum() / num_elements
        if self.reduction == "sum":
            return nls_score.sum()
        if self.reduction is None or self.reduction == "none":
            return nls_score

    def compute(self) -> torch.Tensor:
        """状態に基づいて NLS を計算します。"""
        if self.reduction == "none" or self.reduction is None:
            return self._compute(dim_zero_cat(self.nls_values_list), 1)
        return self._compute(self.nls_score, self.num_elements)


if __name__ == "__main__":
    anls = NormalizedLevenshteinSimilarity(reduction='mean')
    preds = ["rain", "lnaguaeg"]
    target = ["shine", "language"]
    print(anls(preds, target))
```

## 最後に

この実装が完全に正しいと保証できるか？

答えは **いいえ** です。もし問題を発見した場合は、ぜひご指摘ください。心より感謝します！
