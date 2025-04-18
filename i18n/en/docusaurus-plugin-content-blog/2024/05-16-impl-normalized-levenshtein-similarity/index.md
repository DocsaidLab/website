---
slug: impl-normalized-levenshtein-similarity
title: Implementing ANLS
authors: Z. Yuan
image: /en/img/2024/0516.webp
tags: [pytorch, anls]
description: Average Normalized Levenshtein Similarity
---

Average Normalized Levenshtein Similarity, abbreviated as ANLS, is a metric used to compute the similarity between two strings.

<!-- truncate -->

In natural language processing (NLP), it's often necessary to compare the similarity of two strings.

Levenshtein Similarity is a common measure that assesses the "edit distance" between two strings, which is the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one string into the other. However, Levenshtein Similarity itself isn't intuitive as it depends on the lengths of the strings.

To address this issue, we can normalize Levenshtein Similarity to the [0, 1] range, making it easier to understand and compare the similarity between different strings, known as Normalized Levenshtein Similarity (NLS).

As NLS refers to the similarity between sets of strings, we can further extend it to ANLS, which computes the average similarity among multiple sets of strings, thereby quantifying the performance of a model.

And then...

We always struggle to find implementations, so we decided to write one ourself.

## References

- [**torchmetrics.text.EditDistance**](https://lightning.ai/docs/torchmetrics/stable/text/edit.html)

## Import Necessary Libraries

First, we need to import some necessary libraries, especially the `EditDistance` implemented by `torchmetrics`:

```python
from typing import Any, Literal, Optional, Sequence, Union

import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.text import EditDistance
from torchmetrics.utilities.data import dim_zero_cat
```

Since `EditDistance` can already compute the Levenshtein distance, we can directly use it to calculate the edit distance between two strings. However, `EditDistance` doesn't provide normalization functionality, so we need to implement this part ourselves.

## Implement Normalization Functionality

Here, we inherit the interface of `torchmetrics.metric.Metric`, so we need to implement the `update` and `compute` methods:

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

Here are a few key points:

1. Ensure that the input `preds` and `target` are lists of strings, otherwise the function will calculate on a character level.
2. Calculate the maximum length of each string, so that we can perform normalization.

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

## Implement the `reduction` Parameter

We also need to accommodate the `reduction` parameter, where if we specify `mean`, it will be the common ANLS score.

In addition to the usual `mean`, we can also use `sum` or `none` to fulfill different needs.

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

Here, it's noteworthy that when we specify `reduction` as `none`, we need to return all NLS values instead of computing the average. In this case, I referenced the implementation of `torchmetrics.text.EditDistance`, using `dim_zero_cat` to concatenate values in the list together, ensuring that the return value is a `Tensor`.

## Implementation

The complete implementation is as follows:

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

## At Last

Can we guarantee that this implementation is correct?

The answer is no. If you find any issues, please let us know. Thank you very much!
