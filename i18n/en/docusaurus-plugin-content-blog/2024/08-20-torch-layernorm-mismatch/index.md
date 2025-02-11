---
slug: torch-layernorm-mismatch
title: Discrepancy in LayerNorm Calculations?
authors: Z. Yuan
image: /en/img/2024/0820.webp
tags: [PyTorch, LayerNorm]
description: Curious about the numbers? Let's calculate and compare.
---

Today, I decided to manually calculate the values of LayerNorm.

<!-- truncate -->

We all know that the formula for LayerNorm is as follows:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\text{Var[}x\text{]} + \epsilon}} \times \gamma + \beta
$$

Where $\mu$ is the mean of $x$, and $\text{Var}$ is the variance of $x$.

With this information, let's calculate it ourselves, ignoring $\gamma$ and $\beta$ for simplicity:

```python
import torch

x = torch.rand(16, 768)
mu = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
eps = 1e-5
y = (x - mu) / (var + eps).sqrt()
```

This yields the following values:

```python
# tensor([[ 0.1219, -0.0222, -1.4742,  ...,  0.1738, -0.6124, -0.3001],
#         [-1.6009, -1.5814,  1.5357,  ...,  0.1917,  1.3787, -0.2772],
#         [ 0.3738,  1.0520,  0.4403,  ...,  1.1353, -0.7488, -0.9137],
#         ...,
#         [ 0.8823, -1.5427,  0.4725,  ..., -1.2544, -1.5354, -0.4305],
#         [ 1.4548,  0.3059, -0.6732,  ..., -0.7109,  0.4908, -1.2447],
#         [-0.4067,  0.5974, -0.9113,  ..., -0.2511, -0.2279, -0.9675]])
```

Next, let's compare these results with PyTorch's `torch.nn.LayerNorm`:

```python
layer_norm = torch.nn.LayerNorm(768, elementwise_affine=False, bias=False)

y_ln = layer_norm(x)
```

This yields:

```python
# tensor([[ 0.1220, -0.0222, -1.4752,  ...,  0.1739, -0.6128, -0.3003],
#         [-1.6020, -1.5824,  1.5367,  ...,  0.1918,  1.3796, -0.2774],
#         [ 0.3741,  1.0527,  0.4406,  ...,  1.1360, -0.7493, -0.9143],
#         ...,
#         [ 0.8829, -1.5437,  0.4728,  ..., -1.2552, -1.5364, -0.4308],
#         [ 1.4557,  0.3061, -0.6736,  ..., -0.7113,  0.4911, -1.2455],
#         [-0.4069,  0.5978, -0.9119,  ..., -0.2513, -0.2281, -0.9681]])
```

When we compare the two, why are they different?

## Unbiased Estimation

After a quick search, I found that `torch.var` has a parameter called `correction`, which defaults to `1`, meaning it uses an unbiased estimate.

This means that it divides by `N-1` instead of `N`, whereas `torch.nn.LayerNorm` uses `N`.

So let's modify the `torch.var` function by setting `correction=0`:

```python
var = x.var(dim=-1, correction=0, keepdim=True)
```

:::tip
`correction` is an alias for `unbiased`, introduced in PyTorch 2.0.0.

In earlier versions, you would set it using `unbiased=False`:

```python
var = x.var(dim=-1, unbiased=False, keepdim=True)
```

:::

Now, let's compare the results again:

```python
# tensor([[ 0.1220, -0.0222, -1.4752,  ...,  0.1739, -0.6128, -0.3003],
#         [-1.6020, -1.5824,  1.5367,  ...,  0.1918,  1.3796, -0.2774],
#         [ 0.3741,  1.0527,  0.4406,  ...,  1.1360, -0.7493, -0.9143],
#         ...,
#         [ 0.8829, -1.5437,  0.4728,  ..., -1.2552, -1.5364, -0.4308],
#         [ 1.4557,  0.3061, -0.6736,  ..., -0.7113,  0.4911, -1.2455],
#         [-0.4069,  0.5978, -0.9119,  ..., -0.2513, -0.2281, -0.9681]])
```

Now the numbers match perfectly!

## Why Doesn't LayerNorm Use Unbiased Estimation?

To summarize briefly, it's about stability and simplification of calculations.

If you're interested in a deeper dive, here are some key points:

- **Stability in Small Batch Calculations**

  LayerNorm is typically applied to the features of individual samples (e.g., each neuron or feature) rather than across the entire batch. The number of features per sample is usually much larger than the number of samples. Thus, using population variance provides a more stable and accurate estimate, especially with small sample sizes.

- **Reduced Importance of Unbiased Estimation**

  The unbiased nature of sample variance (i.e., dividing by n-1 instead of n) is crucial when estimating population parameters from a sample. However, in deep learning regularization and normalization, like in LayerNorm, the impact of this bias is relatively minor. This is because these computations are used to normalize activations rather than to estimate overall statistics. Using population variance simplifies calculations with minimal effect on training outcomes.

- **Stability in Gradient Calculations**

  Stable gradients are vital during backpropagation. Using population variance leads to smoother and more stable gradient calculations, avoiding additional noise that might arise from small sample sizes. This contributes to better convergence and training performance.

- **Simplified Calculations**

  From a computational perspective, calculating population variance is slightly simpler than sample variance because it omits a subtraction operation (i.e., dividing by n rather than n-1). While this isn't a decisive factor, it's a consideration in the design process.

## Conclusion

This article was inspired by a sudden curiosity.

I hope this explanation helps clarify the issue for you.
