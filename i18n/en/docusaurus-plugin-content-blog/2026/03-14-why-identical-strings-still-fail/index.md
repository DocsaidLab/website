---
slug: why-identical-strings-still-fail
title: They Look the Same. Why Do Strings Still Fail to Match?
authors: Z. Yuan
image: /en/img/2026/0314-string-compare-unicode.svg
tags: [unicode, python, text-processing]
description: Two strings can look identical and still fail to match. The usual suspects are Unicode, invisible characters, and misplaced trust in computers.
---

You have probably seen this before:

Two strings look exactly the same, and the comparison still fails.

Then you stare at the screen for five minutes and begin to wonder whether your eyes have stopped working.

Usually, your eyes are fine.

The computer is just being painfully honest.

<!-- truncate -->

For humans, “looks the same” is often good enough.

For code, it is not.

Code does not care about vibes. It cares about:

- code points
- byte sequences
- normalization forms
- whether invisible characters are hiding inside the string

If any of those differ, the answer may simply be:

> **Different is different.**

Cold, yes.

Wrong? Not really.

## The classic example: `é`

Take these two strings:

```python
s1 = "café"
s2 = "cafe\u0301"

print(s1 == s2)
```

A lot of people expect `True`.

In practice, you usually get:

```python
False
```

Why? Because these two versions of `é` are not represented the same way:

- `é`: a single code point
- `e` + `◌́`: the letter `e` followed by a combining acute accent

They look the same on screen.

They are not the same string underneath.

## Why does this happen?

Because Unicode is not just “what character looks like what.”

It is a system that defines:

- how characters are assigned numbers
- how they can be combined
- how different platforms can represent them

There are three layers worth separating.

### 1. Code point

Unicode assigns an identifier to each character, for example:

- `A` → `U+0041`
- `é` → `U+00E9`

Think of this as the character’s ID card.

### 2. Grapheme

What a user sees as one visible character is not always one code point.

The `e` plus accent example is a classic case.

Humans see one character.

Your program may see two pieces.

### 3. Encoding

Once strings become bytes, you still have encoding involved: UTF-8, UTF-16, and so on.

So “looks the same” can fail at multiple layers.

## The usual traps are not limited to accents

This problem is not only about French text or unusual symbols. Plenty of ordinary data can trigger it.

### 1. Full-width vs half-width characters

```python
s1 = "ABC123"
s2 = "ＡＢＣ１２３"

print(s1 == s2)  # False
```

To humans, this is the same text wearing a wider coat.

To a program, these are different characters.

### 2. Invisible characters

The worst characters are often the ones you cannot see.

For example:

- zero-width spaces
- non-breaking spaces
- directional marks
- control characters copied from web pages or office documents

Once these get into your data, the text still looks clean.
Your comparison logic, however, starts having opinions.

### 3. Case conversion is not always as simple as you think

A lot of people assume case-insensitive comparison means `lower()` and move on.

Not always.

Unicode covers far more than English, and some languages have case rules that are less obedient.

If you actually want Unicode-aware case-insensitive comparison, this is usually closer to what you want:

```python
text.casefold()
```

Not just `lower()`.

## The fix: normalize first, compare second

The standard solution here is **Unicode normalization**.

Python already gives you the tool:

```python
import unicodedata

s1 = "café"
s2 = "cafe\u0301"

n1 = unicodedata.normalize("NFC", s1)
n2 = unicodedata.normalize("NFC", s2)

print(n1 == n2)  # True
```

Now both strings are converted to the same normalized form before comparison.

At that point, the computer finally starts behaving like a reasonable colleague.

## What do NFC, NFD, NFKC, and NFKD actually mean?

The names look unpleasant at first, but they answer only two questions:

1. should characters be decomposed?
2. should compatibility transformations be applied?

### 1. NFC

**Canonical Composition**

It prefers combining decomposed sequences when possible.

For example:

- `e` + accent → `é`

This is usually the safest and most common choice.

If your goal is:

- storing text
- doing stable comparisons
- preserving meaning

then `NFC` is a very reasonable default.

### 2. NFD

**Canonical Decomposition**

It breaks combined characters into components.

This is more useful in specialized text-processing workflows where you actually want to operate on the pieces.

Most business systems do not use it as the default storage form.

### 3. NFKC

**Compatibility Composition**

This goes beyond canonical normalization and also applies compatibility-level transformations.

That means some things such as:

- full-width characters
- compatibility symbols
- visually similar forms that Unicode considers foldable

may be collapsed into a more unified representation.

This is powerful.

It is also not harmless.

It is useful for:

- search indexing
- cleaning user input
- usernames or identifiers where you want more aggressive normalization

But if you are dealing with:

- legally sensitive text
- layout-sensitive content
- data that must preserve original form exactly

then do not reach for `NFKC` casually.

### 4. NFKD

This is the decomposed version of compatibility normalization.

Unless you know exactly why you need it, it is usually not your first choice.

## A more realistic text-cleaning pipeline

In practice, string comparison often needs more than normalization alone.

You may also want:

- Unicode normalization
- case folding
- whitespace cleanup
- removal of invisible formatting characters

For example:

```python
import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


s1 = " Docsaid\u00A0Lab "
s2 = "docsaid lab"

print(normalize_text(s1) == normalize_text(s2))  # True
```

That is already far more reliable than `strip().lower()`.

At least now the confidence is somewhat deserved.

## But do not normalize everything

A common overcorrection looks like this:

> “Normalization works well. I will apply it to every field.”

No.

Some data should not be touched.

For example:

- passwords
- tokens
- signed payloads
- pre-hash source text
- any field that must preserve byte-level fidelity

If you normalize those, you may quietly break the entire pipeline.

Some systems do not fail because comparison is hard.
They fail because someone helpfully “cleaned” the data first.

Engineering has produced many bugs this way.

## So which one should you use?

If you do not want to memorize the spec, remember this rough rule:

- **general text storage / display**: start with `NFC`
- **search, usernames, user-input comparison**: consider `NFKC` + `casefold()`
- **security-sensitive data**: do not normalize casually
- **if matching fails even though text looks identical**: suspect Unicode before suspecting your sanity

That ordering saves time.

## How do you debug this quickly?

When you suspect a string is hiding something, do not just `print(text)`.

That is often useless.

Inspect the representation directly:

```python
text = "cafe\u0301"

print(repr(text))
print([hex(ord(ch)) for ch in text])
```

You will get something like:

```python
'cafe\u0301'
['0x63', '0x61', '0x66', '0x65', '0x301']
```

At that point, you know the database is not targeting you personally and Python has not developed attitude.

There really is a combining mark in the string.

## Final words

String matching failures are often not a sign that your logic is complicated.

They are a sign that you assumed “visually identical” meant “structurally identical.”

That is a reasonable assumption for humans.
It is not a reasonable assumption for computers.

Computers do not fill in the blanks for you.

They quietly return `False` and let you experience character development.

So if you are dealing with any of these:

- copied text from web pages that never matches
- usernames that look identical but fail lookup
- multilingual text behaving strangely in search or deduplication
- a comparison pipeline built on `lower().strip()` and optimism

then the next step is probably not another `if` statement.

It is Unicode normalization.

That looks more like debugging and less like prayer.

## References

- [Unicode Standard Annex #15: Unicode Normalization Forms](https://unicode.org/reports/tr15/)
- [Python `unicodedata` Documentation](https://docs.python.org/3/library/unicodedata.html)
- [Python `str.casefold`](https://docs.python.org/3/library/stdtypes.html#str.casefold)
