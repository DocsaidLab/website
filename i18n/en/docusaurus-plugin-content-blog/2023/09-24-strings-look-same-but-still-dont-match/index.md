---
slug: strings-look-same-but-still-dont-match
title: They Look the Same. Why Does String Matching Still Fail?
authors: Z. Yuan
date: 2023-09-24T09:56:27+08:00
tags: [unicode, python, javascript, text-processing, debugging]
image: /img/2023/0924-unicode-string-traps.svg
description: Two strings can look identical and still fail comparison. The usual suspects are Unicode normalization, invisible characters, and excessive trust in your own eyes.
---

You look at two strings.

They look identical.

You compare them with `==`.

It fails.

At that point people usually go through three stages:

1. suspect their eyesight
2. suspect the encoding
3. suspect the universe

Usually the universe is innocent.

The real problem is simpler:

> **visual equality is not the same thing as identical code points or identical bytes.**

This post covers the usual traps:

1. same glyph, different Unicode composition
2. invisible characters mixed into the text
3. full-width vs half-width characters, strange dashes, strange spaces
4. why `trim()` helps less than people hope
5. when to normalize, and when normalization is the wrong move

Examples use both Python and JavaScript, because both can hurt you here. They just have different manners.

<!-- truncate -->

## First principle: same appearance does not mean same code points

A classic example is `é`.

It can be represented as:

- one character: `U+00E9`
- or `e` plus a combining acute accent: `U+0301`

They render the same.

They are not the same sequence.

### Python

```python
s1 = "é"
s2 = "e\u0301"

print(s1 == s2)          # False
print(len(s1), len(s2))  # 1 2
print([hex(ord(c)) for c in s1])
print([hex(ord(c)) for c in s2])
```

### JavaScript

```js
const s1 = "é";
const s2 = "e\u0301";

console.log(s1 === s2); // false
console.log(s1.length, s2.length); // 1 2
console.log([...s1].map(ch => ch.codePointAt(0).toString(16)));
console.log([...s2].map(ch => ch.codePointAt(0).toString(16)));
```

If you only inspect the rendered text, this looks absurd.

From the computer’s perspective, it is perfectly normal.

## Fix 1: normalize Unicode before comparison

This is not magic, but it is often the first correct step.

Common forms are:

- `NFC`: composed standard form
- `NFD`: decomposed form
- `NFKC`: compatibility normalization, more aggressive
- `NFKD`: compatibility decomposition

For ordinary text matching, start with `NFC`.

### Python

```python
import unicodedata

s1 = "é"
s2 = "e\u0301"

n1 = unicodedata.normalize("NFC", s1)
n2 = unicodedata.normalize("NFC", s2)

print(n1 == n2)  # True
```

### JavaScript

```js
const s1 = "é";
const s2 = "e\u0301";

console.log(s1.normalize("NFC") === s2.normalize("NFC")); // true
```

### When should you use `NFKC`?

Useful cases include:

- forgiving search
- usernames, labels, or identifiers where you want to collapse input variants
- folding full-width Latin letters and digits into ASCII forms

Example:

```python
import unicodedata

print(unicodedata.normalize("NFKC", "ＡＢＣ１２３"))
# ABC123
```

Convenient, yes.

Also dangerous.

`NFKC` does more than standardize representation. It may fold compatibility characters into the same canonical shape.

That is great for search.

It can be terrible for passwords, signatures, legal text, and anything that must preserve the exact original input.

So the short rule is:

- **search / loose matching**: `NFKC` can be reasonable
- **storage / security-sensitive comparison**: usually `NFC`, or even preserve the original exactly

## Fix 2: inspect invisible characters instead of trusting your eyes

Another common failure mode is hidden characters such as:

- zero-width space `U+200B`
- no-break space `U+00A0`
- word joiner `U+2060`
- BOM `U+FEFF`
- tabs, carriage returns, and odd line separators

These often arrive from:

- copied web content
- Excel or Word exports
- IMEs
- OCR pipelines
- third-party APIs

Example:

```python
s1 = "token=abc123"
s2 = "token=abc123\u200b"

print(s1 == s2)  # False
print(repr(s2))
```

If you do not print `repr()`, you may not even notice the extra character.

### Debug helpers

#### Python

```python
def inspect_string(s: str):
    for i, ch in enumerate(s):
        print(i, hex(ord(ch)), repr(ch))
```

#### JavaScript

```js
function inspectString(s) {
  [...s].forEach((ch, i) => {
    console.log(i, "U+" + ch.codePointAt(0).toString(16).toUpperCase(), JSON.stringify(ch));
  });
}
```

This is not elegant.

It is effective.

In debugging, effective beats elegant very quickly.

## `trim()` helps, but it is not a religion

A lot of people respond to string bugs with:

- Python: `s.strip()`
- JavaScript: `s.trim()`

Useful, yes.

Sufficient, no.

Why not?

1. it only touches the edges, not the middle
2. it does not solve composed vs decomposed Unicode
3. it does not normalize different dash-like or space-like characters the way you might expect

## Not every dash is `-`

In practice you will see all of these:

- Hyphen-minus: `-` (`U+002D`)
- Non-breaking hyphen: `‑` (`U+2011`)
- En dash: `–` (`U+2013`)
- Em dash: `—` (`U+2014`)
- Minus sign: `−` (`U+2212`)

Humans read “dash”.

Parsers do not.

If your regex, parser, file naming rule, or split logic only accepts ASCII `-`, the others will break it.

If a field is supposed to be ASCII-only by design, such as:

- slugs
- internal IDs
- command-line options
- file naming conventions

then do not pretend it is flexible. Reject invalid input explicitly.

That is usually cheaper than trying to infer user intent after the fact.

## A safer data-cleaning pattern: preserve the original, derive a canonical form

This is the pattern I trust more.

Do not immediately rewrite user input into something else and hope for the best.

A more stable approach is:

1. **preserve the raw original input**
2. create a **canonical form** for search, deduplication, or loose matching
3. make the rules explicit and reproducible

Python example:

```python
import unicodedata


def canonicalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00A0", " ")
    text = text.replace("\u200B", "")
    text = text.strip()
    return text
```

If you need a looser search key:

```python
import re
import unicodedata


def search_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

These are different tools for different jobs.

Mix them carelessly and you get a bug farm.

## For Unicode-insensitive matching, `casefold()` is usually better than `lower()`

In Python, `casefold()` is often more appropriate than `lower()` for case-insensitive text matching.

```python
print("Straße".lower())
print("Straße".casefold())
```

Output:

```text
straße
strasse
```

JavaScript does not give you a direct `casefold()` equivalent. Usually you are limited to:

- `toLowerCase()`
- `toLocaleLowerCase()`
- plus your own normalization rules

So for serious multilingual matching, do not rely on ad hoc frontend logic alone. Put canonicalization rules in the backend and keep them consistent.

## Do not normalize passwords, signatures, or tokens just because you can

This deserves its own section.

A common overreaction is:

> “Fine, I will normalize everything.”

That usually turns a visible bug into a subtler one.

The following data should not be “helpfully” normalized in a loose way:

- passwords
- HMAC or API signatures
- JWTs or tokens
- hash inputs
- legally or operationally sensitive original text

For those fields you want:

- exact byte stability
- explicit rules
- no silent reinterpretation

You can warn the user.

You can detect suspicious characters.

You should not quietly rewrite the input and pretend that was safe.

## A practical debugging checklist

When two strings look the same but comparison fails, I usually do this:

1. print `repr()` or `JSON.stringify()`
2. list every code point
3. check for zero-width and unusual whitespace characters
4. compare again after `NFC`
5. decide whether the field semantics allow `NFKC`
6. centralize the rule in one shared function instead of re-inventing it everywhere

A lot of Unicode bugs are not caused by Unicode being impossibly complicated.

They are caused by teams doing four different “small fixes” in four different places.

That is democratic.

It is also how string handling becomes an operational problem.

## Two practical helpers

### Python

```python
import re
import unicodedata

ZERO_WIDTH = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\ufeff",
}


def clean_for_search(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch not in ZERO_WIDTH)
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

### JavaScript

```js
function cleanForSearch(text) {
  return text
    .normalize("NFKC")
    .replace(/[\u200B\u200C\u200D\uFEFF]/g, "")
    .toLocaleLowerCase("en-US")
    .replace(/\s+/g, " ")
    .trim();
}
```

This is not universal truth.

It is, however, a lot better than pretending raw `==` is a text-processing strategy.

## Summary

String bugs are annoying because the data often looks fine.

But once code points differ, invisible characters slip in, or normalization rules vary across the stack, your system starts improvising.

The useful rules are not:

- “just trim it”
- “just lowercase it”
- “just normalize everything with `NFKC`”

The useful rules are:

1. **inspect the actual characters**
2. **choose cleaning strength based on field semantics**
3. **preserve the original and derive a canonical form separately**
4. **centralize the rule instead of scattering string folklore across the codebase**

Computers are not being dramatic here.

They are simply refusing to guess what you meant.

Cold behavior, perhaps.

Also professional.
