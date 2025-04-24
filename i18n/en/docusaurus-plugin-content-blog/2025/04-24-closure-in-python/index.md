---
slug: closure-in-python
title: What is Closure?
authors: Z. Yuan
image: /en/img/2025/0424.jpg
tags: [python, closure]
description: A simple introduction to the concept of Closure.
---

When writing code, you may occasionally come across the term "Closure."

It's not a foreign concept. We use it often, though we may not always recognize it by name.

<!-- truncate -->

## Functions as First-Class Objects

In Python, functions are not just syntactic tools; they are full-fledged objects.

You can:

- Assign a function to a variable
- Pass it as an argument
- Return it from another function

```python
def greet(name):
    return f"Hello, {name}"

say_hello = greet
print(say_hello("Alice"))
# => Hello, Alice
```

This means that functions can be treated like data, allowing them to be manipulated and combined with other logic to form modular behavior units.

## Defining Functions Within Functions

Python allows defining functions inside other functions, creating a nested structure:

```python
def outer():
    def inner():
        print("Hello from the inside")
    inner()
```

Here, `inner()` only lives within the scope of `outer()` and cannot be directly called from the outside.

But we can rewrite it in a way that changes its destiny:

- Passing the function as a result

  ```python
  def outer():
      def inner():
          print("I’m still alive.")
      return inner

  escaped = outer()
  escaped()  # => I’m still alive.
  ```

In this code, even though `outer()` has finished, `inner()` can still be called.

The reason is that when `inner()` was "brought out," it carried its execution context along.

## Closure

Now, we come to the main topic.

A Closure is a language feature that allows a function to capture variables from its outer scope and still access them after the function has finished executing.

Here’s an example:

```python
def make_multiplier(factor):
    def multiply(n):
        return n * factor
    return multiply

triple = make_multiplier(3)
double = make_multiplier(2)

print(triple(10))  # 30
print(double(10))  # 20
```

In this case, `factor` is a free variable in `multiply()`:

- **It is not defined inside `multiply()`, but is used within it.**

Even after `make_multiplier()` finishes executing, the `factor` does not disappear.

It is "packaged" inside `multiply()` and returned along with it.

This combination is what we call a **Closure**.

## How to Identify a Closure?

You can observe this through the function's `__closure__` attribute:

```python
>>> triple.__closure__
(<cell at 0x...: int object at 0x...>,)

>>> [c.cell_contents for c in triple.__closure__]
[3]
```

- `__closure__` lists the free variables captured in the function
- `cell_contents` retrieves the actual content of these variables

This is not some mysterious phenomenon, but simply a natural consequence of the language's features.

## Common Uses and Scenarios

- **Function Factories**: Creating custom functions with state based on input parameters
- **Counters/Caches**: Retaining limited state without needing a full class
- **Decorators (`@decorator`)**: A common implementation technique is based on the Closure structure
- **Dependency Injection**: Binding data implicitly to avoid polluting global state

When you need to **retain a small amount of state** and don’t want to use full object-oriented design, Closure is a perfect tool.

## Summary

Closure is not hard to understand. Its essence is simply:

1. **Capture**: Saving the value of free variables
2. **Package**: Bundling them together with the function itself
3. **Continue**: Even after the original scope is gone, the function can still operate normally

When you encounter `__closure__`, don’t be surprised. It’s just a preserved version of the environment at that moment, holding onto the data state.

These values act as memory snapshots of your program, traveling along with the function.
