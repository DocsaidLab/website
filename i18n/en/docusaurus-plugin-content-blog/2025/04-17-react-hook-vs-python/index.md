---
slug: react-hook-vs-python
title: What exactly is React Hook hooking?
authors: Z. Yuan
image: /en/img/2025/0417.jpg
tags: [react, hook]
description: Deconstructing the concepts behind React Hook.
---

A few years ago, when I first started learning React, I felt like React was living in a parallel universe and was hard to understand.

Look at this Hook—doesn’t it seem counterintuitive?

<!-- truncate -->

## Starting with Python

You might be like me, more comfortable with Python, so let’s start there!

Let's take a look at the simplest counter. In React, we would write it like this:

```jsx
import { useState } from "react";

function Counter() {
  const [count, setCount] = useState(0);

  function handleClick() {
    setCount(count + 1);
  }

  return <button onClick={handleClick}>Clicked! Count = {count}</button>;
}

export default Counter;
```

If you had to write the counter in Python, how would you do it?

You think it’s simple? Then let’s do it using OOP!

## Traditional OOP Approach

Here’s how I would write it. This problem itself is not difficult:

```python
class Counter:

    def __init__(self):
        self.count = 0

    def handle_click(self):
        self.count += 1
        self.render()

    def render(self):
        print(f"Clicked! Count = {self.count}")


# Usage
counter = Counter()
counter.handle_click()  # Click once
counter.handle_click()  # Click again
```

But this approach doesn’t quite match up with the React example, so let’s try a different method:

## Functional + State Binding

Instead of using classes, let’s define a function that separates the "state" and "render logic," using pure functions and closures to manage the state:

```python
def use_state(initial_value):

    state = {"value": initial_value}

    def get_value():
        return state["value"]

    def set_value(new_value):
        state["value"] = new_value

    return get_value, set_value
```

By now, does this feel a bit more familiar?

Next, let’s create a state:

```python
get_count, set_count = use_state(0)
```

This is pure Python now—don’t tell me I’m writing React anymore.

Now, we have the `get_count` and `set_count` functions, and we’ll write a "click event" function:

```python
def render():
    print(f"Clicked! Count = {get_count()}")

def handle_click():
    set_count(get_count() + 1)
    render()
```

Let’s simulate two clicks:

```python
handle_click()
handle_click()
```

The output is:

```python
Clicked! Count = 1
Clicked! Count = 2
```

Hmm? Isn’t this exactly what we did in React just now?

## OOP vs Functional

You might ask, why doesn’t React use OOP?

Isn’t using classes more intuitive? Like the `Counter` class we just wrote in Python, the state is clear at a glance!

Let’s revisit the previous example and see the actual difference between these two approaches when it comes to **state management**.

Writing a counter in Python with OOP is like building a desktop application, where the entire component is wrapped in a class:

- `self.count` is the state.
- `handle_click` is the event handler.
- `render` is the view update.

The advantage here is that the structure is clear, and the behavior is well-encapsulated.

But here’s the problem: as the component grows more complex, you may need to manage multiple states (e.g., a countdown, background color, error messages, etc.), all of which are stored in `self`. Before long, your code starts to look like this:

```python
self.count
self.timer
self.error_message
self.loading
self.is_dark_mode
self.is_visible
self.is_editable
self.is_submitting
self.is_valid
self.is_disabled
```

And that’s when you start questioning life.

## Functional Thinking

React uses Hooks to break state logic down into smaller functions, letting you “assemble” the component’s functionality.

Remember our `use_state`?

```python
get_count, set_count = use_state(0)
```

Now, you have two functions: `get_count()` and `set_count()`, which you can use anywhere, without needing a class to store this information.

**This is the essence of Hooks: the state is tied there, and as long as you can access `get_count()` or `set_count()`, you can retrieve and update it anytime. It’s like holding a hook (Hook) that you can use to grab that state and pull it out whenever you need it.**

Don’t believe me? Let’s try adding a feature: "remind every 5 clicks."

- **With OOP**

  ```python
  class Counter:

      def __init__(self):
          self.count = 0

      def handle_click(self):
          self.count += 1
          if self.count % 5 == 0:
              print("🎉 You clicked 5 times!")
          self.render()

      def render(self):
          print(f"Clicked! Count = {self.count}")
  ```

- **With Functional**

  ```python
  get_count, set_count = use_state(0)

  def use_celebrate_every(n):
      def check():
          if get_count() % n == 0 and get_count() != 0:
              print(f"🎉 You clicked {n} times!")
      return check

  celebrate = use_celebrate_every(5)

  def handle_click():
      set_count(get_count() + 1)
      celebrate()
      render()

  def render():
      print(f"Clicked! Count = {get_count()}")
  ```

  Here, we’ve separated the "reminder every n clicks" logic into its own Hook: `use_celebrate_every(n)`, which you can use for any component that needs "periodic reminders."

  This is the power of functional programming: you break down functionality into modular blocks, making it easier to decompose, recombine, and reuse.

---

But functional programming does seem more complex!

Don’t worry, let’s look at another example.

In fact, if you’re just writing a simple button, OOP is far more intuitive than functional programming.

But React isn’t designed for you to just “write a button”—it’s built to let you “assemble an entire application.”

So the real question isn’t, “Which is simpler?”

It’s, “**When things get complicated, which one can still hold up?**”

## Think of it in a Different Scenario

Let’s say you start by writing a simple component like this:

```jsx
function LoginForm() {
  const [username, setUsername] = useState("");
  return (
    <input value={username} onChange={(e) => setUsername(e.target.value)} />
  );
}
```

No problem, nice and simple.

But then the product manager comes along, and says:

1. Show a loading animation when logging in.
2. The error message should disappear automatically after 3 seconds.
3. Remember the user's username and store it in localStorage.
4. Auto-focus the password field after the username is entered.
5. Lock the account for 30 seconds after 3 consecutive failed attempts.

Now you need to handle:

- Loading state
- Error message display and auto-clear
- Side effects: localStorage operations, focus control
- Time logic: account lock timer

## What if You Use OOP?

You might write it like this:

```python
class LoginForm:

    def __init__(self):
        self.username = ""
        self.loading = False
        self.error_message = ""
        self.failed_attempts = 0
        self.lock_until = None
        # Then you write a bunch of methods: handle_input, handle_submit, handle_error...
```

The states are all mixed together, and the logic is spread across ten different methods. And if something goes wrong? It’ll blow up in your face.

Want to reuse the "error message disappearing after 3 seconds" behavior? Sorry, that’s private to the `LoginForm` class, so no one else can reuse it. You’ll either need to rewrite the same code or directly extend that class.

## But What if You Use Functional Programming?

You could break it down like this:

```jsx
const [username, setUsername] = useState("");
const [loading, setLoading] = useState(false);
const errorMessage = useTimedError(3000); // Error message shows for 3 seconds
useLocalStorage("last-username", username); // Automatically store the username
useAutoFocus(ref); // Auto-focus the input field
useLoginRateLimit(); // Limit the number of failed login attempts
```

Each requirement is encapsulated into its own **Hook module**, with no entanglements between them.

It’s like putting together building blocks, rather than untying knots.

Doesn’t that feel much easier?

## Why Give Up OOP?

In fact, React initially did have `class` components. You might still remember writing this:

```jsx
class Counter extends React.Component {
  state = { count: 0 };
  handleClick = () => this.setState({ count: this.state.count + 1 });
  render() {
    return (
      <button onClick={this.handleClick}>Count = {this.state.count}</button>
    );
  }
}
```

But once you have a need for **shared logic** (like form validation, detecting device sizes, making API requests, etc.), these behaviors can’t be broken down into reusable modules. You’re forced to use **higher-order components** or **pass functions into functional components** to break things apart. The result? The code gets messier and messier, and you end up feeling like you're writing Java.

No offense to Java, of course.

## So

You say functional programming looks complicated? Yes, at first it is.

But the complexity of a single button is not the same as the complexity of a complete application.

**The design of Hooks is not meant to make "getting started easier," but to ensure that "it doesn't break when things scale up."**

When you’ve written the 8th component that needs "auto-save," "error handling," and "window detection," you’ll be glad you used Hooks, rather than cramming everything into a weird object.
