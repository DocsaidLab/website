---
slug: whatsapp-online-but-no-listener
title: "OpenClaw × WhatsApp: A Runtime State Split Triggered by the Bundler"
authors: Z. Yuan
tags: [openclaw, whatsapp, debugging, javascript, bundling]
image: /img/2026/0315-whatsapp-no-listener.svg
description: An analysis of the state-consistency problem in OpenClaw's WhatsApp path.
---

I finally had a bit of free time, so I decided to try wiring OpenClaw into WhatsApp.

The basic flow went smoothly:
when a user sent a message from WhatsApp, the AI agent replied normally.

But when I tried to proactively push a message from OpenClaw to WhatsApp, the system kept responding with:

```
No active WhatsApp Web listener
```

What made it strange was that every other signal still looked healthy.

<!-- truncate -->

## The Problem

The overall symptom looked inconsistent:

- the gateway log showed that the WhatsApp inbound listener had started
- the dashboard opened normally
- inbound messages could still trigger agent replies
- but the active send path always returned:

```
No active WhatsApp Web listener
```

In other words:

- the monitor path could see the listener
- the send path believed the listener did not exist

That suggested the issue was not the WhatsApp connection, and not the gateway service itself. The problem was that listener state was being observed inconsistently inside the system.

## Initial Triage, But Not the Cause

At first, the most reasonable suspects were these:

- WhatsApp session invalidation
- a broken QR pairing flow
- the gateway service not starting correctly
- a lifecycle timing race condition around listener initialization

All of those were plausible, but they could not explain one key signal:

> the monitor explicitly recorded that a listener existed, yet the send path still reported that none existed.

That meant the listener state had not disappeared. It was being seen as different versions by different modules.

## Root Cause: Bundling Split the Runtime State

Inside OpenClaw's WhatsApp integration, there is a piece of shared state:

```
active web listener registry
```

By design:

- the monitor path registers the listener
- the send path reads the listener

In theory, both sides should share the same module state.

But after bundling, that assumption stopped being true.

In the build output:

- the monitor code and the send code landed in different bundle chunks
- the module-scoped store was initialized separately in each chunk

The result looked like this:

```
monitor chunk -> store A
send chunk    -> store B
```

Each side was locally reasonable:

- the monitor really did write the listener
- the send path really could not find the listener

But they were operating on two completely different copies of runtime state.

That is also why:

- the logs looked correct
- the error message looked correct too
- but the overall behavior was still inconsistent

## Why the First Patch Was Not Enough

The first patch was applied inside the Homebrew-installed copy of OpenClaw.

That was useful for validating the diagnosis, but it was not a maintainable fix.

The reason was simple:

- you are modifying installed artifacts
- package updates can overwrite the patch
- the next failure sends you back to patching it again

So the final fix had to move back into:

```
~/openclaw
```

The change needed to be made in the source tree and rebuilt there, so runtime behavior, source, and tests would stay aligned.

## Final Remediation Strategy

The remediation goal was straightforward:

> ensure that the monitor path and the send path always share the same listener store.

The fix was to move the module-scoped state into a global runtime store:

```javascript
const STORE_KEY = Symbol.for("openclaw.whatsapp.active-web-listener-store");
```

Then mount the listener registry on:

```
globalThis[STORE_KEY]
```

The benefits are:

- different bundle chunks still resolve the same Symbol key
- module reload does not reinitialize the state
- as long as execution stays inside the same JavaScript runtime, it shares the same store

In other words:

```
module state   -> unreliable
global runtime -> stable
```

## Regression Test

When a runtime-state bug is fixed without tests, it is easy for it to come back during a later build change.

So this change also added regression coverage for:

- module reload
- lazy-load boundaries
- bundle chunk boundaries

The tests ensure that:

- the listener registry always resolves to the same store
- the send path does not silently obtain a fresh registry

## A Verification Trap

There is an easy source of false confidence when verifying this kind of bug.

`openclaw message send` is not the best smoke test.

The reason is:

- the CLI command starts its own process
- the send path is lazy-loaded

So it can only prove:

```
the CLI process can find a listener
```

It does not necessarily prove:

```
the long-running gateway service has recovered its listener sharing
```

A more accurate verification path is to call:

```
gateway send RPC
```

That was the path used for the final smoke test in this case.

## A Broader Engineering Lesson

When a system shows all of the following at the same time:

- the monitor is visible
- the service is alive
- the active operation fails
- the shared object is missing

the problem is often not in the transport layer, but in runtime state ownership.

The boundaries worth checking first are:

- process boundaries
- lazy-loading boundaries
- bundle chunk boundaries
- global state boundaries

On the surface, this kind of failure looks like a network problem or a session problem. In practice, it is often a case where state got duplicated or reinitialized across different runtime contexts.

If runtime state and module boundaries move earlier in the debugging order, diagnosis usually gets much faster.
