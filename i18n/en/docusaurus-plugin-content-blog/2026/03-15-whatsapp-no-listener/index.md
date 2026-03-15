---
slug: whatsapp-online-but-no-listener
title: WhatsApp Looks Online. Why Does OpenClaw Still Say There Is No Listener?
authors: Z. Yuan
tags: [openclaw, whatsapp, debugging, javascript, bundling]
image: /img/2026/0315-whatsapp-no-listener.svg
description: The gateway said WhatsApp was listening, yet outbound send returned No active WhatsApp Web listener. The connection was not dead. Shared state had been split across bundle chunks.
---

It looked online.

The log said it was listening.

You hit send.

It replied:

```text
No active WhatsApp Web listener
```

This kind of bug is annoying because every side appears to be telling the truth.

The monitor says it registered a listener.

The send path says it cannot find one.

Neither side is lying.

They just live in different universes.

> **The problem was not a dead WhatsApp connection. The same runtime state had been split into two copies by the bundler.**

This is the cleaned-up version of the issue, the fix, and why the final repair had to move back into `~/openclaw`.

<!-- truncate -->

## Start with the symptom: it looked alive, but outbound send still failed

The behavior was contradictory:

- the gateway log said WhatsApp inbound monitoring was active
- the dashboard still opened normally
- outbound send still returned `No active WhatsApp Web listener`

If you stop at the surface, you end up chasing the usual suspects:

- maybe the WhatsApp session died
- maybe QR pairing broke
- maybe the gateway service was only half alive
- maybe a restart race left something in a bad state

Those are all reasonable guesses.

They were just wrong this time.

## The real cause: the listener got split in two

There is a shared piece of state in the WhatsApp path: the currently active web listener.

In theory:

- the monitor path registers the listener
- the send path reads it back and uses it for outbound send

The trouble started after bundling.

The monitor path and send path landed in different chunks.

Both imported `active-listener`, but after build they no longer shared the same module state.

So the runtime ended up like this:

- chunk A had the real listener
- chunk B had an empty registry
- the log message was true in its world
- the send error was also true in its world

This was not a normal overwrite bug.

It was closer to making two copies of the same office whiteboard, then wondering why both teams kept reporting different reality with a straight face.

## Why the earlier Homebrew-side fix was not enough

The first round of mitigation happened inside the Homebrew-installed copy.

That was useful for proving the diagnosis.

It was not a stable place to stop.

The reason is boring and important:

- you are patching installed output
- package updates can wipe the fix
- the next regression sends you back to the same chase

So the real job was not just "make it send again."

It was to move the main system back into the `~/openclaw` repo, build from source, and run the service from the maintained codebase.

That way:

- the fix does not disappear during package upgrades
- the regression test lives next to the code
- the gateway is actually running the code you just inspected

## The fix was simple: move shared state from the module to `globalThis`

Once the problem became "chunk boundaries do not share module state," the fix stopped being mysterious.

We moved the active-listener registry onto `globalThis`, keyed by `Symbol.for(...)`.

The idea is plain:

- no matter which chunk the monitor path comes from
- no matter which chunk the send path comes from
- if they are in the same JavaScript runtime
- they reach the same store

The core direction looked like this:

```ts
const STORE_KEY = Symbol.for("openclaw.whatsapp.active-web-listener-store");
```

Then the old module-local singleton became a shared store initialized off `globalThis`.

Not glamorous.

Just effective.

## The code fix needed a test with teeth

Bugs like this are slippery because they can look solved until the next bundling change or lazy-load adjustment quietly reintroduces them.

So this was not only a code patch in `active-listener.ts`.

It also got a regression test.

The test does more than verify set/get behavior.

It checks that:

- the store survives module reload boundaries
- the shared listener still points at the same backing state
- the send path does not accidentally grow a second universe later

In other words, the exact hole we fell into now has a tripwire.

## Validation has a trap too: `message send` is not the right proof

One more detail turned out to matter.

`openclaw message send` is not the best way to prove that the main-system WhatsApp push path is fixed.

Not because it never works.

Because it exercises the CLI process and its own lazy-loaded outbound path.

That means you can accidentally prove only this:

- the CLI process can send
- the long-running gateway service is still wrong

If the question is "is the main service fixed," the better check is the gateway `send` RPC.

That is the path we used for the final smoke test, and that is the path that confirmed the repo-based main system was actually healthy again.

## The broader lesson is not really about WhatsApp

This bug looked like transport failure.

It was not.

It was a runtime-boundary bug:

- process boundary
- lazy-loading boundary
- bundle chunk boundary
- global-state boundary

These bugs have a particular personality:

- the logs are often not false
- they are just reporting from different worlds
- only when you line them up do you realize the state was never shared

So the next time a system looks like this:

- "it is online"
- "the monitor is running"
- "the service is alive"
- "but this one active operation says the shared object does not exist"

do not blame the transport first.

Check whether the state actually crosses the boundary you think it does.

Quite often the ghost is not in the network.

It is in your own runtime.
