---
slug: update-docusaurus-to-3-6-0
title: Update Docusaurus to 3.6.0
authors: Zephyr
image: /en/img/2024/1109.webp
tags: [Docusaurus, Update]
description: Troubleshooting issues during the update process
---

Docusaurus has released version 3.6.0, which includes updates to the bundling tool, significantly speeding up build times.

However, we ran into some issues during the update!

<!-- truncate -->

## Update Details

If you're not familiar with the recent release, you can check out the latest blog post from the Docusaurus team:

- [**Docusaurus 3.6**](https://docusaurus.io/blog/releases/3.6)

  <iframe
    src="https://docusaurus.io/blog/releases/3.6"
    width="80%"
    height="300px"
    center="true"
    ></iframe>

## Issue Description

The update itself proceeded without issues, but Docusaurus introduced a new feature in this version that allows adding a config setting:

```js title="docusaurus.config.js"
const config = {
  future: {
    experimental_faster: true,
  },
};
```

When we added this setting to our `docusaurus.config.js` file, we encountered the following error:

```shell
yarn run v1.22.22
$ docusaurus start
[INFO] Starting the development server...
[SUCCESS] Docusaurus website is running at: http://localhost:3000/
● Client ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ (83%) sealing chunk ids
Segmentation fault (core dumped)
error Command failed with exit code 139.
info Visit https://yarnpkg.com/en/docs/cli/run for documentation about this command.
```

---

When we saw this error, it was frustrating.

A simple `Segmentation fault`? That’s all? Really?

## Troubleshooting

Since we couldn't find any solutions in the official issue tracker, we had to troubleshoot this manually.

After some investigation, we discovered that the issue stems from using certain Chinese characters in `_category_.json` files.

Or, more accurately, specific Chinese characters cause the issue, though we’re unsure exactly which ones.

For example, one of our files originally looked like this:

```json title="_category_.json"
{
  "label": "元富證券",
  "position": 1,
  "link": {
    "type": "generated-index"
  }
}
```

Changing `元富證券` to `中文` allowed the project to run successfully!

:::tip
Both labels are in Chinese. Why does one work while the other doesn’t?
:::

Replacing `中文` with an English label also worked:

```json title="_category_.json"
{
  "label": "English Label",
  "position": 1,
  "link": {
    "type": "generated-index"
  }
}
```

## Additional Issues

We also discovered another issue: the new setting doesn’t support special characters in file names.

For instance, one of our files was named `Bézier`, which caused an error due to the accented character.

After removing the accent, everything ran smoothly.

## Conclusion

In the end, we decided not to enable this new feature.

Since our website is relatively small, build speed isn’t a major bottleneck, but this feature would require us to change multiple files.

Maybe we’ll revisit it later!

## 2024-11-20 Update

The official update notification has been received, and this time the version has been updated to **v3.6.2**, which resolves the issues mentioned in the previous update.

In this version, we can now successfully use the `experimental_faster` setting:

```js title="docusaurus.config.js"
const config = {
  future: {
    experimental_faster: true,
  },
};
```

Testing shows that the **Segmentation fault** issue no longer occurs.

However...

After starting the development environment, modifying files triggers the following error:

```shell
Panic occurred at runtime. Please file an issue on GitHub with the backtrace below: https://github.com/web-infra-dev/rspack/issues
Message:  Chunk(ChunkUkey(Ukey(606), PhantomData<rspack_core::chunk::Chunk>)) not found in ChunkByUkey
Location: crates/rspack_core/src/lib.rs:328

Run with COLORBT_SHOW_HIDDEN=1 environment variable to disable frame filtering.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ BACKTRACE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 1: start_thread
    at ./nptl/pthread_create.c:447
 2: clone3
    at ./misc/../sysdeps/unix/sysv/linux/x86_64/clone3.S:78
Aborted (core dumped)
error Command failed with exit code 134.
info Visit https://yarnpkg.com/en/docs/cli/run for documentation about this command.
```

It seems to be an issue with Rspack. We quickly found a related issue on GitHub:

- [**web-infra-dev/rspack: [Bug]: using docusaurus edit mdx or md file, process crash. #8480**](https://github.com/web-infra-dev/rspack/issues/8480)

It looks like we're not alone in this! We'll have to wait for a further fix.

## 2024-11-24 Update

Continuing from the previous issue, this time we updated to v3.6.3.

The Rspack issue has been fixed, and we can now happily use it as normal!
