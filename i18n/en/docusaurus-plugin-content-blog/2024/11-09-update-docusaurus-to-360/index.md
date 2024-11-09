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
