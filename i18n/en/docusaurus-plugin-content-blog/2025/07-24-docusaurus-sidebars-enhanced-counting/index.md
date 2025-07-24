---
slug: docusaurus-sidebars-enhanced-counting
title: "Advanced Implementation of Docusaurus Sidebar Counting: Recursive Counting and Stable Slugs"
authors: Z. Yuan
image: /en/img/2025/0724.jpg
tags: [Docusaurus, Sidebar, Enhancement]
description: Recursive counting of subfolders and slug stability optimization.
---

A long time ago, I wrote an article explaining how to modify Docusaurus Sidebar to automatically count the number of articles.

- Background: [**Automatically Count Article Numbers in Docusaurus Sidebar**](/en/blog/customized-docusaurus-sidebars-auto-count)

After using it for several months, I found there was still room for improvement. So this time, I made **three improvements** to solve pain points and enhance the experience all at once.

<!-- truncate -->

## Issues with the Original Version

1. **Counting is not precise enough**: It only counts Markdown files directly under the current folder, ignoring deeper nested directories.

   For example, if the file structure is

   ```bash
   papers/
   ├── classic-cnns/
   │   ├── alexnet.md
   │   ├── vgg/
   │   │   ├── vgg16.md
   │   │   └── vgg19.md
   │   └── resnet/
   │       └── resnet50.md
   ```

   The old version would only display `Classic CNNs (1)`, but the expected result should be: `Classic CNNs (4)`.

2. **Unstable slug**: Category links depend on Docusaurus’s automatic slug generation mechanism, occasionally causing path changes.

   Since Docusaurus automatically generates slugs based on the category names we provide, whenever we change or add categories, the slug also changes, causing original links to break.

   In the old version, links looked like this: `papers/category/classic-cnns-11`.

   After adding a new article, this link might change to `papers/category/classic-cnns-12`. Such changes confuse users, as they may have already shared the original link, which is now invalid.

   Beyond user frustration, Google’s crawlers lose trust in these frequently changing links, negatively impacting the website’s SEO ranking.

## Three Major Improvements

### 1. Recursive Counting of All Subfolders

The old Sidebar only counted `.md` files directly inside the current folder. This version uses deep recursion to correctly count all Markdown files including those in all subfolders, ensuring the numbers on category labels are accurate.

```js
/**
 * Recursively counts all Markdown (.md) files under a given directory.
 * Ignores hidden entries (starting with '.') and Docusaurus config files like '_category_.json'.
 *
 * @param {string} dirPath - Absolute path to the target directory.
 * @returns {number} Total number of Markdown files found.
 */
function countMarkdownFiles(dirPath) {
  let count = 0;

  // Read all items (files and subdirectories) under the current directory
  for (const name of fs.readdirSync(dirPath)) {
    // Ignore hidden files and _category_.json config files
    if (name.startsWith(".") || name === "_category_.json") continue;

    const fullPath = path.join(dirPath, name); // Get full path
    const stat = fs.statSync(fullPath); // Get file or directory status

    if (stat.isDirectory()) {
      // If it is a folder, recursively count Markdown files inside
      count += countMarkdownFiles(fullPath);
    } else if (stat.isFile() && name.endsWith(".md")) {
      // If it is a Markdown file, increment count
      count += 1;
    }
  }

  return count;
}
```

### 2. Stable Slug Generation

In Docusaurus, each category link is automatically converted to a URL slug.

However, this can cause “unpredictable” changes, such as inconsistencies in URLs when paths contain spaces, Chinese characters, or special symbols. To ensure **stability and controllability** of links, we implement a `toSlug` function that encodes paths into stable, predictable slugs.

```js
/**
 * Converts a relative POSIX path into a URL-safe slug.
 * Ensures cross-platform consistency by normalizing slashes and applying encodeURIComponent.
 *
 * @param {string} relPath - Relative path like 'classic-cnns/vgg'.
 * @returns {string} Encoded URL slug like 'classic-cnns/vgg'.
 */
function toSlug(relPath) {
  return relPath
    .replace(/\\/g, "/") // Replace Windows backslashes with POSIX separators
    .split("/") // Split into directory segments
    .map(encodeURIComponent) // URL encode each segment
    .join("/"); // Join back with '/' for a clean, stable slug
}
```

Then in `buildCategoryItem()`, we first try to read the custom slug from `_category_.json`. If none is set, we fallback to using the above `toSlug()` default:

```js
const defaultSlug = `/category/${toSlug(relativeDirPath)}`;

const link = {
  type: "generated-index",
  slug: metadata.link?.slug || defaultSlug, // Prefer custom slug, fallback otherwise
  title: metadata.link?.title || baseLabel,
  ...metadata.link, // Preserve other fields such as description
};
```

### 3. Folder Sorting Optimization

In Docusaurus Sidebar auto-generation, folders are by default sorted alphabetically if no order is controlled.

In practice, we often want to:

- **Prioritize folders containing `_category_.json`** (representing explicitly defined categories)
- **Sort other undefined folders alphabetically afterwards**

This gives the Sidebar better organization and guides readers to browse “designed topic groups” first.

Here is the implementation of this sorting logic:

```js
/**
 * Sorts subdirectories such that:
 * - Folders with `_category_.json` come first (i.e., manually defined categories)
 * - Others follow, sorted alphabetically
 *
 * @param {string} dir - Absolute path to parent directory
 * @returns {string[]} Sorted list of subdirectory names
 */
function getSortedSubDirs(dir) {
  return fs
    .readdirSync(dir)
    .filter((name) => {
      const fullPath = path.join(dir, name);
      return !name.startsWith(".") && fs.statSync(fullPath).isDirectory();
    })
    .sort((a, b) => {
      const aHasCategory = fs.existsSync(path.join(dir, a, "_category_.json"));
      const bHasCategory = fs.existsSync(path.join(dir, b, "_category_.json"));

      // Folders with _category_.json come first
      if (aHasCategory && !bHasCategory) return -1;
      if (bHasCategory && !aHasCategory) return 1;

      // Otherwise sort alphabetically
      return a.localeCompare(b);
    });
}
```

## Summary

Now, the Sidebar’s numbers, links, and hierarchy are all properly handled.

It should be good to go for a while!

Wish me luck.

## Preview

- Full code is available here 👉 [**sidebarsPapers.js**](https://github.com/DocsaidLab/website/blob/main/sidebarsPapers.js)
- See the results directly here 👉 [**Paper Notes**](https://docsaid.org/en/papers/intro)
