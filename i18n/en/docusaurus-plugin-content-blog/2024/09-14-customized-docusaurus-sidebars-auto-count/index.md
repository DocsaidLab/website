---
slug: customized-docusaurus-sidebars-auto-count
title: Automatically Count Articles in Docusaurus Sidebar
authors: Z. Yuan
image: /en/img/2024/0914.webp
tags: [Docusaurus, Sidebar]
description: Adding some new features to the sidebar.
---

Docusaurus provides a very feature-rich Sidebar by default, but sometimes it doesn't fit all our needs.

Let's tweak it a bit this time.

<!-- truncate -->

The goal is simple. The original Sidebar displays the categories we've specified. When the site starts, it looks for the `_category_.json` in each directory level, which contains something like this:

```json
{
  "label": "Classic CNNs",
  "link": {
    "type": "generated-index"
  }
}
```

This would display:

- Classic CNNs
- ... (other categories)

What I'd like to do is count the number of items under each folder and display it directly on the page, like this:

- Classic CNNs (8)
- ... (other categories)

---

Can we just add the count directly in the `_category_.json` file?

```json
{
  "label": "Classic CNNs (8)",
  "link": {
    "type": "generated-index"
  }
}
```

And manually update the count every time we add an article?

**Absolutely not! We can't write code like that.**

## Reference Material

To solve this issue, as usual, let's first check Docusaurus' official documentation:

- [**Docusaurus Sidebar**](https://docusaurus.io/docs/sidebar)

---

The default Sidebar is autogenerated through the `autogenerated` option:

```jsx
/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [{ type: "autogenerated", dirName: "." }],

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    'intro',
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
   */
};

export default sidebars;
```

---

After reviewing it, it seems there is no built-in feature that fits our needs. Let's build one ourselves.

## Implementation

I'll include comments directly in the code. Here are a few places to note where modifications may be required based on your setup:

1. Line 8: `const baseDir = path.join(__dirname, "papers");`

   The `papers` here refers to our folder name. Ensure your directory path is correct.

---

2. Line 20: `sidebarItems.push("intro");`

   The `intro` here refers to the name of our homepage. Ensure your homepage name is correct. If there's no homepage, you can remove this line.

---

3. Line 72: `return stat.isDirectory() || (stat.isFile() && item.endsWith(".md"));`

   Here, `.md` is the format of our articles. Adjust it based on your format.

---

Here's the implementation. You can see the result directly on the webpage: [**Papers**](/papers/intro).

```jsx showLineNumbers title="/sidebars.js"
const fs = require("fs"); // Import Node.js file system module to work with files and directories
const path = require("path"); // Import Node.js path module to handle file paths

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
// This is a type definition for Docusaurus sidebar configuration to enable IDE autocompletion
function generateSidebar() {
  // Set the base directory path, pointing to the 'papers' folder. Modify as needed.
  const baseDir = path.join(__dirname, "papers");

  // Read all subdirectories under 'papers', filtering out hidden ones (starting with a dot)
  const categories = fs.readdirSync(baseDir).filter((item) => {
    const itemPath = path.join(baseDir, item);
    // Ensure it's a directory and not hidden
    return fs.statSync(itemPath).isDirectory() && !item.startsWith(".");
  });

  const sidebarItems = []; // Array to store sidebar items

  // Add a fixed 'intro' item at the beginning
  sidebarItems.push("intro");

  // Iterate through all category directories
  categories.forEach((category) => {
    const categoryPath = path.join(baseDir, category); // Get the full path for each category
    const count = countItemsInDirectory(categoryPath); // Count the number of items in the directory

    // Try to read the '_category_.json' file within the category directory to get label and link
    const categoryJsonPath = path.join(categoryPath, "_category_.json");
    let label = category; // Default label is the directory name
    let link = undefined; // Default link is undefined
    if (fs.existsSync(categoryJsonPath)) {
      // If the '_category_.json' file exists
      const categoryJson = JSON.parse(
        fs.readFileSync(categoryJsonPath, "utf8")
      ); // Read and parse the JSON file
      label = categoryJson.label || category; // Use the label from the JSON file or default to directory name
      link = categoryJson.link; // Use the link from the JSON file
    }

    // Append the item count to the label
    label = `${label} (${count})`;

    // Create a sidebar item, with type 'category', indicating this is a category
    const sidebarItem = {
      type: "category",
      label: label, // Display label
      items: [{ type: "autogenerated", dirName: category }], // Automatically generate document items under the category
    };

    if (link) {
      // If a link is provided, add it to the category
      sidebarItem.link = link;
    }

    sidebarItems.push(sidebarItem); // Add the category item to the sidebar array
  });

  // Return an object containing the sidebar configuration
  return {
    papersSidebar: sidebarItems,
  };
}

// Count the valid items (including subdirectories and Markdown files) in a specified directory
function countItemsInDirectory(dirPath) {
  const items = fs.readdirSync(dirPath).filter((item) => {
    const itemPath = path.join(dirPath, item);
    // Exclude '_category_.json' and hidden files (starting with a dot)
    if (item === "_category_.json" || item.startsWith(".")) return false;
    const stat = fs.statSync(itemPath);
    // Only count directories and .md files
    return stat.isDirectory() || (stat.isFile() && item.endsWith(".md"));
  });
  return items.length; // Return the count of items
}

// Export the generated sidebar configuration for Docusaurus to use
export default generateSidebar();
```
