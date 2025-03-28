---
slug: should-you-choose-docusaurus
title: Should You Choose Docusaurus?
authors: Z. Yuan
image: /en/img/2025/0328.jpg
tags: [docusaurus, static-site, blog]
description: A year of practical experience shared.
---

Is Docusaurus easy to use?

This is the most common question I get asked after running Docsaid for over a year.

The answer is yes, but the reasons it’s easy to use might not be what you expect.

<!-- truncate -->

## Why did I initially choose Docusaurus?

Actually, my first choice was WordPress.org. (Surprised?)

It has a large user base, is widely used, and has a well-established ecosystem. Many features come with ready-made plugins, such as SEO, social sharing, sitemaps, and so on. Plus, WordPress has a very user-friendly interface, which makes it easy for those who are not familiar with code to get started.

But hey, I can write code!

Sometimes, being able to write code is a curse. When you don’t know much, it’s fine to follow tutorials online without questioning much, as there aren’t many choices.

But once you realize there are other options, you begin to question whether this choice is the right one: is this really the best approach?

Especially when you know you "can" do certain things, like:

- **Why can’t I write articles using Markdown?**

  Actually, you can, but it requires installing extra plugins. Writing with Markdown doesn’t have the developer tools like syntax highlighting, version control, or easy metadata integration.

  ***

- **Why can’t I manage versions with Git?**

  You can, but most modifications in WordPress are done via the backend, making it hard to sync with Git. Plus, much of the content is stored in the database, not as files.

  ***

- **Why can’t I call external APIs?**

  You can, but you have to bypass CORS, write custom JS, and figure out how to inject frontend code without affecting the theme.

  ***

You think you’re saving time, but in reality, you end up wasting more time dealing with these issues.

The solutions to these problems aren’t simple plugins but require you to write your own code.

In the end, if you still have to write your own code, then why bother with WordPress?

Also, the WordPress plugin ecosystem isn’t perfect. Many plugins are developed by third parties, and their quality can vary widely. Some even affect website performance or security.

So, was it really worth it?

:::tip
Personally, I think it wasn’t.
:::

## What other options are there?

After carefully considering my needs, my most basic requirement is: writing articles.

Writing articles itself is already quite a task, and if I had to spend energy on formatting tools, it wouldn't take long before I'd be mentally drained.

So, I don't just want to "write articles," I also want to "write Markdown articles," so I can focus on the content rather than the formatting.

:::tip
**So, does this mean formatting is ignored?**

Do you have a misconception about Markdown? It already has good layouts that can help organize content quickly.

Furthermore, formatting can be handled with CSS and JS, which I can write myself. This is much simpler than using drag-and-drop editors.
:::

There are many static site generators based on Markdown, each with its own strengths.

We gathered some information online and here’s a quick comparison:

<div style={{
  whiteSpace: 'nowrap',
  overflowX: 'auto',
  fontSize: '0.8rem',
  lineHeight: '0.8',
  justifyContent: 'center',
  display: 'flex',
}}>

| Feature                | Docusaurus       | Hugo         | Jekyll            | Hexo          | Astro            |
| ---------------------- | ---------------- | ------------ | ----------------- | ------------- | ---------------- |
| Document-oriented      | ✅ Strong        | ⚠️ Fair      | ⚠️ Acceptable     | ⚠️ Acceptable | ❌ Weak          |
| Multilingual Support   | ✅ Strong        | ❌ Poor      | ⚠️ Acceptable     | ⚠️ Acceptable | ⚠️ Acceptable    |
| Deployment Ease        | ✅ Strong        | ✅ Strong    | ✅ Strong         | ✅ Strong     | ✅ Strong        |
| Frontend Interactivity | ⚠️ Acceptable    | ❌ Poor      | ❌ Poor           | ❌ Poor       | ✅ Strong        |
| Markdown Support       | ✅ Strong        | ✅ Strong    | ⚠️ Acceptable     | ⚠️ Acceptable | ✅ Strong        |
| Git/Version Control    | ✅ Strong        | ✅ Strong    | ⚠️ Acceptable     | ⚠️ Acceptable | ✅ Strong        |
| Themes & Community     | ⚠️ Weak          | ✅ Strong    | ⚠️ Acceptable     | ✅ Strong     | ⚠️ Weak          |
| Arch Flexibility       | ⚠️ Acceptable    | ✅ Strong    | ⚠️ Acceptable     | ⚠️ Acceptable | ✅ Strong        |
| Learning Curve         | ❌ Difficult     | ⚠️ Moderate  | ✅ Easy           | ✅ Easy       | ⚠️ Moderate      |
| Build Speed            | ⚠️ Acceptable    | ✅ Very Fast | ⚠️ Average        | ⚠️ Average    | ✅ Fast          |
| SEO Friendliness       | ✅ Strong        | ✅ Strong    | ⚠️ Acceptable     | ⚠️ Acceptable | ✅ Strong        |
| Plugins & Ecosystem    | ⚠️ Weak          | ✅ Strong    | ⚠️ Acceptable     | ⚠️ Acceptable | ⚠️ Weak          |
| Data Integration       | ⚠️ Acceptable    | ✅ Strong    | ⚠️ Acceptable     | ❌ Weak       | ✅ Strong        |
| Programming Language   | ⚛️ React         | 🐹 Go        | 💎 Ruby           | 🟢 Node.js    | 📜 JS/TS         |
| Maintenance Frequency  | ✅ Strong        | ✅ Strong    | ⚠️ Acceptable     | ❌ Weak       | ✅ Strong        |
| Suitable for           | 👨‍💻 Technical Doc | ✍️ Blogs     | 📝 Personal Pages | 🌏 Chinese    | 🚧 Customization |

</div>

### Docusaurus

1. **Best for technical documentation and multilingual sites**: Excels in "documentation-oriented" and "multilingual support," solidifying its position as the top choice for development documentation websites.
2. **Frontend interaction is flexible, but requires React experience**: Supports the React framework, providing decent interactivity but also a steeper "learning curve," making it better suited for experienced frontend developers.
3. **Frequent updates and excellent version control integration, ideal for team collaboration**: Well-suited for team development scenarios requiring Git workflows, with an active update frequency.

### Hugo

1. **Static site compilation is incredibly fast, perfect for speed and efficiency seekers**: Hugo is built using Go, with the fastest compile time among all frameworks, enabling quick deployment.
2. **Strong flexibility in structure and plugin integration**: Supports highly customizable features and data source integration, making it a popular choice for large content websites.
3. **Weak multilingual and frontend interaction support, not suitable for internationalization or dynamic applications**: These weaknesses make Hugo more suitable for single-language content websites or blogs.

### Jekyll

1. **Most beginner-friendly learning curve, ideal for newcomers to static sites**: Jekyll's simple syntax makes it easy for non-technical users to get started with GitHub Pages deployment.
2. **Most features are average, making it unsuitable for complex applications**: Although deployment and SEO are good, it lacks standout strengths, which has led to its gradual replacement by other frameworks.
3. **Ecosystem and maintenance momentum are declining, with limited future scalability**: Compared to other frameworks, its update frequency and community activity are notably declining, making it more suitable for low-maintenance personal pages.

### Hexo

1. **Active Chinese community, ideal for Chinese-speaking users**: While international support is limited, there is abundant Chinese-language resources, making it easy to find tutorials and theme packages.
2. **Easy to learn and deploy, ideal for writing-focused users**: Built on Node.js, Hexo is very friendly for those familiar with JavaScript, allowing for rapid website creation without much hassle.
3. **Weak plugin and multilingual support, unsuitable for long-term large-scale maintenance projects**: More suitable for simple blogs or personal websites, it struggles with handling multiple languages and enterprise-level documentation.

### Astro

1. **Strongest frontend interaction and modern technology integration capabilities**: Supports modern frameworks like React, Vue, and Svelte, making it ideal for building rich interactive and static hybrid websites.
2. **Highly flexible architecture with excellent data integration capabilities, perfect for highly customized projects**: Can pull from multiple data sources and supports complex architectures, making it useful for content-driven or design-oriented websites.
3. **Multilingual and theme ecosystems are still developing, with a moderate learning curve**: For users needing internationalization and ready-to-use templates, it’s not yet mature enough.

## What Has It Done for Me?

I didn't have enough time to try everything individually, so based on the available information, I chose Docusaurus.

- Document-oriented design with automatic generation of sidebars and breadcrumbs
- Built-in multilingual support, with my site currently having Chinese, English, and Japanese versions
- Easy Markdown / MDX writing, with the ability to insert React components
- The default theme is good enough without needing a lot of custom CSS
- One-click deployment to GitHub Pages, suitable for self-hosted sites

If you ask me which feature I’m most satisfied with, I would say it’s "MDX writing."

For React developers, the ability to insert React components with MDX greatly enhances the flexibility of content presentation. If something is missing, just create a custom React component.

Additionally, I almost never have to deal with frontend bundling, optimization, routing configuration, or multilingual setup details.

Overall, it has handled about 80% of my website needs, and the remaining 20% consists of React components and articles that I wrote myself.

## What Are the Limitations?

Although Docusaurus is very convenient for setting up documentation websites or knowledge bases, there are some obvious limitations and downsides in the process:

1. **Not suitable for highly customized dynamic websites**

   Docusaurus is designed for static content generation, so it's not suitable for websites that require extensive dynamic interaction or real-time data updates, such as e-commerce sites, forums, or membership systems. For websites that need dynamic rendering (SSR), frequent data interactions, or backend integration, frameworks like Next.js, Remix, or Nuxt are more appropriate.

   ***

2. **React is the only built-in supported frontend framework**

   Docusaurus uses React and MDX technology, and does not natively support other frontend frameworks (such as Vue, Svelte, Angular). If your project team is familiar with other frameworks or wants to switch frameworks in the future without making significant changes to the content, Docusaurus may feel limiting.

   ***

3. **Limited customization for styles and UI**

   While Docusaurus offers a built-in theme and allows custom CSS, most UI adjustments require overriding built-in components or redefining the theme. If you need extensive UI customization or a complete redesign of the visual style, it might lead to significant additional work.

   ***

4. **Limited plugin ecosystem**

   Although Docusaurus’s plugin ecosystem is relatively active, it is still less extensive compared to more mature ecosystems like Gatsby, Next.js, or Hugo. If an official or community plugin is not available for certain needs, you’ll have to develop your own, potentially increasing the maintenance cost.

   ***

5. **Performance issues with large-scale websites**

   When the content scale becomes very large (over thousands of pages), Docusaurus’s build time may significantly increase. Every time new content is added or adjusted, the entire website content needs to be rebuilt. If the website content is frequently updated, this could pose significant efficiency issues. In such cases, a more mature and optimized framework like Hugo might be a better fit.

   :::tip
   This issue has been improved in the recent V3.6.0 release, with significant build speed improvements. The detailed user experience still needs to be verified by the community.
   :::

   ***

6. **Some features need to be implemented through third-party services**

   Docusaurus focuses on static site generation, so features like search (e.g., Algolia), comment systems (e.g., Disqus or GitHub Issues), user login, or database integration must be implemented through third-party services. The internal system doesn’t offer built-in solutions.

   ***

7. **Learning curve depends on familiarity with React**

   While Docusaurus is very friendly for Markdown writers, if you want to make full use of MDX, add React components, or customize features, users must be familiar with React and JSX. For non-frontend developers or beginners, this might increase the difficulty of getting started.

---

Overall, Docusaurus is best suited for quickly building knowledge bases, documentation, or technical content websites. If your needs exceed this scope, the limitations mentioned above could affect the user experience and development efficiency.

## My Usage Status

:::info
Updated in March 2025.
:::

My website, Docsaid, has been using the Docusaurus framework for over a year now:

- Accumulated over 170 paper notes
- Supports Chinese, English, and Japanese language switching
- Includes modules such as blog, docs, papers, playground, etc.
- I built my own backend for user registration, login, and API token issuance
- Developed over 20 React components to support different presentation needs for articles

Although Docusaurus is not perfect, for content-heavy websites like mine, it remains the most convenient and stable solution.

## Conclusion: Who Is Docusaurus Suitable For?

I recommend it to the following types of users:

- Those who want to build content-oriented websites (technical documentation, learning notes, open-source manuals)
- Those who want multilingual support but don't want to maintain a complex website structure
- Those who want to set up a website with low maintenance cost quickly
- Those who are willing to embrace the React ecosystem (MDX/JS config)

I will continue using it and will keep improving my knowledge of Docusaurus’s usage techniques and component extensions. If you’re considering a site-building tool, it’s worth trying out Docusaurus first to see if it suits your needs.

If you have any specific topics you'd like me to cover about Docusaurus, feel free to leave a comment below, and I'll prioritize writing about it.

Good luck with your website setup!
