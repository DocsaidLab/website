---
slug: works-locally-breaks-after-deploy
title: Works Locally, Breaks After Deploy? It’s Usually the Path
authors: Z. Yuan
image: /img/2025/0618-static-site-baseurl-trap.svg
tags: [docusaurus, deployment, frontend, debugging]
description: If your site works locally and breaks in production, it usually isn't dark magic. It's your baseUrl, asset paths, cache, and deployment assumptions forming a small but effective conspiracy.
---

There is a category of bug that is both common and aggressively unoriginal.

Everything looks fine in local development:

- the page loads
- the image shows up
- the CSS survives
- the links still pretend to be useful

Then you deploy, and the site starts performing modern interpretive dance:

- images return 404
- JS chunks fail to load
- CSS paths go missing
- subpages work until refresh, then die
- `/docs/intro` works when clicked, but opening it directly makes the server act surprised

At that point, the usual reaction is:

> But it worked locally.

Yes.

Local environments are often excellent at lying to you.

<!-- truncate -->

The real issue usually isn't that the code randomly broke overnight.

It's that you carried a set of **assumptions that only held locally** into a production environment that had no intention of cooperating.

One of the most common versions of that problem is this:

> **Paths.**

In plainer terms: the file is in one place, and you told the browser to look somewhere else.

## Why this happens so often

Because local development is usually forgiving:

- the dev server patches over routing mistakes
- the site often runs at the root path `/`
- there is no CDN, reverse proxy, or subdirectory deployment
- caching is lighter, so mistakes don't linger as long
- people click around locally, but rarely test direct deep links

So a lot of path-related errors never fully surface until production.

Then the environment changes slightly, and the site responds with a 404 and no emotional support.

Typical changes include:

- the site is hosted under `/blog/` or `/docs/` instead of `/`
- static assets get a CDN prefix
- the server does not provide SPA fallback
- a reverse proxy strips or rewrites the path prefix incorrectly
- what you thought was an “absolute path” was really just “absolute from the domain root”

That is usually enough.

## The classic trap: `/img/a.png`

Consider this Markdown:

```md
![cover](/img/cover.png)
```

Or this JSX:

```tsx
<img src="/img/cover.png" alt="cover" />
```

If your site is deployed at:

```text
https://example.com/
```

this is usually fine.

But if the real deployment target is:

```text
https://example.com/docs/
```

then the browser interprets `/img/cover.png` as:

```text
https://example.com/img/cover.png
```

not:

```text
https://example.com/docs/img/cover.png
```

So in your head, it was “an image inside my site.”

In the browser's head, it was “an image at the domain root.”

The browser is not confused.

It is merely less imaginative than you.

## `baseUrl` is not decorative

If you are using Docusaurus, this problem is often tied to `baseUrl`.

For example:

```ts
const config = {
  url: 'https://example.com',
  baseUrl: '/docs/',
};
```

That means the site actually lives under `/docs/`.

If you then hardcode this:

```tsx
<img src="/img/cover.png" alt="cover" />
```

you are bypassing the framework's deployment-aware path handling.

A safer approach is to let the framework build the path:

```tsx
import useBaseUrl from '@docusaurus/useBaseUrl';

export default function Hero() {
  const imageUrl = useBaseUrl('/img/cover.png');
  return <img src={imageUrl} alt="cover" />;
}
```

That tends to survive both `/` and `/docs/` deployments.

If you truly know the site will only ever live at the root, hardcoding may be acceptable.

But a lot of bugs begin with:

> We thought deployment would never change.

Production enjoys disproving that sentence.

## It is not just images

People often notice missing images first.

The more annoying breakages are usually these:

- script chunk paths are wrong
- lazy-loaded assets point to stale locations
- font URLs miss the correct prefix
- manifest, favicon, or social image paths point to the wrong place

The result is the kind of page that looks half-built and slightly offended:

- some styles load, some don't
- buttons work, icons vanish
- the homepage is fine, inner pages look dismantled

This is why path bugs are often misdiagnosed as:

- broken builds
- stale caches
- package version regressions

Those things can happen too.

But checking the path is usually the cheapest first move.

And the probability is embarrassingly high.

## Relative paths are also happy to betray you

Suppose you write:

```html
<img src="img/cover.png" alt="cover" />
```

If the current page is:

```text
https://example.com/posts/hello/
```

then the browser may resolve it as:

```text
https://example.com/posts/hello/img/cover.png
```

which is not the same as:

```text
https://example.com/img/cover.png
```

That produces a very annoying class of symptoms:

- homepage works
- some article pages work
- deeper pages fail

The relative path is not malfunctioning.

It is following the rules exactly.

You just were not thinking about those rules five minutes ago.

## Another old favorite: navigation works, refresh dies

This one is also common.

You have a frontend route like:

```text
/docs/intro
```

Clicking into it from the homepage works.

Because the frontend router takes over.

But opening that URL directly, or refreshing the page, returns a 404 from the server.

That usually does not mean the frontend route is broken.

It means the server was never told:

> When this path does not match a real file, serve `index.html` and let the frontend router decide.

For Nginx, that often looks like this:

```nginx
location /docs/ {
    try_files $uri $uri/ /docs/index.html;
}
```

Without a fallback like that, many client-side routes fail only when directly requested.

Which makes the bug wonderfully deceptive:

- internal navigation works
- bookmarks, shared links, and refreshes fail

So it can hide for a long time.

Until someone uses the site like a normal person.

## How to debug this faster

Open DevTools and go to **Network**.

Before rebuilding everything, check these:

1. **Which URL actually returned 404?**
2. **Is it missing a prefix, or does it have an extra one?**
3. **Did root-relative `/...` get used where a deployment-aware path was needed?**
4. **Is the missing resource HTML, a JS chunk, CSS, or an asset?**
5. **When an inner page fails on refresh, is the server missing a fallback rule?**

A lot of debugging time gets wasted staring at the page and saying, “why is it gone?”

The page will not explain itself.

The failing URL usually will.

## A practical checklist

This is the order I usually use.

### 1. Confirm the real deployment location

Figure out where the site actually lives:

- `/`
- `/docs/`
- `/product/site/`
- behind a CDN path prefix

Use the real URL, not the one still living in your memory.

### 2. Check framework config

For Docusaurus, look at:

- `url`
- `baseUrl`
- `trailingSlash`

Other frameworks have their equivalents:

- `base`
- `assetPrefix`
- `publicPath`

Different names, same category of pain.

### 3. Find hardcoded paths

Especially these patterns:

```text
src="/..."
href="/..."
url(/...)
fetch('/...')
```

They are not always wrong.

They are simply suspicious often enough to deserve attention.

### 4. Test direct entry to inner pages

Do not stop at the homepage.

Open something like:

```text
https://example.com/docs/some/page
```

If internal navigation works but direct entry fails, the problem is usually routing or server configuration, not page content.

### 5. Clear caches and test again

Especially:

- service workers
- CDN cache
- mismatched HTML and hashed assets

Sometimes you fixed the problem and the cache is just committed to preserving the old mistake.

## A habit that saves time later

The principle is simple:

> **If a project may be deployed under different prefixes, stop hardcoding asset paths like they are eternal truths.**

Let the framework handle them when possible.

If not, centralize the logic.

For example:

```ts
import useBaseUrl from '@docusaurus/useBaseUrl';

export function useAsset(path: string) {
  return useBaseUrl(path);
}
```

Then use it consistently:

```tsx
const logo = useAsset('/img/logo.svg');
```

This will not solve every deployment bug.

But it does reduce the number of future archaeological digs through `"/img/..."` strings.

## Final

“Works locally, breaks after deploy” is often not a mysterious production-only bug.

It is just production finally refusing to cover for your assumptions.

And among all the possible causes, **path handling** is one of the cheapest and most profitable things to suspect first.

So the next time you see this pattern:

- homepage works, inner pages act strange
- some images vanish
- JS or CSS disappears only in production
- clicking works, refreshing gives 404

Do not start by blaming the framework.

Do not immediately accuse browser cache, Node version, the moon phase, or some bundler release note you only half read.

Check the path first.

A lot of the time, the bug is not deep.

You just walked into the wrong directory with confidence.
