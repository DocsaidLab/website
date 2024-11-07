---
slug: customized-docusaurus-404-page
title: Customizing the Docusaurus 404 Page
authors: Zephyr
image: /en/img/2024/0910.webp
tags: [Docusaurus, 404NotFound]
description: The default 404 page needs a revamp!
---

Docusaurus is a static site generator developed by Meta, designed for building open-source documentation websites.

It provides a simple way to create and maintain websites, supporting custom themes and plugins.

<!-- truncate -->

If you're unfamiliar with Docusaurus, you can check it out here: [**Docusaurus Official Website**](https://docusaurus.io/)

Our site is also built using Docusaurus, but after launching, we noticed that the default 404 page was quite basic.

To enhance user experience, we decided to create a custom 404 page.

## References

To solve this issue, we first found a discussion page within the Docusaurus project:

- [**How can I customize the 404 page?**](https://github.com/facebook/docusaurus/discussions/6030)

Based on this discussion, we implemented the solution.

Here's our step-by-step process.

## Exporting the 404 Page Configuration

:::warning
Starting from this step, we’ll be modifying Docusaurus' source code.

If there are destructive version updates in the future, these modifications may cause the website to malfunction. Please make sure you have the ability to maintain the website before proceeding.
:::

In Docusaurus, when a 404 error occurs, it redirects to the `NotFound` page of the `@docusaurus/theme-classic` theme.

We need to export this page’s configuration by running the following command:

```bash
npm run swizzle @docusaurus/theme-classic NotFound
```

During the process, select `JavaScript`, and then choose `--eject`. This will generate a `NotFound` directory under the `src/theme` folder.

If you're curious about the original code, you can find it here:

- [**docusaurus-theme-classic/src/theme/NotFound**](https://github.com/facebook/docusaurus/tree/e8c6787ec20adc975dd6cd292a731d01206afe92/packages/docusaurus-theme-classic/src/theme/NotFound)

Inside the folder, you'll find an `index.js` file. Initially, we don’t need to worry about this file. Instead, look inside the `Content` subfolder, where you'll find another `index.js` file. This is the one we’ll be modifying.

Here’s the original code:

```jsx
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from "react";
import clsx from "clsx";
import Translate from "@docusaurus/Translate";
import type { Props } from "@theme/NotFound/Content";
import Heading from "@theme/Heading";

export default function NotFoundContent({ className }: Props): JSX.Element {
  return (
    <main className={clsx("container margin-vert--xl", className)}>
      <div className="row">
        <div className="col col--6 col--offset-3">
          <Heading as="h1" className="hero__title">
            <Translate
              id="theme.NotFound.title"
              description="The title of the 404 page"
            >
              Page Not Found
            </Translate>
          </Heading>
          <p>
            <Translate
              id="theme.NotFound.p1"
              description="The first paragraph of the 404 page"
            >
              We could not find what you were looking for.
            </Translate>
          </p>
          <p>
            <Translate
              id="theme.NotFound.p2"
              description="The 2nd paragraph of the 404 page"
            >
              Please contact the owner of the site that linked you to the
              original URL and let them know their link is broken.
            </Translate>
          </p>
        </div>
      </div>
    </main>
  );
}
```

## Start Modifying

We want to add a few features:

1. A cute icon.
2. A countdown timer that automatically redirects to the homepage.
3. Custom text to provide more information to the reader.

### Countdown Timer

First, we’ll add a countdown timer using `useEffect`.

```jsx
import React, { useEffect, useState } from "react";

const [countdown, setCountdown] = useState(15);

useEffect(() => {
  const timer = setInterval(() => {
    setCountdown((prevCountdown) =>
      prevCountdown > 0 ? prevCountdown - 1 : 0
    );
  }, 1000);

  if (countdown === 0) {
    window.location.href = "/";
  }

  return () => clearInterval(timer);
}, [countdown]);
```

This will automatically redirect to the homepage once the countdown reaches zero.

### Adding an Icon

We found a cute icon on a free icon website:

- [**Freepik**](https://www.freepik.com/icons/error)

After downloading it, we placed it in the `static/img` directory and referenced it in `index.js`.

```jsx
<img
  src="/img/error-icon.png"
  alt="Error icon"
  style={{
    width: "150px",
    height: "150px",
    marginBottom: "20px",
    animation: "bounce 1s infinite",
  }}
/>

<style>{`
    @keyframes bounce {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-10px);
    }
    }
`}</style>
```

This `<img>` tag displays an error icon, with some additional styling to make it bounce:

- `src="/img/error-icon.png"`: The image source, pointing to a local file.
- `alt="Error icon"`: Alternative text, displayed if the image can’t load.
- The `style` attribute defines the icon's size and adds a bounce animation.

### Custom Text

The default 404 page looks something like this:

<div style={{ textAlign: 'center' }}>
<iframe
  src="https://docusaurus.io/non-exist"
  width="80%"
  height="500px"
  center="true"
></iframe>
</div>

---

We updated it with the following text:

```jsx
<p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
Sorry, we couldn't find the page you were looking for.
</p>
<p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
It’s possible that the site structure has changed, and you might have clicked an outdated link.
</p>
<p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
Please use the navigation bar above to find the information you're looking for.
</p>
```

Since we frequently update our site, some paths may be outdated in Google’s index, causing users to land on the wrong page.

We use this opportunity to inform users that:

- The page they’re looking for likely still exists, but has been moved!

We hope that with this message, users can navigate the site and find what they need.

---

Feel free to customize this section to suit your needs.

### Final Result

Here’s what our final 404 page looks like:

<br /><br />

<div className="row" style={{
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    flexDirection: 'column',
    textAlign: 'center',
    animation: 'fadeIn 0.5s ease-in-out',
  }}>

<img
src="/img/error-icon.png"
alt="Error icon"
style={{
      width: '150px',
      height: '150px',
      marginBottom: '20px',
      animation: 'bounce 1s infinite',
    }}
/>

  <div>
    <p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
      很抱歉，我們無法找到您要的頁面。
    </p>
    <p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
      網頁結構已經修改了，而您可能選到過時的連結。
    </p>
    <p style={{ fontSize: '1.2rem', marginBottom: '20px' }}>
      請點擊上方導航欄，或許可以找到您要的資訊。
    </p>
  </div>

  <style>{`
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    @keyframes bounce {
      0%, 100% {
        transform: translateY(0);
      }
      50% {
        transform: translateY(-10px);
      }
    }
  `}</style>

</div>

### Full Code

Finally, here is the complete code:

```jsx title='src/theme/NotFound/Content/index.js'
import Translate from "@docusaurus/Translate";
import Heading from "@theme/Heading";
import clsx from "clsx";
import React, { useEffect, useState } from "react";

export default function NotFoundContent({ className }) {
  const [countdown, setCountdown] = useState(15);

  useEffect(() => {
    const timer = setInterval(() => {
      setCountdown((prevCountdown) =>
        prevCountdown > 0 ? prevCountdown - 1 : 0
      );
    }, 1000);

    if (countdown === 0) {
      window.location.href = "/";
    }

    return () => clearInterval(timer);
  }, [countdown]);

  return (
    <main className={clsx("container margin-vert--xl", className)}>
      <div
        className="row"
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          flexDirection: "column",
          textAlign: "center",
          animation: "fadeIn 0.5s ease-in-out",
        }}
      >
        <img
          src="/img/error-icon.png"
          alt="Error icon"
          style={{
            width: "150px",
            height: "150px",
            marginBottom: "20px",
            animation: "bounce 1s infinite",
          }}
        />

        <div>
          <Heading as="h1" className="hero__title">
            <Translate
              id="theme.NotFound.title"
              description="The title of the 404 page"
            >
              Page Not Found
            </Translate>
          </Heading>
          <p style={{ fontSize: "1.2rem", marginBottom: "20px" }}>
            Sorry, we couldn't find the page you were looking for.
          </p>
          <p style={{ fontSize: "1.2rem", marginBottom: "20px" }}>
            It’s possible that the site structure has changed, and you might
            have clicked an outdated link.
          </p>
          <p style={{ fontSize: "1.2rem", marginBottom: "20px" }}>
            Please use the navigation bar above to find the information you're
            looking for.
          </p>
          <p aria-live="polite" style={{ fontSize: "1rem", color: "#555" }}>
            {countdown > 0
              ? `Redirecting to the homepage in ${countdown} seconds...`
              : "Redirecting..."}
          </p>
        </div>

        <style>{`
          @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
          }
          @keyframes bounce {
            0%, 100% {
              transform: translateY(0);
            }
            50% {
              transform: translateY(-10px);
            }
          }
        `}</style>
      </div>
    </main>
  );
}
```
