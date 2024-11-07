---
slug: python-js-basic-command-equivalents
title: Equivalent Basic Commands between Python and JS
authors: Zephyr
image: /en/img/2024/0407.webp
tags: [npm, pip]
description: Mapping basic cmds between Py and JS.
---

As engineers, we are destined to keep learning new technologies.

While we are more accustomed to Python, we discovered some similar commands when starting to learn JavaScript, such as `npm`, `npx`, and `nvm`.

So we tried to map these commands, hoping it might ease the transition into new skills.

<!-- truncate -->

I'm interested in aligning these commands, hoping it might ease the transition into new skills.

## npm vs. pip

:::tip
**They are both: Package Managers**
:::

npm (Node Package Manager) and pip essentially serve the same purpose: they are package managers for Node.js and Python respectively. Package managers are crucial for sharing, reusing, and managing repositories or modules.

- **Installing Packages**: In npm, we use the `npm install <package-name>` command to add a library to our project. Similarly, pip achieves the same goal through `pip install <package-name>`.
- **Version Control**: npm tracks package versions through the `package.json` file, ensuring that every member of a development team uses the same version of a library. Pip relies on `requirements.txt` or newer tools like pipenv and poetry to achieve similar functionality.
- **Package Publishing**: npm enables developers to publish their packages to the npm registry for use by the global Node.js community. Pip provides this capability through PyPI (Python Package Index), allowing the sharing of Python packages.

## npx vs. -m Flag

:::tip
**They are both: Tools for Direct Command Execution**
:::

npx (npm package runner) and Python's `-m` flag address the need to execute package commands directly in the terminal without global installation.

- **Direct Execution**: npx allows you to directly execute any package installed in the project's local `node_modules` folder (or fetch it from the npm registry if not installed), while Python achieves similar results through the `-m` flag, allowing direct execution of modules, such as starting a simple HTTP server with `python -m http.server`.

:::note
**npm run vs. npx run**

- npm run: In JavaScript projects, npm run is used to execute scripts defined in the package.json file. This is a common approach to perform project-specific tasks like testing, building, or deploying.
- npx run: While npx is typically used to execute single commands or packages, it primarily serves to execute packages not globally installed. npx run is not a standard command; common usage of npx doesn't include the keyword "run" but directly follows with the package name or command.
  :::

## nvm, pyenv, and conda

:::tip
**They are all: Version Management Tools**
:::

Switching between different versions of Node.js or Python can be cumbersome without proper tools. nvm (Node Version Manager), pyenv, and conda provide solutions to this problem, allowing developers to install and switch between multiple versions of Node.js or Python on the same machine.

- **Version Switching**: nvm uses commands like `nvm use <version>` to switch Node.js versions. Pyenv and conda offer similar functionalities for Python; pyenv switches versions through `pyenv global <version>` or `pyenv local <version>`, while conda uses `conda activate <environment-name>` to switch to different environments, each capable of having different Python versions and packages.
- **Multi-Version Management**: These tools facilitate managing multiple versions on the same machine, addressing potential conflicts due to version discrepancies.

## Conclusion

While these commands may not align perfectly, they do share some similarities, hopefully making the transition a bit smoother.
