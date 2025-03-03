---
slug: error-record
title: Daily Error Troubleshooting Log
authors: Z. Yuan
tags: [error, record]
image: /en/img/2024/0204.webp
description: A log of simple issues and their solutions.
---

We always encounter a lot of problems when writing code.

Here we record some trivial issues and their solutions.

:::tip
This article will be continuously updated.
:::

<!-- truncate -->

## 1. Error when running `npx docusaurus start`

- **Error Message:**

  ```bash
  file:///home/user/workspace/blog/node_modules/@docusaurus/core/bin/docusaurus.mjs:30
  process.env.BABEL_ENV ??= 'development';
                      ^^^

  SyntaxError: Unexpected token '??='
  ```

- **Solution:**

  The `??=` operator requires Node.js version 15.0.0 or higher.

  ```bash
  nvm install node
  nvm use node
  ```

## 2. 'choco' command not recognized

- **Error Message:**

  ```shell
  PS C:\Windows\System32> choco install git -y
  >>
  choco : The term 'choco' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or if a path was included, verify that the path is correct and try again.
  At line:1 char:1
  + choco install git -y
  + ~~~~~
      + CategoryInfo          : ObjectNotFound: (choco:String) [], CommandNotFoundException
      + FullyQualifiedErrorId : CommandNotFoundException
  ```

- **Solution:**

  This indicates that Chocolatey was not successfully installed, often due to not running PowerShell as an Administrator.

  Run PowerShell as an Administrator, and then execute the Chocolatey installation command again.

  ```shell
  Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
  ```

## 3. Chocolatey installation failure

- **Error Message:**

  ```shell
  Warning: An existing Chocolatey installation was detected. Installation will not continue. This script will not overwrite existing installations.
  If there is no Chocolatey installation at 'C:\ProgramData\chocolatey', delete the folder and attempt the installation again.

  Please use choco upgrade chocolatey to handle upgrades of Chocolatey itself.
  If the existing installation is not functional or a prior installation did not complete, follow these steps:
  - Backup the files at the path listed above so you can restore your previous installation if needed.
  - Remove the existing installation manually.
  - Rerun this installation script.
  - Reinstall any packages previously installed, if needed (refer to the lib folder in the backup).

  Once installation is completed, the backup folder is no longer needed and can be deleted.
  ```

- **Solution:**

  This indicates that Chocolatey is already installed. Please remove the old installation before reinstalling.

  ```shell
  Remove-Item "C:\ProgramData\chocolatey" -Recurse -Force
  ```

## 4. Remote port forwarding

- **Description:**

  You’ve started a service on a remote machine, such as TensorBoard, but can't access it directly, so you need to forward the port through your local machine.

- **Solution:**

  Assuming the service is running on port 6006 on the remote machine, and you want to access it on the same port on your local machine.

  When using SSH to log in, you can forward the port using the `-L` parameter:

  ```bash
  ssh -L 6006:localhost:6006 user@remote_ip_address
  ```

  This way, you can access the TensorBoard service on the remote machine via `http://localhost:6006` on your local machine.

## 5. Inconsistent Web Rendering Behavior in Development and Deployment Environments

- **Description:**

  You've set the layout style of the blog in `custom.css`:

  ```css
  .container {
    max-width: 90%;
    padding: 0 15px;
    margin: 0 auto;
  }
  ```

  In the deployment phase, this style seems to be overridden by other higher-priority styles, but in the development phase, this style is normal.

- **Solution:**

  Be more specific in selecting the target:

  ```css
  body .container {
    max-width: 90%;
    padding: 0 15px;
    margin: 0 auto;
  }
  ```

## 6. Turbojpeg Warning When Reading Images

- **Description**

  When reading images, the following warning messages appear:

  ```shell
  turbojpeg.py:940: UserWarning: Corrupt JPEG data: 18 extraneous bytes before marker 0xc4
  turbojpeg.py:940: UserWarning: Corrupt JPEG data: bad Huffman code
  turbojpeg.py:940: UserWarning: Corrupt JPEG data: premature end of data segment
  ```

- **Solution**

  To avoid these annoying warnings, you should filter out the problematic images:

  ```python
  import cv2
  import warnings

  data = ['test1.jpg', 'test2.jpg', 'test3.jpg']

  for d in data:
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always", UserWarning)

      # Read the image and see if there are any warnings
      cv2.imread(d)

      # Remove the data if warnings are present
      if w:
        data.remove(d)

  # Saving the filtered data
  ```

## 7. `Docusaurus` Deployment: `showLastUpdateTime: true` Not Working

- **Description**

  In `docusaurus.config.js`, you set `showLastUpdateTime: true` and `showLastUpdateAuthor: true,` but after deployment, you find that it has no effect. The rendered page displays the same time and author?

- **Solution**

  The problem is caused by an incorrect setting when checking out the branch during deployment, which prevents `git` from correctly obtaining the last update time and author.

  Change it like this:

  ```yaml
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
  ```

  Setting `fetch-depth: 0` will solve the problem.

## 8. Checking Error Logs in a Docker Container

- **Description**

  A service is running inside a Docker container, but it encounters an error and fails to function properly. It is necessary to check the error logs.

- **Solution**

  First, identify the target container's ID:

  ```bash
  docker ps
  ```

  Then, access the container and check the logs:

  ```bash
  docker exec -it container_id /bin/bash
  cat /path/to/logfile
  ```

  Alternatively, view the logs directly:

  ```bash
  docker logs container_id
  ```

## 9. Checking the i18n Status in Docusaurus

- **Description**

  How can we check the current language status in `Docusaurus`?

- **Solution**

  You can use `i18n` to retrieve the current language status:

  ```javascript
  import useDocusaurusContext from "@docusaurus/useDocusaurusContext";

  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();
  ```

  Then, you can use the `currentLocale` variable to obtain the corresponding language data.
