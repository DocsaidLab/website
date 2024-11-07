---
slug: file-crawler-python-implementation
title: Python Implementation of a Web File Downloader
authors: Zephyr
image: /en/img/2024/0923.webp
tags: [Python, File Crawler]
description: Implement a simple web file downloader.
---

We came across a webpage containing hundreds of PDF file links.

As engineers, if we were to download them manually, it would be highly inefficient, right?

So, what we need here is a small script that will help us download all the files.

<!-- truncate -->

## Install Required Packages

First, you need to install the necessary packages. If you haven't installed them yet, you can do so using the following command:

```bash
pip install requests beautifulsoup4 urllib3
```

## The Code

Without further ado, since the script is already written, let's dive straight into the code!

The parts highlighted are the ones you’ll need to modify yourself. Adjust the script according to your needs.

```python {13,16} title="file_crawler.py"
import os
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Simulating a browser's headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
}

# Web page URL
url = "put_your_url_here"

# Target file format
target_format = ".pdf"

# Send an HTTP GET request with headers
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Use BeautifulSoup to parse the HTML
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all <a> tags and filter those with href attributes matching the target format
    target_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(target_format):  # Specify the file format you want to download
            target_links.append(urljoin(url, href))

    # Create a folder to save the files
    os.makedirs("downloads", exist_ok=True)

    # Download each file
    for url in target_links:
        file_name = url.split("/")[-1]  # Extract the filename from the URL
        file_path = os.path.join("downloads", file_name)

        # Send a request to download the file
        response = requests.get(url, headers=headers)  # Add headers again
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {file_name}")
        else:
            print(f"Failed to download: {url}")
else:
    print(f"Unable to access the webpage, status code: {response.status_code}")
```

## Running the Script

Once you’re done, you can simply run the script to download all the files matching the target format.

```bash
python file_crawler.py
```
