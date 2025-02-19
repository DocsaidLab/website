---
slug: download-from-google-drive-using-python
title: Download Files from Google Drive Using Python
authors: Z. Yuan
image: /en/img/2025/0219.webp
tags: [python, google-drive, download]
description: Small files, large files, all can be downloaded.
---

We wrote a Python script to download files from Google Drive, but sometimes it works fine, while other times it only gives us a strange HTML file...?

The script must be the problem, so we need to fix it.

<!-- truncate -->

## Why Do We Only Get HTML?

When we send a GET request to Google Drive to download a file, if the file is small (usually less than 100MB), Google may respond directly with the file content, allowing us to download it successfully.

However, if the file is larger, Google will show a "virus scan warning" page before the download, notifying the user that the file has not been fully scanned for viruses and offering a button to confirm the download.

If you're using a browser, you can manually click the button to download the file. However, when using a Python script, unless there's an additional mechanism to simulate the button click or parse the download link from the HTML page, you'll only get the HTML of the warning page itself instead of the actual file.

:::info
The unexpected HTML file might look something like this:

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Google Drive - Virus scan warning</title>
    <meta http-equiv="content-type" content="text/html; charset=utf-8" />
   ... <!-- other HTML content -->
</html>
```

:::

## Solution: Two-Step Request

Now that we understand the issue, we can fix it with the following steps:

1. **First request**: Send a GET request to `https://docs.google.com/uc?export=download` with the file ID.
2. **Check the response**: If the HTTP header contains `content-disposition`, then we have the file content, and we can download it directly. If not, it means we are still on the virus scan warning page, so we need to send another request.
3. **Extract the confirmation token**:
   - **From cookies**: Google may place a `download_warning_xxxxx` key in the cookies, which contains the token. For example: `token = session.cookies.get('download_warning_xxxxx')`
   - **From HTML**: Sometimes, Google may not place the token in the cookies but in the HTML form. For example:
     ```html
     <form
       id="download-form"
       action="https://drive.usercontent.google.com/download"
       method="get"
     >
       <input type="hidden" name="confirm" value="t" />
       ...
     </form>
     ```
     In this case, we can use [**BeautifulSoup**](https://www.crummy.com/software/BeautifulSoup/) to scrape all the hidden fields, including parameters like `confirm` or `uuid`.
4. **Combine the two requests**: After obtaining the `token`, include it in the request, or directly use the form’s `action` URL and corresponding hidden parameters, then send the second request to begin the download.

## Code Implementation

First, install the necessary packages:

```bash
pip install requests tqdm beautifulsoup4
```

Then use the following Python function, providing the file ID, the name you want to save the file as, and the target directory to automatically determine whether a second request is needed and download the file successfully:

```python title="download_from_google.py"
import os
import re
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

def download_from_google(file_id: str, file_name: str, target: str = "."):
    """
    Downloads a file from Google Drive, handling potential confirmation tokens for large files.

    Args:
        file_id (str):
            The ID of the file to download from Google Drive.
        file_name (str):
            The name to save the downloaded file as.
        target (str, optional):
            The directory to save the file in. Defaults to the current directory (".").

    Raises:
        Exception: If the download fails or the file cannot be created.

    Notes:
        This function handles both small and large files. For large files, it automatically processes
        Google's confirmation token to bypass warnings about virus scans or file size limits.

    Example:
        Download a file to the current directory:
            download_from_google(
                file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                file_name="example_file.txt"
            )

        Download a file to a specific directory:
            download_from_google(
                file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                file_name="example_file.txt",
                target="./downloads"
            )
    """
    # First try: docs.google.com/uc?export=download&id=FileID
    base_url = "https://docs.google.com/uc"
    session = requests.Session()
    params = {
        "export": "download",
        "id": file_id
    }
    response = session.get(base_url, params=params, stream=True)

    # If Content-Disposition is present, the file is directly available
    if "content-disposition" not in response.headers:
        # Try to get the token from cookies
        token = None
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break

        # If no token in cookies, extract it from the HTML
        if not token:
            soup = BeautifulSoup(response.text, "html.parser")
            # Common case: HTML contains a form with id="download-form"
            download_form = soup.find("form", {"id": "download-form"})
            if download_form and download_form.get("action"):
                # Extract action URL, which might be drive.usercontent.google.com/download
                download_url = download_form["action"]
                # Collect all hidden inputs
                hidden_inputs = download_form.find_all("input", {"type": "hidden"})
                form_params = {}
                for inp in hidden_inputs:
                    if inp.get("name") and inp.get("value") is not None:
                        form_params[inp["name"]] = inp["value"]

                # Re-send the GET request with these parameters
                response = session.get(download_url, params=form_params, stream=True)
            else:
                # Otherwise, search for confirm=xxx in HTML
                match = re.search(r'confirm=([0-9A-Za-z-_]+)', response.text)
                if match:
                    token = match.group(1)
                    # Include the confirm token in the request
                    params["confirm"] = token
                    response = session.get(base_url, params=params, stream=True)
                else:
                    raise Exception("Unable to find the download link or confirmation token in the response. Download failed.")

        else:
            # Use the token obtained from cookies and resend the request
            params["confirm"] = token
            response = session.get(base_url, params=params, stream=True)

    # Ensure the download directory exists
    os.makedirs(target, exist_ok=True)
    file_path = os.path.join(target, file_name)

    # Start downloading the file in chunks, with a progress bar
    try:
        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, "wb") as f, tqdm(
            desc=file_name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"File successfully downloaded to: {file_path}")

    except Exception as e:
        raise Exception(f"File download failed: {e}")
```

## How to Use

Assuming you have a file ID: `YOUR_FILE_ID`, you want to download it as `big_model.onnx` and store it in the `./models` directory, just call it like this:

```python
download_from_google(
    file_id="YOUR_FILE_ID",
    file_name="big_model.onnx",
    target="./models"
)
```

Once done, you'll find the successfully downloaded `big_model.onnx` in the `./models` folder, and you’ll see the download progress in the terminal.

## Command-Line Tool

If you'd prefer to use the command line directly, we can add a wrapper for you:

```python title="download_from_google_cli.py"
from download_from_google import download_from_google
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download files from Google Drive.")
    parser.add_argument("--file-id", required=True, help="Google Drive file ID.")
    parser.add_argument("--file-name", required=True, help="Output file name.")
    parser.add_argument("--target", default=".", help="Output directory. Defaults to current folder.")
    args = parser.parse_args()

    download_from_google(file_id=args.file_id, file_name=args.file_name, target=args.target)

if __name__ == "__main__":
    main()
```

Save the code above as `download_from_google_cli.py`, then run it from the command line:

```bash
python download_from_google_cli.py \
 --file-id YOUR_FILE_ID \
 --file-name big_model.onnx \
 --target ./models
```

If all goes well, the file will start downloading and show a progress bar.

We've tested with files of 70MB and 900MB, both working fine. As for a 900GB file... (🤔 🤔 🤔)

We haven't tested that, and we don't have a file that large at hand, so we’ll update when we try it next time!
