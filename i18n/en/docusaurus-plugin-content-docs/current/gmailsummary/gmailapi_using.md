---
sidebar_position: 4
---

# Gmail API Calls

Now that the setup is complete, we can begin using the Gmail API.

First, locate the `credentials.json` file you downloaded earlier and place it in the root directory of your project.

Next, let's open Google's provided tutorial document: [**Python quickstart**](https://developers.google.com/gmail/api/quickstart/python)

## Installing Dependencies

You need to install the Google client library for Python:

```bash
pip install -U google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

## Configuration Example

1. In your working directory, create a file named `quickstart.py`.
    - You can directly reference the source code provided by Google: [**source code**](https://github.com/googleworkspace/python-samples/blob/main/gmail/quickstart/quickstart.py)

2. Include the following code in `quickstart.py`:

    ```python title="quickstart.py"
    import os.path

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    # If modifying these scopes, delete the file token.json.
    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


    def main():
    """Shows basic usage of the Gmail API. Lists the user's Gmail labels."""
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        # Call the Gmail API
        service = build("gmail", "v1", credentials=creds)
        results = service.users().labels().list(userId="me").execute()
        labels = results.get("labels", [])

        if not labels:
            print("No labels found.")
            return
        print("Labels:")
        for label in labels:
            print(label["name"])

    except HttpError as error:
        # TODO(developer) - Handle errors from gmail API.
        print(f"An error occurred: {error}")

    if __name__ == "__main__":
        main()
    ```

## Running the Example

Run `quickstart.py`:

```bash
python quickstart.py
```

When you run `quickstart.py` for the first time, it will prompt you for authorization. Click "Allow".

![gmail_19](./resources/gmail19.jpg)

You'll see output similar to the following:

```bash
Labels:
CHAT
SENT
INBOX
IMPORTANT
TRASH
DRAFT
SPAM
CATEGORY_FORUMS
CATEGORY_UPDATES
CATEGORY_PERSONAL
CATEGORY_PROMOTIONS
CATEGORY_SOCIAL
STARRED
UNREAD
```

Additionally, a `token.json` file will be retrieved. This file will be used for subsequent runs of `quickstart.py` without needing to authorize again.

## Getting Started

Next, we'll begin using the Gmail API to parse email contents.

We've implemented three parts: creating a client, fetching emails, and parsing emails.

First, import the necessary packages:

```python
from base64 import urlsafe_b64decode
from datetime import datetime, timedelta
from typing import Dict, List

import pytz
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
```

### Creating a Client

When creating a Gmail API client, load `token.json`, which stores the user's access and refresh tokens, and automatically refreshes the access token when it expires.

```python
def build_service():
    creds = None
    token_file = 'token.json'
    creds = Credentials.from_authorized_user_file(
        token_file, scopes=['https://www.googleapis.com/auth/gmail.readonly'])
    service = build('gmail', 'v1', credentials=creds)
    return service
```

### Fetching Emails

Next, define a function to retrieve email contents from the client:

```python
def get_messages(
    service,
    user_id='me',
    after_date=None,
    subject_filter: str = None,
    max_results: int = 500
) -> List[Dict[str, str]]:

    tz = pytz.timezone('Asia/Taipei')
    if not after_date:
        now = datetime.now(tz)
        after_date = (now - timedelta(days=1)).strftime('%Y/%m/%d')

    messages = []
    try:
        query = ''
        if after_date:
            query += f' after:{after_date}'
        if subject_filter:
            query += f' subject:("{subject_filter}")'

        response = service.users().messages().list(
            userId=user_id, q=query, maxResults=max_results).execute()

        messages.extend(response.get('messages', []))

        # Handle pagination with nextPageToken
        while 'nextPageToken' in response:
            page_token = response['nextPageToken']
            response = service.users().messages().list(
                userId=user_id, q=query, maxResults=max_results, pageToken=page_token).execute()
            messages.extend(response.get('messages', []))

    except Exception as error:
        print(f'An error occurred: {error}')

    if not messages:
        print("No messages found.")

    return messages
```

### Parsing Emails

After retrieving the data, it exists with a lot of metadata. We need to parse it into a readable format.

```python
def parse_message(service, msg_id, user_id='me'):

    try:
        message = service.users().messages().get(
            userId=user_id, id=msg_id, format='full').execute()
        headers = message['payload']['headers']
        parts = message['payload'].get('parts', [])
        email_data = {
            'Date': None,
            'Subject': None,
            'Text': None
        }

        # Parse headers to get send time, subject, sender, and recipients
        for header in headers:
            if header['name'] == 'Date':
                email_data['Date'] = header['value']
            elif header['name'] == 'Subject':
                email_data['Subject'] = header['value']

        # Parse email body
        for part in parts:
            if part['mimeType'] == 'text/plain' or part['mimeType'] == 'text/html':
                data = part['body']['data']
                text = urlsafe_b64decode(data.encode('ASCII')).decode('UTF-8')
                email_data['Text'] = text
                break  # Take only the first matching part

        return email_data

    except Exception as error:
        print(f'An error occurred: {error}')
        return None
```

## Conclusion

With that, we've covered the basic usage of the Gmail API.

Don't run it yet as we still need to do some preparation work.

We need to integrate with the OpenAI API so we can send email contents to ChatGPT for analysis.