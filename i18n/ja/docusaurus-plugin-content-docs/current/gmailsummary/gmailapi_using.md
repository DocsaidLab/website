---
sidebar_position: 4
---

# Gmail API の呼び出し

設定が完了したら、Gmail API を使用し始めることができます。

まず、先ほどダウンロードした`credentials.json`ファイルを見つけ、それをプロジェクトのルートディレクトリに配置します。

次に、Google が提供しているチュートリアルを開きます：[**Python quickstart**](https://developers.google.com/gmail/api/quickstart/python)

## パッケージのインストール

Python 用の Google クライアントライブラリをインストールする必要があります：

```bash
pip install -U google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

## 設定例

1. 作業ディレクトリに`quickstart.py`という名前のファイルを作成します。

   - Google が提供するソースコードをそのまま使うこともできます：[**source code**](https://github.com/googleworkspace/python-samples/blob/main/gmail/quickstart/quickstart.py)

2. 以下のコードを`quickstart.py`に追加します：

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

## 実行例

`quickstart.py`を実行します：

```bash
python quickstart.py
```

`quickstart.py`を初めて実行すると、認証を求められます。「Allow」をクリックしてください。

![gmail_19](./resources/gmail19.jpg)

次のような出力が表示されます：

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

また、`token.json`というファイルが取得され、次回`quickstart.py`を実行する際に再度認証を求められることはなくなります。

## 使用開始

次に、Gmail API を使用してメール内容を解析する準備を始めます。

以下の三つの部分を実装します：クライアントの作成、メールの取得、メールの解析。

まず必要なパッケージをインポートします：

```python
from base64 import urlsafe_b64decode
from datetime import datetime, timedelta
from typing import Dict, List

import pytz
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
```

### クライアントの作成

Gmail API クライアントを作成するとき、`token.json`に保存されたユーザーのアクセスおよびリフレッシュトークンをロードし、アクセス令牌が期限切れの場合は自動的にリフレッシュされます。

```python
def build_service():
    creds = None
    token_file = 'token.json'
    creds = Credentials.from_authorized_user_file(
        token_file, scopes=['https://www.googleapis.com/auth/gmail.readonly'])
    service = build('gmail', 'v1', credentials=creds)
    return service
```

### メールの取得

次に、ユーザーからメール内容を取得する関数を定義します：

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

### メールの解析

データを取得した後、その内容は大量のメタデータを含んでいるため、読みやすい形式に解析する必要があります。

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

        # 解析ヘッダー情報（送信日時、件名、送信者、受信者）
        for header in headers:
            if header['name'] == 'Date':
                email_data['Date'] = header['value']
            elif header['name'] == 'Subject':
                email_data['Subject'] = header['value']

        # メール本文の解析
        for part in parts:
            if part['mimeType'] == 'text/plain' or part['mimeType'] == 'text/html':
                data = part['body']['data']
                text = urlsafe_b64decode(data.encode('ASCII')).decode('UTF-8')
                email_data['Text'] = text
                break  # 最初に一致した部分のみを取得

        return email_data

    except Exception as error:
        print(f'An error occurred: {error}')
        return None
```

## まとめ

ここまでで、Gmail API の基本的な使用方法について説明しました。

次に進む前に、いくつか準備が必要です。

OpenAI の API と接続し、メール内容を ChatGPT に送信して解析を行えるようにします。
