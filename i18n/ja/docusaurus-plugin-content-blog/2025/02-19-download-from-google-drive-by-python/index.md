---
slug: download-from-google-drive-using-python
title: PythonでGoogle Driveからファイルをダウンロードする
authors: Z. Yuan
image: /ja/img/2025/0219.webp
tags: [python, google-drive, download]
description: 小さなファイルも大きなファイルもダウンロードできます。
---

私たちは Google Drive からファイルをダウンロードする Python プログラムを作成しましたが、時々はうまくいくのに、時々は意味不明な HTML だけがダウンロードされてしまうことがあります…？

これは確実にプログラムに問題がありますので、修正が必要です。

<!-- truncate -->

## なぜ HTML だけがダウンロードされるのか？

Google Drive に GET リクエストを送信してファイルをダウンロードしようとすると、ファイルが小さい（通常 100MB 未満）場合、Google は直接ファイル内容を返し、問題なくダウンロードできます。

しかし、ファイルが大きい場合、Google は「ウイルススキャン警告ページ」を表示し、そのファイルが完全にスキャンされていないことを警告し、ユーザーにダウンロードを確認するボタンを提供します。

ブラウザを使用している場合、手動でボタンをクリックしてダウンロードできますが、Python プログラムを使用している場合、追加のメカニズムでボタンをクリックするシミュレーションを行ったり、HTML ページ内のダウンロードリンクを解析しない限り、この警告ページ自体の HTML がダウンロードされてしまい、実際のファイルはダウンロードできません。

:::info
予期しない HTML ファイルは、おそらく以下の内容になります：

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Google Drive - Virus scan warning</title>
    <meta http-equiv="content-type" content="text/html; charset=utf-8" />
   ... <!-- その他のHTML内容 -->
</html>
```

:::

## 解決方法：二回目のリクエスト

原因がわかったので、次の方法で解決できます：

1. **最初のリクエスト**：ファイル ID を使って`https://docs.google.com/uc?export=download`に GET リクエストを送信します。
2. **レスポンスの確認**：もし HTTP ヘッダーに`content-disposition`が含まれていれば、ファイル本体を取得できたことになりますので、そのままダウンロードできます。含まれていなければ、ウイルススキャン警告ページに滞在していることになるので、もう一度リクエストを送る必要があります。
3. **確認トークンの取得**：
   - **クッキーから取得**：Google はクッキー内に`download_warning_xxxxx`のようなキーを置き、その中にトークンが含まれている場合があります。例えば、`token = session.cookies.get('download_warning_xxxxx')`
   - **HTML から取得**：場合によっては Google がトークンをクッキーではなく HTML のフォーム内に置くことがあります。例えば：
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
     この場合、[**BeautifulSoup**](https://www.crummy.com/software/BeautifulSoup/)を使ってすべての hidden フィールドを取得し、`confirm`や`uuid`などのパラメータを取得します。
4. **二回目のリクエストの組み合わせ**：`token`を取得した後、それをリクエストに含めるか、フォームの`action` URL と対応する hidden パラメータを全て含めて、再度リクエストを送信することで、実際のファイルをダウンロードできます。

## コード実装

必要なパッケージをインストールします：

```bash
pip install requests tqdm beautifulsoup4
```

次に、以下の Python 関数を使用して、ファイル ID、保存するファイル名、ダウンロード後の保存先フォルダを渡すことで、二回目のリクエストが必要かどうかを自動的に判断し、ダウンロードを成功させます：

```python title="download_from_google.py"
import os
import re
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

def download_from_google(file_id: str, file_name: str, target: str = "."):
    """
    Google Driveからファイルをダウンロードし、大きなファイルのために確認トークンを処理します。

    引数：
        file_id (str):
            Google DriveからダウンロードするファイルのID。
        file_name (str):
            ダウンロードしたファイルの保存名。
        target (str, optional):
            ファイルを保存するディレクトリ。デフォルトはカレントディレクトリ（"."）。

    例外：
        Exception: ダウンロードに失敗した場合や、ファイルを作成できない場合。

    備考：
        この関数は小さなファイルと大きなファイルの両方を処理します。大きなファイルの場合、Googleの確認トークンを自動的に処理し、ウイルススキャン警告やファイルサイズ制限を回避します。

    例：
        カレントディレクトリにファイルをダウンロード：
            download_from_google(
                file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                file_name="example_file.txt"
            )

        特定のディレクトリにファイルをダウンロード：
            download_from_google(
                file_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                file_name="example_file.txt",
                target="./downloads"
            )
    """
    # 最初の試み：docs.google.com/uc?export=download&id=ファイルID
    base_url = "https://docs.google.com/uc"
    session = requests.Session()
    params = {
        "export": "download",
        "id": file_id
    }
    response = session.get(base_url, params=params, stream=True)

    # すでにContent-Dispositionが含まれていれば、直接ファイルを取得できたことを意味します
    if "content-disposition" not in response.headers:
        # 最初にクッキーからトークンを取得
        token = None
        for k, v in response.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break

        # クッキーにトークンがない場合は、HTMLから解析
        if not token:
            soup = BeautifulSoup(response.text, "html.parser")
            # よく見られるケース：HTML内にform#download-formがある
            download_form = soup.find("form", {"id": "download-form"})
            if download_form and download_form.get("action"):
                # action内のURLを取得、通常はdrive.usercontent.google.com/download
                download_url = download_form["action"]
                # すべてのhiddenフィールドを収集
                hidden_inputs = download_form.find_all("input", {"type": "hidden"})
                form_params = {}
                for inp in hidden_inputs:
                    if inp.get("name") and inp.get("value") is not None:
                        form_params[inp["name"]] = inp["value"]

                # これらのパラメータで再度GETリクエスト
                # 注意：元のactionは相対パスの可能性があるので、ここでは完全なURLを使用
                response = session.get(download_url, params=form_params, stream=True)
            else:
                # もしくはHTML内に直接confirm=xxxが含まれていることもある
                match = re.search(r'confirm=([0-9A-Za-z-_]+)', response.text)
                if match:
                    token = match.group(1)
                    # confirmトークンを含めて再度docs.google.comにリクエスト
                    params["confirm"] = token
                    response = session.get(base_url, params=params, stream=True)
                else:
                    raise Exception("レスポンス内でダウンロードリンクまたは確認パラメータが見つかりませんでした。ダウンロード失敗。")

        else:
            # クッキーから取得したトークンを使って再リクエスト
            params["confirm"] = token
            response = session.get(base_url, params=params, stream=True)

    # ダウンロード先のディレクトリが存在するか確認
    os.makedirs(target, exist_ok=True)
    file_path = os.path.join(target, file_name)

    # ファイルをchunkごとにローカルに書き込み、進行状況バーを表示
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

        print(f"ファイルが正常にダウンロードされました: {file_path}")

    except Exception as e:
        raise Exception(f"ファイルのダウンロードに失敗しました: {e}")
```

## 使用方法

ファイル ID が`YOUR_FILE_ID`で、`big_model.onnx`という名前で`./models`フォルダに保存する場合、以下のように呼び出します：

```python
download_from_google(
    file_id="YOUR_FILE_ID",
    file_name="big_model.onnx",
    target="./models"
)
```

完了後、`./models`フォルダに`big_model.onnx`が正常にダウンロードされ、コマンドラインで進行状況が表示されます。

## コマンドラインツール

コマンドラインで操作したい場合、次のようにコードを追加してラップすることができます：

```python title="download_from_google_cli.py"
from download_from_google import download_from_google
import argparse

def main():
    parser = argparse.ArgumentParser(description="Google Driveからファイルをダウンロードします。")
    parser.add_argument("--file-id", required=True, help="Google DriveのファイルID。")
    parser.add_argument("--file-name", required=True, help="出力ファイル名。")
    parser.add_argument("--target", default=".", help="出力ディレクトリ。デフォルトはカレントディレクトリ。")
    args = parser.parse_args()

    download_from_google(file_id=args.file_id, file_name=args.file_name, target=args.target)

if __name__ == "__main__":
    main()
```

このコードを`download_from_google_cli.py`として保存すれば、次のようにコマンドラインから実行できます：

```bash
python download_from_google_cli.py \
 --file-id YOUR_FILE_ID \
 --file-name big_model.onnx \
 --target ./models
```

特に問題がなければ、ダウンロードが開始され、進行状況が表示されます。

私たちは 70MB および 900MB のファイルでテストしましたが、どちらも正常にダウンロードできました。900GB のファイルについては…（🤔 🤔 🤔）

試したことはありませんが、そのような大きなファイルは手元にありませんので、また後日機会があれば更新します！
