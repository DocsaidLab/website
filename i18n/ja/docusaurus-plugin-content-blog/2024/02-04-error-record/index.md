---
slug: error-record
title: 日常エラー解消記録
authors: Z. Yuan
tags: [error, record]
image: /ja/img/2024/0204.webp
description: 簡単な問題とその解決方法を記録。
---

プログラミングをしていると、いつもさまざまな問題に直面します。

ここでは、些細な問題とその解決方法を記録します。

:::tip
この記事は随時更新されます。
:::

<!-- truncate -->

## 1. `npx docusaurus start` を実行した際に以下のエラーが発生

- **説明：**

  ```bash
  file:///home/user/workspace/blog/node_modules/@docusaurus/core/bin/docusaurus.mjs:30
  process.env.BABEL_ENV ??= 'development';
                      ^^^

  SyntaxError: Unexpected token '??='
  ```

- **解決方法：**

  `??=` 演算子は、Node.js 15.0.0 以上でサポートされています。

  ```bash
  nvm install node
  nvm use node
  ```

## 2. choco コマンドが認識されない

- **説明：**

  ```shell
  PS C:\Windows\System32> choco install git -y
  >>
  choco : 'choco' という用語は、コマンドレット、関数、スクリプト ファイル、または操作可能なプログラムの名前として認識されません。名前が正しいことを確認し、パスが含まれている場合は、そのパスが正しいことを確認して再試行してください。
  行:1 文字:1
  + choco install git -y
  + ~~~~~
      + CategoryInfo          : ObjectNotFound: (choco:String) [], CommandNotFoundException
      + FullyQualifiedErrorId : CommandNotFoundException
  ```

- **解決方法：**

  これは Chocolatey のインストールが成功していないことを意味します。失敗の原因として多いのは、PowerShell を「管理者として実行」していないことです。

  PowerShell を「管理者として実行」してから、Chocolatey のインストールコマンドを再実行してください。

  ```shell
  Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
  ```

## 3. Chocolatey のインストール失敗

- **説明：**

  ```shell
  警告: An existing Chocolatey installation was detected. Installation will not continue. This script will not overwrite existing installations.
  If there is no Chocolatey installation at 'C:\ProgramData\chocolatey', delete the folder and attempt the installation again.

  Please use choco upgrade chocolatey to handle upgrades of Chocolatey itself.
  If the existing installation is not functional or a prior installation did not complete, follow these steps:
  - Backup the files at the path listed above so you can restore your previous installation if needed.
  - Remove the existing installation manually.
  - Rerun this installation script.
  - Reinstall any packages previously installed, if needed (refer to the lib folder in the backup).

  Once installation is completed, the backup folder is no longer needed and can be deleted.
  ```

- **解決方法：**

  これはすでに Chocolatey がインストールされていることを意味します。古いインストールを削除してから、再インストールしてください。

  ```shell
  Remove-Item "C:\ProgramData\chocolatey" -Recurse -Force
  ```

## 4. リモートマシンのポート転送

- **説明：**

  リモートマシンでサービス（例：TensorBoard）を起動したが、直接アクセスできない場合、ローカルマシン経由でポート転送を行う必要があります。

- **解決方法：**

  リモートマシン上のサービスがポート 6006 で動作しており、ローカルマシンで同じポート 6006 でアクセスしたい場合、SSH でログインする際に `-L` パラメータを使用してポート転送を設定します：

  ```bash
  ssh -L 6006:localhost:6006 user@remote_ip_address
  ```

  これにより、ローカルマシンから `http://localhost:6006` を通じて、リモートマシン上の TensorBoard サービスにアクセスできるようになります。

## 5. 開発環境とデプロイ環境でのウェブレンダリングの不一致

- **説明：**

  `custom.css` でブログのレイアウトスタイルを設定しました：

  ```css
  .container {
    max-width: 90%;
    padding: 0 15px;
    margin: 0 auto;
  }
  ```

  デプロイ時、このスタイルが他のより高い優先順位のスタイルに上書きされましたが、開発環境では正常に適用されています。

- **解決方法：**

  より具体的にターゲットを指定します：

  ```css
  body .container {
    max-width: 90%;
    padding: 0 15px;
    margin: 0 auto;
  }
  ```

## 6. TurboJPEG で画像を読み込む際に警告が表示される

- **説明：**

  画像を読み込む際、以下のような警告メッセージが表示されます：

  ```shell
  turbojpeg.py:940: UserWarning: Corrupt JPEG data: 18 extraneous bytes before marker 0xc4
  turbojpeg.py:940: UserWarning: Corrupt JPEG data: bad Huffman code
  turbojpeg.py:940: UserWarning: Corrupt JPEG data: premature end of data segment
  ```

- **解決方法：**

  見ていて煩わしいので、これらのデータをフィルタリングして除去します：

  ```python
  import cv2
  import warnings

  data = ['test1.jpg', 'test2.jpg', 'test3.jpg']

  for d in data:
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always", UserWarning)

      # 画像を読み込み、警告があるか確認
      cv2.imread(d)

      # 警告があれば削除
      if w:
        data.remove(d)

  # 最後に JSON やその他の方法で洗浄後のデータを記録可能
  ```

## 7. `Docusaurus` デプロイ後に `showLastUpdateTime: true` が機能しない

- **説明：**

  `docusaurus.config.js` に `showLastUpdateTime: true` と `showLastUpdateAuthor: true` を設定しましたが、デプロイ後、全ページで同じ時間と作者が表示されてしまう。

- **解決方法：**

  デプロイ時に分岐のチェックアウト設定が間違っていたため、`git` が正しく最終更新時間と作者を取得できませんでした。

  以下のように変更します：

  ```yaml
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
  ```

  `fetch-depth: 0` を設定するだけで、この問題は解決します。

## 8. Docker コンテナ内のエラーログを確認する

- **説明：**

  Docker コンテナ内でサービスを実行しているが、エラーが発生し正常に動作しないため、エラーログを確認する必要がある。

- **解決方法：**

  まず、対象のコンテナ ID を確認する：

  ```bash
  docker ps
  ```

  次に、コンテナ内に入ってログを確認する：

  ```bash
  docker exec -it container_id /bin/bash
  cat /path/to/logfile
  ```

  または、直接ログを表示する：

  ```bash
  docker logs container_id
  ```
