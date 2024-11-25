---
sidebar_position: 6
---

# OpenAI API の呼び出し

OpenAI の API は多くの機能を提供していますが、今回は「Text generation models」機能を使用します。この機能により、入力されたテキストを基にモデルが続きを書いてくれるようになります。

以下では、OpenAI API の使用方法を簡単に紹介します。

## パッケージのインストール

まず、`openai`パッケージをインストールする必要があります。インストールしないと使用できません。

```bash
pip install openai
```

## API の使用

次に、API を使い始める方法を見てみましょう。以下は OpenAI が提供するサンプルコードです：

```python
# OpenAIのサンプルコード
from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")

completion = client.chat.completions.create(
model="gpt-3.5-turbo",
messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
]
)

print(completion.choices[0].message)
```

これを参考にして、私たちのニーズに合うようにコードを変更しました：[**openai_api.py**](https://github.com/DocsaidLab/GmailSummary/blob/main/openai_api.py)

```python
import json
import os
from typing import Dict, List

import tiktoken
from openai import OpenAI


def chatgpt_summary(results: List[Dict[str, str]], model: str = 'gpt-3.5-turbo') -> str:

    # `OPENAI_API_KEY` 環境変数を設定する必要があります
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    prompt = '''
        受信したメールの内容に基づき、これらはGmail APIからの解析結果です。重要な情報を抽出してください。
        これには、バグ修正、機能追加、議論されたトピック、特に言及された成果や課題が含まれますが、それに限りません。
        重要だと思われる内容を詳細に記述してください。
    '''

    prompt_final = '''
        前述のすべての内容を整理し、重要な記述を見つけてください。これには、バグ修正、機能追加、
        議論されたトピック、特に言及された成果や課題が含まれますが、それに限りません。
        重要だと思われる内容を詳細に記述してください。
        最後に、内容に専門用語が含まれている可能性があることを考慮し、それに関連する説明や補足を加えてください。
        繁体字の中国語で記事を作成し、できるだけ詳細に記述してください。読者はこの分野の専門家であるため、
        記事を書く際に関連する技術的な詳細についても記述し、段落ごとに説明を行い、記述の完全性を保ってください。
    '''


    # 分段，每 20 個內容分一段
    results_seg = [results[i:i + 20] for i in range(0, len(results), 20)]

    responses = []
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    for i, seg in enumerate(results_seg):
        content = json.dumps(seg)

        # 估計 token
        tokens = enc.encode(content)
        print(f'Segment {i}: Length of tokens: {len(tokens)}')

        if len(tokens) > 16000:
            continue

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{content}\n\n{prompt}"},
            ],
            temperature=0.2,
        ).choices[0].message.content

        responses.append(response)

    # 彙整分段結果
    all_content = '\n\n'.join(responses)
    tokens = enc.encode(all_content)
    print(
        f'Summary all segments, length of tokens: {len(tokens)}...', end=' ', flush=True)

    summary = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{all_content}\n\n{prompt_final}"},
        ],
        temperature=0.2,
    ).choices[0].message.content
    print('Done.')

    return summary
```

このコードでは、Gmail API から取得したデータを ChatGPT に渡し、関連する重要な情報を要約するために API を呼び出します。また、複数のセグメントに分けて内容を送信し、最終的にすべてのセグメントをまとめて最終的な要約を取得します。
