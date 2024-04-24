---
sidebar_position: 6
---

# OpenAI API Calls

The OpenAI API offers a variety of functionalities, and for this project, we'll be using the "Text generation models" feature. This feature allows us to input a piece of text, and the model will continue writing based on that input.

Let's briefly go over how to use the OpenAI API.

## Installing the Package

First, we need to install the `openai` package to use the API.

```bash
pip install openai
```

## Using the API

Next, we can start using the API. Let's take a look at the usage example provided by OpenAI:

```python
# An example from OpenAI
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

We've adapted the provided example to a version that suits our needs better. You can find it in [**openai_api.py**](https://github.com/DocsaidLab/GmailSummary/blob/main/openai_api.py).

```python
import json
import os
from typing import Dict, List

import tiktoken
from openai import OpenAI


def chatgpt_summary(results: List[Dict[str, str]], model: str = 'gpt-3.5-turbo') -> str:

    # Setting `OPENAI_API_KEY` environment variable is required
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    prompt = '''
        Based on the parsed content from emails received,
        please extract key information,
        including but not limited to bug fixes, feature additions,
        discussion topics, and any notable achievements or challenges mentioned.
        Please provide detailed descriptions of what you consider important.
    '''

    prompt_final = '''
        Summarize and consolidate all the content from earlier.
        Identify key textual descriptions, including but not limited to bug fixes,
        feature additions, discussion topics, and any notable achievements or
        challenges mentioned. Please provide detailed descriptions
        of what you consider important. Additionally,
        considering there may be some proprietary terms in the content,
        please provide corresponding explanations and elaborations.
        Write the article in Traditional Chinese and elaborate on relevant
        engineering details as much as possible.
        Assume the readers are experts in the field, so feel free to describe
        additional engineering details.
        Please use paragraph descriptions and maintain narrative integrity.
    '''

    # Segmentation: every 20 contents form one segment
    results_seg = [results[i:i + 20] for i in range(0, len(results), 20)]

    responses = []
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    for i, seg in enumerate(results_seg):
        content = json.dumps(seg)

        # Estimate tokens
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

    # Aggregate segment results
    all_content = '\n\n'.join(responses)
    tokens = enc.encode(all_content)
    print(
        f'Summary all segments, length of tokens: {len(tokens)}...',
        end=' ', flush=True
    )

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