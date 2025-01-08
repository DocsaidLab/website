---
slug: /
sidebar_position: 1
---

# Open Source Projects

The page you're viewing is meant for technical documentation.

- If you're interested in reading related papers, please visit: [**Papers**](https://docsaid.org/papers/intro).
- For more technical insights and discussions, please visit: [**Blog**](https://docsaid.org/blog).

## 📂 Public Projects Overview

Currently, we have several completed projects available on GitHub, including:

### Tools and Integrations

- [**AutoTraderX**](./autotraderx/index.md):

  This is a record of our practice integrating with Taiwan's securities trading brokers. We have explored the "Yuanta Securities" API so far, and we plan to explore "Fubon Securities" next, but we have not yet scheduled the time.

  :::tip
  If you're asking about our development experience? It’s probably one filled with lingering fear. 😓

  We hope other brokers will provide a better development experience.
  :::

  ***

- [**Capybara**](./capybara/index.md):

  This project defines structures commonly used in computer vision, such as `Boxes`, `Polygons`, and `Keypoints`.

  In addition, it contains tools for image processing (OpenCV), model architecture (PyTorch), inference tools (ONNXRuntime), and environment configuration, all of which are frequently used in our work.

  ***

- [**DocsaidKit (soon to be deprecated)**](./docsaidkit/index.md):

  After some time of usage, we've decided to split this toolkit by removing the PyTorch-related training tools and keeping only the model inference and image processing functionalities.

  The project has now been split into three parts:

  - [**Capybara**](https://github.com/DocsaidLab/Capybara): Model inference and image processing.
  - [**Chameleon**](https://github.com/DocsaidLab/Chameleon): Purely PyTorch training tools.
  - [**Otter**](https://github.com/DocsaidLab/Otter): PyTorch-Lightning-based training tools.

  By splitting these modules, we gain more flexibility during training and deployment, making them easier to maintain.

  :::tip
  You might wonder why these packages have such names? Did we drink too much? 🤔🤔🤔

  Not at all! If you look at the papers from major institutions, they often come up with strange names like these. We’re just paying tribute to the masters... (?)
  :::

  ***

- [**GmailSummary**](./gmailsummary/index.md):

  This project is a record of our practice integrating Gmail and OpenAI, and its functionality may be rendered obsolete by future updates to Google and OpenAI's APIs.

  This project worked for several months but has now stopped because we've exhausted the funds we allocated to OpenAI.

  ***

- [**WordCanvas**](./wordcanvas/index.md):

  Previously, we created several tools for synthetic training data, but they felt too scattered. So, we abstracted some basic functionalities and integrated them into a new tool. The main function of this project is to render font files into images.

### Deep Learning Projects

- [**DocAligner**](./docaligner/index.md):

  This is a document alignment project that locates the four corners of a document.

  Although this feature is simple, it can be applied in many scenarios. Currently, it only locates the four corners, but I will add more functionalities if I have time.

  ***

- [**DocClassifier**](./docclassifier/index.md):

  This is a document classification project that classifies documents into different categories.

  This project offers a training module. All of my model projects use the same construction logic, so if you're interested in other models, you can refer to this project to build your own training environment.

  ***

- [**MRZScanner**](./mrzscanner/index.md):

  This function recognizes the MRZ region on documents.

  Initially, I aimed to build an end-to-end model, but the results didn't meet expectations. However, I still achieved some small outcomes, so I organized it into an open-source project to help others in need.

## 🚧 Development and Unreleased Projects

In addition to the above public projects, there are other projects currently under development or in internal testing.

If you have any particular interests or ideas, feel free to contact me.

## 🌍 Multilingual Support

Currently, we write primarily in Chinese, and then translate into other languages.

Given our limited resources, we cannot handle all the translation work ourselves. Therefore, various GPTs in the market help us with the translations, and we manually proofread the results to eliminate visible errors.

If you encounter:

- **Broken or incorrect links**
- **Incorrect translations**
- **Misunderstandings**

Feel free to leave a comment at the end of the article, and we will schedule time to fix it.

:::info
Another way is to raise an issue in the GitHub discussion forum:

- [**Help Us Improve Translation Accuracy**](https://github.com/orgs/DocsaidLab/discussions/12)
- [**翻訳の正確性向上のためのご協力をお願いします**](https://github.com/orgs/DocsaidLab/discussions/13)

Alternatively, you can directly submit a PR, and after confirmation, we can merge it into the main project branch, saving time and effort.
:::

## 🍹 Finally

If you have any questions or are interested in our work, feel free to email us:

- **docsaidlab@gmail.com**

You can choose to send an email or leave a comment on the website; we will see it.

Thank you for reading and supporting us. We hope this site can offer you help and inspiration!
