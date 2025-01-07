---
slug: /
sidebar_position: 1
---

# Open Source Project

The page you're currently viewing is used for writing technical documentation.

- If you're interested in reading related paper shares, please visit: [**Papers**](https://docsaid.org/en/papers/intro).
- For more technical insights and discussions, please check out: [**Blog**](https://docsaid.org/en/blog).

## 📂 List of Public Projects

Currently, I have made several completed projects public on GitHub, including:

### Tools and Integrations

- [**AutoTraderX**](./autotraderx/index.md):

  This is a record I left while practicing integrating the systems of Taiwan's securities brokers. I have only explored the API of "MasterLink" so far, and I plan to explore "Fubon Securities" next, but I haven't scheduled the time yet.

  :::tip
  If you ask me about my development experience? It's probably a bit of a nightmare. 😓
  :::

  ***

- [**Capybara**](./capybara/index.md):

  This defines some commonly used structures in the field of computer vision, such as `Boxes`, `Polygons` and `Keypoints`.

  Additionally, it includes some image processing (opencv), model architectures (pytorch), inference tools (onnxruntime), and environment configurations. These are tools we frequently use in our work.

  ***

- [**DocsaidKit (soon to be deprecated)**](./docsaidkit/index.md):

  After some time of use, we decided to split this toolkit by removing the PyTorch-related training tools and only keeping the model inference and image processing functions.

  The project was eventually split into three parts:

  - [**Capybara**](https://github.com/DocsaidLab/Capybara): Functions related to model inference and image processing.
  - [**Chameleon**](https://github.com/DocsaidLab/Chameleon): Pure PyTorch training tools.
  - [**Otter**](https://github.com/DocsaidLab/Otter): Training tools based on PyTorch-Lightning.

  :::tip
  You might ask, what's up with the names of these packages? Did you drink too much? 🤔🤔🤔

  Not at all! As you can see, many papers from major institutions nowadays feature strange names. We are just paying tribute to the masters!
  :::

  ***

- [**GmailSummary**](./gmailsummary/index.md):

  This is a record I left while practicing integrating Gmail and OpenAI. The content may become outdated after future updates to Google and OpenAI's APIs.

  This project worked for a few months, but I have already exhausted the funds for OpenAI, so it is no longer functional.

  ***

- [**WordCanvas**](./wordcanvas/index.md):

  This project was originally created to abstract out some basic features for generating synthetic training data, and it integrates them into a tool that mainly renders font files into images.

### Deep Learning Projects

- [**DocAligner**](./docaligner/index.md):

  This is a document alignment project, which locates the four corner points of a document.

  Though the function is simple, it can be applied in many scenarios. Currently, it only locates the four corner points, but I plan to add more features if I have time.

  ***

- [**DocClassifier**](./docclassifier/index.md):

  This is a document classification project, which classifies documents into different categories.

  This project offers a training module. I follow the same building logic for each of my model projects. If you're interested in other models, you can refer to this project to set up your own training environment.

  ***

- [**MRZScanner**](./mrzscanner/index.md):

  This function recognizes the MRZ (Machine Readable Zone) area on documents.

  I initially aimed to develop an end-to-end model, but the results were not as expected. However, I still made some small progress, so I turned it into an open-source project, hoping to help those who need it.

## 🚧 In Development and Unreleased Projects

In addition to the above public projects, there are some projects in development or in internal testing.

If you're particularly interested in any topic or idea, feel free to contact me.

## 🌍 Multilingual Support

I write the content primarily in Chinese, and then translate it into other languages.

However, my ability is limited, and I can't handle all the translation work by myself, so I rely on various `GPTs` available on the market to help with the task.

The standard process is: I extract paragraphs from each article and submit them to `GPTs` for translation. After receiving the translation, I manually proofread it to fix any obvious errors.

If you come across:

- **Broken or incorrect links**
- **Incorrect translations**
- **Misunderstandings**

Feel free to leave a comment below, and I will prioritize fixing them.

:::info
Another way is to raise an issue on the GitHub discussion board:

- [**Help Us Improve Translation Accuracy**](https://github.com/orgs/DocsaidLab/discussions/12)

Alternatively, you can raise an issue on the GitHub discussion board.
:::

## 🔄 Adjusting Models

This might be the topic you're most interested in.

Based on the topics I define and the models I provide, I believe they will solve most application scenarios.

However, I also know that some scenarios may require better model performance, which means you might need to collect your own dataset and fine-tune the model.

You might get stuck at this step, as most people do, but don’t worry.

### Scenario One

You know that the project functions I provide meet your needs, but you're not sure how to adjust them.

In this case, you can send me an email with your needs and the dataset you want to solve. I can help with model fine-tuning so you can achieve better results.

No charges, but no timeline guarantees either. I can't promise it will be executed. (This is important!)

Although I work on open-source projects, I’m not just idling around. When the opportunity arises, the model will naturally be updated. You just need to send an email, and "maybe" get a better model. It could be a win-win!

### Scenario Two

You want to develop a specific function but aren’t in a hurry.

In that case, let’s discuss it via email. If I find it interesting, I’d be happy to help you develop it. However, I’d like you to prepare a sufficient dataset, because even if I’m interested, I may not have the time to gather enough data, or some specific data might require special channels to obtain.

This scenario is the same as the previous one — no charges, but no timeline guarantees either. I can’t promise execution.

:::tip
If the specific function is for public model competitions, the answer is no. These competitions often have copyrights and related restrictions, and if complaints arise, the organizers will come after me.
:::

### Scenario Three

You need a specific feature developed quickly.

When time is your primary concern, we can turn to a commissioned development collaboration. Based on your needs, I will propose a reasonable price according to my development time.

Generally speaking, I will retain ownership of the project, but you are free to use it. I do not recommend buying out a project, as it doesn’t align with the continuous progress concept. With technological advancements, today’s solution might soon be replaced by a newer method. If you buy out a project, over time, you may find that the investment no longer holds its original value.

:::tip
You may not fully understand the issue of project ownership.

Think about it: perhaps you just want to "drink the milk," not actually "raise the cow."

- How tiring is raising a cow? (You need to maintain the project with engineers)
- It takes up space and is hard to take care of. (Building training machines, renting cloud machines is expensive, buying hardware is prone to failure)
- It’s sensitive to temperature. (Tuning models until you question your life)
- It could suddenly break down. (Not achieving the expected results)
- It’s quite a loss. (Spending money to buy out a project)
  :::

Moreover, the most valuable part of most projects is the dataset, followed by the thought process behind the solution. Without releasing private datasets, the most you can do with the code is for viewing purposes.

If, after careful consideration, you still insist on buying out the project, I won’t stop you. Go ahead.

## ✉️ Contact Information

If you have any questions or are interested in my work, feel free to reach out to me anytime!

This is the email I use for job applications: **docsaidlab@gmail.com**. You can send an email, or simply find an article on this website and leave a comment below. I will see it.

## 🍹 Final Note

Unless given your permission, we will never open-source the data you provide in any form of development projects. The data will only be used to update the model.

Thank you for reading and supporting. we hope **DOCSAID** can bring you help and inspiration!

＊

2024 © Zephyr
