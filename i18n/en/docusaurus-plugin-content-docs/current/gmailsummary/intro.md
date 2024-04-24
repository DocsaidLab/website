---
sidebar_position: 1
---

# Introduction

The core functionality of this project is "**Email Summary**".

- [**GmailSummary Github**](https://github.com/DocsaidLab/GmailSummary)

![title](./resources/title.jpg)

:::info
This project isn't a ready-to-use Python tool; instead, it's an example of API integration. If you happen to have similar needs, you can refer to the instructions in this project for corresponding modifications and further development.
:::

## Overview

In daily life, we often start receiving activity update emails from GitHub repositories because we've clicked the "Watch" option on those repositories. These updates include but are not limited to discussions on new features, issue reports, pull requests (PR), and bug reports.

For example, if you follow some GitHub projects and opt for "All activity":

- [**albumentations**](https://github.com/albumentations-team/albumentations): About 10 emails per day.
- [**onnxruntime**](https://github.com/microsoft/onnxruntime): About 200 emails per day.
- [**PyTorch**](https://github.com/pytorch/pytorch): About 1,500 emails per day.

One can imagine that if you follow even more projects, you'll receive over 5,000 emails per day.

＊

**Does anyone really read all these emails "without missing any"?**

＊

Well, I don't. Usually, I just delete them without even looking.

Therefore, as engineers seeking efficiency (or laziness), we must think about how to solve this problem.

## Problem Breakdown

To tackle the issue of a large number of emails, we can break it down into two parts: automatic downloading and automatic analysis.

### Automatic Downloading

To be able to automatically download these emails from Gmail and then extract key information.

Let's briefly consider some feasible solutions:

1. **Using Services like Zapier or IFTTT**

    - [**Zapier**](https://zapier.com/): This is an automation platform focused on enhancing productivity, supporting connections to over 3,000 different web applications, including Gmail, Slack, Mailchimp, etc. This platform allows users to create automated workflows to achieve automatic interaction between various applications.
    - [**IFTTT**](https://ifttt.com/): IFTTT is a free web service that allows users to create "if this, then that" automated tasks, known as "Applets." Each Applet consists of a trigger and an action. When the trigger condition is met, the Applet will automatically execute the action.

2. **Using Gmail API**

    - [**Gmail API**](https://developers.google.com/gmail/api): This allows us to perform operations like reading emails, writing emails, searching emails, etc., programmatically.

:::tip
Since we're already going to write code, let's not consider the first solution and go with Gmail API.
:::

### Automatic Analysis

After retrieving a large number of emails, we need to analyze these emails to extract key information.

This part isn't much of a challenge in the era of ChatGPT. We can use ChatGPT for natural language processing to extract key information from the emails.

## Conclusion

We break down the entire process into several parts:

1. **Automatic Email Downloading**: Using Gmail API.
2. **Email Content Parsing**: Implementing logic ourselves.
3. **Email Content Summarization**: Using ChatGPT.
4. **Output & Scheduling**: Outputting in Markdown format and scheduling using crontab.

The above is the core functionality of this project, and we showcase the results on the **Daily News** page.

Next, we'll introduce the implementation of these parts one by one.