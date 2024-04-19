---
slug: /
sidebar_position: 1
---

# Introduction

Welcome to my website.

I'm Zephyr, an AI engineer from Taiwan and the founder of this organization.

## What's Here

::: info
I would like to inform you that at present, only the technical documentation has been translated into English. I am committed to making all content fully accessible and plan to progressively update the remaining sections in the near future. Thank you for your patience and understanding as I work to enhance the offerings.
:::

- For paper reviews, see next door: [**Papers**](https://docsaid.org/papers/intro).
- For the blog section, it's the door after next: [**Blog**](https://docsaid.org/blog). -->

＊

I have completed several projects on Github, including:

1. [**DocsaidKit**](https://github.com/DocsaidLab/DocsaidKit): This is a toolbox of small utilities I use regularly.
2. [**DocAligner**](https://github.com/DocsaidLab/DocAligner): This project provides text localization functionality, mainly used to pinpoint the four corners of a document.
3. [**DocClassifier**](https://github.com/DocsaidLab/DocClassifier): This project is used for calculating text similarity, inspired by facial recognition systems.
4. [**GmailSummary**](https://github.com/DocsaidLab/GmailSummary): This project summarizes large volumes of Gmail messages, integrating Gmail with OpenAI.
5. [**WordCanvas**](https://github.com/DocsaidLab/WordCanvas): This project generates text images to solve the problem of insufficient training data for OCR models.

---

Of course, there's more, but most are still in the planning stages, so I won't go into details now.

:::info
**DocClassifier** has a special origin. It was conceived by a facial recognition system expert, also my friend, [**Jack, L.**](https://github.com/Jack-Lin-NTU). Knowing that I have a website, he completed the initial development and feasibility testing, then entrusted the idea to me to finalize the details and release it here.

Special thanks to him for his contribution.
:::

## Are You an Engineer?

Yes.

I enjoy using scientific knowledge to solve practical problems and find joy in doing so.

＊

Years ago, those who worked with images called themselves "computer vision engineers," those who worked with text were "natural language processing engineers," and distinctions even existed for "machine learning engineers" and "deep learning engineers," with a hierarchy of disdain among these groups within certain small circles.

Regardless of the title, everyone looked down on "AI engineers":

- Hmph! Just another senseless title used by companies/positions/job seekers to sound impressive!

As time passed, we were astounded to find that text, images, and audio data from various dimensions have unified under the wave of "foundation models," becoming essential for topping leaderboards, scoring high, and publishing papers!

People realized that these engineers were all the same, just working in different dimensions doing similar things.

Soon, no one differentiated between types of engineers because academic research topics had become interdisciplinary, requiring knowledge across various fields to continue research. Thus, it became challenging to define what exactly one is.

Looking back, we see:

- Yes, that's right! AI Engineer!

## Github Organization Account

> As recommended by Github: Want to code together? Start an organization!

The purpose of creating an organization is simple; it's not really about starting a big enterprise but about finding that Github's organizational accounts offer more features than personal accounts, allowing others to join in coding and research interesting topics together, importantly, without extra cost.

With so many advantages, why not create one for fun?

＊

The hardest part about this whole thing was coming up with a name.

Analyzing text is my daily job, including image text recognition, image fraud detection, topic classification, keyword extraction, and more. From my perspective, text isn't just words; it can be an image, a video, a song, a dataset, or even a person's behavior. Thus, anything analyzable or worth analyzing, in my view, can be considered text (chaotic evil?).

So ultimately, I chose a name related to this field, called **DOCSAID**.

This name combines "DOC" and "SAID," roughly meaning:

- **The moment a document is created, it has already expressed its intended content.**

So, what exactly did these documents say? Now we just need to analyze them!

Interestingly, after choosing this name, I found it also contains the letters AI, which was a delightful surprise.

## What Is This Place?

This is a website generated through automatic deployment with Github Pages.

:::info
Surprise! So if Github goes down, so does this site.

That's why we need to take good care of Github and not let it break. (What a conclusion!)
:::

I've always had a habit of blogging. I used **Dot Blog** for about two years before giving it up because it was too cumbersome.

Then I tried **Wordpress.org** for about half a year before

 giving up on it too.

- **Engineers should write documentation in Markdown! Isn't Elementor laggy?**

Recently, while wandering around, I stumbled upon Meta's open-source [**Docusaurus**](https://docusaurus.io/), found its features adequate, and the deployment process straightforward:

- **Writing and deploying with Markdown, +10 points**
- **Automatic deployment, +10 points**
- **Usable with React, +10 points**
- **Many reasons to add points, simply put, it's very useful.**

So, I decided it was the right choice.

Docusaurus allows me not only to write articles but also provides technical document management, which you are viewing now. It has given me something new to work on:

- Writing technical documentation.

Documenting is truly a refining activity; it helps engineers clarify their thoughts, reorganize their knowledge, and assist others in understanding their ideas quickly. The only downside is "time consumption"; spending a significant amount of time on documentation may slow down engineering progress.

However, I believe it is worth it.

＊

You can explore the contents via the sidebar menu. I've completed some parts already.

If you find some content missing, it means I'm still working on it, so please be patient.

## How to Adjust Models

This is probably the topic you're most interested in.

Based on the topics I've defined, along with the models I provide, I believe they can solve most application scenarios.

I also know that some scenarios might require better model performance, thus requiring you to collect your dataset and fine-tune the model.

You might get stuck at this step, like most people do, but don't worry.

- **Scenario One: You know my project's functions fit your needs, but you don't know how to adjust:**

    In this case, you can send me an email with your requirements and provide the "dataset you want to solve." I can help you fine-tune the model, so you can achieve better model performance.

    There's no charge, but I can't be pressed for time, and I can't guarantee it will be done. (This is important!)

    Although I'm working on open-source projects, I'm not just twiddling my thumbs. When the time comes, the model will naturally update, and just by sending an email, you might get a better model. At the very least, it could be considered a win-win, right?

- **Scenario Two: You want to develop a specific function:**

    Then write to me to discuss it. If I find it interesting, I'll be happy to help you develop it, but I hope you can prepare a substantial dataset first, because even if I'm interested, I might not have the time to obtain enough data or some special data might require specific channels to acquire.

    This scenario, like the one above, does not involve charges, but I can't be pressed for time, and I can't guarantee execution.

    :::tip
    If the specific function is for those public model competitions? The answer is no. Those competitions often have copyrights and related restrictions. If complained, the organizers might trouble me.
    :::

- **Scenario Three: You are pursuing rapid development of a specific function:**

    When time becomes your primary consideration, we can turn to a commissioned development cooperation mode. Based on your requirements, I will offer a reasonable quote. The calculation is straightforward: based on the time I need to invest, plus some necessary development costs, that's it.

    Take the [**DocAligner**](https://github.com/DocsaidLab/DocAligner) project as an example, from scratch, based on my own development speed, including literature review, initial exploration, model design testing, etc., normally requires a six-month development cycle. If this were a commissioned development case, the quote would be around thirty thousand dollars. This price covers the entire process from data collection, model training, to system deployment, just as you see it. If a complex, large-scale project involving multiple models communicating and coordinating is needed, we can first discuss specific needs and scheduling, then provide a quote.

    Additionally, you might overlook the "necessary costs" mentioned above, referring to training machines. If it's a small project, like those I've open-sourced, then my own training machines are sufficient to complete the tasks, and there are no cloud machine rental fees. But for a previous facial recognition project, due to the large size of the model, only renting cloud machines was an option. With the dataset and machines in place, it took two months to complete a model. In such scenarios, development costs come to about nine thousand dollars, but cloud training machine rental costs amount to twenty thousand dollars (a money pit), and this cost must be considered.

    Also, I won't accept proposals like: "I want the OOO function of XXX's LLM," although I am equally skilled at building generative models, but only on a small scale. Most things related to LLMs are likely to involve expenditures in the hundreds of millions of dollars. Currently, I remain a user as far as training is concerned. Sorry for the disturbance.

    Furthermore, the purpose of my commissioned development is to rapidly meet your needs. If I succeed, it's a win-win outcome, and we can happily settle accounts. If the goals are not met, then consider it a lack of skill on my part, and I won't charge you any development fees, but if cloud machines are rented, those costs still apply. Although I can absorb the development costs myself, the expenses for cloud machines are too high for me to bear alone.

    :::tip
    The ownership issues of commissioned development must be discussed upfront. Generally, I retain ownership of the project, but you are free to use it. If you want to buy out the project, I will provide a separate quote.

    I do not recommend proposals to buy out projects, as this does not align with the philosophy of continuous improvement. With technological advancements, today's solutions might soon be superseded by updated methods. If you buy out a project, as time passes, you might find this investment has lost its original value.
    :::
    :::tip
    You might not understand the issue of project ownership.

    But think about it carefully; maybe you just want to "drink milk," not really "raise a cow."

    - How tiring is it to raise a cow? (Needing to maintain engineers for the project)
    - It takes up space and is hard to care for. (Setting up training machines, renting cloud machines is expensive, buying main machines tends to break down)
    - It's sensitive to cold and heat. (Model tuning drives you to doubt life)
    - And it dies on a whim. (Fails to meet expected results)
    - Truly a loss. (Spent money to buy out the project)

    Moreover, the most valuable part of most projects is the dataset, followed by the thought process of the solution. Without open-sourcing the private dataset, having a piece of code is mostly just for viewing. If, after careful consideration, you still insist on buying out the project, I won't stop you; come on then.
    :::

    :::info
    - **In all forms of development projects, we absolutely do not open-source the data you provide unless permitted by you.**
    - **Normally, data is only used for updating models.**
    - **To submit datasets: docsaidlab@gmail.com**
    :::

## Conclusion 🍹

I am an AI engineer, not an omnipotent one; I can't solve all problems.

You can tell from this site that my skill tree is somewhat skewed; I'm best at: "Given an input, requiring a specific output," and I'll try to solve it with models, just like my open-source projects. As for other areas, I'm still learning, hoping to become more comprehensive and solve more problems in the future.

＊

In this era, we are fortunate to have big companies like Google, OpenAI, Meta, NVIDIA, and an active open-source community driving technological development. Their contributions allow us to learn cutting-edge technologies and gain rich knowledge.

I believe every small contribution is part of our collective effort to create a better life.

If my projects can help you in any way, feel free to use them.

＊

2024 Zephyr
