---
sidebar_position: 1
---

import { Timeline } from "antd";
import Link from '@docusaurus/Link';
import recentUpdates from './recent_updates_data.json';

# Research Paper Notes

## Recent Updates

<Timeline mode="alternate">
  {recentUpdates.map((item, idx) => {
    const convertMdLinkToRoute = (mdLink) => {
      return mdLink
        .replace(/^.\//, '/papers/')  // 將 "./" 換成 "/papers/"
        .replace(/\.md$/, '')         // 移除 .md 副檔名
        .replace(/\/index$/, '')      // 移除 /index
        // 將最後一段若是 YYYY-xxxx 格式，移除 YYYY-
        // 例如: /papers/transformers/2101-switch-transformer -> /papers/transformers/switch-transformer
        .replace(/\/(\d{4}-)/, '/');
    };

    const finalRoute = convertMdLinkToRoute(item.link);

    return (
      <Timeline.Item key={idx} label={item.date}>
        <Link to={finalRoute}>{item.combinedTitle}</Link>
      </Timeline.Item>
    );

})}
</Timeline>

:::info
This block will automatically read the paper notes written in the last 30 days from our commit history.

Therefore, it's normal to see different content every day, and it also reminds us to write more notes.
:::

## Daily Life

Reading papers is a truly enjoyable activity!

If you've been an engineer for many years, you probably understand this sentiment.

＊

In our usual development projects, we tend to focus on practical issues like performance, stability, and maintainability. Therefore, the technologies and tools we use are relatively fixed and don't change much. Furthermore, a significant amount of our time and mental energy is consumed by communicating with clients, leaving us exhausted and with little energy to keep up with new technologies.

Being able to take some time out of our daily work to read a paper is a rare and leisurely pleasure. We don't have to be like those researchers who are stressed out every day about the progress of their papers. Just by reading the papers, we can understand the difficulties and efforts of the researchers, as well as the latest academic developments.

The truths of this universe lie in the minute observations within each paper. These observations might be incorrect, biased, or merely wishful thinking by the researchers, but they could also hit the core of the matter. These are all essential steps in the pursuit of truth.

We believe that the attitude towards reading papers should be relaxed, with a reverence for knowledge.

We think, we record, and we persistently move forward.

We always believe: knowledge is power.

## The Era of Large Language Models

Since the advent of ChatGPT, reading papers has become much easier, but this doesn't mean we can relinquish the responsibility of thinking.

Here, we document our insights from reading papers and our understanding of the papers might be biased or incorrect.

If there is any discrepancy with the original papers, **please refer to the original paper first**.

## Finding Papers

If you want to find a paper, we suggest pasting the title into the search bar at the top right corner for quicker results.

Our writing format is as follows:

- The title includes: [Publication Date] + [Author's Name or Common Reference Name]
- The content starts with a brief discussion
- Then defines the problem the paper aims to solve
- Introduces the solution method
- Ends with discussion and conclusion

Thus, this is not a direct translation of the paper but more of a guided reading.

## And

Writing notes is very time-consuming and labor-intensive. The time it takes to write one note is enough for me to read five papers.

At the same time, it's a continuous task, usually taking two days to write one. As the duration increases, my writing style changes. For example, earlier articles tend to be more detailed, but that's actually because of inexperience—I wasn't sure what the main points of the paper were, so writing it turned into a translation. (I apologize for that... 😓)

As I continued writing, I started to recognize the main points of the papers, which allowed me to skip some sections and focus solely on the core contributions. This led to a more concise presentation of the article.

In the end, the quality of the content still depends on luck, and not every article will be written perfectly. If you're willing, you can also publish your notes here or revise existing notes. Just send us a pull request, and we'll be very happy to receive it.

:::info
Don't worry about the multilingual part; we can handle it. Just write your notes in your preferred language.
:::

If you come across any interesting articles, feel free to share them with us, and when we have time, we'll take a look.

If we have time.

＊

2024 © Zephyr
