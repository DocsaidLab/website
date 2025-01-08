---
sidebar_position: 1
---

import { Timeline } from "antd";
import Link from '@docusaurus/Link';
import recentUpdates from './recent_updates_data.json';

# Paper Notes

## Recent Updates

<Timeline mode="alternate">
  {recentUpdates.map((item, idx) => {
    const convertMdLinkToRoute = (mdLink) => {
      return mdLink
        .replace(/^.\//, '/papers/')  // Replace "./" with "/papers/"
        .replace(/\.md$/, '')         // Remove .md extension
        .replace(/\/index$/, '')      // Remove /index
        // Remove the year prefix if in YYYY-xxxx format
        // For example: /papers/transformers/2101-switch-transformer -> /papers/transformers/switch-transformer
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
This section automatically reads the most recent 30 days' worth of paper notes from our commit history.

Therefore, it's normal to see different content each day, which also encourages us to write more notes ourselves.
:::

## Daily Routine

Reading papers is a very enjoyable activity!

If you are an experienced engineer, you probably understand what I mean.

＊

In our regular development projects, we focus on real-world issues like performance, stability, and maintainability. As a result, the technologies and tools we use are relatively fixed and don't change much. Moreover, the workload and client communication often drain our energy, leaving us with little time to focus on new technologies.

Finding time to read papers during our daily work is a rare and leisurely entertainment. We don’t have to be as stressed as researchers who have to worry about their paper deadlines. Through the papers, we can understand the difficulties and efforts faced by the researchers, as well as the latest academic advancements.

The truth of the universe lies in the small observations within each paper. These observations may be incorrect, biased, or based on the researcher’s wishes, but they may also strike at the core. These are all essential steps in the exploration of truth.

We believe that the attitude towards reading papers should be relaxed, and the yearning for knowledge should be sincere.

We think, we record, and we persistently move forward.

We firmly believe: knowledge is power.

## The Era of Large Language Models

Since the emergence of ChatGPT, reading papers has become much easier, but this doesn't mean we can stop thinking critically.

Here, we record some insights from reading papers, and the understanding of the papers may contain biases or errors.

If there is any discrepancy with the paper, **please refer to the original paper first**.

## Finding Papers

If you're looking for a paper, we suggest pasting the title into the search bar at the top right corner for quicker results.

Our standard writing format is:

- Title: [**Publication Date**] + [**Author's Name or Common Industry Name for the Paper**]
- The content mainly starts with a few introductory remarks
- Then, we define the problem the paper aims to solve
- Next, we introduce the proposed solution
- Finally, we discuss and conclude

So this is not a translation of the paper, but more like paper notes.

## Additionally

Writing notes is not an easy task; it can take as long as reading five papers to write one note. Therefore, we don't have many paper notes due to time constraints, but we will continue to update them.

You might ask why not just have GPT write all the notes?

Of course, that’s possible! But notes that aren't thought through by the brain have no value.

Additionally, this is a continuous task, and the writing style usually changes over time. Typically, the first ten papers will be more formal, while later ones will reflect a more personalized style.

In conclusion, the quality of the articles depends on the circumstances, and not every paper can be written perfectly. If you're willing, you can also contribute your own notes or rewrite existing notes by submitting a Pull Request; we welcome that very much.

:::info
For multilingual support, we can handle it for you. Feel free to write your notes in any language you prefer.
:::

## Finally

If you would like us to write about specific papers, please leave a comment below, and we will check them out when we have time.

Thank you for your reading and support, and we hope this site can bring you help and inspiration!
