---
sidebar_position: 1
---

import { Timeline } from "antd";
import Link from '@docusaurus/Link';
import recentUpdates from './recent_updates_data.json';

# 論文筆記

## 近期更新

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
這個區塊會自動從我們的提交紀錄中讀取最近 30 天內撰寫的論文筆記。

因此，每天看到不一樣的內容是正常的，也順便督促我們自己要多寫點筆記。
:::

## 日常

看論文是件非常享受的事情！

如果你也是從業多年的工程師，大概能體會這個意思。

＊

在平常的開發項目中，我們大都著眼在一些現實的問題上，比如性能、穩定性、可維護性等等。因此，可以用的技術和工具也比較固定，不會有太多變化。不僅如此，大量的工作和客戶溝通會消耗我們大量的精神，筋疲力竭的我們往往也沒有太多精力去關注新技術。

能夠在日常工作中，抽出時間看點論文是個難得清閒的娛樂。我們也不必像是那些研究人員一樣，每天都要為了論文的進度而焦頭爛額。只需要透過論文的陳述，就能了解到研究者遇到的困難和努力，當然，還有最新的學術進展。

這個宇宙的真理就存在於每篇論文內的微小觀察，這些觀察可能不正確，可能有偏頗，可能是研究者的一廂情願，但也可能直指核心。這些都是探索真理的必經之途。

我們認為看論文的姿態是放鬆的，對知識的憧憬是虔誠的。

我們思考，我們紀錄，我們持之以恆地向前走。

我們始終相信：知識就是力量。

## 大語言模型時代

從 ChatGPT 問世以來，閱讀論文變得更加輕鬆，但這不意味著我們可以放下思考的責任。

我們在這裡記錄了一些讀論文的心得，其中對論文的理解可能會存在偏頗或錯誤。

如果有任何和論文不符的地方，**請優先參考原始論文的內容**。

## 找論文

如果你想找論文，我們會建議你將論文題目貼上右上角的搜尋，這樣比較快。

我們寫作的固定格式為：

- 標題為：[**發表年月**] + [**作者取名或業界對該論文的泛稱**]
- 內容主要是先簡單聊個幾句
- 然後定義論文想解決問題
- 接著是介紹解決的方法
- 最後是討論和結論

所以這不是翻譯論文，比較像是論文筆記。

## 還有

寫筆記不是一件容易的事情，寫一篇的時間可能抵得過看五篇論文。所以論文筆記沒有很多篇，因為我們的時間有限，但我們會持續更新。

你可能會問說為何不全部扔給 GPT 寫？

當然可以！但是沒有經過大腦思考的筆記，沒有價值。

另外，這也是個連續性的工作，不管是哪個作者，持續時間長了，寫作風格都會有所變化。通常前十篇文章最生疏，後面的文章就會比較有自己的風格。

總之，文章內容還是要看緣分，不一定每篇都能寫的很好。如果你願意也可以在這裡發表你的筆記或改寫現有的筆記，直接發 Pull Request 給我們就好，非常歡迎。

:::info
多國語系的部分我們可以幫你搞定，用你喜歡的語言寫筆記即可。
:::

## 最後

如果你希望我們寫一些特定的論文，可以直接在下方留言，我們有空的話會去看看。

感謝你的閱讀與支持，希這裡能為你帶來幫助與啟發！
