# StockAnalysis Demo

每個人多少都有在投資股票，對於股票的分析，也是投資決策中不可或缺的重要一環。

然而，股票分析是一個龐大且多維度的工作，考量到時間與資源有限，我們將這項任務分為幾個階段，逐步完善並建構一個全面的股票分析工具。

:::info
**使用本網頁功能時有幾個注意事項：**

1. 請先閱讀：[**免責聲明**](#-免責聲明)，以確保你對本網頁的使用有正確的認識。
2. FinMind API 存在使用次數限制，詳情請參考：[**使用次數限制說明**](#-使用次數限制說明)
3. 我們不打算重新做一套 K 線分析工具，前面幾個階段都是為了最後的建模分析做準備，相關指標可能不會再增加。
4. 歷史成交資訊來自於 FinMind API，數值可能涵蓋盤後交易，因此數值可能與其他網站顯示的成交量有所不同。
5. 目前僅支援取得臺灣股市的資料，其他市場的股票資料留給未來的版本。
   :::

## 階段一：基本分析

在第一階段，我們專注於最基礎的股票價格與交易量分析，透過視覺化呈現，協助使用者掌握股票的基本面與技術面走勢。執行內容大概包含：

1.  **串接資料來源**：使用 FinMind 作為股票資料來源，透過 API 取得台股的歷史股價資料。
2.  **圖表化呈現**

    - 將原始數據轉化為易於理解的圖表，包括折線圖、柱狀圖及 K 線圖。
    - 加入互動功能，使用者可以自由調整時間範圍與個別股票。

3.  **技術指標運算**

    - 計算並呈現常見的技術指標：
      - **布林通道**（Bollinger Bands）：顯示價格波動範圍與均線。
      - **MACD**（平滑異同移動平均線）：辨識市場動能與趨勢反轉信號。
      - **RSI**（相對強弱指標）：判斷市場超買或超賣情況。
      - **KDJ** 指標：反映股價波動的動能與短期買賣機會。

透過圖表與指標，讓使用者能快速了解股票的歷史價量表現，並套用基礎 K 線理論進行初步的買賣點判斷。

### 程式功能

import StockAnalysisPage from '@site/src/components/StockAnalysis';

<StockAnalysisPage />

## 階段二：新聞分析

將股票市場中的即時新聞與價格走勢進行結合，從中提取語意資訊，分析新聞對股票短期走勢的潛在影響。

預計執行內容包括：

1. **新聞資料來源整合**：串接新聞平台的 API，例如 Google News、Yahoo Finance News、FinMind 等，取得即時股票相關新聞與報導。
2. **語意特徵提取**：串接 openai 的模型，進行新聞語意分析，提取新聞報導的情緒、主題、重要性等特徵。
3. **新聞與股票走勢交叉比對**
   - 將新聞報導與股票的價格、成交量變動進行時間對齊，觀察兩者的相關性。
   - 透過數據統計與視覺化圖表，展示新聞情緒對短期股價波動的潛在影響。

分析結果將揭示新聞情緒與股價走勢之間的關聯性，協助使用者洞察市場反應，找出潛在的市場機會或風險。

### 🚧 程式功能 🚧

（尚未安排時間）

## 階段三：建模分析

既然我們常用深度學習的技術來解決問題，沒道理不用在股票上。

在這個階段，我們將引入深度學習技術，建構預測模型，進一步提升股票分析的精準度與智能化程度。

預計執行內容包括：

1. **數據準備與特徵工程**：建立訓練用的數據集
2. **模型選擇與訓練**：根據數據規模與問題性質，選擇適合的預測模型。
3. **模型評估與迭代優化**：模型回測與交叉驗證，評估模型的準確性與穩定性。
4. **結果視覺化與解釋**：提供解釋性模型分析，讓使用者理解預測結果背後的主要因素。

透過建模分析，系統將具備一定程度的股價趨勢預測能力，為投資者提供更智能的參考決策。

### 🚧 程式功能 🚧

（尚未安排時間）

## 📊 使用次數限制說明

本平台透過後端代理服務，調用 **FinMind** API 以提供即時股票資料。然而，該 API 存在每小時 **600 次** 的使用次數限制，所有使用者共享此額度。

關於這件事情，我們感到很抱歉，目前我們無法負擔更多的 API 費用，因此只能提供最低限度的使用次數。如果未來有更多的資源，我們會提供更多的使用次數。

如果你發現數據或圖表未能正常顯示，這可能是因為當前時段的 API 調用次數已用完。請稍後再試，或等候系統於下一小時重置額度。

- [**FinMind 官方網站**](https://finmindtrade.com/)
- [**FinMind GitHub 文件**](https://github.com/FinMind/FinMind)

:::info
每周日早上零點至早上七點為 Finmind API 維護時間，此時段內無法取得股票資料。
:::

## 📢 免責聲明

以下內容僅供參考，**不構成任何投資建議或理財建議**。請仔細閱讀，審慎評估風險。

1. **非專業投資建議**：本平台所提供的分析與數據，皆基於歷史資料及技術指標進行演算，僅供參考。相關內容不代表對市場走勢的保證或預測，亦不具備任何法律效力。
2. **投資風險與自我承擔**：投資涉及風險，市場行情瞬息萬變，任何投資行為均存在資產虧損的可能性。根據本平台所提供的數據或分析進行投資，應由投資人自行決定並承擔所有風險與後果。
3. **數據與資訊的限制**：本平台所引用的數據及分析結果，可能受限於資訊來源的延遲、不完整或錯誤，演算法也非絕對可靠。因此，若數據或結果與你的預期有所出入，請以官方資訊及具權威性的專業機構數據為準。
4. **歷史績效不代表未來結果**：所有分析與數據均基於過去市場表現，歷史績效無法保證未來結果。市場變化受多種因素影響，具高度不確定性，任何依據此內容所做的投資判斷均存在風險。
5. **建議諮詢專業機構**：在進行任何投資決策前，建議你諮詢具備合法資格的專業投資顧問、持牌理財專家或其他值得信賴的專業人士，以獲取更全面且客觀的建議。

本平台對內容的正確性、完整性及時效性不作任何明示或暗示的保證，亦不承擔任何直接或間接的責任。請投資人根據自身風險承受能力，謹慎評估並做出決策。
