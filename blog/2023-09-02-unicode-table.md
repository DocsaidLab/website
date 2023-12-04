---
slug: unicode-table
title: Unicode 編碼區段表
authors: TSE
tags: [unicode]
---

Unicode 是一個由非營利組織 Unicode 聯盟所維護的國際性字符編碼標準。該標準旨在統一世界上大部分的文字系統，從而實現跨平臺和多語言環境下的信息交流。Unicode 是 ISO/IEC 10646標準的實現，也成為全球性的通用字元集。

<!--truncate-->

在 Unicode 標準中，每個字符都對應到一個唯一的數字代碼，這個代碼被稱為「區段」（Code Point）。區段是 Unicode 標準中一個至關重要的概念，用於確定每個獨特字符的位置。

Unicode 的區段範圍從U+0000 到 U+10FFFF，這意味著它可以容納 1,114,112 個不同的字符。這些區段按照不同的功能和用途被劃分成多個子集，例如：

- 基本多文種平面（Basic Multilingual Plane, BMP）：U+0000 到 U+FFFF

    包括了拉丁字母、希臘字母、漢字等常用字符。

- 補充平面（Supplementary Planes）：U+010000 到 U+10FFFF

    包括了更多的象形文字、補充字符等。

有鑒於我們每次需要查找 Unicode 編碼時，總是得花費半小時去把 wiki 的資料表翻出來，所以還是把一些基本的資訊儲存起來吧。

## Unicode 區段表

如果您有更近一步的需求，我們還是推薦您去查原始表格：[Unicode區段](https://zh.wikipedia.org/zh-tw/Unicode%E5%8D%80%E6%AE%B5)

| 平面               | 區段範圍                  | 漢語名稱                 | 英語名稱                                             |
|------------------|-----------------------|----------------------|--------------------------------------------------|
| &nbsp;0&nbsp;BMP | 0000 &#8211; 007F     | 基本拉丁字母               | Basic Latin                                      |
| &nbsp;0&nbsp;BMP | 0080 &#8211; 00FF     | 拉丁字母補充-1             | Latin-1 Supplement                               |
| &nbsp;0&nbsp;BMP | 0100 &#8211; 017F     | 拉丁字母擴展-A             | Latin Extended-A                                 |
| &nbsp;0&nbsp;BMP | 0180 &#8211; 024F     | 拉丁字母擴展-B             | Latin Extended-B                                 |
| &nbsp;0&nbsp;BMP | 0250 &#8211; 02AF     | 國際音標擴展               | IPA Extensions                                   |
| &nbsp;0&nbsp;BMP | 02B0 &#8211; 02FF     | 佔位修飾符號               | Spacing Modifier Letters                         |
| &nbsp;0&nbsp;BMP | 0300 &#8211; 036F     | 組合附加符號               | Combining Diacritical Marks                      |
| &nbsp;0&nbsp;BMP | 0370 &#8211; 03FF     | 希臘字母和科普特字母           | Greek and Coptic                                 |
| &nbsp;0&nbsp;BMP | 0400 &#8211; 04FF     | 西里爾字母                | Cyrillic                                         |
| &nbsp;0&nbsp;BMP | 0500 &#8211; 052F     | 西里爾字母補充              | Cyrillic Supplement                              |
| &nbsp;0&nbsp;BMP | 0530 &#8211; 058F     | 亞美尼亞字母               | Armenian                                         |
| &nbsp;0&nbsp;BMP | 0590 &#8211; 05FF     | 希伯來文字母               | Hebrew                                           |
| &nbsp;0&nbsp;BMP | 0600 &#8211; 06FF     | 阿拉伯文字母               | Arabic                                           |
| &nbsp;0&nbsp;BMP | 0700 &#8211; 074F     | 敘利亞字母                | Syriac                                           |
| &nbsp;0&nbsp;BMP | 0750 &#8211; 077F     | 阿拉伯文補充               | Arabic Supplement                                |
| &nbsp;0&nbsp;BMP | 0780 &#8211; 07BF     | 它拿字母                 | Thaana                                           |
| &nbsp;0&nbsp;BMP | 07C0 &#8211; 07FF     | 西非書面文字               | NKo                                              |
| &nbsp;0&nbsp;BMP | 0800 &#8211; 083F     | 撒瑪利亞字母               | Samaritan                                        |
| &nbsp;0&nbsp;BMP | 0840 &#8211; 085F     | 曼達安字母                | Mandaic                                          |
| &nbsp;0&nbsp;BMP | 0860 &#8211; 086F     | 敘利亞文補充               | Syriac Supplement                                |
| &nbsp;0&nbsp;BMP | 0870 &#8211; 089F     | 阿拉伯字母擴展-B            | Arabic Extended-B                                |
| &nbsp;0&nbsp;BMP | 08A0 &#8211; 08FF     | 阿拉伯字母擴展-A            | Arabic Extended-A                                |
| &nbsp;0&nbsp;BMP | 0900 &#8211; 097F     | 天城文                  | Devanagari                                       |
| &nbsp;0&nbsp;BMP | 0980 &#8211; 09FF     | 孟加拉文                 | Bengali                                          |
| &nbsp;0&nbsp;BMP | 0A00 &#8211; 0A7F     | 古木基文                 | Gurmukhi                                         |
| &nbsp;0&nbsp;BMP | 0A80 &#8211; 0AFF     | 古吉拉特文                | Gujarati                                         |
| &nbsp;0&nbsp;BMP | 0B00 &#8211; 0B7F     | 奧里亞文                 | Oriya                                            |
| &nbsp;0&nbsp;BMP | 0B80 &#8211; 0BFF     | 泰米爾文                 | Tamil                                            |
| &nbsp;0&nbsp;BMP | 0C00 &#8211; 0C7F     | 泰盧固文                 | Telugu                                           |
| &nbsp;0&nbsp;BMP | 0C80 &#8211; 0CFF     | 卡納達文                 | Kannada                                          |
| &nbsp;0&nbsp;BMP | 0D00 &#8211; 0D7F     | 馬拉雅拉姆文               | Malayalam                                        |
| &nbsp;0&nbsp;BMP | 0D80 &#8211; 0DFF     | 僧伽羅文                 | Sinhala                                          |
| &nbsp;0&nbsp;BMP | 0E00 &#8211; 0E7F     | 泰文                   | Thai                                             |
| &nbsp;0&nbsp;BMP | 0E80 &#8211; 0EFF     | 寮文                   | Lao                                              |
| &nbsp;0&nbsp;BMP | 0F00 &#8211; 0FFF     | 藏文                   | Tibetan                                          |
| &nbsp;0&nbsp;BMP | 1000 &#8211; 109F     | 緬甸文                  | Myanmar                                          |
| &nbsp;0&nbsp;BMP | 10A0 &#8211; 10FF     | 喬治亞字母                | Georgian                                         |
| &nbsp;0&nbsp;BMP | 1100 &#8211; 11FF     | 諺文字母                 | Hangul Jamo                                      |
| &nbsp;0&nbsp;BMP | 1200 &#8211; 137F     | 衣索比亞字母               | Ethiopic                                         |
| &nbsp;0&nbsp;BMP | 1380 &#8211; 139F     | 衣索比亞字母補充             | Ethiopic Supplement                              |
| &nbsp;0&nbsp;BMP | 13A0 &#8211; 13FF     | 切羅基文                 | Cherokee                                         |
| &nbsp;0&nbsp;BMP | 1400 &#8211; 167F     | 統一加拿大原住民音節文字         | Unified Canadian Aboriginal Syllabics            |
| &nbsp;0&nbsp;BMP | 1680 &#8211; 169F     | 歐甘字母                 | Ogham                                            |
| &nbsp;0&nbsp;BMP | 16A0 &#8211; 16FF     | 盧恩字母                 | Runic                                            |
| &nbsp;0&nbsp;BMP | 1700 &#8211; 171F     | 他加祿字母                | Tagalog                                          |
| &nbsp;0&nbsp;BMP | 1720 &#8211; 173F     | 哈努諾文                 | Hanunoo                                          |
| &nbsp;0&nbsp;BMP | 1740 &#8211; 175F     | 布希德字母                | Buhid                                            |
| &nbsp;0&nbsp;BMP | 1760 &#8211; 177F     | 塔格班瓦字母               | Tagbanwa                                         |
| &nbsp;0&nbsp;BMP | 1780 &#8211; 17FF     | 高棉文                  | Khmer                                            |
| &nbsp;0&nbsp;BMP | 1800 &#8211; 18AF     | 蒙古文                  | Mongolian                                        |
| &nbsp;0&nbsp;BMP | 18B0 &#8211; 18FF     | 統一加拿大原住民音節文字擴充       | Unified Canadian Aboriginal Syllabics Extended   |
| &nbsp;0&nbsp;BMP | 1900 &#8211; 194F     | 林布文                  | Limbu                                            |
| &nbsp;0&nbsp;BMP | 1950 &#8211; 197F     | 德宏傣文                 | Tai Le                                           |
| &nbsp;0&nbsp;BMP | 1980 &#8211; 19DF     | 新傣仂文                 | New Tai Le                                       |
| &nbsp;0&nbsp;BMP | 19E0 &#8211; 19FF     | 高棉文符號                | Khmer Symbols                                    |
| &nbsp;0&nbsp;BMP | 1A00 &#8211; 1A1F     | 布吉文                  | Buginese                                         |
| &nbsp;0&nbsp;BMP | 1A20 &#8211; 1AAF     | 老傣文                  | Tai Tham                                         |
| &nbsp;0&nbsp;BMP | 1AB0 &#8211; 1AFF     | 組合附加符號擴展             | Combining Diacritical Marks Extended             |
| &nbsp;0&nbsp;BMP | 1B00 &#8211; 1B7F     | 峇里字母                 | Balinese                                         |
| &nbsp;0&nbsp;BMP | 1B80 &#8211; 1BBF     | 巽他字母                 | Sundanese                                        |
| &nbsp;0&nbsp;BMP | 1BC0 &#8211; 1BFF     | 巴塔克字母                | Batak                                            |
| &nbsp;0&nbsp;BMP | 1C00 &#8211; 1C4F     | 絨巴文                  | Lepcha                                           |
| &nbsp;0&nbsp;BMP | 1C50 &#8211; 1C7F     | 桑塔利文                 | Ol Chiki                                         |
| &nbsp;0&nbsp;BMP | 1C80 &#8211; 1C8F     | 西里爾字母擴展-C            | Cyrillic Extended-C                              |
| &nbsp;0&nbsp;BMP | 1C90 &#8211; 1CBF     | 喬治亞字母擴展              | Georgian Extended                                |
| &nbsp;0&nbsp;BMP | 1CC0 &#8211; 1CCF     | 巽他字母補充               | Sundanese Supplement                             |
| &nbsp;0&nbsp;BMP | 1CD0 &#8211; 1CFF     | 吠陀擴展                 | Vedic Extensions                                 |
| &nbsp;0&nbsp;BMP | 1D00 &#8211; 1D7F     | 音標擴展                 | Phonetic Extensions                              |
| &nbsp;0&nbsp;BMP | 1D80 &#8211; 1DBF     | 音標擴展補充               | Phonetic Extensions Supplement                   |
| &nbsp;0&nbsp;BMP | 1DC0 &#8211; 1DFF     | 組合附加符號補充             | Combining Diacritical Marks Supplement           |
| &nbsp;0&nbsp;BMP | 1E00 &#8211; 1EFF     | 拉丁字母擴展附加             | Latin Extended Additional                        |
| &nbsp;0&nbsp;BMP | 1F00 &#8211; 1FFF     | 希臘字母擴展               | Greek Extended                                   |
| &nbsp;0&nbsp;BMP | 2000 &#8211; 206F     | 一般標點                 | General Punctuation                              |
| &nbsp;0&nbsp;BMP | 2070 &#8211; 209F     | 上標及下標                | Superscripts and Subscripts                      |
| &nbsp;0&nbsp;BMP | 20A0 &#8211; 20CF     | 貨幣符號                 | Currency Symbols                                 |
| &nbsp;0&nbsp;BMP | 20D0 &#8211; 20FF     | 符號用組合附加符號            | Combining Diacritical Marks for Symbols          |
| &nbsp;0&nbsp;BMP | 2100 &#8211; 214F     | 類字母符號                | Letterlike Symbols                               |
| &nbsp;0&nbsp;BMP | 2150 &#8211; 218F     | 數字形式                 | Number Forms                                     |
| &nbsp;0&nbsp;BMP | 2190 &#8211; 21FF     | 箭頭                   | Arrows                                           |
| &nbsp;0&nbsp;BMP | 2200 &#8211; 22FF     | 數學運算子                | Mathematical Operators                           |
| &nbsp;0&nbsp;BMP | 2300 &#8211; 23FF     | 雜項技術符號               | Miscellaneous Technical                          |
| &nbsp;0&nbsp;BMP | 2400 &#8211; 243F     | 控制圖形                 | Control Pictures                                 |
| &nbsp;0&nbsp;BMP | 2440 &#8211; 245F     | 光學字元識別               | Optical Character Recognition                    |
| &nbsp;0&nbsp;BMP | 2460 &#8211; 24FF     | 圍繞字母數字               | Enclosed Alphanumerics                           |
| &nbsp;0&nbsp;BMP | 2500 &#8211; 257F     | 制表符                  | Box Drawing                                      |
| &nbsp;0&nbsp;BMP | 2580 &#8211; 259F     | 方塊元素                 | Block Elements                                   |
| &nbsp;0&nbsp;BMP | 25A0 &#8211; 25FF     | 幾何圖形                 | Geometric Shapes                                 |
| &nbsp;0&nbsp;BMP | 2600 &#8211; 26FF     | 雜項符號                 | Miscellaneous Symbols                            |
| &nbsp;0&nbsp;BMP | 2700 &#8211; 27BF     | 裝飾符號                 | Dingbats                                         |
| &nbsp;0&nbsp;BMP | 27C0 &#8211; 27EF     | 雜項數學符號-A             | Miscellaneous Mathematical Symbols-A             |
| &nbsp;0&nbsp;BMP | 27F0 &#8211; 27FF     | 追加箭頭-A               | Supplemental Arrows-A                            |
| &nbsp;0&nbsp;BMP | 2800 &#8211; 28FF     | 點字圖案                 | Braille Patterns                                 |
| &nbsp;0&nbsp;BMP | 2900 &#8211; 297F     | 追加箭頭-B               | Supplemental Arrows-B                            |
| &nbsp;0&nbsp;BMP | 2980 &#8211; 29FF     | 雜項數學符號-B             | Miscellaneous Mathematical Symbols-B             |
| &nbsp;0&nbsp;BMP | 2A00 &#8211; 2AFF     | 補充數學運算子              | Supplemental Mathematical Operators              |
| &nbsp;0&nbsp;BMP | 2B00 &#8211; 2BFF     | 雜項符號和箭頭              | Miscellaneous Symbols and Arrows                 |
| &nbsp;0&nbsp;BMP | 2C00 &#8211; 2C5F     | 格拉哥里字母               | Glagolitic                                       |
| &nbsp;0&nbsp;BMP | 2C60 &#8211; 2C7F     | 拉丁字母擴展-C             | Latin Extended-C                                 |
| &nbsp;0&nbsp;BMP | 2C80 &#8211; 2CFF     | 科普特字母                | Coptic                                           |
| &nbsp;0&nbsp;BMP | 2D00 &#8211; 2D2F     | 喬治亞字母補充              | Georgian Supplement                              |
| &nbsp;0&nbsp;BMP | 2D30 &#8211; 2D7F     | 提非納文                 | Tifinagh                                         |
| &nbsp;0&nbsp;BMP | 2D80 &#8211; 2DDF     | 衣索比亞字母擴充             | Ethiopic Extended                                |
| &nbsp;0&nbsp;BMP | 2DE0 &#8211; 2DFF     | 西里爾字母擴展-A            | Cyrillic Extended-A                              |
| &nbsp;0&nbsp;BMP | 2E00 &#8211; 2E7F     | 補充標點                 | Supplemental Punctuation                         |
| &nbsp;0&nbsp;BMP | 2E80 &#8211; 2EFF     | 中日韓漢字部首補充            | CJK Radicals Supplement                          |
| &nbsp;0&nbsp;BMP | 2F00 &#8211; 2FDF     | 康熙部首                 | Kangxi Radicals                                  |
| &nbsp;0&nbsp;BMP | 2FF0 &#8211; 2FFF     | 表意文字描述字元             | Ideographic Description Characters               |
| &nbsp;0&nbsp;BMP | 3000 &#8211; 303F     | 中日韓符號和標點             | CJK Symbols and Punctuation                      |
| &nbsp;0&nbsp;BMP | 3040 &#8211; 309F     | 平假名                  | Hiragana                                         |
| &nbsp;0&nbsp;BMP | 30A0 &#8211; 30FF     | 片假名                  | Katakana                                         |
| &nbsp;0&nbsp;BMP | 3100 &#8211; 312F     | 注音符號                 | Bopomofo                                         |
| &nbsp;0&nbsp;BMP | 3130 &#8211; 318F     | 諺文相容字母               | Hangul Compatibility Jamo                        |
| &nbsp;0&nbsp;BMP | 3190 &#8211; 319F     | 漢文訓讀符號               | Kanbun                                           |
| &nbsp;0&nbsp;BMP | 31A0 &#8211; 31BF     | 注音符號擴展               | Bopomofo Extended                                |
| &nbsp;0&nbsp;BMP | 31C0 &#8211; 31EF     | 中日韓筆畫                | CJK Strokes                                      |
| &nbsp;0&nbsp;BMP | 31F0 &#8211; 31FF     | 片假名語音擴展              | Katakana Phonetic Extensions                     |
| &nbsp;0&nbsp;BMP | 3200 &#8211; 32FF     | 中日韓圍繞字元及月份           | Enclosed CJK Letters and Months                  |
| &nbsp;0&nbsp;BMP | 3300 &#8211; 33FF     | 中日韓相容字元              | CJK Compatibility                                |
| &nbsp;0&nbsp;BMP | 3400 &#8211; 4DBF     | 中日韓統一表意文字擴充區A        | CJK Unified Ideographs Extension A               |
| &nbsp;0&nbsp;BMP | 4DC0 &#8211; 4DFF     | 易經六十四卦符號             | Yijing Hexagram Symbols                          |
| &nbsp;0&nbsp;BMP | 4E00 &#8211; 9FFF     | 中日韓統一表意文字&nbsp;(基本區) | CJK Unified Ideographs                           |
| &nbsp;0&nbsp;BMP | A000 &#8211; A48F     | 彝文音節                 | Yi Syllables                                     |
| &nbsp;0&nbsp;BMP | A490 &#8211; A4CF     | 彝文部首                 | Yi Radicals                                      |
| &nbsp;0&nbsp;BMP | A4D0 &#8211; A4FF     | 傈僳文                  | Lisu                                             |
| &nbsp;0&nbsp;BMP | A500 &#8211; A63F     | 瓦伊文                  | Vai                                              |
| &nbsp;0&nbsp;BMP | A640 &#8211; A69F     | 西里爾字母擴展-B            | Cyrillic Extended-B                              |
| &nbsp;0&nbsp;BMP | A6A0 &#8211; A6FF     | 巴姆穆文字                | Bamum                                            |
| &nbsp;0&nbsp;BMP | A700 &#8211; A71F     | 聲調修飾符號               | Modifier Tone Letters                            |
| &nbsp;0&nbsp;BMP | A720 &#8211; A7FF     | 拉丁字母擴展-D             | Latin Extended-D                                 |
| &nbsp;0&nbsp;BMP | A800 &#8211; A82F     | 錫爾赫特文                | Syloti Nagri                                     |
| &nbsp;0&nbsp;BMP | A830 &#8211; A83F     | 通用印度數字形式             | Common Indic Number Forms                        |
| &nbsp;0&nbsp;BMP | A840 &#8211; A87F     | 八思巴文                 | Phags-pa                                         |
| &nbsp;0&nbsp;BMP | A880 &#8211; A8DF     | 索拉什特拉文               | Saurashtra                                       |
| &nbsp;0&nbsp;BMP | A8E0 &#8211; A8FF     | 天城文擴展                | Devanagari Extended                              |
| &nbsp;0&nbsp;BMP | A900 &#8211; A92F     | 克耶字母                 | Kayah Li                                         |
| &nbsp;0&nbsp;BMP | A930 &#8211; A95F     | 勒姜字母                 | Rejang                                           |
| &nbsp;0&nbsp;BMP | A960 &#8211; A97F     | 諺文字母擴展-A             | Hangul Jamo Extended-A                           |
| &nbsp;0&nbsp;BMP | A980 &#8211; A9DF     | 爪哇字母                 | Javanese                                         |
| &nbsp;0&nbsp;BMP | A9E0 &#8211; A9FF     | 緬甸文擴展-B              | Myanmar Extended-B                               |
| &nbsp;0&nbsp;BMP | AA00 &#8211; AA5F     | 占文                   | Cham                                             |
| &nbsp;0&nbsp;BMP | AA60 &#8211; AA7F     | 緬甸文擴展-A              | Myanmar Extended-A                               |
| &nbsp;0&nbsp;BMP | AA80 &#8211; AADF     | 傣越文                  | Tai Viet                                         |
| &nbsp;0&nbsp;BMP | AAE0 &#8211; AAFF     | 梅泰文擴充                | Meetei Mayek Extensions                          |
| &nbsp;0&nbsp;BMP | AB00 &#8211; AB2F     | 衣索比亞字母擴充-A           | Ethiopic Extended-A                              |
| &nbsp;0&nbsp;BMP | AB30 &#8211; AB6F     | 拉丁字母擴展-E             | Latin Extended-E                                 |
| &nbsp;0&nbsp;BMP | AB70 &#8211; ABBF     | 切羅基文補充               | Cherokee Supplement                              |
| &nbsp;0&nbsp;BMP | ABC0 &#8211; ABFF     | 梅泰文                  | Meetei Mayek                                     |
| &nbsp;0&nbsp;BMP | AC00 &#8211; D7AF     | 諺文音節                 | Hangul Syllables                                 |
| &nbsp;0&nbsp;BMP | D7B0 &#8211; D7FF     | 諺文字母擴展-B             | Hangul Jamo Extended-B                           |
| &nbsp;0&nbsp;BMP | D800 &#8211; DB7F     | 高半代用區                | High Surrogates                                  |
| &nbsp;0&nbsp;BMP | DB80 &#8211; DBFF     | 高半私人代用區              | High Private Use Surrogates                      |
| &nbsp;0&nbsp;BMP | DC00 &#8211; DFFF     | 低半代用區                | Low Surrogates                                   |
| &nbsp;0&nbsp;BMP | E000 &#8211; F8FF     | 私用區                  | Private Use Area                                 |
| &nbsp;0&nbsp;BMP | F900 &#8211; FAFF     | 中日韓相容表意文字            | CJK Compatibility Ideographs                     |
| &nbsp;0&nbsp;BMP | FB00 &#8211; FB4F     | 字母表達形式               | Alphabetic Presentation Forms                    |
| &nbsp;0&nbsp;BMP | FB50 &#8211; FDFF     | 阿拉伯字母表達形式-A          | Arabic Presentation Forms-A                      |
| &nbsp;0&nbsp;BMP | FE00 &#8211; FE0F     | 變體選擇符                | Variation Selectors                              |
| &nbsp;0&nbsp;BMP | FE10 &#8211; FE1F     | 豎排形式                 | Vertical Forms                                   |
| &nbsp;0&nbsp;BMP | FE20 &#8211; FE2F     | 組合用半符號               | Combining Half Marks                             |
| &nbsp;0&nbsp;BMP | FE30 &#8211; FE4F     | 中日韓相容形式              | CJK Compatibility Forms                          |
| &nbsp;0&nbsp;BMP | FE50 &#8211; FE6F     | 小寫變體形式               | Small Form Variants                              |
| &nbsp;0&nbsp;BMP | FE70 &#8211; FEFF     | 阿拉伯字母表達形式-B          | Arabic Presentation Forms-B                      |
| &nbsp;0&nbsp;BMP | FF00 &#8211; FFEF     | 半形及全形字元              | Halfwidth and Fullwidth Forms                    |
| &nbsp;0&nbsp;BMP | FFF0 &#8211; FFFF     | 特殊                   | Specials                                         |
| &nbsp;1&nbsp;SMP | 10000 &#8211; 1007F   | 線形文字B音節文字            | Linear B Syllabary                               |
| &nbsp;1&nbsp;SMP | 10080 &#8211; 100FF   | 線形文字B表意文字            | Linear B Ideograms                               |
| &nbsp;1&nbsp;SMP | 10100 &#8211; 1013F   | 愛琴海數字                | Aegean Numbers                                   |
| &nbsp;1&nbsp;SMP | 10140 &#8211; 1018F   | 古希臘數字                | Ancient Greek Numbers                            |
| &nbsp;1&nbsp;SMP | 10190 &#8211; 101CF   | 古代符號                 | Ancient Symbols                                  |
| &nbsp;1&nbsp;SMP | 101D0 &#8211; 101FF   | 斐斯托斯圓盤               | Phaistos Disc                                    |
| &nbsp;1&nbsp;SMP | 10280 &#8211; 1029F   | 呂基亞字母                | Lycian                                           |
| &nbsp;1&nbsp;SMP | 102A0 &#8211; 102DF   | 卡里亞字母                | Carian                                           |
| &nbsp;1&nbsp;SMP | 102E0 &#8211; 102FF   | 科普特閏餘數字              | Coptic Epact Numbers                             |
| &nbsp;1&nbsp;SMP | 10300 &#8211; 1032F   | 古義大利字母               | Old Italic                                       |
| &nbsp;1&nbsp;SMP | 10330 &#8211; 1034F   | 哥特字母                 | Gothic                                           |
| &nbsp;1&nbsp;SMP | 10350 &#8211; 1037F   | 古彼爾姆文                | Old Permic                                       |
| &nbsp;1&nbsp;SMP | 10380 &#8211; 1039F   | 烏加里特字母               | Ugaritic                                         |
| &nbsp;1&nbsp;SMP | 103A0 &#8211; 103DF   | 古波斯楔形文字              | Old Persian                                      |
| &nbsp;1&nbsp;SMP | 10400 &#8211; 1044F   | 德瑟雷特字母               | Deseret                                          |
| &nbsp;1&nbsp;SMP | 10450 &#8211; 1047F   | 蕭伯納字母                | Shavian                                          |
| &nbsp;1&nbsp;SMP | 10480 &#8211; 104AF   | 奧斯曼亞字母               | Osmanya                                          |
| &nbsp;1&nbsp;SMP | 104B0 &#8211; 104FF   | 歐塞奇字母                | Osage                                            |
| &nbsp;1&nbsp;SMP | 10500 &#8211; 1052F   | 愛爾巴桑字母               | Elbasan                                          |
| &nbsp;1&nbsp;SMP | 10530 &#8211; 1056F   | 高加索阿爾巴尼亞字母           | Caucasian Albanian                               |
| &nbsp;1&nbsp;SMP | 10570 &#8211; 105BF   | 維斯庫奇文                | Vithkuqi                                         |
| &nbsp;1&nbsp;SMP | 10600 &#8211; 1077F   | 線形文字A                | Linear A                                         |
| &nbsp;1&nbsp;SMP | 10780 &#8211; 107BF   | 拉丁字母擴展-F             | Latin Extended-F                                 |
| &nbsp;1&nbsp;SMP | 10800 &#8211; 1083F   | 賽普勒斯音節文字             | Cypriot Syllabary                                |
| &nbsp;1&nbsp;SMP | 10840 &#8211; 1085F   | 帝國亞拉姆文               | Imperial Aramaic                                 |
| &nbsp;1&nbsp;SMP | 10860 &#8211; 1087F   | 帕爾邁拉字母               | Palmyrene                                        |
| &nbsp;1&nbsp;SMP | 10880 &#8211; 108AF   | 納巴泰字母                | Nabataean                                        |
| &nbsp;1&nbsp;SMP | 108E0 &#8211; 108FF   | 哈特拉文                 | Hatran                                           |
| &nbsp;1&nbsp;SMP | 10900 &#8211; 1091F   | 腓尼基字母                | Phoenician                                       |
| &nbsp;1&nbsp;SMP | 10920 &#8211; 1093F   | 呂底亞字母                | Lydian                                           |
| &nbsp;1&nbsp;SMP | 10980 &#8211; 1099F   | 麥羅埃文聖書體              | Meroitic Hieroglyphs                             |
| &nbsp;1&nbsp;SMP | 109A0 &#8211; 109FF   | 麥羅埃文草書體              | Meroitic Cursive                                 |
| &nbsp;1&nbsp;SMP | 10A00 &#8211; 10A5F   | 佉盧文                  | Kharoshthi                                       |
| &nbsp;1&nbsp;SMP | 10A60 &#8211; 10A7F   | 古南阿拉伯字母              | Old South Arabian                                |
| &nbsp;1&nbsp;SMP | 10A80 &#8211; 10A9F   | 古北阿拉伯字母              | Old North Arabian                                |
| &nbsp;1&nbsp;SMP | 10AC0 &#8211; 10AFF   | 摩尼字母                 | Manichaean                                       |
| &nbsp;1&nbsp;SMP | 10B00 &#8211; 10B3F   | 阿維斯陀字母               | Avestan                                          |
| &nbsp;1&nbsp;SMP | 10B40 &#8211; 10B5F   | 碑刻帕提亞文               | Inscriptional Parthian                           |
| &nbsp;1&nbsp;SMP | 10B60 &#8211; 10B7F   | 碑刻巴列維文               | Inscriptional Pahlavi                            |
| &nbsp;1&nbsp;SMP | 10B80 &#8211; 10BAF   | 詩篇巴列維文               | Psalter Pahlavi                                  |
| &nbsp;1&nbsp;SMP | 10C00 &#8211; 10C4F   | 古突厥文                 | Old Turkic                                       |
| &nbsp;1&nbsp;SMP | 10C80 &#8211; 10CFF   | 古匈牙利字母               | Old Hungarian                                    |
| &nbsp;1&nbsp;SMP | 10D00 &#8211; 10D3F   | 哈乃斐羅興亞文字             | Hanifi Rohingya                                  |
| &nbsp;1&nbsp;SMP | 10E60 &#8211; 10E7F   | 盧米文數字                | Rumi Numeral Symbols                             |
| &nbsp;1&nbsp;SMP | 10E80 &#8211; 10EBF   | 雅茲迪文                 | Yezidi                                           |
| &nbsp;1&nbsp;SMP | 10EC0 &#8211; 10EFF   | 阿拉伯字母擴展-C            | Arabic Extended-C                                |
| &nbsp;1&nbsp;SMP | 10F00 &#8211; 10F2F   | 古粟特字母                | Old Sogdian                                      |
| &nbsp;1&nbsp;SMP | 10F30 &#8211; 10F6F   | 粟特字母                 | Sogdian                                          |
| &nbsp;1&nbsp;SMP | 10F70 &#8211; 10FAF   | 回鶻字母                 | Old Uyghur                                       |
| &nbsp;1&nbsp;SMP | 10FB0 &#8211; 10FDF   | 花剌子模字母               | Chorasmian                                       |
| &nbsp;1&nbsp;SMP | 10FE0 &#8211; 10FFF   | 埃利邁文                 | Elymaic                                          |
| &nbsp;1&nbsp;SMP | 11000 &#8211; 1107F   | 婆羅米文                 | Brahmi                                           |
| &nbsp;1&nbsp;SMP | 11080 &#8211; 110CF   | 凱提文                  | Kaithi                                           |
| &nbsp;1&nbsp;SMP | 110D0 &#8211; 110FF   | 索拉僧平文字               | Sora Sompeng                                     |
| &nbsp;1&nbsp;SMP | 11100 &#8211; 1114F   | 查克馬文                 | Chakma                                           |
| &nbsp;1&nbsp;SMP | 11150 &#8211; 1117F   | 馬哈佳尼文                | Mahajani                                         |
| &nbsp;1&nbsp;SMP | 11180 &#8211; 111DF   | 夏拉達文                 | Sharada                                          |
| &nbsp;1&nbsp;SMP | 111E0 &#8211; 111FF   | 古僧伽羅文數字              | Sinhala Archaic Numbers                          |
| &nbsp;1&nbsp;SMP | 11200 &#8211; 1124F   | 可吉文                  | Khojki                                           |
| &nbsp;1&nbsp;SMP | 11280 &#8211; 112AF   | 穆爾塔尼文                | Multani                                          |
| &nbsp;1&nbsp;SMP | 112B0 &#8211; 112FF   | 庫達瓦迪文                | Khudawadi                                        |
| &nbsp;1&nbsp;SMP | 11300 &#8211; 1137F   | 古蘭塔文                 | Grantha                                          |
| &nbsp;1&nbsp;SMP | 11400 &#8211; 1147F   | 紐瓦字母                 | Newa                                             |
| &nbsp;1&nbsp;SMP | 11480 &#8211; 114DF   | 底羅僕多文                | Tirhuta                                          |
| &nbsp;1&nbsp;SMP | 11580 &#8211; 115FF   | 悉曇文字                 | Siddham                                          |
| &nbsp;1&nbsp;SMP | 11600 &#8211; 1165F   | 莫迪文                  | Modi                                             |
| &nbsp;1&nbsp;SMP | 11660 &#8211; 1167F   | 蒙古文補充                | Mongolian Supplement                             |
| &nbsp;1&nbsp;SMP | 11680 &#8211; 116CF   | 塔克里文                 | Takri                                            |
| &nbsp;1&nbsp;SMP | 11700 &#8211; 1174F   | 阿洪姆文                 | Ahom                                             |
| &nbsp;1&nbsp;SMP | 11800 &#8211; 1184F   | 多格拉文                 | Dogra                                            |
| &nbsp;1&nbsp;SMP | 118A0 &#8211; 118FF   | 瓦蘭齊地文                | Warang Citi                                      |
| &nbsp;1&nbsp;SMP | 11900 &#8211; 1195F   | 島嶼字母                 | Dhives Akuru (Dives Akuru)                       |
| &nbsp;1&nbsp;SMP | 119A0 &#8211; 119FF   | 南迪城文                 | Nandinagari                                      |
| &nbsp;1&nbsp;SMP | 11A00 &#8211; 11A4F   | 札那巴札爾方形字母            | Zanabazar Square                                 |
| &nbsp;1&nbsp;SMP | 11A50 &#8211; 11AAF   | 索永布文字                | Soyombo                                          |
| &nbsp;1&nbsp;SMP | 11AB0 &#8211; 11ABF   | 加拿大原住民音節文字擴展-A       | Unified Canadian Aboriginal Syllabics Extended-A |
| &nbsp;1&nbsp;SMP | 11AC0 &#8211; 11AFF   | 包欽豪文                 | Pau Cin Hau                                      |
| &nbsp;1&nbsp;SMP | 11B00 &#8211; 11B5F   | 天城文擴展-A              | Devanagari Extended-A                            |
| &nbsp;1&nbsp;SMP | 11C00 &#8211; 11C6F   | 拜克舒基文                | Bhaiksuki                                        |
| &nbsp;1&nbsp;SMP | 11C70 &#8211; 11CBF   | 瑪欽文                  | Marchen                                          |
| &nbsp;1&nbsp;SMP | 11D00 &#8211; 11D5F   | 馬薩拉姆貢德文字             | Masaram Gondi                                    |
| &nbsp;1&nbsp;SMP | 11D60 &#8211; 11DAF   | 貢賈拉貢德文字              | Gunjala Gondi                                    |
| &nbsp;1&nbsp;SMP | 11EE0 &#8211; 11EFF   | 望加錫文                 | Makasar                                          |
| &nbsp;1&nbsp;SMP | 11F00 &#8211; 11F5F   | 卡維文                  | Kawi                                             |
| &nbsp;1&nbsp;SMP | 11FB0 &#8211; 11FBF   | 老傈僳文補充               | Lisu Supplement                                  |
| &nbsp;1&nbsp;SMP | 11FC0 &#8211; 11FFF   | 泰米爾文補充               | Tamil Supplement                                 |
| &nbsp;1&nbsp;SMP | 12000 &#8211; 123FF   | 楔形文字                 | Cuneiform                                        |
| &nbsp;1&nbsp;SMP | 12400 &#8211; 1247F   | 楔形文字數字和標點符號          | Cuneiform Numbers and Punctuation                |
| &nbsp;1&nbsp;SMP | 12480 &#8211; 1254F   | 早期王朝楔形文字             | Early Dynastic Cuneiform                         |
| &nbsp;1&nbsp;SMP | 12F90 &#8211; 12FFF   | 賽普勒斯-米諾斯文字           | Cypro-Minoan                                     |
| &nbsp;1&nbsp;SMP | 13000 &#8211; 1342F   | 埃及聖書體                | Egyptian Hieroglyphs                             |
| &nbsp;1&nbsp;SMP | 13430 &#8211; 1345F   | 埃及聖書體格式控制            | Egyptian Hieroglyph Format Controls              |
| &nbsp;1&nbsp;SMP | 14400 &#8211; 1467F   | 安納托利亞象形文字            | Anatolian Hieroglyphs                            |
| &nbsp;1&nbsp;SMP | 16800 &#8211; 16A3F   | 巴姆穆文字補充              | Bamum Supplement                                 |
| &nbsp;1&nbsp;SMP | 16A40 &#8211; 16A6F   | 默祿文                  | Mro                                              |
| &nbsp;1&nbsp;SMP | 16A70 &#8211; 16ACF   | 唐薩文                  | Tangsa                                           |
| &nbsp;1&nbsp;SMP | 16AD0 &#8211; 16AFF   | 巴薩文                  | Bassa Vah                                        |
| &nbsp;1&nbsp;SMP | 16B00 &#8211; 16B8F   | 救世苗文                 | Pahawh Hmong                                     |
| &nbsp;1&nbsp;SMP | 16E40 &#8211; 16E9F   | 梅德法伊德林文              | Medefaidrin                                      |
| &nbsp;1&nbsp;SMP | 16F00 &#8211; 16F9F   | 柏格理苗文                | Miao                                             |
| &nbsp;1&nbsp;SMP | 16FE0 &#8211; 16FFF   | 表意符號和標點符號            | Ideographic Symbols and Punctuation              |
| &nbsp;1&nbsp;SMP | 17000 &#8211; 187FF   | 西夏文                  | Tangut                                           |
| &nbsp;1&nbsp;SMP | 18800 &#8211; 18AFF   | 西夏文部件                | Tangut Components                                |
| &nbsp;1&nbsp;SMP | 18B00 &#8211; 18CFF   | 契丹小字                 | Khitan Small Script                              |
| &nbsp;1&nbsp;SMP | 18D00 &#8211; 18D7F   | 西夏文補充                | Tangut Supplement                                |
| &nbsp;1&nbsp;SMP | 1AFF0 &#8211; 1AFFF   | 假名擴展-B               | Kana Extended-B                                  |
| &nbsp;1&nbsp;SMP | 1B000 &#8211; 1B0FF   | 假名補充                 | Kana Supplement                                  |
| &nbsp;1&nbsp;SMP | 1B100 &#8211; 1B12F   | 假名擴展-A               | Kana Extended-A                                  |
| &nbsp;1&nbsp;SMP | 1B130 &#8211; 1B16F   | 小型假名擴充               | Small Kana Extension                             |
| &nbsp;1&nbsp;SMP | 1B170 &#8211; 1B2FF   | 女書                   | Nushu                                            |
| &nbsp;1&nbsp;SMP | 1BC00 &#8211; 1BC9F   | 杜普雷速記                | Duployan                                         |
| &nbsp;1&nbsp;SMP | 1BCA0 &#8211; 1BCAF   | 速記格式控制符              | Shorthand Format Controls                        |
| &nbsp;1&nbsp;SMP | 1CF00 &#8211; 1CFCF   | 贊玫尼聖歌音樂符號            | Znamenny Musical Notation                        |
| &nbsp;1&nbsp;SMP | 1D000 &#8211; 1D0FF   | 拜占庭音樂符號              | Byzantine Musical Symbols                        |
| &nbsp;1&nbsp;SMP | 1D100 &#8211; 1D1FF   | 音樂符號                 | Musical Symbols                                  |
| &nbsp;1&nbsp;SMP | 1D200 &#8211; 1D24F   | 古希臘音樂記號              | Ancient Greek Musical Notation                   |
| &nbsp;1&nbsp;SMP | 1D2C0 &#8211; 1D2DF   | 卡克托維克數字              | Kaktovik Numerals                                |
| &nbsp;1&nbsp;SMP | 1D2E0 &#8211; 1D2FF   | 瑪雅數字                 | Mayan Numerals                                   |
| &nbsp;1&nbsp;SMP | 1D300 &#8211; 1D35F   | 太玄經符號                | Tai Xuan Jing Symbols                            |
| &nbsp;1&nbsp;SMP | 1D360 &#8211; 1D37F   | 算籌                   | Counting Rod Numerals                            |
| &nbsp;1&nbsp;SMP | 1D400 &#8211; 1D7FF   | 字母和數字符號              | Mathematical Alphanumeric Symbols                |
| &nbsp;1&nbsp;SMP | 1D800 &#8211; 1DAAF   | 薩頓書寫符號               | Sutton SignWriting                               |
| &nbsp;1&nbsp;SMP | 1DF00 &#8211; 1DFFF   | 拉丁字母擴展-G             | Latin Extended-G                                 |
| &nbsp;1&nbsp;SMP | 1E000 &#8211; 1E02F   | 格拉哥里字母補充             | Glagolitic Supplement                            |
| &nbsp;1&nbsp;SMP | 1E030 &#8211; 1E08F   | 西里爾字母擴展-D            | Cyrillic Extended-D                              |
| &nbsp;1&nbsp;SMP | 1E100 &#8211; 1E14F   | 創世紀苗文                | Nyiakeng Puachue Hmong                           |
| &nbsp;1&nbsp;SMP | 1E290 &#8211; 1E2BF   | 投投文                  | Toto                                             |
| &nbsp;1&nbsp;SMP | 1E2C0 &#8211; 1E2FF   | 文喬字母                 | Wancho                                           |
| &nbsp;1&nbsp;SMP | 1E4D0 &#8211; 1E4FF   | 蒙達里字母                | Nag Mundari                                      |
| &nbsp;1&nbsp;SMP | 1E7E0 &#8211; 1E7FF   | 衣索比亞字母擴充-B           | Ethiopic Extended-B                              |
| &nbsp;1&nbsp;SMP | 1E800 &#8211; 1E8DF   | 門德基卡庫文               | Mende Kikakui                                    |
| &nbsp;1&nbsp;SMP | 1E900 &#8211; 1E95F   | 阿德拉姆字母               | Adlam                                            |
| &nbsp;1&nbsp;SMP | 1EC70 &#8211; 1ECBF   | 印度西亞格數字              | Indic Siyaq Numbers                              |
| &nbsp;1&nbsp;SMP | 1ED00 &#8211; 1ED4F   | 奧斯曼西亞格數字             | Ottoman Siyaq Numbers                            |
| &nbsp;1&nbsp;SMP | 1EE00 &#8211; 1EEFF   | 阿拉伯字母數字符號            | Arabic Mathematical Alphabetic Symbols           |
| &nbsp;1&nbsp;SMP | 1F000 &#8211; 1F02F   | 麻將牌                  | Mahjong Tiles                                    |
| &nbsp;1&nbsp;SMP | 1F030 &#8211; 1F09F   | 多米諾骨牌                | Domino Tiles                                     |
| &nbsp;1&nbsp;SMP | 1F0A0 &#8211; 1F0FF   | 撲克牌                  | Playing Cards                                    |
| &nbsp;1&nbsp;SMP | 1F100 &#8211; 1F1FF   | 圍繞字母數字補充             | Enclosed Alphanumeric Supplement                 |
| &nbsp;1&nbsp;SMP | 1F200 &#8211; 1F2FF   | 圍繞表意文字補充             | Enclosed Ideographic Supplement                  |
| &nbsp;1&nbsp;SMP | 1F300 &#8211; 1F5FF   | 雜項符號和象形文字            | Miscellaneous Symbols and Pictographs            |
| &nbsp;1&nbsp;SMP | 1F600 &#8211; 1F64F   | 表情符號                 | Emoticons                                        |
| &nbsp;1&nbsp;SMP | 1F650 &#8211; 1F67F   | 裝飾符號                 | Ornamental Dingbats                              |
| &nbsp;1&nbsp;SMP | 1F680 &#8211; 1F6FF   | 交通和地圖符號              | Transport and Map Symbols                        |
| &nbsp;1&nbsp;SMP | 1F700 &#8211; 1F77F   | 鍊金術符號                | Alchemical Symbols                               |
| &nbsp;1&nbsp;SMP | 1F780 &#8211; 1F7FF   | 幾何圖形擴展               | Geometric Shapes Extended                        |
| &nbsp;1&nbsp;SMP | 1F800 &#8211; 1F8FF   | 追加箭頭-C               | Supplemental Arrows-C                            |
| &nbsp;1&nbsp;SMP | 1F900 &#8211; 1F9FF   | 補充符號和象形文字            | Supplemental Symbols and Pictographs             |
| &nbsp;1&nbsp;SMP | 1FA00 &#8211; 1FA6F   | 棋類符號                 | Chess Symbols                                    |
| &nbsp;1&nbsp;SMP | 1FA70 &#8211; 1FAFF   | 符號和象形文字擴充-A          | Symbols and Pictographs Extended-A               |
| &nbsp;1&nbsp;SMP | 1FB00 &#8211; 1FBFF   | 遺留計算符號               | Symbols for Legacy Computing                     |
| &nbsp;2&nbsp;SIP | 20000 &#8211; 2A6DF   | 中日韓統一表意文字擴充區B        | CJK Unified Ideographs Extension B               |
| &nbsp;2&nbsp;SIP | 2A700 &#8211; 2B73F   | 中日韓統一表意文字擴充區C        | CJK Unified Ideographs Extension C               |
| &nbsp;2&nbsp;SIP | 2B740 &#8211; 2B81F   | 中日韓統一表意文字擴充區D        | CJK Unified Ideographs Extension D               |
| &nbsp;2&nbsp;SIP | 2B820 &#8211; 2CEAF   | 中日韓統一表意文字擴充區E        | CJK Unified Ideographs Extension E               |
| &nbsp;2&nbsp;SIP | 2CEB0 &#8211; 2EBEF   | 中日韓統一表意文字擴充區F        | CJK Unified Ideographs Extension F               |
| &nbsp;2&nbsp;SIP | 2F800 &#8211; 2FA1F   | 中日韓相容表意文字補充區         | CJK Compatibility Ideographs Supplement          |
| &nbsp;3&nbsp;TIP | 30000 &#8211; 3134F   | 中日韓統一表意文字擴充區G        | CJK Unified Ideographs Extension G               |
| &nbsp;3&nbsp;TIP | 31350 &#8211; 323AF   | 中日韓統一表意文字擴充區H        | CJK Unified Ideographs Extension H               |
| 14&nbsp;SSP      | E0000 &#8211; E007F   | 標籤                   | Tags                                             |
| 14&nbsp;SSP      | E0100 &#8211; E01EF   | 變體選擇符補充              | Variation Selectors Supplement                   |
| 15&nbsp;PUA-A    | F0000 &#8211; FFFFF   | 補充私人使用區-A            | Supplementary Private Use Area-A                 |
| 16&nbsp;PUA-B    | 100000 &#8211; 10FFFF | 補充私人使用區-B            | Supplementary Private Use Area-B                 |


## 參考文獻

- [Unicode字元平面對映](https://zh.wikipedia.org/zh-tw/Unicode%E5%AD%97%E7%AC%A6%E5%B9%B3%E9%9D%A2%E6%98%A0%E5%B0%84)
- [Unicode區段](https://zh.wikipedia.org/zh-tw/Unicode%E5%8D%80%E6%AE%B5)
