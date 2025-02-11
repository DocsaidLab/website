---
slug: unicode-table
title: Unicode Table
authors: Z. Yuan
tags: [unicode]
image: /en/img/2023/0902.webp
description: Unicode Table for easy reference.
---

Unicode is an international character encoding standard maintained by the non-profit organization Unicode Consortium. The standard unifies most of the world's writing systems, enabling information exchange across platforms and languages.

<!-- truncate -->

In the Unicode standard, each character corresponds to a unique numerical code, known as a "code point." The code point is an essential concept in the Unicode standard, used to determine the position of each unique character. The code points range from `U+0000` to `U+10FFFF`, meaning it can accommodate 1,114,112 different characters.

These code points are divided into multiple subsets based on different functions and purposes, such as:

- **Basic Multilingual Plane (BMP)**: Includes common characters like Latin letters, Greek letters, Chinese characters, etc.
- **Supplementary Planes**: Includes additional ideographic characters, supplementary characters, etc.

Given that we always spend half an hour looking up Unicode encoding every time we need it, it's better to store some basic information.

:::tip
If you require further details, we recommend consulting the original table: [**Unicode Code Points**](https://en.wikipedia.org/wiki/Unicode#Code_points)
:::

## 參考資料

- [Unicode 字元平面對映](https://zh.wikipedia.org/zh-tw/Unicode%E5%AD%97%E7%AC%A6%E5%B9%B3%E9%9D%A2%E6%98%A0%E5%B0%84)
- [Unicode 區段](https://zh.wikipedia.org/zh-tw/Unicode%E5%8D%80%E6%AE%B5)

## Unicode Code Point Range Table

If you require further details, we recommend consulting the original table: [**Unicode Code Points**](https://en.wikipedia.org/wiki/Unicode#Code_points)

|  Plane   | Code Point Range |         Chinese Name         |                   English Name                   |
| :------: | :--------------: | :--------------------------: | :----------------------------------------------: |
|  0 BMP   |    0000-007F     |         基本拉丁字母         |                   Basic Latin                    |
|  0 BMP   |    0080-00FF     |        拉丁字母補充-1        |                Latin-1 Supplement                |
|  0 BMP   |    0100-017F     |        拉丁字母擴展-A        |                 Latin Extended-A                 |
|  0 BMP   |    0180-024F     |        拉丁字母擴展-B        |                 Latin Extended-B                 |
|  0 BMP   |    0250-02AF     |         國際音標擴展         |                  IPA Extensions                  |
|  0 BMP   |    02B0-02FF     |         佔位修飾符號         |             Spacing Modifier Letters             |
|  0 BMP   |    0300-036F     |         組合附加符號         |           Combining Diacritical Marks            |
|  0 BMP   |    0370-03FF     |     希臘字母和科普特字母     |                 Greek and Coptic                 |
|  0 BMP   |    0400-04FF     |          西里爾字母          |                     Cyrillic                     |
|  0 BMP   |    0500-052F     |        西里爾字母補充        |               Cyrillic Supplement                |
|  0 BMP   |    0530-058F     |         亞美尼亞字母         |                     Armenian                     |
|  0 BMP   |    0590-05FF     |         希伯來文字母         |                      Hebrew                      |
|  0 BMP   |    0600-06FF     |         阿拉伯文字母         |                      Arabic                      |
|  0 BMP   |    0700-074F     |          敘利亞字母          |                      Syriac                      |
|  0 BMP   |    0750-077F     |         阿拉伯文補充         |                Arabic Supplement                 |
|  0 BMP   |    0780-07BF     |           它拿字母           |                      Thaana                      |
|  0 BMP   |    07C0-07FF     |         西非書面文字         |                       NKo                        |
|  0 BMP   |    0800-083F     |         撒瑪利亞字母         |                    Samaritan                     |
|  0 BMP   |    0840-085F     |          曼達安字母          |                     Mandaic                      |
|  0 BMP   |    0860-086F     |         敘利亞文補充         |                Syriac Supplement                 |
|  0 BMP   |    0870-089F     |       阿拉伯字母擴展-B       |                Arabic Extended-B                 |
|  0 BMP   |    08A0-08FF     |       阿拉伯字母擴展-A       |                Arabic Extended-A                 |
|  0 BMP   |    0900-097F     |            天城文            |                    Devanagari                    |
|  0 BMP   |    0980-09FF     |           孟加拉文           |                     Bengali                      |
|  0 BMP   |    0A00-0A7F     |           古木基文           |                     Gurmukhi                     |
|  0 BMP   |    0A80-0AFF     |          古吉拉特文          |                     Gujarati                     |
|  0 BMP   |    0B00-0B7F     |           奧里亞文           |                      Oriya                       |
|  0 BMP   |    0B80-0BFF     |           泰米爾文           |                      Tamil                       |
|  0 BMP   |    0C00-0C7F     |           泰盧固文           |                      Telugu                      |
|  0 BMP   |    0C80-0CFF     |           卡納達文           |                     Kannada                      |
|  0 BMP   |    0D00-0D7F     |         馬拉雅拉姆文         |                    Malayalam                     |
|  0 BMP   |    0D80-0DFF     |           僧伽羅文           |                     Sinhala                      |
|  0 BMP   |    0E00-0E7F     |             泰文             |                       Thai                       |
|  0 BMP   |    0E80-0EFF     |             寮文             |                       Lao                        |
|  0 BMP   |    0F00-0FFF     |             藏文             |                     Tibetan                      |
|  0 BMP   |    1000-109F     |            緬甸文            |                     Myanmar                      |
|  0 BMP   |    10A0-10FF     |          喬治亞字母          |                     Georgian                     |
|  0 BMP   |    1100-11FF     |           諺文字母           |                   Hangul Jamo                    |
|  0 BMP   |    1200-137F     |         衣索比亞字母         |                     Ethiopic                     |
|  0 BMP   |    1380-139F     |       衣索比亞字母補充       |               Ethiopic Supplement                |
|  0 BMP   |    13A0-13FF     |           切羅基文           |                     Cherokee                     |
|  0 BMP   |    1400-167F     |   統一加拿大原住民音節文字   |      Unified Canadian Aboriginal Syllabics       |
|  0 BMP   |    1680-169F     |           歐甘字母           |                      Ogham                       |
|  0 BMP   |    16A0-16FF     |           盧恩字母           |                      Runic                       |
|  0 BMP   |    1700-171F     |          他加祿字母          |                     Tagalog                      |
|  0 BMP   |    1720-173F     |           哈努諾文           |                     Hanunoo                      |
|  0 BMP   |    1740-175F     |          布希德字母          |                      Buhid                       |
|  0 BMP   |    1760-177F     |         塔格班瓦字母         |                     Tagbanwa                     |
|  0 BMP   |    1780-17FF     |            高棉文            |                      Khmer                       |
|  0 BMP   |    1800-18AF     |            蒙古文            |                    Mongolian                     |
|  0 BMP   |    18B0-18FF     | 統一加拿大原住民音節文字擴充 |  Unified Canadian Aboriginal Syllabics Extended  |
|  0 BMP   |    1900-194F     |            林布文            |                      Limbu                       |
|  0 BMP   |    1950-197F     |           德宏傣文           |                      Tai Le                      |
|  0 BMP   |    1980-19DF     |           新傣仂文           |                    New Tai Le                    |
|  0 BMP   |    19E0-19FF     |          高棉文符號          |                  Khmer Symbols                   |
|  0 BMP   |    1A00-1A1F     |            布吉文            |                     Buginese                     |
|  0 BMP   |    1A20-1AAF     |            老傣文            |                     Tai Tham                     |
|  0 BMP   |    1AB0-1AFF     |       組合附加符號擴展       |       Combining Diacritical Marks Extended       |
|  0 BMP   |    1B00-1B7F     |           峇里字母           |                     Balinese                     |
|  0 BMP   |    1B80-1BBF     |           巽他字母           |                    Sundanese                     |
|  0 BMP   |    1BC0-1BFF     |          巴塔克字母          |                      Batak                       |
|  0 BMP   |    1C00-1C4F     |            絨巴文            |                      Lepcha                      |
|  0 BMP   |    1C50-1C7F     |           桑塔利文           |                     Ol Chiki                     |
|  0 BMP   |    1C80-1C8F     |       西里爾字母擴展-C       |               Cyrillic Extended-C                |
|  0 BMP   |    1C90-1CBF     |        喬治亞字母擴展        |                Georgian Extended                 |
|  0 BMP   |    1CC0-1CCF     |         巽他字母補充         |               Sundanese Supplement               |
|  0 BMP   |    1CD0-1CFF     |           吠陀擴展           |                 Vedic Extensions                 |
|  0 BMP   |    1D00-1D7F     |           音標擴展           |               Phonetic Extensions                |
|  0 BMP   |    1D80-1DBF     |         音標擴展補充         |          Phonetic Extensions Supplement          |
|  0 BMP   |    1DC0-1DFF     |       組合附加符號補充       |      Combining Diacritical Marks Supplement      |
|  0 BMP   |    1E00-1EFF     |       拉丁字母擴展附加       |            Latin Extended Additional             |
|  0 BMP   |    1F00-1FFF     |         希臘字母擴展         |                  Greek Extended                  |
|  0 BMP   |    2000-206F     |           一般標點           |               General Punctuation                |
|  0 BMP   |    2070-209F     |          上標及下標          |           Superscripts and Subscripts            |
|  0 BMP   |    20A0-20CF     |           貨幣符號           |                 Currency Symbols                 |
|  0 BMP   |    20D0-20FF     |      符號用組合附加符號      |     Combining Diacritical Marks for Symbols      |
|  0 BMP   |    2100-214F     |          類字母符號          |                Letterlike Symbols                |
|  0 BMP   |    2150-218F     |           數字形式           |                   Number Forms                   |
|  0 BMP   |    2190-21FF     |             箭頭             |                      Arrows                      |
|  0 BMP   |    2200-22FF     |          數學運算子          |              Mathematical Operators              |
|  0 BMP   |    2300-23FF     |         雜項技術符號         |             Miscellaneous Technical              |
|  0 BMP   |    2400-243F     |           控制圖形           |                 Control Pictures                 |
|  0 BMP   |    2440-245F     |         光學字元辨識         |          Optical Character Recognition           |
|  0 BMP   |    2460-24FF     |         圍繞字母數字         |              Enclosed Alphanumerics              |
|  0 BMP   |    2500-257F     |            制表符            |                   Box Drawing                    |
|  0 BMP   |    2580-259F     |           方塊元素           |                  Block Elements                  |
|  0 BMP   |    25A0-25FF     |           幾何圖形           |                 Geometric Shapes                 |
|  0 BMP   |    2600-26FF     |           雜項符號           |              Miscellaneous Symbols               |
|  0 BMP   |    2700-27BF     |           裝飾符號           |                     Dingbats                     |
|  0 BMP   |    27C0-27EF     |        雜項數學符號-A        |       Miscellaneous Mathematical Symbols-A       |
|  0 BMP   |    27F0-27FF     |          追加箭頭-A          |              Supplemental Arrows-A               |
|  0 BMP   |    2800-28FF     |           點字圖案           |                 Braille Patterns                 |
|  0 BMP   |    2900-297F     |          追加箭頭-B          |              Supplemental Arrows-B               |
|  0 BMP   |    2980-29FF     |        雜項數學符號-B        |       Miscellaneous Mathematical Symbols-B       |
|  0 BMP   |    2A00-2AFF     |        補充數學運算子        |       Supplemental Mathematical Operators        |
|  0 BMP   |    2B00-2BFF     |        雜項符號和箭頭        |         Miscellaneous Symbols and Arrows         |
|  0 BMP   |    2C00-2C5F     |         格拉哥里字母         |                    Glagolitic                    |
|  0 BMP   |    2C60-2C7F     |        拉丁字母擴展-C        |                 Latin Extended-C                 |
|  0 BMP   |    2C80-2CFF     |          科普特字母          |                      Coptic                      |
|  0 BMP   |    2D00-2D2F     |        喬治亞字母補充        |               Georgian Supplement                |
|  0 BMP   |    2D30-2D7F     |           提非納文           |                     Tifinagh                     |
|  0 BMP   |    2D80-2DDF     |       衣索比亞字母擴充       |                Ethiopic Extended                 |
|  0 BMP   |    2DE0-2DFF     |       西里爾字母擴展-A       |               Cyrillic Extended-A                |
|  0 BMP   |    2E00-2E7F     |           補充標點           |             Supplemental Punctuation             |
|  0 BMP   |    2E80-2EFF     |      中日韓漢字部首補充      |             CJK Radicals Supplement              |
|  0 BMP   |    2F00-2FDF     |           康熙部首           |                 Kangxi Radicals                  |
|  0 BMP   |    2FF0-2FFF     |       表意文字描述字元       |        Ideographic Description Characters        |
|  0 BMP   |    3000-303F     |       中日韓符號和標點       |           CJK Symbols and Punctuation            |
|  0 BMP   |    3040-309F     |            平假名            |                     Hiragana                     |
|  0 BMP   |    30A0-30FF     |            片假名            |                     Katakana                     |
|  0 BMP   |    3100-312F     |           注音符號           |                     Bopomofo                     |
|  0 BMP   |    3130-318F     |         諺文相容字母         |            Hangul Compatibility Jamo             |
|  0 BMP   |    3190-319F     |         漢文訓讀符號         |                      Kanbun                      |
|  0 BMP   |    31A0-31BF     |         注音符號擴展         |                Bopomofo Extended                 |
|  0 BMP   |    31C0-31EF     |          中日韓筆畫          |                   CJK Strokes                    |
|  0 BMP   |    31F0-31FF     |        片假名語音擴展        |           Katakana Phonetic Extensions           |
|  0 BMP   |    3200-32FF     |     中日韓圍繞字元及月份     |         Enclosed CJK Letters and Months          |
|  0 BMP   |    3300-33FF     |        中日韓相容字元        |                CJK Compatibility                 |
|  0 BMP   |    3400-4DBF     |  中日韓統一表意文字擴充區 A  |        CJK Unified Ideographs Extension A        |
|  0 BMP   |    4DC0-4DFF     |       易經六十四卦符號       |             Yijing Hexagram Symbols              |
|  0 BMP   |    4E00-9FFF     | 中日韓統一表意文字 (基本區)  |              CJK Unified Ideographs              |
|  0 BMP   |    A000-A48F     |           彝文音節           |                   Yi Syllables                   |
|  0 BMP   |    A490-A4CF     |           彝文部首           |                   Yi Radicals                    |
|  0 BMP   |    A4D0-A4FF     |            傈僳文            |                       Lisu                       |
|  0 BMP   |    A500-A63F     |            瓦伊文            |                       Vai                        |
|  0 BMP   |    A640-A69F     |       西里爾字母擴展-B       |               Cyrillic Extended-B                |
|  0 BMP   |    A6A0-A6FF     |          巴姆穆文字          |                      Bamum                       |
|  0 BMP   |    A700-A71F     |         聲調修飾符號         |              Modifier Tone Letters               |
|  0 BMP   |    A720-A7FF     |        拉丁字母擴展-D        |                 Latin Extended-D                 |
|  0 BMP   |    A800-A82F     |          錫爾赫特文          |                   Syloti Nagri                   |
|  0 BMP   |    A830-A83F     |       通用印度數字形式       |            Common Indic Number Forms             |
|  0 BMP   |    A840-A87F     |           八思巴文           |                     Phags-pa                     |
|  0 BMP   |    A880-A8DF     |         索拉什特拉文         |                    Saurashtra                    |
|  0 BMP   |    A8E0-A8FF     |          天城文擴展          |               Devanagari Extended                |
|  0 BMP   |    A900-A92F     |           克耶字母           |                     Kayah Li                     |
|  0 BMP   |    A930-A95F     |           勒姜字母           |                      Rejang                      |
|  0 BMP   |    A960-A97F     |        諺文字母擴展-A        |              Hangul Jamo Extended-A              |
|  0 BMP   |    A980-A9DF     |           爪哇字母           |                     Javanese                     |
|  0 BMP   |    A9E0-A9FF     |         緬甸文擴展-B         |                Myanmar Extended-B                |
|  0 BMP   |    AA00-AA5F     |             占文             |                       Cham                       |
|  0 BMP   |    AA60-AA7F     |         緬甸文擴展-A         |                Myanmar Extended-A                |
|  0 BMP   |    AA80-AADF     |            傣越文            |                     Tai Viet                     |
|  0 BMP   |    AAE0-AAFF     |          梅泰文擴充          |             Meetei Mayek Extensions              |
|  0 BMP   |    AB00-AB2F     |      衣索比亞字母擴充-A      |               Ethiopic Extended-A                |
|  0 BMP   |    AB30-AB6F     |        拉丁字母擴展-E        |                 Latin Extended-E                 |
|  0 BMP   |    AB70-ABBF     |         切羅基文補充         |               Cherokee Supplement                |
|  0 BMP   |    ABC0-ABFF     |            梅泰文            |                   Meetei Mayek                   |
|  0 BMP   |    AC00-D7AF     |           諺文音節           |                 Hangul Syllables                 |
|  0 BMP   |    D7B0-D7FF     |        諺文字母擴展-B        |              Hangul Jamo Extended-B              |
|  0 BMP   |    D800-DB7F     |          高半代用區          |                 High Surrogates                  |
|  0 BMP   |    DB80-DBFF     |        高半私人代用區        |           High Private Use Surrogates            |
|  0 BMP   |    DC00-DFFF     |          低半代用區          |                  Low Surrogates                  |
|  0 BMP   |    E000-F8FF     |            私用區            |                 Private Use Area                 |
|  0 BMP   |    F900-FAFF     |      中日韓相容表意文字      |           CJK Compatibility Ideographs           |
|  0 BMP   |    FB00-FB4F     |         字母表達形式         |          Alphabetic Presentation Forms           |
|  0 BMP   |    FB50-FDFF     |     阿拉伯字母表達形式-A     |           Arabic Presentation Forms-A            |
|  0 BMP   |    FE00-FE0F     |          變體選擇符          |               Variation Selectors                |
|  0 BMP   |    FE10-FE1F     |           豎排形式           |                  Vertical Forms                  |
|  0 BMP   |    FE20-FE2F     |         組合用半符號         |               Combining Half Marks               |
|  0 BMP   |    FE30-FE4F     |        中日韓相容形式        |             CJK Compatibility Forms              |
|  0 BMP   |    FE50-FE6F     |         小寫變體形式         |               Small Form Variants                |
|  0 BMP   |    FE70-FEFF     |     阿拉伯字母表達形式-B     |           Arabic Presentation Forms-B            |
|  0 BMP   |    FF00-FFEF     |        半形及全形字元        |          Halfwidth and Fullwidth Forms           |
|  0 BMP   |    FFF0-FFFF     |             特殊             |                     Specials                     |
|  1 SMP   |   10000-1007F    |     線形文字 B 音節文字      |                Linear B Syllabary                |
|  1 SMP   |   10080-100FF    |     線形文字 B 表意文字      |                Linear B Ideograms                |
|  1 SMP   |   10100-1013F    |          愛琴海數字          |                  Aegean Numbers                  |
|  1 SMP   |   10140-1018F    |          古希臘數字          |              Ancient Greek Numbers               |
|  1 SMP   |   10190-101CF    |           古代符號           |                 Ancient Symbols                  |
|  1 SMP   |   101D0-101FF    |         斐斯托斯圓盤         |                  Phaistos Disc                   |
|  1 SMP   |   10280-1029F    |          呂基亞字母          |                      Lycian                      |
|  1 SMP   |   102A0-102DF    |          卡里亞字母          |                      Carian                      |
|  1 SMP   |   102E0-102FF    |        科普特閏餘數字        |               Coptic Epact Numbers               |
|  1 SMP   |   10300-1032F    |         古義大利字母         |                    Old Italic                    |
|  1 SMP   |   10330-1034F    |           哥特字母           |                      Gothic                      |
|  1 SMP   |   10350-1037F    |          古彼爾姆文          |                    Old Permic                    |
|  1 SMP   |   10380-1039F    |         烏加里特字母         |                     Ugaritic                     |
|  1 SMP   |   103A0-103DF    |        古波斯楔形文字        |                   Old Persian                    |
|  1 SMP   |   10400-1044F    |         德瑟雷特字母         |                     Deseret                      |
|  1 SMP   |   10450-1047F    |          蕭伯納字母          |                     Shavian                      |
|  1 SMP   |   10480-104AF    |         奧斯曼亞字母         |                     Osmanya                      |
|  1 SMP   |   104B0-104FF    |          歐塞奇字母          |                      Osage                       |
|  1 SMP   |   10500-1052F    |         愛爾巴桑字母         |                     Elbasan                      |
|  1 SMP   |   10530-1056F    |     高加索阿爾巴尼亞字母     |                Caucasian Albanian                |
|  1 SMP   |   10570-105BF    |          維斯庫奇文          |                     Vithkuqi                     |
|  1 SMP   |   10600-1077F    |          線形文字 A          |                     Linear A                     |
|  1 SMP   |   10780-107BF    |        拉丁字母擴展-F        |                 Latin Extended-F                 |
|  1 SMP   |   10800-1083F    |       賽普勒斯音節文字       |                Cypriot Syllabary                 |
|  1 SMP   |   10840-1085F    |         帝國亞拉姆文         |                 Imperial Aramaic                 |
|  1 SMP   |   10860-1087F    |         帕爾邁拉字母         |                    Palmyrene                     |
|  1 SMP   |   10880-108AF    |          納巴泰字母          |                    Nabataean                     |
|  1 SMP   |   108E0-108FF    |           哈特拉文           |                      Hatran                      |
|  1 SMP   |   10900-1091F    |          腓尼基字母          |                    Phoenician                    |
|  1 SMP   |   10920-1093F    |          呂底亞字母          |                      Lydian                      |
|  1 SMP   |   10980-1099F    |        麥羅埃文聖書體        |               Meroitic Hieroglyphs               |
|  1 SMP   |   109A0-109FF    |        麥羅埃文草書體        |                 Meroitic Cursive                 |
|  1 SMP   |   10A00-10A5F    |            佉盧文            |                    Kharoshthi                    |
|  1 SMP   |   10A60-10A7F    |        古南阿拉伯字母        |                Old South Arabian                 |
|  1 SMP   |   10A80-10A9F    |        古北阿拉伯字母        |                Old North Arabian                 |
|  1 SMP   |   10AC0-10AFF    |           摩尼字母           |                    Manichaean                    |
|  1 SMP   |   10B00-10B3F    |         阿維斯陀字母         |                     Avestan                      |
|  1 SMP   |   10B40-10B5F    |         碑刻帕提亞文         |              Inscriptional Parthian              |
|  1 SMP   |   10B60-10B7F    |         碑刻巴列維文         |              Inscriptional Pahlavi               |
|  1 SMP   |   10B80-10BAF    |         詩篇巴列維文         |                 Psalter Pahlavi                  |
|  1 SMP   |   10C00-10C4F    |           古突厥文           |                    Old Turkic                    |
|  1 SMP   |   10C80-10CFF    |         古匈牙利字母         |                  Old Hungarian                   |
|  1 SMP   |   10D00-10D3F    |       哈乃斐羅興亞文字       |                 Hanifi Rohingya                  |
|  1 SMP   |   10E60-10E7F    |          盧米文數字          |               Rumi Numeral Symbols               |
|  1 SMP   |   10E80-10EBF    |           雅茲迪文           |                      Yezidi                      |
|  1 SMP   |   10EC0-10EFF    |       阿拉伯字母擴展-C       |                Arabic Extended-C                 |
|  1 SMP   |   10F00-10F2F    |          古粟特字母          |                   Old Sogdian                    |
|  1 SMP   |   10F30-10F6F    |           粟特字母           |                     Sogdian                      |
|  1 SMP   |   10F70-10FAF    |           回鶻字母           |                    Old Uyghur                    |
|  1 SMP   |   10FB0-10FDF    |         花剌子模字母         |                    Chorasmian                    |
|  1 SMP   |   10FE0-10FFF    |           埃利邁文           |                     Elymaic                      |
|  1 SMP   |   11000-1107F    |           婆羅米文           |                      Brahmi                      |
|  1 SMP   |   11080-110CF    |            凱提文            |                      Kaithi                      |
|  1 SMP   |   110D0-110FF    |         索拉僧平文字         |                   Sora Sompeng                   |
|  1 SMP   |   11100-1114F    |           查克馬文           |                      Chakma                      |
|  1 SMP   |   11150-1117F    |          馬哈佳尼文          |                     Mahajani                     |
|  1 SMP   |   11180-111DF    |           夏拉達文           |                     Sharada                      |
|  1 SMP   |   111E0-111FF    |        古僧伽羅文數字        |             Sinhala Archaic Numbers              |
|  1 SMP   |   11200-1124F    |            可吉文            |                      Khojki                      |
|  1 SMP   |   11280-112AF    |          穆爾塔尼文          |                     Multani                      |
|  1 SMP   |   112B0-112FF    |          庫達瓦迪文          |                    Khudawadi                     |
|  1 SMP   |   11300-1137F    |           古蘭塔文           |                     Grantha                      |
|  1 SMP   |   11400-1147F    |           紐瓦字母           |                       Newa                       |
|  1 SMP   |   11480-114DF    |          底羅僕多文          |                     Tirhuta                      |
|  1 SMP   |   11580-115FF    |           悉曇文字           |                     Siddham                      |
|  1 SMP   |   11600-1165F    |            莫迪文            |                       Modi                       |
|  1 SMP   |   11660-1167F    |          蒙古文補充          |               Mongolian Supplement               |
|  1 SMP   |   11680-116CF    |           塔克里文           |                      Takri                       |
|  1 SMP   |   11700-1174F    |           阿洪姆文           |                       Ahom                       |
|  1 SMP   |   11800-1184F    |           多格拉文           |                      Dogra                       |
|  1 SMP   |   118A0-118FF    |          瓦蘭齊地文          |                   Warang Citi                    |
|  1 SMP   |   11900-1195F    |           島嶼字母           |            Dhives Akuru (Dives Akuru)            |
|  1 SMP   |   119A0-119FF    |           南迪城文           |                   Nandinagari                    |
|  1 SMP   |   11A00-11A4F    |      札那巴札爾方形字母      |                 Zanabazar Square                 |
|  1 SMP   |   11A50-11AAF    |          索永布文字          |                     Soyombo                      |
|  1 SMP   |   11AB0-11ABF    |  加拿大原住民音節文字擴展-A  | Unified Canadian Aboriginal Syllabics Extended-A |
|  1 SMP   |   11AC0-11AFF    |           包欽豪文           |                   Pau Cin Hau                    |
|  1 SMP   |   11B00-11B5F    |         天城文擴展-A         |              Devanagari Extended-A               |
|  1 SMP   |   11C00-11C6F    |          拜克舒基文          |                    Bhaiksuki                     |
|  1 SMP   |   11C70-11CBF    |            瑪欽文            |                     Marchen                      |
|  1 SMP   |   11D00-11D5F    |       馬薩拉姆貢德文字       |                  Masaram Gondi                   |
|  1 SMP   |   11D60-11DAF    |        貢賈拉貢德文字        |                  Gunjala Gondi                   |
|  1 SMP   |   11EE0-11EFF    |           望加錫文           |                     Makasar                      |
|  1 SMP   |   11F00-11F5F    |            卡維文            |                       Kawi                       |
|  1 SMP   |   11FB0-11FBF    |         老傈僳文補充         |                 Lisu Supplement                  |
|  1 SMP   |   11FC0-11FFF    |         泰米爾文補充         |                 Tamil Supplement                 |
|  1 SMP   |   12000-123FF    |           楔形文字           |                    Cuneiform                     |
|  1 SMP   |   12400-1247F    |    楔形文字數字和標點符號    |        Cuneiform Numbers and Punctuation         |
|  1 SMP   |   12480-1254F    |       早期王朝楔形文字       |             Early Dynastic Cuneiform             |
|  1 SMP   |   12F90-12FFF    |     賽普勒斯-米諾斯文字      |                   Cypro-Minoan                   |
|  1 SMP   |   13000-1342F    |          埃及聖書體          |               Egyptian Hieroglyphs               |
|  1 SMP   |   13430-1345F    |      埃及聖書體格式控制      |       Egyptian Hieroglyph Format Controls        |
|  1 SMP   |   14400-1467F    |      安納托利亞象形文字      |              Anatolian Hieroglyphs               |
|  1 SMP   |   16800-16A3F    |        巴姆穆文字補充        |                 Bamum Supplement                 |
|  1 SMP   |   16A40-16A6F    |            默祿文            |                       Mro                        |
|  1 SMP   |   16A70-16ACF    |            唐薩文            |                      Tangsa                      |
|  1 SMP   |   16AD0-16AFF    |            巴薩文            |                    Bassa Vah                     |
|  1 SMP   |   16B00-16B8F    |           救世苗文           |                   Pahawh Hmong                   |
|  1 SMP   |   16E40-16E9F    |        梅德法伊德林文        |                   Medefaidrin                    |
|  1 SMP   |   16F00-16F9F    |          柏格理苗文          |                       Miao                       |
|  1 SMP   |   16FE0-16FFF    |      表意符號和標點符號      |       Ideographic Symbols and Punctuation        |
|  1 SMP   |   17000-187FF    |            西夏文            |                      Tangut                      |
|  1 SMP   |   18800-18AFF    |          西夏文部件          |                Tangut Components                 |
|  1 SMP   |   18B00-18CFF    |           契丹小字           |               Khitan Small Script                |
|  1 SMP   |   18D00-18D7F    |          西夏文補充          |                Tangut Supplement                 |
|  1 SMP   |   1AFF0-1AFFF    |          假名擴展-B          |                 Kana Extended-B                  |
|  1 SMP   |   1B000-1B0FF    |           假名補充           |                 Kana Supplement                  |
|  1 SMP   |   1B100-1B12F    |          假名擴展-A          |                 Kana Extended-A                  |
|  1 SMP   |   1B130-1B16F    |         小型假名擴充         |               Small Kana Extension               |
|  1 SMP   |   1B170-1B2FF    |             女書             |                      Nushu                       |
|  1 SMP   |   1BC00-1BC9F    |          杜普雷速記          |                     Duployan                     |
|  1 SMP   |   1BCA0-1BCAF    |        速記格式控制符        |            Shorthand Format Controls             |
|  1 SMP   |   1CF00-1CFCF    |      贊玫尼聖歌音樂符號      |            Znamenny Musical Notation             |
|  1 SMP   |   1D000-1D0FF    |        拜占庭音樂符號        |            Byzantine Musical Symbols             |
|  1 SMP   |   1D100-1D1FF    |           音樂符號           |                 Musical Symbols                  |
|  1 SMP   |   1D200-1D24F    |        古希臘音樂記號        |          Ancient Greek Musical Notation          |
|  1 SMP   |   1D2C0-1D2DF    |        卡克托維克數字        |                Kaktovik Numerals                 |
|  1 SMP   |   1D2E0-1D2FF    |           瑪雅數字           |                  Mayan Numerals                  |
|  1 SMP   |   1D300-1D35F    |          太玄經符號          |              Tai Xuan Jing Symbols               |
|  1 SMP   |   1D360-1D37F    |             算籌             |              Counting Rod Numerals               |
|  1 SMP   |   1D400-1D7FF    |        字母和數字元號        |        Mathematical Alphanumeric Symbols         |
|  1 SMP   |   1D800-1DAAF    |         薩頓書寫符號         |                Sutton SignWriting                |
|  1 SMP   |   1DF00-1DFFF    |        拉丁字母擴展-G        |                 Latin Extended-G                 |
|  1 SMP   |   1E000-1E02F    |       格拉哥里字母補充       |              Glagolitic Supplement               |
|  1 SMP   |   1E030-1E08F    |       西里爾字母擴展-D       |               Cyrillic Extended-D                |
|  1 SMP   |   1E100-1E14F    |          創世紀苗文          |              Nyiakeng Puachue Hmong              |
|  1 SMP   |   1E290-1E2BF    |            投投文            |                       Toto                       |
|  1 SMP   |   1E2C0-1E2FF    |           文喬字母           |                      Wancho                      |
|  1 SMP   |   1E4D0-1E4FF    |          蒙達里字母          |                   Nag Mundari                    |
|  1 SMP   |   1E7E0-1E7FF    |      衣索比亞字母擴充-B      |               Ethiopic Extended-B                |
|  1 SMP   |   1E800-1E8DF    |         門德基卡庫文         |                  Mende Kikakui                   |
|  1 SMP   |   1E900-1E95F    |         阿德拉姆字母         |                      Adlam                       |
|  1 SMP   |   1EC70-1ECBF    |        印度西亞格數字        |               Indic Siyaq Numbers                |
|  1 SMP   |   1ED00-1ED4F    |       奧斯曼西亞格數字       |              Ottoman Siyaq Numbers               |
|  1 SMP   |   1EE00-1EEFF    |      阿拉伯字母數字元號      |      Arabic Mathematical Alphabetic Symbols      |
|  1 SMP   |   1F000-1F02F    |            麻將牌            |                  Mahjong Tiles                   |
|  1 SMP   |   1F030-1F09F    |          多米諾骨牌          |                   Domino Tiles                   |
|  1 SMP   |   1F0A0-1F0FF    |            撲克牌            |                  Playing Cards                   |
|  1 SMP   |   1F100-1F1FF    |       圍繞字母數字補充       |         Enclosed Alphanumeric Supplement         |
|  1 SMP   |   1F200-1F2FF    |       圍繞表意文字補充       |         Enclosed Ideographic Supplement          |
|  1 SMP   |   1F300-1F5FF    |      雜項符號和象形文字      |      Miscellaneous Symbols and Pictographs       |
|  1 SMP   |   1F600-1F64F    |           表情符號           |                    Emoticons                     |
|  1 SMP   |   1F650-1F67F    |           裝飾符號           |               Ornamental Dingbats                |
|  1 SMP   |   1F680-1F6FF    |        交通和地圖符號        |            Transport and Map Symbols             |
|  1 SMP   |   1F700-1F77F    |          鍊金術符號          |                Alchemical Symbols                |
|  1 SMP   |   1F780-1F7FF    |         幾何圖形擴展         |            Geometric Shapes Extended             |
|  1 SMP   |   1F800-1F8FF    |          追加箭頭-C          |              Supplemental Arrows-C               |
|  1 SMP   |   1F900-1F9FF    |      補充符號和象形文字      |       Supplemental Symbols and Pictographs       |
|  1 SMP   |   1FA00-1FA6F    |           棋類符號           |                  Chess Symbols                   |
|  1 SMP   |   1FA70-1FAFF    |     符號和象形文字擴充-A     |        Symbols and Pictographs Extended-A        |
|  1 SMP   |   1FB00-1FBFF    |         遺留計算符號         |           Symbols for Legacy Computing           |
|  2 SIP   |   20000-2A6DF    |  中日韓統一表意文字擴充區 B  |        CJK Unified Ideographs Extension B        |
|  2 SIP   |   2A700-2B73F    |  中日韓統一表意文字擴充區 C  |        CJK Unified Ideographs Extension C        |
|  2 SIP   |   2B740-2B81F    |  中日韓統一表意文字擴充區 D  |        CJK Unified Ideographs Extension D        |
|  2 SIP   |   2B820-2CEAF    |  中日韓統一表意文字擴充區 E  |        CJK Unified Ideographs Extension E        |
|  2 SIP   |   2CEB0-2EBEF    |  中日韓統一表意文字擴充區 F  |        CJK Unified Ideographs Extension F        |
|  2 SIP   |   2F800-2FA1F    |   中日韓相容表意文字補充區   |     CJK Compatibility Ideographs Supplement      |
|  3 TIP   |   30000-3134F    |  中日韓統一表意文字擴充區 G  |        CJK Unified Ideographs Extension G        |
|  3 TIP   |   31350-323AF    |  中日韓統一表意文字擴充區 H  |        CJK Unified Ideographs Extension H        |
|  14 SSP  |   E0000-E007F    |             標籤             |                       Tags                       |
|  14 SSP  |   E0100-E01EF    |        變體選擇符補充        |          Variation Selectors Supplement          |
| 15 PUA-A |   F0000-FFFFF    |       補充私人使用區-A       |         Supplementary Private Use Area-A         |
| 16 PUA-B |  100000-10FFFF   |       補充私人使用區-B       |         Supplementary Private Use Area-B         |
