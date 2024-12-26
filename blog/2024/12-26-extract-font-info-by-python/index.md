---
slug: extract-font-info-by-python
title: 取得字型檔案的資訊
authors: Zephyr
image: /img/2024/1226.webp
tags: [font-tools, Python]
description: 透過 Python 取得字型檔案的資訊。
---

雖然我們經常在使用各種字型，但每次都會卡在取得字型參數的環節。

用完就忘記，下次再重新查一次？

這樣顯得我們不夠專業，必須寫點程式來解決問題。

<!-- truncate -->

## 安裝 `fontTools`

我們使用 `fontTools` 來取得字型相關資訊，這是一個在 Python 社群廣受好評的處理字型檔案套件，我們可以透過他來操作並解析各式 TTF、OTF 檔案。

首先，使用以下指令安裝 `fontTools`：

```bash
pip install fonttools
```

:::info
如果你對 `fontTools` 有興趣，可以參考他們的專案：[**fontTools-github**](https://github.com/fonttools/fonttools)
:::

安裝完成後，就能執行程式了。

## 實作重點

在開始使用程式碼之前，先看一下我們的實作重點：

1. **移除字串中不需要的控制字元 (`remove_control_characters`)**

   這個函數用來清理字串中的控制字元或不可見字元，這些字元可能會干擾後續的處理，因此我們需要先將他們移除。此外，有些字元是多個 Unicode 組合而成，我們有引入 `unicodedata` 套件，進行字串的標準化處理，將這些組合字元轉換成單一字元，確保字串的一致性。

   :::tip
   如果你不需要這個功能，可以將 `normalize` 設為 `False`。
   :::

2. **提取字型資訊 (`extract_font_info`)**

   從字型中取得各種資訊，並進行彙整，輸出成易於閱讀的結構化字典，其中含有非常多的鍵值，說明如下：

   - `fileName`: 字型檔在系統中的路徑。
   - `tables`: 列出字型檔案中可用的所有 Tables。
   - `nameTable`: 以 `nameID` 為索引的原始 name table。
   - `nameTableReadable`: 將常見 `nameID`（例如字型家族、版本）映射成更易讀的 key。
   - `cmapTable`: 對應各種編碼（platformID、platEncID）與 glyph 名稱的映射關係。
   - `headTable`: 字型的基本參數資訊，例如 `unitsPerEm`、`xMin`、`yMin` 等。
   - `hheaTable`: 水平排版資訊，包括上緣 (ascent)、下緣 (descent) 與行距 (lineGap)。
   - `OS2Table`: 字重 (usWeightClass)、字型寬度 (usWidthClass) 以及嵌入限制 (fsType) 等。
   - `postTable`: PostScript 相關資訊，如是否為等寬字型 (isFixedPitch)、字型傾斜角度 (italicAngle)。
   - `layoutMetrics`: 整合了多個表格後的排版度量資訊（包括 boundingBox、unitsPerEm、行距等）。
   - `summary`: 字型概要，幫你快速查閱字型家族 (fontFamily)、子家族 (fontSubfamily)、版本資訊 (version)、字重 (weightClass) 及是否為斜體 (isItalic)。

---

其中比較重要的是 `cmapTable`，這個表格對應了各種編碼與字元的對應關係，在我們的實作中，我們將這個表格進行了更進一步的處理，將編碼轉換成可讀的平臺名稱和編碼名稱：

1.  **平臺名稱 (`platform_name`)**

    `platformID` 是用來表示字型所支持的平臺，常見的平臺代碼包括：

    - `0`: Unicode（通用的字型標準）
    - `1`: Macintosh（Mac 系統專用字型）
    - `3`: Windows（Windows 系統專用字型）

    在程式中，這些代碼會透過字典轉換成對應的文字描述：

    ```python
    platform_name = {
        0: 'Unicode',
        1: 'Macintosh',
        3: 'Windows'
    }.get(cmap.platformID, f"Platform {cmap.platformID}")
    ```

    這段程式碼會先檢查 `platformID` 是否在字典內，如果找到對應值，就返回名稱（如 `'Unicode'`）；若找不到，則直接返回 `Platform {cmap.platformID}`，用以處理不在字典內的自訂平臺代碼。

2.  **編碼名稱 (`encoding_name`)**

    字型中的編碼方式則由 `platformID` 和 `platEncID` 共同決定，常見的組合及其含義如下：

    - `(0, 0)`: Unicode 1.0
    - `(0, 3)`: Unicode 2.0+
    - `(0, 4)`: Unicode 2.0+ with BMP（基本多文種平面）
    - `(1, 0)`: Mac Roman（Macintosh 的羅馬字母編碼）
    - `(3, 1)`: Windows Unicode BMP（Windows 的基本多文種平面編碼）
    - `(3, 10)`: Windows Unicode Full（Windows 的完整 Unicode 編碼）

    程式中，這些組合被存放在一個嵌套的字典中，並透過 `(platformID, platEncID)` 的元組進行查找：

    ```python
    encoding_name = {
        (0, 0): 'Unicode 1.0',
        (0, 3): 'Unicode 2.0+',
        (0, 4): 'Unicode 2.0+ with BMP',
        (1, 0): 'Mac Roman',
        (3, 1): 'Windows Unicode BMP',
        (3, 10): 'Windows Unicode Full'
    }.get((cmap.platformID, cmap.platEncID), f"Encoding {cmap.platEncID}")
    ```

    如果找到對應的組合，程式會返回對應的文字描述；若無法匹配，則返回預設值 `Encoding {cmap.platEncID}`，用於處理未知的編碼。

## 程式碼

以下是完整程式，你可以將輸出資訊匯出成 JSON 儲存，以便後續做更進一步的分析或追蹤。

```python
import re
import unicodedata
from pathlib import Path
from typing import List, Union

from fontTools.ttLib import TTFont


def load_ttfont(font_path: Union[str, Path], **kwargs) -> TTFont:
    """Load a TrueType font file."""
    if isinstance(font_path, Path):
        font_path = str(font_path)
    return TTFont(font_path, **kwargs)


def remove_control_characters(text: str, normalize: bool = True) -> str:
    """
    Remove control characters and invisible formatting characters from a string.

    Args:
        text (str): The input string.
        normalize (bool): Whether to normalize the text to remove inconsistencies.

    Returns:
        str: The sanitized string with control and invisible characters removed.
    """
    # Remove basic control characters (C0 and C1 control codes)
    sanitized = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

    # Remove specific Unicode control and invisible formatting characters
    sanitized = re.sub(
        r'[\u200B-\u200F\u2028-\u202F\u2060-\u206F]', '', sanitized)

    # Remove directional formatting characters (optional, adjust if needed)
    sanitized = re.sub(r'[\u202A-\u202E]', '', sanitized)

    # Optionally, normalize the text to remove any leftover inconsistencies
    if normalize:
        sanitized = unicodedata.normalize('NFKC', sanitized)

    return sanitized


def extract_font_info(
    font_path: Union[str, Path],
    normalize: bool = True
) -> dict:
    """Extract detailed metadata and structural information from a font file.

    Args:
        font_path (Union[str, Path]): Path to the font file.

    Returns:
        dict: A dictionary containing font metadata and tables, including:

            - fileName (str): Path to the font file.
            - tables (list): List of available tables in the font.
            - nameTable (dict): Raw name table values, keyed by nameID.
            - nameTableReadable (dict): Readable name table with keys:
                * copyright (str): Copyright information.
                * fontFamily (str): Font family name.
                * fontSubfamily (str): Font subfamily name.
                * uniqueID (str): Unique identifier for the font.
                * fullName (str): Full font name.
                * version (str): Font version string.
                * postScriptName (str): PostScript name.
            - cmapTable (dict): Character-to-glyph mappings, keyed by encoding.
            - cmapTableIndex (list): List of encoding descriptions.
            - headTable (dict): Font header table with keys:
                * unitsPerEm (int): Units per em.
                * xMin (int): Minimum x-coordinate of the glyph bounding box.
                * yMin (int): Minimum y-coordinate of the glyph bounding box.
                * xMax (int): Maximum x-coordinate of the glyph bounding box.
                * yMax (int): Maximum y-coordinate of the glyph bounding box.
            - hheaTable (dict): Horizontal header table with keys:
                * ascent (int): Typographic ascent.
                * descent (int): Typographic descent.
                * lineGap (int): Line gap.
            - OS2Table (dict): OS/2 table with keys:
                * usWeightClass (int): Weight class.
                * usWidthClass (int): Width class.
                * fsType (int): Embedding restrictions.
            - postTable (dict): PostScript table with keys:
                * isFixedPitch (bool): Whether the font is monospaced.
                * italicAngle (float): Italic angle of the font.
            - layoutMetrics (dict): Font layout metrics with keys:
                * unitsPerEm (int): Units per em.
                * boundingBox (dict): Bounding box coordinates:
                    - xMin (int): Minimum x-coordinate.
                    - yMin (int): Minimum y-coordinate.
                    - xMax (int): Maximum x-coordinate.
                    - yMax (int): Maximum y-coordinate.
                * ascent (int): Typographic ascent.
                * descent (int): Typographic descent.
                * lineGap (int): Line gap.
            - summary (dict): High-level font summary with keys:
                * fontFamily (str): Font family name.
                * fontSubfamily (str): Font subfamily name.
                * version (str): Font version.
                * weightClass (int): Weight class.
                * isItalic (bool): Whether the font is italic.
    """

    if isinstance(font_path, Path):
        font_path = str(font_path)

    font = TTFont(font_path)
    font_info = {}

    # File name and available tables
    font_info['fileName'] = font_path
    font_info['tables'] = list(font.keys())

    # Parse name table
    name_table = {}
    for record in font['name'].names:
        try:
            raw_string = record.string.decode('utf-16-be').strip()
            clean_string = remove_control_characters(raw_string, normalize)
            name_table[record.nameID] = clean_string
        except UnicodeDecodeError:
            name_table[record.nameID] = remove_control_characters(
                record.string.decode(errors='ignore'), normalize)
    font_info['nameTable'] = name_table

    # Readable name table for common nameIDs
    name_table_readable = {
        'copyright': name_table.get(0, ''),
        'fontFamily': name_table.get(1, ''),
        'fontSubfamily': name_table.get(2, ''),
        'uniqueID': name_table.get(3, ''),
        'fullName': name_table.get(4, ''),
        'version': name_table.get(5, ''),
        'postScriptName': name_table.get(6, ''),
    }
    font_info['nameTableReadable'] = {
        k: remove_control_characters(v, normalize)
        for k, v in name_table_readable.items()
    }

    # Parse cmap table
    cmap_table = {}
    cmap_table_index = []

    for cmap in font['cmap'].tables:
        platform_name = {
            0: 'Unicode',
            1: 'Macintosh',
            3: 'Windows'
        }.get(cmap.platformID, f"Platform {cmap.platformID}")

        encoding_name = {
            (0, 0): 'Unicode 1.0',
            (0, 3): 'Unicode 2.0+',
            (0, 4): 'Unicode 2.0+ with BMP',
            (1, 0): 'Mac Roman',
            (3, 1): 'Windows Unicode BMP',
            (3, 10): 'Windows Unicode Full'
        }.get((cmap.platformID, cmap.platEncID), f"Encoding {cmap.platEncID}")

        cmap_entries = {}
        for codepoint, glyph_name in cmap.cmap.items():
            char = chr(codepoint)
            cmap_entries[remove_control_characters(char, normalize)] = \
                remove_control_characters(glyph_name, normalize)

        key = f"{platform_name}, {encoding_name}"
        cmap_table[key] = cmap_entries
        cmap_table_index.append(key)

    font_info['cmapTable'] = cmap_table
    font_info['cmapTableIndex'] = cmap_table_index

    # Parse head table
    head = font['head']
    head_table = {
        'unitsPerEm': head.unitsPerEm,
        'xMin': head.xMin,
        'yMin': head.yMin,
        'xMax': head.xMax,
        'yMax': head.yMax,
    }
    font_info['headTable'] = head_table

    # Parse hhea table
    hhea = font['hhea']
    hhea_table = {
        'ascent': hhea.ascent,
        'descent': hhea.descent,
        'lineGap': hhea.lineGap,
    }
    font_info['hheaTable'] = hhea_table

    # Parse OS/2 table
    os2 = font['OS/2']
    os2_table = {
        'usWeightClass': os2.usWeightClass,
        'usWidthClass': os2.usWidthClass,
        'fsType': os2.fsType,
    }
    font_info['OS2Table'] = os2_table

    # Parse post table
    post = font['post']
    post_table = {
        'isFixedPitch': post.isFixedPitch,
        'italicAngle': post.italicAngle,
    }
    font_info['postTable'] = post_table

    # Combine layout-related metrics
    font_info['layoutMetrics'] = {
        'unitsPerEm': head_table['unitsPerEm'],
        'boundingBox': {
            'xMin': head_table['xMin'],
            'yMin': head_table['yMin'],
            'xMax': head_table['xMax'],
            'yMax': head_table['yMax']
        },
        'ascent': hhea_table['ascent'],
        'descent': hhea_table['descent'],
        'lineGap': hhea_table['lineGap']
    }

    # Font summary
    font_info['summary'] = {
        'fontFamily': name_table_readable['fontFamily'],
        'fontSubfamily': name_table_readable['fontSubfamily'],
        'version': name_table_readable['version'],
        'weightClass': os2.usWeightClass,
        'isItalic': post_table['italicAngle'] != 0
    }

    return font_info
```

## 輸出結果示範

我們以 `OcrB-Regular.ttf` 這個字型檔案為例，先呼叫函數，然後輸出至 JSON 檔案：

```python
import json

font_infos = extract_font_info('OcrB-Regular.ttf')
json.dump(font_infos, open('OcrB-Regular-Info.json', 'w'),
          indent=2, ensure_ascii=False)
```

輸出結果如下：

```json
{
  "fileName": "/path/to/your/folder/OcrB-Regular.ttf",
  "tables": [
    "GlyphOrder",
    "head",
    "hhea",
    "maxp",
    "OS/2",
    "hmtx",
    "hdmx",
    "cmap",
    "fpgm",
    "prep",
    "cvt ",
    "loca",
    "glyf",
    "name",
    "post"
  ],
  "nameTable": {
    "0": "This is a copyrighted typeface program",
    "1": "OcrB",
    "2": "Regular",
    "3": "Altsys Fontographer 3.5  OcrB Regular",
    "4": "OcrB Regular",
    "5": "Altsys Fontographer 3.5  4/15/93",
    "6": "OcrB Regular"
  },
  "nameTableReadable": {
    "copyright": "This is a copyrighted typeface program",
    "fontFamily": "OcrB",
    "fontSubfamily": "Regular",
    "uniqueID": "Altsys Fontographer 3.5  OcrB Regular",
    "fullName": "OcrB Regular",
    "version": "Altsys Fontographer 3.5  4/15/93",
    "postScriptName": "OcrB Regular"
  },
  "cmapTable": {
    "Unicode, Unicode 1.0": {
      " ": "nonbreakingspace",
      "!": "exclam",
      "\"": "quotedbl",
      "#": "numbersign",
      "$": "dollar",
      "%": "percent",
      "&": "ampersand",
      "'": "quotesingle",
      "(": "parenleft",
      ")": "parenright",
      "*": "asterisk",
      "+": "plus",
      ",": "comma",
      "-": "hyphen",
      ".": "period",
      "/": "slash",
      "0": "zero",
      "1": "one",
      "2": "two",
      "3": "three",
      "4": "four",
      "5": "five",
      "6": "six",
      "7": "seven",
      "8": "eight",
      "9": "nine",
      ":": "colon",
      ";": "semicolon",
      "<": "less",
      "=": "equal",
      ">": "greater",
      "?": "question",
      "@": "at",
      "A": "A",
      "B": "B",
      "C": "C",
      "D": "D",
      "E": "E",
      "F": "F",
      "G": "G",
      "H": "H",
      "I": "I",
      "J": "J",
      "K": "K",
      "L": "L",
      "M": "M",
      "N": "N",
      "O": "O",
      "P": "P",
      "Q": "Q",
      "R": "R",
      "S": "S",
      "T": "T",
      "U": "U",
      "V": "V",
      "W": "W",
      "X": "X",
      "Y": "Y",
      "Z": "Z",
      "[": "bracketleft",
      "\\": "backslash",
      "]": "bracketright",
      "^": "asciicircum",
      "_": "underscore",
      "`": "grave",
      "a": "a",
      "b": "b",
      "c": "c",
      "d": "d",
      "e": "e",
      "f": "f",
      "g": "g",
      "h": "h",
      "i": "i",
      "j": "j",
      "k": "k",
      "l": "l",
      "m": "m",
      "n": "n",
      "o": "o",
      "p": "p",
      "q": "q",
      "r": "r",
      "s": "zcaron",
      "t": "t",
      "u": "u",
      "v": "v",
      "w": "w",
      "x": "x",
      "y": "y",
      "z": "z",
      "{": "braceleft",
      "|": "bar",
      "}": "braceright",
      "¡": "exclamdown",
      "¢": "cent",
      "£": "sterling",
      "¤": "currency",
      "¥": "yen",
      "§": "section",
      " ̈": "dieresis",
      "«": "guillemotleft",
      "­": "hyphen",
      " ̄": "macron",
      " ́": "acute",
      "·": "periodcentered",
      " ̧": "cedilla",
      "»": "guillemotright",
      "¿": "questiondown",
      "À": "Agrave",
      "Á": "Aacute",
      "Â": "Acircumflex",
      "Ã": "Atilde",
      "Ä": "Adieresis",
      "Å": "Aring",
      "Æ": "AE",
      "Ç": "Ccedilla",
      "È": "Egrave",
      "É": "Eacute",
      "Ê": "Ecircumflex",
      "Ë": "Edieresis",
      "Ì": "Igrave",
      "Í": "Iacute",
      "Î": "Icircumflex",
      "Ï": "Idieresis",
      "Ð": "Eth",
      "Ñ": "Ntilde",
      "Ò": "Ograve",
      "Ó": "Oacute",
      "Ô": "Ocircumflex",
      "Õ": "Otilde",
      "Ö": "Odieresis",
      "×": ".null",
      "Ø": "Oslash",
      "Ù": "Ugrave",
      "Ú": "Uacute",
      "Û": "Ucircumflex",
      "Ü": "Udieresis",
      "Ý": "Yacute#1",
      "Þ": "Thorn",
      "ß": "germandbls",
      "à": "agrave",
      "á": "aacute",
      "â": "acircumflex",
      "ã": "atilde",
      "ä": "adieresis",
      "å": "aring",
      "æ": "ae",
      "ç": "ccedilla",
      "è": "egrave",
      "é": "eacute",
      "ê": "ecircumflex",
      "ë": "edieresis",
      "ì": "igrave",
      "í": "iacute",
      "î": "icircumflex",
      "ï": "idieresis",
      "ð": "Yacute",
      "ñ": "ntilde",
      "ò": "ograve",
      "ó": "oacute",
      "ô": "ocircumflex",
      "õ": "otilde",
      "ö": "odieresis",
      "ø": "oslash",
      "ù": "ugrave",
      "ú": "uacute",
      "û": "ucircumflex",
      "ü": "udieresis",
      "ý": "yacute",
      "þ": "thorn",
      "ÿ": "ydieresis",
      "ı": "dotlessi",
      "Ł": "Lslash",
      "ł": "lslash",
      "Œ": "OE",
      "œ": "oe",
      "Š": "Scaron",
      "š": "scaron",
      "Ÿ": "Ydieresis",
      "Ž": "Zcaron",
      "ʺ": "hungarumlaut",
      "ˆ": "circumflex",
      "ˇ": "caron",
      "ˉ": "macron",
      " ̆": "breve",
      " ̇": "dotaccent",
      " ̊": "ring",
      " ̨": "ogonek",
      " ̃": "tilde",
      "–": "endash",
      "—": "emdash",
      "‘": "quoteleft",
      "‚": "quotesinglbase",
      "“": "quotedblleft",
      "”": "quotedblright",
      "„": "quotedblbase",
      "†": "dagger",
      "‡": "daggerdbl",
      "...": "ellipsis",
      "‹": "guilsinglleft",
      "›": "guilsinglright",
      "−": "minus",
      "∙": "periodcentered"
    },
    "Macintosh, Mac Roman": {
      "": "udieresis",
      " ": "dagger",
      "!": "exclam",
      "\"": "quotedbl",
      "#": "numbersign",
      "$": "dollar",
      "%": "percent",
      "&": "ampersand",
      "'": "quotesingle",
      "(": "parenleft",
      ")": "parenright",
      "*": "asterisk",
      "+": "plus",
      ",": "comma",
      "-": "hyphen",
      ".": "period",
      "/": "slash",
      "0": "zero",
      "1": "one",
      "2": "two",
      "3": "three",
      "4": "four",
      "5": "five",
      "6": "six",
      "7": "seven",
      "8": "eight",
      "9": "nine",
      ":": "colon",
      ";": "semicolon",
      "<": "less",
      "=": "equal",
      ">": "greater",
      "?": "question",
      "@": "at",
      "A": "A",
      "B": "B",
      "C": "C",
      "D": "D",
      "E": "E",
      "F": "F",
      "G": "G",
      "H": "H",
      "I": "I",
      "J": "J",
      "K": "K",
      "L": "L",
      "M": "M",
      "N": "N",
      "O": "O",
      "P": "P",
      "Q": "Q",
      "R": "R",
      "S": "S",
      "T": "T",
      "U": "U",
      "V": "V",
      "W": "W",
      "X": "X",
      "Y": "Y",
      "Z": "Z",
      "[": "bracketleft",
      "\\": "backslash",
      "]": "bracketright",
      "^": "asciicircum",
      "_": "underscore",
      "`": "grave",
      "a": "a",
      "b": "b",
      "c": "c",
      "d": "d",
      "e": "e",
      "f": "f",
      "g": "g",
      "h": "h",
      "i": "i",
      "j": "j",
      "k": "k",
      "l": "l",
      "m": "m",
      "n": "n",
      "o": "o",
      "p": "p",
      "q": "q",
      "r": "r",
      "s": "s",
      "t": "t",
      "u": "u",
      "v": "v",
      "w": "w",
      "x": "x",
      "y": "y",
      "z": "z",
      "{": "braceleft",
      "|": "bar",
      "}": "braceright",
      "¢": "cent",
      "£": "sterling",
      "¤": "section",
      "§": "germandbls",
      "«": "acute",
      "¬": "dieresis",
      "®": "AE",
      " ̄": "Oslash",
      " ́": "yen",
      "3⁄4": "ae",
      "¿": "oslash",
      "À": "questiondown",
      "Á": "exclamdown",
      "Ç": "guillemotleft",
      "È": "guillemotright",
      "É": "ellipsis",
      "Ê": "nonbreakingspace",
      "Ë": "Agrave",
      "Ì": "Atilde",
      "Í": "Otilde",
      "Î": "OE",
      "Ï": "oe",
      "Ð": "endash",
      "Ñ": "emdash",
      "Ò": "quotedblleft",
      "Ó": "quotedblright",
      "Ô": "quoteleft",
      "Ø": "ydieresis",
      "Ù": "Ydieresis",
      "Û": "currency",
      "Ü": "guilsinglleft",
      "Ý": "guilsinglright",
      "à": "daggerdbl",
      "á": "periodcentered",
      "â": "quotesinglbase",
      "ã": "quotedblbase",
      "å": "Acircumflex",
      "æ": "Ecircumflex",
      "ç": "Aacute",
      "è": "Edieresis",
      "é": "Egrave",
      "ê": "Iacute",
      "ë": "Icircumflex",
      "ì": "Idieresis",
      "í": "Igrave",
      "î": "Oacute",
      "ï": "Ocircumflex",
      "ñ": "Ograve",
      "ò": "Uacute",
      "ó": "Ucircumflex",
      "ô": "Ugrave",
      "õ": "dotlessi",
      "ö": "circumflex",
      "÷": "tilde",
      "ø": "macron",
      "ù": "breve",
      "ú": "dotaccent",
      "û": "ring",
      "ü": "cedilla",
      "ý": "hungarumlaut",
      "þ": "ogonek",
      "ÿ": "caron"
    },
    "Windows, Windows Unicode BMP": {
      " ": "nonbreakingspace",
      "!": "exclam",
      "\"": "quotedbl",
      "#": "numbersign",
      "$": "dollar",
      "%": "percent",
      "&": "ampersand",
      "'": "quotesingle",
      "(": "parenleft",
      ")": "parenright",
      "*": "asterisk",
      "+": "plus",
      ",": "comma",
      "-": "hyphen",
      ".": "period",
      "/": "slash",
      "0": "zero",
      "1": "one",
      "2": "two",
      "3": "three",
      "4": "four",
      "5": "five",
      "6": "six",
      "7": "seven",
      "8": "eight",
      "9": "nine",
      ":": "colon",
      ";": "semicolon",
      "<": "less",
      "=": "equal",
      ">": "greater",
      "?": "question",
      "@": "at",
      "A": "A",
      "B": "B",
      "C": "C",
      "D": "D",
      "E": "E",
      "F": "F",
      "G": "G",
      "H": "H",
      "I": "I",
      "J": "J",
      "K": "K",
      "L": "L",
      "M": "M",
      "N": "N",
      "O": "O",
      "P": "P",
      "Q": "Q",
      "R": "R",
      "S": "S",
      "T": "T",
      "U": "U",
      "V": "V",
      "W": "W",
      "X": "X",
      "Y": "Y",
      "Z": "Z",
      "[": "bracketleft",
      "\\": "backslash",
      "]": "bracketright",
      "^": "asciicircum",
      "_": "underscore",
      "`": "grave",
      "a": "a",
      "b": "b",
      "c": "c",
      "d": "d",
      "e": "e",
      "f": "f",
      "g": "g",
      "h": "h",
      "i": "i",
      "j": "j",
      "k": "k",
      "l": "l",
      "m": "m",
      "n": "n",
      "o": "o",
      "p": "p",
      "q": "q",
      "r": "r",
      "s": "zcaron",
      "t": "t",
      "u": "u",
      "v": "v",
      "w": "w",
      "x": "x",
      "y": "y",
      "z": "z",
      "{": "braceleft",
      "|": "bar",
      "}": "braceright",
      "¡": "exclamdown",
      "¢": "cent",
      "£": "sterling",
      "¤": "currency",
      "¥": "yen",
      "§": "section",
      " ̈": "dieresis",
      "«": "guillemotleft",
      "­": "hyphen",
      " ̄": "macron",
      " ́": "acute",
      "·": "periodcentered",
      " ̧": "cedilla",
      "»": "guillemotright",
      "¿": "questiondown",
      "À": "Agrave",
      "Á": "Aacute",
      "Â": "Acircumflex",
      "Ã": "Atilde",
      "Ä": "Adieresis",
      "Å": "Aring",
      "Æ": "AE",
      "Ç": "Ccedilla",
      "È": "Egrave",
      "É": "Eacute",
      "Ê": "Ecircumflex",
      "Ë": "Edieresis",
      "Ì": "Igrave",
      "Í": "Iacute",
      "Î": "Icircumflex",
      "Ï": "Idieresis",
      "Ð": "Eth",
      "Ñ": "Ntilde",
      "Ò": "Ograve",
      "Ó": "Oacute",
      "Ô": "Ocircumflex",
      "Õ": "Otilde",
      "Ö": "Odieresis",
      "×": ".null",
      "Ø": "Oslash",
      "Ù": "Ugrave",
      "Ú": "Uacute",
      "Û": "Ucircumflex",
      "Ü": "Udieresis",
      "Ý": "Yacute#1",
      "Þ": "Thorn",
      "ß": "germandbls",
      "à": "agrave",
      "á": "aacute",
      "â": "acircumflex",
      "ã": "atilde",
      "ä": "adieresis",
      "å": "aring",
      "æ": "ae",
      "ç": "ccedilla",
      "è": "egrave",
      "é": "eacute",
      "ê": "ecircumflex",
      "ë": "edieresis",
      "ì": "igrave",
      "í": "iacute",
      "î": "icircumflex",
      "ï": "idieresis",
      "ð": "Yacute",
      "ñ": "ntilde",
      "ò": "ograve",
      "ó": "oacute",
      "ô": "ocircumflex",
      "õ": "otilde",
      "ö": "odieresis",
      "ø": "oslash",
      "ù": "ugrave",
      "ú": "uacute",
      "û": "ucircumflex",
      "ü": "udieresis",
      "ý": "yacute",
      "þ": "thorn",
      "ÿ": "ydieresis",
      "ı": "dotlessi",
      "Ł": "Lslash",
      "ł": "lslash",
      "Œ": "OE",
      "œ": "oe",
      "Š": "Scaron",
      "š": "scaron",
      "Ÿ": "Ydieresis",
      "Ž": "Zcaron",
      "ʺ": "hungarumlaut",
      "ˆ": "circumflex",
      "ˇ": "caron",
      "ˉ": "macron",
      " ̆": "breve",
      " ̇": "dotaccent",
      " ̊": "ring",
      " ̨": "ogonek",
      " ̃": "tilde",
      "–": "endash",
      "—": "emdash",
      "‘": "quoteleft",
      "‚": "quotesinglbase",
      "“": "quotedblleft",
      "”": "quotedblright",
      "„": "quotedblbase",
      "†": "dagger",
      "‡": "daggerdbl",
      "...": "ellipsis",
      "‹": "guilsinglleft",
      "›": "guilsinglright",
      "−": "minus",
      "∙": "periodcentered"
    }
  },
  "cmapTableIndex": [
    "Unicode, Unicode 1.0",
    "Macintosh, Mac Roman",
    "Windows, Windows Unicode BMP"
  ],
  "headTable": {
    "unitsPerEm": 1000,
    "xMin": -89,
    "yMin": -337,
    "xMax": 691,
    "yMax": 744
  },
  "hheaTable": {
    "ascent": 744,
    "descent": -337,
    "lineGap": 0
  },
  "OS2Table": {
    "usWeightClass": 400,
    "usWidthClass": 5,
    "fsType": 2
  },
  "postTable": {
    "isFixedPitch": 0,
    "italicAngle": 0.0
  },
  "layoutMetrics": {
    "unitsPerEm": 1000,
    "boundingBox": {
      "xMin": -89,
      "yMin": -337,
      "xMax": 691,
      "yMax": 744
    },
    "ascent": 744,
    "descent": -337,
    "lineGap": 0
  },
  "summary": {
    "fontFamily": "OcrB",
    "fontSubfamily": "Regular",
    "version": "Altsys Fontographer 3.5  4/15/93",
    "weightClass": 400,
    "isItalic": false
  }
}
```

## 參考資料

- [**Character to Glyph Mapping Table**](https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6cmap.html?utm_source=chatgpt.com)
- [**cmap — Character to Glyph Index Mapping Table**](https://learn.microsoft.com/en-us/typography/opentype/spec/cmap?utm_source=chatgpt.com)
