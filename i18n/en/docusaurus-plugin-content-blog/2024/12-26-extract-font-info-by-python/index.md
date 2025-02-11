---
slug: extract-font-info-by-python
title: Extract Font File Information
authors: Z. Yuan
image: /en/img/2024/1226.webp
tags: [font-tools, Python]
description: Retrieve font file information using Python.
---

Although we frequently use various fonts, we often get stuck when it comes to retrieving font parameters.

We forget after using them, and then look them up again next time?

This makes us seem unprofessional, so we need to write a program to solve the problem.

<!-- truncate -->

## Install `fontTools`

We use `fontTools` to retrieve font-related information. It is a widely praised Python package for handling font files, allowing us to manipulate and parse various TTF and OTF files.

First, install `fontTools` using the following command:

```bash
pip install fonttools
```

:::info
If you're interested in `fontTools`, you can refer to their project on GitHub: [**fontTools-github**](https://github.com/fonttools/fonttools)
:::

Once installed, you can start running the program.

## Key Implementation Points

Before we start using the code, let’s review the key implementation points:

1. **Remove Control Characters (`remove_control_characters`)**

   This function is used to clean up control or invisible characters from a string, as these characters may interfere with subsequent processing. We need to remove them first. Additionally, some characters are made up of multiple Unicode combinations. We use the `unicodedata` module to normalize the string, converting these composite characters into a single character to ensure consistency.

   :::tip
   If you don’t need this functionality, you can set `normalize` to `False`.
   :::

2. **Extract Font Information (`extract_font_info`)**

   This function extracts various pieces of information from a font and organizes them into a structured, easy-to-read dictionary. It includes a variety of keys, described as follows:

   - `fileName`: The path of the font file on the system.
   - `tables`: Lists all available tables in the font file.
   - `nameTable`: The raw name table indexed by `nameID`.
   - `nameTableReadable`: Maps common `nameID` values (e.g., font family, version) to more readable keys.
   - `cmapTable`: A mapping of various encodings (platformID, platEncID) to glyph names.
   - `headTable`: Basic font parameter information such as `unitsPerEm`, `xMin`, `yMin`, etc.
   - `hheaTable`: Horizontal layout information, including ascent, descent, and line gap.
   - `OS2Table`: Information about weight (usWeightClass), width (usWidthClass), and embedding restrictions (fsType).
   - `postTable`: PostScript-related information, such as whether the font is monospaced (isFixedPitch) and the italic angle (italicAngle).
   - `layoutMetrics`: Typography metrics derived from multiple tables, including bounding box, unitsPerEm, line spacing, etc.
   - `summary`: A quick overview of the font, including font family, subfamily, version, weight class, and whether it is italic.

---

One of the most important parts is the `cmapTable`, which maps various encodings to corresponding characters. In our implementation, we further process this table to convert encodings into readable platform names and encoding names:

1. **Platform Name (`platform_name`)**

   `platformID` represents the platform supported by the font. Common platform codes include:

   - `0`: Unicode (general font standard)
   - `1`: Macintosh (Mac system-specific fonts)
   - `3`: Windows (Windows system-specific fonts)

   In the code, these codes are converted to their corresponding descriptions via a dictionary:

   ```python
   platform_name = {
       0: 'Unicode',
       1: 'Macintosh',
       3: 'Windows'
   }.get(cmap.platformID, f"Platform {cmap.platformID}")
   ```

   This code first checks if the `platformID` exists in the dictionary. If a corresponding value is found, it returns the name (e.g., `'Unicode'`). If not, it returns `Platform {cmap.platformID}` to handle custom platform codes not in the dictionary.

2. **Encoding Name (`encoding_name`)**

   The encoding method in the font is determined by both `platformID` and `platEncID`. Common combinations and their meanings are as follows:

   - `(0, 0)`: Unicode 1.0
   - `(0, 3)`: Unicode 2.0+
   - `(0, 4)`: Unicode 2.0+ with BMP (Basic Multilingual Plane)
   - `(1, 0)`: Mac Roman (Macintosh Roman alphabet encoding)
   - `(3, 1)`: Windows Unicode BMP (Windows Basic Multilingual Plane encoding)
   - `(3, 10)`: Windows Unicode Full (Windows Full Unicode encoding)

   In the code, these combinations are stored in a nested dictionary and looked up using the tuple `(platformID, platEncID)`:

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

   If a matching combination is found, the program returns the corresponding description. If no match is found, it defaults to `Encoding {cmap.platEncID}`, used to handle unknown encodings.

## Code

Here is the complete code. You can export the output information as JSON for further analysis or tracking.

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

## Example Output

Let’s take the font file `OcrB-Regular.ttf` as an example. We will call the function and then export the results to a JSON file:

```python
import json

font_infos = extract_font_info('OcrB-Regular.ttf')
json.dump(font_infos, open('OcrB-Regular-Info.json', 'w'),
          indent=2, ensure_ascii=False)
```

The output will be as follows:

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

## References

- [**Character to Glyph Mapping Table**](https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6cmap.html?utm_source=chatgpt.com)
- [**cmap — Character to Glyph Index Mapping Table**](https://learn.microsoft.com/en-us/typography/opentype/spec/cmap?utm_source=chatgpt.com)
