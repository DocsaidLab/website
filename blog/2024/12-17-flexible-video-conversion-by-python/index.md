---
slug: flexible-video-conversion-by-python
title: 批次影片轉檔
authors: Z. Yuan
image: /img/2024/1217.webp
tags: [Media-Processing, Python, ffmpeg]
description: 使用 Python 與 ffmpeg 建立指定格式的批次轉換流程。
---

收到一批 MOV 的影音檔，但是系統不支援讀取，要轉成 MP4 才行。

只好自己動手寫點程式。

<!-- truncate -->

## 設計草稿

轉檔工具毫無懸念的就是 ffmpeg，這個開源工具支援幾乎所有的影音格式，並且可以透過命令列參數來控制轉檔的方式。

本來我們想把這個功能直接寫到前端，讓其他人愛怎麼轉就怎麼轉...

但是在瀏覽器的調用上遇到困難，花了一小時無法排除問題，所以只好回到本機上處理。

在本機上轉檔相比之下容易很多，首先我們嘗試寫成 Bash，但是想一想又覺得 Python 好像比較好維護，所以最後選用 Python 搭配 ffmpeg 來完成這個功能。

## 什麼是 FFMPEG？

[ffmpeg](https://ffmpeg.org/) 是一款功能非常強大的開源多媒體處理工具，廣泛應用於影音格式的轉換、串流、剪輯與合併等多種工作任務。

它支援許多常見與非常用的影音格式，同時內建大量編解碼器，可透過單純的命令列操作，快速完成轉檔、剪接、字幕嵌入、重新取樣與壓縮，以及跨平台的影音串流。

由於 ffmpeg 是開源專案，並且在各種作業系統（Linux、macOS、Windows）都可輕鬆安裝與執行，因此成為媒體相關工作流程中不可或缺的工具。

在一般情境中，我們可以透過簡單的指令就完成最常見的轉檔需求，例如將一個 MOV 檔案轉成 MP4：

```bash
ffmpeg -i input.mov -c copy output.mp4
```

此時，`-i` 指定輸入檔案路徑，而 `-c copy` 則代表直接複製來源檔案的影音軌道（即不重新編碼），這能極大縮短處理時間並保持原始品質。若是想要對品質、編碼參數、輸出解析度、比特率、聲道數進行調整，ffmpeg 也提供相當彈性的指令列參數供使用者客製化設定。

總之，是個非常棒的工具，要學起來啊！

## 環境準備

我們基於 Ubuntu 作業系統進行開發，類似的 Linux 系統也可以使用。

1. **Python 環境**：確保已安裝 Python 3.x：

   ```bash
   python3 --version
   ```

2. **ffmpeg 安裝**：在 Ubuntu 環境中可透過下列指令安裝：

   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

   安裝完成後檢查版本：

   ```bash
   ffmpeg -version
   ```

3. **程式碼結構**：於專案資料夾中建立 `convert.py`（檔名可自行決定），將下方程式碼貼入。

## 程式碼範例

```python
import subprocess
import sys
from pathlib import Path

def convert_videos(input_dir: Path, src_format: str, dest_format: str):
    # 檢查目標資料夾是否存在
    if not input_dir.is_dir():
        print(f"錯誤: 目標資料夾 '{input_dir}' 不存在。")
        sys.exit(1)

    # 自動建立輸出資料夾
    output_dir = input_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 格式字首補齊
    if not src_format.startswith("."):
        src_format = f".{src_format}"
    if not dest_format.startswith("."):
        dest_format = f".{dest_format}"

    # 遍歷所有來源格式的檔案（大小寫不敏感）
    video_files = [f for f in input_dir.rglob("*") if f.suffix.casefold() == src_format.casefold()]

    if not video_files:
        print(f"未找到任何 {src_format} 檔案。")
        sys.exit(0)

    for file in video_files:
        output_file = output_dir / f"{file.stem}{dest_format}"
        print(f"正在轉換: '{file}' -> '{output_file}'")

        # 使用 ffmpeg 轉換檔案
        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", str(file),
                    "-c", "copy",
                    str(output_file)
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"轉換成功: '{output_file}'")
        except subprocess.CalledProcessError as e:
            print(f"轉換失敗: '{file}'")
            print(e.stderr.decode())

    print(f"所有檔案處理完成。輸出資料夾: '{output_dir}'")


if __name__ == "__main__":
    # 檢查使用者是否提供參數
    if len(sys.argv) != 4:
        print(f"用法: python {sys.argv[0]} <目標資料夾> <來源格式> <目標格式>")
        print(f"範例: python {sys.argv[0]} 'videos' 'MOV' 'mp4'")
        sys.exit(1)

    input_dir = Path(sys.argv[1]).resolve()
    src_format = sys.argv[2]
    dest_format = sys.argv[3]

    convert_videos(input_dir, src_format, dest_format)
```

## 使用方式

1. **準備來源檔案**：將欲轉換的檔案（如 MOV、AVI、MKV 等）放入指定的資料夾（如 `videos`）。

2. **執行轉換**：進入程式檔所在目錄後執行指令：

   ```bash
   python3 convert.py videos MOV mp4
   ```

   若您要將 AVI 檔案轉為 MKV，則可改成：

   ```bash
   python3 convert.py videos avi mkv
   ```

   執行後，程式會在 `videos/output` 資料夾中產生已轉換完成的檔案。

3. **檢查結果**：確認 `output` 資料夾有正確產生目標格式的影片，即完成任務。

## 進階運用

如果你想對檔案進行壓縮與品質調整，可在 ffmpeg 指令中加入特定參數，如：

```bash
ffmpeg -i input.avi -c:v libx264 -crf 20 output.mp4
```

並在程式中調整對 ffmpeg 的呼叫方式。

## 結語

就是這樣，我們在開發中順手寫了個簡單的功能，希望對你有所幫助。

可以開始轉檔了！
