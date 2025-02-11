---
slug: flexible-video-conversion-by-python
title: Batch Video Conversion
authors: Z. Yuan
image: /en/img/2024/1217.webp
tags: [Media-Processing, Python, ffmpeg]
description: Build a batch conversion process with Python and ffmpeg to convert to a specified format.
---

A batch of MOV video files were received, but the system does not support reading them. They need to be converted to MP4 for compatibility.

I had to write some code myself.

<!-- truncate -->

## Design Draft

The go-to tool for video conversion is undoubtedly ffmpeg. This open-source tool supports almost all audio and video formats and allows you to control the conversion process through command-line parameters.

Initially, we wanted to implement this functionality directly in the frontend, so others could convert files however they wanted...

However, we encountered difficulties when calling it from the browser and spent an hour trying to resolve the issue without success. So, we decided to handle the conversion locally.

Converting locally is much easier. At first, we considered writing a Bash script, but then we realized Python might be easier to maintain, so we chose to use Python together with ffmpeg to accomplish the task.

## What is FFMPEG?

[ffmpeg](https://ffmpeg.org/) is an extremely powerful open-source multimedia processing tool, widely used for tasks like format conversion, streaming, editing, and merging multimedia files.

It supports a wide range of common and uncommon audio and video formats and includes a large number of codecs. With simple command-line operations, ffmpeg allows you to quickly perform tasks like conversion, cutting, embedding subtitles, resampling, compression, and cross-platform streaming.

Since ffmpeg is an open-source project and can be easily installed and run on various operating systems (Linux, macOS, Windows), it has become an indispensable tool in media workflows.

In general, we can accomplish common conversion tasks with simple commands, such as converting a MOV file to MP4:

```bash
ffmpeg -i input.mov -c copy output.mp4
```

Here, `-i` specifies the input file path, and `-c copy` means copying the video and audio streams directly (without re-encoding), which significantly reduces processing time and maintains the original quality. If you want to adjust quality, encoding parameters, output resolution, bitrate, or channels, ffmpeg provides highly flexible command-line parameters for customization.

In short, it's an excellent tool that you should learn to use!

## Environment Setup

We are developing on an Ubuntu system, and similar Linux systems can be used as well.

1. **Python Environment**: Ensure that Python 3.x is installed:

   ```bash
   python3 --version
   ```

2. **Install ffmpeg**: On Ubuntu, you can install ffmpeg with the following commands:

   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

   After installation, check the version:

   ```bash
   ffmpeg -version
   ```

3. **Code Structure**: Create a `convert.py` file in your project folder (you can name it differently), and paste the following code into it.

## Example Code

```python
import subprocess
import sys
from pathlib import Path

def convert_videos(input_dir: Path, src_format: str, dest_format: str):
    # Check if the target directory exists
    if not input_dir.is_dir():
        print(f"Error: The target directory '{input_dir}' does not exist.")
        sys.exit(1)

    # Automatically create an output directory
    output_dir = input_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure format strings start with a period
    if not src_format.startswith("."):
        src_format = f".{src_format}"
    if not dest_format.startswith("."):
        dest_format = f".{dest_format}"

    # Iterate over all files with the source format (case insensitive)
    video_files = [f for f in input_dir.rglob("*") if f.suffix.casefold() == src_format.casefold()]

    if not video_files:
        print(f"No {src_format} files found.")
        sys.exit(0)

    for file in video_files:
        output_file = output_dir / f"{file.stem}{dest_format}"
        print(f"Converting: '{file}' -> '{output_file}'")

        # Use ffmpeg to convert the file
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
            print(f"Conversion successful: '{output_file}'")
        except subprocess.CalledProcessError as e:
            print(f"Conversion failed: '{file}'")
            print(e.stderr.decode())

    print(f"All files processed. Output directory: '{output_dir}'")


if __name__ == "__main__":
    # Check if the user has provided parameters
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <input directory> <source format> <destination format>")
        print(f"Example: python {sys.argv[0]} 'videos' 'MOV' 'mp4'")
        sys.exit(1)

    input_dir = Path(sys.argv[1]).resolve()
    src_format = sys.argv[2]
    dest_format = sys.argv[3]

    convert_videos(input_dir, src_format, dest_format)
```

## How to Use

1. **Prepare Source Files**: Place the files to be converted (e.g., MOV, AVI, MKV) into the specified directory (e.g., `videos`).

2. **Run the Conversion**: Navigate to the directory containing the script and run the following command:

   ```bash
   python3 convert.py videos MOV mp4
   ```

   If you want to convert AVI files to MKV, use:

   ```bash
   python3 convert.py videos avi mkv
   ```

   After running the command, the program will generate the converted files in the `videos/output` folder.

3. **Check the Results**: Ensure that the `output` folder contains the correctly converted files in the desired format.

## Advanced Use

If you want to compress and adjust the quality of the files, you can add specific parameters to the ffmpeg command, such as:

```bash
ffmpeg -i input.avi -c:v libx264 -crf 20 output.mp4
```

You can modify the script to adjust how ffmpeg is called for this purpose.

## Conclusion

That's it! We wrote a simple script during development, and I hope it's helpful to you.

You can now start converting your files!
