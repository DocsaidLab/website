---
sidebar_position: 8
---

# Resources

Text synthesis tools are primarily used for automatically generating image datasets, especially in scenarios where large amounts of annotated data are required to train deep learning models. These tools enhance model adaptability to different environments, fonts, colors, and backgrounds by embedding synthetic text in images to simulate real-world occurrences of text.

Below are some well-known text synthesis tools:

## SynthText

- [**SynthText**](https://github.com/ankush-me/SynthText)

SynthText is an open-source project initiated by Ankush Gupta, Andrea Vedaldi, and Andrew Zisserman, and presented at CVPR 2016 (Conference on Computer Vision and Pattern Recognition). The project aims to generate synthetic text images for studies on text localization in natural images. These synthetic images are primarily used for training machine learning models, especially in the fields of optical character recognition and text detection.

SynthText utilizes natural scene images as backgrounds and synthesizes seemingly natural text within them. This process requires the depth and segmentation information of the background images. The project's main dependencies include pygame, opencv, PIL, numpy, matplotlib, h5py, and scipy.

## SynthText3D

- [**SynthText3D**](https://github.com/MhLiao/SynthText3D)

SynthText3D, based on Unreal Engine 4.16 and the UnrealCV plugin, simulates the appearance of real-world text by embedding it within virtual environments. The project uses custom virtual cameras and text rendering techniques to capture the interaction between text and its surrounding environment.

The project offers a dataset containing 10K images synthesized from 30 different scenes. The data synthesized using SynthText3D demonstrated good performance in benchmark tests on ICDAR 2015, ICDAR 2013, and MLT competitions.

## UnrealText

- [**UnrealText**](https://github.com/Jyouhou/UnrealText/)

UnrealText is an innovative project that uses a 3D graphics engine to synthesize scene text images. This project closely integrates relevant academic papers, specifically detailed in the paper ["**UnrealText: Synthesizing Realistic Scene Text Images from the Unreal World**"](https://arxiv.org/abs/2003.10608).

UnrealText utilizes Unreal Engine, a powerful game development tool, to create highly realistic text scenes. The project not only provides source code and instructions but also includes a demo UE project and 30 precompiled scene executables. Additionally, large-scale Latin/English and multilingual synthetic scene text datasets have been released.

## SynthTIGER

- [**SynthTIGER**](https://github.com/clovaai/synthtiger)

SynthTIGER (Synthetic Text Image Generator) is an open-source project developed by NAVER Corp., designed to generate synthetic text images to support the training of optical character recognition (OCR) models. The tool was specifically designed to enhance the performance of text recognition models and was presented at the International Conference on Document Analysis and Recognition (ICDAR) in 2021.

SynthTIGER supports the generation of multilingual and diversified text styles, producing rich and varied training data. It includes customization options for fonts, color mappings, and templates, allowing users to adjust the generation process according to their specific needs. It also offers large datasets split into multiple small files for easy download and use.

## TextRenderer

- [**text_renderer**](https://github.com/Sanster/text_renderer)

Text Renderer is an open-source project developed by the oh-my-ocr organization for generating line images of text for training deep learning OCR models. The project focuses on modular design, allowing users to easily add different components such as corpora, effects, and layouts.

Text Renderer integrates imgaug to support image augmentation, rendering multiple corpora with various effects, and layouts handle the arrangement of multiple corpora. Additionally, Text Renderer supports generating vertical text and can produce lmdb datasets compatible with PaddleOCR.

## TextRecognitionDataGenerator

- [**TextRecognitionDataGenerator**](https://github.com/Belval/TextRecognitionDataGenerator)

TextRecognitionDataGenerator (TRDG) is an open-source project developed by Belval, specifically for generating text image samples for training optical character recognition (OCR) software. This project supports multilingual text generation, including non-Latin scripts, and can generate images through different fonts, backgrounds, and text effects such as tilting, twisting, and blurring. TRDG is known for its modularity and ease of expansion, making it an essential tool for OCR model developers and researchers.

TRDG can be used via a command-line interface (CLI) and also offers a Python module interface, allowing it to be directly integrated into training pipelines. Users can generate images with random text or specify specific text content through simple commands. Additionally, TRDG supports generating vertical text and character-level masks, increasing the flexibility and applicability of generated text images.
