---
sidebar_position: 5
---

# SystemInfo

このツールはシステム情報を取得するためのものです。CPU、メモリ、ディスク、ネットワークなど、システムのさまざまな情報を取得できます。

## get_package_versions

> [get_package_versions() -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/system_info.py#L14)

- **説明**：よく使われる深層学習やデータ分析関連のライブラリのバージョン情報を取得します。PyTorch、PyTorch Lightning、TensorFlow、Keras、NumPy、Pandas、Scikit-learn、OpenCV などのライブラリのバージョン情報を含みます。

- **返り値**

  - **dict**：インストールされているライブラリのバージョン情報を含む辞書。

- **例**

  ```python
  import docsaidkit as D

  versions_info = D.get_package_versions()
  print(versions_info)
  # versions_info = {
  #     'PyTorch Version': '1.9.0',
  #     'PyTorch Lightning Version': '1.3.8',
  #     'TensorFlow Version': '2.5.0',
  #     'Keras Version': '2.4.3',
  #     'NumPy Version': '1.19.5',
  #     'Pandas Version': '1.1.5',
  #     'Scikit-learn Version': '0.24.2',
  #     'OpenCV Version': '4.5.2'
  # }
  ```

## get_gpu_cuda_versions

> [get_gpu_cuda_versions() -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/system_info.py#L84)

- **説明**：GPU と CUDA のバージョン情報を取得します。PyTorch、TensorFlow、CuPy などのライブラリを使用して CUDA のバージョンを取得し、`nvidia-smi`コマンドを使用して Nvidia ドライバのバージョンを取得します。

- **返り値**

  - **dict**：CUDA および GPU ドライバのバージョン情報を含む辞書。

- **例**

  ```python
  import docsaidkit as D

  gpu_cuda_info = D.get_gpu_cuda_versions()
  print(gpu_cuda_info)
  # gpu_cuda_info = {
  #     'CUDA Version': '11.1',
  #     'NVIDIA Driver Version': '460.32.03'
  # }
  ```

## get_cpu_info

> [get_cpu_info() -> str](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/system_info.py#L134)

- **説明**：異なるプラットフォームで CPU のモデル名を取得します。

- **返り値**

  - **str**：CPU モデル名、見つからない場合は "N/A" を返します。

- **例**

  ```python
  import docsaidkit as D

  cpu_info = D.get_cpu_info()
  print(cpu_info)
  # cpu_info = 'Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz'
  ```

## get_system_info

> [get_system_info() -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/system_info.py#L163)

- **説明**：システムの情報を取得します。OS のバージョン、CPU 情報、メモリ、ディスク使用量などを含みます。

- **返り値**

  - **dict**：システム情報を含む辞書。

- **例**

  ```python
  import docsaidkit as D

  system_info = D.get_system_info()
  print(system_info)
  # system_info = {
  #     'OS Version': 'Linux-5.4.0-80-generic-x86_64-with-glibc2.29',
  #     'CPU Model': 'Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz',
  #     'Physical CPU Cores': 6,
  #     'Logical CPU Cores (incl. hyper-threading)': 12,
  #     'Total RAM (GB)': 15.52,
  #     'Available RAM (GB)': 7.52,
  #     'Disk Total (GB)': 476.94,
  #     'Disk Used (GB)': 18.94,
  #     'Disk Free (GB)': 458.0,
  #     'GPU Info': 'NVIDIA GeForce GTX 1660 Ti',
  #     'IPV4 Address': ['xxx.xxx.xxx.xxx'],
  #     'IPV4 Address (External)': 'xxx.xxx.xxx.xxx',
  #     'MAC Address': [
  #         'xx:xx:xx:xx:xx:xx'
  #     ]
  # }
  ```
