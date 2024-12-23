# SystemInfo

これはシステム情報を取得するためのツールです。CPU、メモリ、ディスク、ネットワークなどのシステム情報を取得するのに役立ちます。

## get_package_versions

> [get_package_versions() -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/system_info.py#L14)

- **説明**：PyTorch、PyTorch Lightning、TensorFlow、Keras、NumPy、Pandas、Scikit-learn、OpenCV などの一般的な深層学習およびデータ分析ライブラリのバージョン情報を取得します。

- **返り値**

  - **dict**：インストールされているパッケージのバージョン情報を含む辞書。

- **例**

  ```python
  import capybara as cb

  versions_info = cb.get_package_versions()
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

> [get_gpu_cuda_versions() -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/system_info.py#L84)

- **説明**：GPU と CUDA のバージョン情報を取得します。PyTorch、TensorFlow、CuPy などのライブラリを使用して CUDA バージョンを取得し、`nvidia-smi` コマンドで Nvidia ドライバーのバージョンを取得します。

- **返り値**

  - **dict**：CUDA と GPU ドライバーのバージョン情報を含む辞書。

- **例**

  ```python
  import capybara as cb

  gpu_cuda_info = cb.get_gpu_cuda_versions()
  print(gpu_cuda_info)
  # gpu_cuda_info = {
  #     'CUDA Version': '11.1',
  #     'NVIDIA Driver Version': '460.32.03'
  # }
  ```

## get_cpu_info

> [get_cpu_info() -> str](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/system_info.py#L134)

- **説明**：プラットフォームに応じて CPU のモデル名を取得します。

- **返り値**

  - **str**：CPU のモデル名。見つからない場合は "N/A" を返します。

- **例**

  ```python
  import capybara as cb

  cpu_info = cb.get_cpu_info()
  print(cpu_info)
  # cpu_info = 'Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz'
  ```

## get_system_info

> [get_system_info() -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/system_info.py#L155)

- **説明**：システム情報を取得します。これには、OS バージョン、CPU 情報、メモリ、ディスク使用量などが含まれます。

- **返り値**

  - **dict**：システム情報を含む辞書。

- **例**

  ```python
  import capybara as cb

  system_info = cb.get_system_info()
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
