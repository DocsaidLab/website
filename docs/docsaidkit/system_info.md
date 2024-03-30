---
sidebar_position: 5
---

# SystemInfo

這是一個用來獲取系統資訊的工具。它可以幫助你獲取 CPU、記憶體、磁盤、網路等系統資訊。

## get_package_versions

> [get_package_versions() -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/system_info.py#L14)

- **說明**：獲取常用深度學習和數據分析相關套件的版本。包含 PyTorch、PyTorch Lightning、TensorFlow、Keras、NumPy、Pandas、Scikit-learn、OpenCV 等套件的版本資訊。

- **傳回值**
    - **dict**：包含已安裝套件版本資訊的字典。

- **範例**

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

- **說明**：獲取 GPU 和 CUDA 版本資訊。嘗試使用 PyTorch、TensorFlow 和 CuPy 等套件來獲取 CUDA 版本，並使用 `nvidia-smi` 命令來獲取 Nvidia 驅動程式版本。

- **傳回值**
    - **dict**：包含 CUDA 和 GPU 驅動程式版本資訊的字典。

- **範例**

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

- **說明**：根據不同平台獲取 CPU 型號名稱。

- **傳回值**
    - **str**：CPU 型號名稱，如果找不到則返回 "N/A"。

- **範例**

    ```python
    import docsaidkit as D

    cpu_info = D.get_cpu_info()
    print(cpu_info)
    # cpu_info = 'Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz'
    ```

## get_system_info

> [get_system_info() -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/system_info.py#L163)

- **說明**：獲取系統資訊，包括作業系統版本、CPU 資訊、記憶體、磁盤使用量等。

- **傳回值**
    - **dict**：包含系統資訊的字典。

- **範例**

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
