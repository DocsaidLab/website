---
sidebar_position: 5
---

# SystemInfo

This tool is designed to fetch system information. It can help you retrieve information about your CPU, memory, disk, network, and more.

## get_package_versions

> [get_package_versions() -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/system_info.py#L14)

- **Description**: Fetches the versions of commonly used deep learning and data analysis packages. This includes version information for packages like PyTorch, PyTorch Lightning, TensorFlow, Keras, NumPy, Pandas, Scikit-learn, OpenCV, etc.

- **Returns**
    - **dict**: A dictionary containing the version information of installed packages.

- **Example**

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

- **Description**: Retrieves GPU and CUDA version information. Attempts to fetch CUDA versions using packages like PyTorch, TensorFlow, and CuPy, and uses the `nvidia-smi` command to get Nvidia driver versions.

- **Returns**
    - **dict**: A dictionary containing CUDA and GPU driver version information.

- **Example**

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

- **Description**: Retrieves the CPU model name based on the platform.

- **Returns**
    - **str**: CPU model name, or "N/A" if not found.

- **Example**

    ```python
    import docsaidkit as D

    cpu_info = D.get_cpu_info()
    print(cpu_info)
    # cpu_info = 'Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz'
    ```

## get_system_info

> [get_system_info() -> dict](https://github.com/DocsaidLab/DocsaidKit/blob/71170598902b6f8e89a969f1ce27ed4fd05b2ff2/docsaidkit/utils/system_info.py#L163)

- **Description**: Retrieves comprehensive system information, including OS version, CPU details, memory, disk usage, and more.

- **Returns**
    - **dict**: A dictionary containing detailed system information.

- **Example**

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
    #