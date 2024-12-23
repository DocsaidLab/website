# SystemInfo

This is a tool to retrieve system information, including CPU, memory, disk, network, and more.

## get_package_versions

> [get_package_versions() -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/system_info.py#L14)

- **Description**: Retrieves the versions of common deep learning and data analysis packages, including PyTorch, PyTorch Lightning, TensorFlow, Keras, NumPy, Pandas, Scikit-learn, OpenCV, and more.

- **Returns**

  - **dict**: A dictionary containing version information of installed packages.

- **Example**

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

- **Description**: Retrieves GPU and CUDA version information. It attempts to get the CUDA version using packages such as PyTorch, TensorFlow, and CuPy, and uses the `nvidia-smi` command to fetch the Nvidia driver version.

- **Returns**

  - **dict**: A dictionary containing CUDA and GPU driver version information.

- **Example**

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

- **Description**: Retrieves the CPU model name based on the platform.

- **Returns**

  - **str**: The CPU model name. Returns "N/A" if not found.

- **Example**

  ```python
  import capybara as cb

  cpu_info = cb.get_cpu_info()
  print(cpu_info)
  # cpu_info = 'Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz'
  ```

## get_system_info

> [get_system_info() -> dict](https://github.com/DocsaidLab/Capybara/blob/975d62fba4f76db59e715c220f7a2af5ad8d050e/capybara/utils/system_info.py#L155)

- **Description**: Retrieves system information, including operating system version, CPU info, memory, disk usage, etc.

- **Returns**

  - **dict**: A dictionary containing system information.

- **Example**

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
