# SystemInfo

This module provides utilities to retrieve system information, including CPU, memory, disk, GPU/CUDA, and network.

Note: this module depends on `psutil`. Install `capybara-docsaid[system]` first:

```bash
pip install "capybara-docsaid[system]"
```

## get_package_versions

> [get_package_versions() -> dict](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/system_info.py)

- **Description**: Retrieves versions of common deep learning / data analysis packages, including PyTorch, PyTorch Lightning, TensorFlow, Keras, NumPy, Pandas, Scikit-learn, OpenCV, etc.

- **Returns**

  - **dict**: A dictionary of installed package versions.

- **Example**

  ```python
  from capybara.utils.system_info import get_package_versions

  versions_info = get_package_versions()
  print(versions_info)
  ```

## get_gpu_cuda_versions

> [get_gpu_cuda_versions() -> dict](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/system_info.py)

- **Description**: Retrieves GPU and CUDA version information. It tries to infer CUDA version via PyTorch/TensorFlow/CuPy and uses `nvidia-smi` to read NVIDIA driver version.

- **Returns**

  - **dict**: A dictionary containing CUDA and driver version information.

- **Example**

  ```python
  from capybara.utils.system_info import get_gpu_cuda_versions

  gpu_cuda_info = get_gpu_cuda_versions()
  print(gpu_cuda_info)
  ```

## get_cpu_info

> [get_cpu_info() -> str](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/system_info.py)

- **Description**: Returns CPU model name based on the current platform.

- **Returns**

  - **str**: CPU model name; returns `"N/A"` when not found.

- **Example**

  ```python
  from capybara.utils.system_info import get_cpu_info

  cpu_info = get_cpu_info()
  print(cpu_info)
  ```

## get_external_ip

> [get_external_ip() -> str](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/system_info.py)

- **Description**: Retrieves external IP via `https://httpbin.org/ip`. On request failure, returns a string like `"Error obtaining IP: ..."`.

- **Example**

  ```python
  from capybara.utils.system_info import get_external_ip

  ip = get_external_ip()
  print(ip)
  ```

## get_system_info

> [get_system_info() -> dict](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/system_info.py)

- **Description**: Retrieves system information, including OS version, CPU info, memory, disk usage, etc.

- **Notes**

  - NIC name is hard-coded to `enp5s0` to fetch `IPV4 Address` / `MAC Address`. If your environment uses a different NIC name, these fields may be empty lists.
  - GPU info is fetched via `nvidia-smi`; if unavailable, it returns `"N/A or Error"`.

- **Returns**

  - **dict**: A dictionary containing system information.

- **Example**

  ```python
  from capybara.utils.system_info import get_system_info

  system_info = get_system_info()
  print(system_info)
  ```
