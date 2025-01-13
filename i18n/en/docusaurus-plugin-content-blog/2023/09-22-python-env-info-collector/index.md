---
slug: python-env-info-collector
title: Recording Model Training Environment Issues
authors: Zephyr
tags: [python, training-log]
image: /en/img/2023/0922.webp
description: A custom logging tool.
---

When a model's training goes awry, we always want to know the reasons behind it. At such times, it's essential to inspect the environment information of the training host, such as Python version, PyTorch version, CUDA version, GPU information, CPU information, RAM information, disk information, IP address, and more.

<!-- truncate -->

In this article, we share a custom Python tool we've crafted to swiftly review this information. While it's not exhaustive, it should suffice for basic troubleshooting needs.

Typically, we record environment information in the training host's logs at the start of training.

## Installation

Let's start by installing the necessary packages:

```bash
pip install psutil requests
```

:::tip
The complete code is available on GitHub, and we've also included it at the end of this article.

- [**system_info.py**](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/system_info.py)
  :::

## `get_package_versions`

Assuming you have installed `capybara` and it's already in your project, you can test it with the following command:

```python
from capybara import get_package_versions

get_package_versions()
```

Upon execution, you'll receive the following result:

```json
{
  "PyTorch Version": "2.1.1+cu121",
  "PyTorch Lightning Version": "2.1.2",
  "TensorFlow Error": "No module named 'tensorflow'",
  "Keras Error": "No module named 'keras'",
  "NumPy Version": "1.24.4",
  "Pandas Version": "2.0.3",
  "Scikit-learn Version": "1.3.2",
  "OpenCV Version": "4.8.1"
}
```

- PyTorch Version: PyTorch version
- PyTorch Lightning Version: PyTorch Lightning version
- TensorFlow Error: TensorFlow version
- Keras Error: Keras version
- NumPy Version: NumPy version
- Pandas Version: Pandas version
- Scikit-learn Version: Scikit-learn version
- OpenCV Version: OpenCV version

## `get_gpu_cuda_versions`

Test program:

```python
from capybara import get_gpu_cuda_versions

get_gpu_cuda_versions()
```

Upon execution, you'll receive the following result:

```json
{
  "CUDA Version": "12.1",
  "NVIDIA Driver Version": "535.129.03"
}
```

- CUDA Version: CUDA version
- NVIDIA Driver Version: NVIDIA driver version

## `get_system_info`

Test program:

```python
from capybara import get_system_info

get_system_info()
```

Upon execution, you'll receive the following result:

```json
{
  "OS Version": "Linux-6.2.0-37-generic-x86_64-with-glibc2.34",
  "CPU Model": "13th Gen Intel(R) Core(TM) i9-13900K",
  "Physical CPU Cores": 24,
  "Logical CPU Cores (incl. hyper-threading)": 32,
  "Total RAM (GB)": 125.56,
  "Available RAM (GB)": 110.9,
  "Disk Total (GB)": 1832.21,
  "Disk Used (GB)": 188.94,
  "Disk Free (GB)": 1550.12,
  "GPU Info": "NVIDIA GeForce RTX 4090",
  "IPV4 Address": ["192.168.---.---"],
  "IPV4 Address (External)": "---.---.---.---",
  "MAC Address": ["--.--.--.--.--.--"]
}
```

- OS Version: Operating system version
- CPU Model: CPU model
- Physical CPU Cores: Number of physical CPU cores
- Logical CPU Cores (incl. hyper-threading): Number of logical CPU cores (including hyper-threading)
- Total RAM (GB): Total RAM capacity (GB)
- Available RAM (GB): Available RAM capacity (GB)
- Disk Total (GB): Total disk capacity (GB)
- Disk Used (GB): Used disk capacity (GB)
- Disk Free (GB): Free disk capacity (GB)
- GPU Info: GPU information
- IPV4 Address: Internal IPV4 address
- IPV4 Address (External): External IPV4 address
- MAC Address: MAC address

## Considerations and Alternatives

Since we're writing this function on Ubuntu, there might be unexpected developments on other operating systems.

Here are a few points to note:

- Due to OS limitations, some functions may not run on all platforms. For example, `get_cpu_info` won't display the complete CPU model on Windows. In such cases, consider using other tools or manually obtaining this information.
- If you're on Windows, you can't directly use `nvidia-smi` to get GPU information. Make sure you have NVIDIA drivers and related tools installed and execute it in a command prompt window.
- External IP address is fetched from `https://httpbin.org/ip`, so ensure an active internet connection.

## Code

```python showLineNumbers
import platform
import socket
import subprocess

import psutil
import requests


def get_package_versions():
    """
    Get versions of commonly used packages in deep learning and data science.

    Returns:
        dict: Dictionary containing versions of installed packages.
    """
    versions_info = {}

    # PyTorch
    try:
        import torch
        versions_info["PyTorch Version"] = torch.__version__
    except Exception as e:
        versions_info["PyTorch Error"] = str(e)

    # PyTorch Lightning
    try:
        import pytorch_lightning as pl
        versions_info["PyTorch Lightning Version"] = pl.__version__
    except Exception as e:
        versions_info["PyTorch Lightning Error"] = str(e)

    # TensorFlow
    try:
        import tensorflow as tf
        versions_info["TensorFlow Version"] = tf.__version__
    except Exception as e:
        versions_info["TensorFlow Error"] = str(e)

    # Keras
    try:
        import keras
        versions_info["Keras Version"] = keras.__version__
    except Exception as e:
        versions_info["Keras Error"] = str(e)

    # NumPy
    try:
        import numpy as np
        versions_info["NumPy Version"] = np.__version__
    except Exception as e:
        versions_info["NumPy Error"] = str(e)

    # Pandas
    try:
        import pandas as pd
        versions_info["Pandas Version"] = pd.__version__
    except Exception as e:
        versions_info["Pandas Error"] = str(e)

    # Scikit-learn
    try:
        import sklearn
        versions_info["Scikit-learn Version"] = sklearn.__version__
    except Exception as e:
        versions_info["Scikit-learn Error"] = str(e)

    # OpenCV
    try:
        import cv2
        versions_info["OpenCV Version"] = cv2.__version__
    except Exception as e:
        versions_info["OpenCV Error"] = str(e)

    # ... and so on for any other packages you"re interested in

    return versions_info


def get_gpu_cuda_versions():
    """
    Get GPU and CUDA versions using popular Python libraries.

    Returns:
        dict: Dictionary containing CUDA and GPU driver versions.
    """

    cuda_version = None

    # Attempt to retrieve CUDA version using PyTorch
    try:
        import torch
        cuda_version = torch.version.cuda
    except ImportError:
        pass

    # If not retrieved via PyTorch, try using TensorFlow
    if not cuda_version:
        try:
            import tensorflow as tf
            cuda_version = tf.version.COMPILER_VERSION
        except ImportError:
            pass

    # If still not retrieved, try using CuPy
    if not cuda_version:
        try:
            import cupy
            cuda_version = cupy.cuda.runtime.runtimeGetVersion()
        except ImportError:
            cuda_version = "Error: None of PyTorch, TensorFlow, or CuPy are installed."

    # Try to get Nvidia driver version using nvidia-smi command
    try:
        smi_output = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits"
        ]).decode("utf-8").strip()
        nvidia_driver_version = smi_output.split("\n")[0]
    except Exception as e:
        nvidia_driver_version = f"Error getting NVIDIA driver version: {e}"

    return {
        "CUDA Version": cuda_version,
        "NVIDIA Driver Version": nvidia_driver_version
    }


def get_cpu_info():
    """
    Retrieve the CPU model name based on the platform.

    Returns:
        str: CPU model name or "N/A" if not found.
    """
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        # For macOS
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command, shell=True).strip().decode()
    elif platform.system() == "Linux":
        # For Linux
        command = "cat /proc/cpuinfo | grep "model name" | uniq"
        return subprocess.check_output(command, shell=True).strip().decode().split(":")[1].strip()
    else:
        return "N/A"


def get_external_ip():
    try:
        response = requests.get("https://httpbin.org/ip")
        return response.json()["origin"]
    except Exception as e:
        return f"Error obtaining IP: {e}"


def get_system_info():
    """
    Fetch system information like OS version, CPU info, RAM, Disk usage, etc.

    Returns:
        dict: Dictionary containing system information.
    """
    info = {
        "OS Version": platform.platform(),
        "CPU Model": get_cpu_info(),
        "Physical CPU Cores": psutil.cpu_count(logical=False),
        "Logical CPU Cores (incl. hyper-threading)": psutil.cpu_count(logical=True),
        "Total RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "Available RAM (GB)": round(psutil.virtual_memory().available / (1024 ** 3), 2),
        "Disk Total (GB)": round(psutil.disk_usage("/").total / (1024 ** 3), 2),
        "Disk Used (GB)": round(psutil.disk_usage("/").used / (1024 ** 3), 2),
        "Disk Free (GB)": round(psutil.disk_usage("/").free / (1024 ** 3), 2)
    }

    # Try to fetch GPU information using nvidia-smi command
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"]
        ).decode("utf-8").strip()
        info["GPU Info"] = gpu_info
    except Exception:
        info["GPU Info"] = "N/A or Error"

    # Get network information
    addrs = psutil.net_if_addrs()
    info["IPV4 Address"] = [
        addr.address for addr in addrs.get("enp5s0", []) if addr.family == socket.AF_INET
    ]

    info["IPV4 Address (External)"] = get_external_ip()

    # Determine platform and choose correct address family for MAC
    if hasattr(socket, "AF_LINK"):
        AF_LINK = socket.AF_LINK
    elif hasattr(psutil, "AF_LINK"):
        AF_LINK = psutil.AF_LINK
    else:
        raise Exception(
            "Cannot determine the correct AF_LINK value for this platform.")

    info["MAC Address"] = [
        addr.address for addr in addrs.get("enp5s0", []) if addr.family == AF_LINK
    ]

    return info
```
