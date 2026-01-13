# SystemInfo

このモジュールは CPU/メモリ/ディスク/GPU/CUDA/ネットワークなどのシステム情報を取得するためのユーティリティです。

注意：このモジュールは `psutil` に依存します。先に `capybara-docsaid[system]` をインストールしてください：

```bash
pip install "capybara-docsaid[system]"
```

## get_package_versions

> [get_package_versions() -> dict](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/system_info.py)

- **説明**：PyTorch、PyTorch Lightning、TensorFlow、Keras、NumPy、Pandas、Scikit-learn、OpenCV などの一般的なパッケージのバージョン情報を取得します。

- **戻り値**

  - **dict**：インストール済みパッケージのバージョン情報。

- **例**

  ```python
  from capybara.utils.system_info import get_package_versions

  versions_info = get_package_versions()
  print(versions_info)
  ```

## get_gpu_cuda_versions

> [get_gpu_cuda_versions() -> dict](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/system_info.py)

- **説明**：GPU と CUDA バージョン情報を取得します。PyTorch/TensorFlow/CuPy から CUDA バージョンを推定し、`nvidia-smi` から NVIDIA driver version を読み取ります。

- **戻り値**

  - **dict**：CUDA/driver version などを含む辞書。

- **例**

  ```python
  from capybara.utils.system_info import get_gpu_cuda_versions

  gpu_cuda_info = get_gpu_cuda_versions()
  print(gpu_cuda_info)
  ```

## get_cpu_info

> [get_cpu_info() -> str](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/system_info.py)

- **説明**：現在のプラットフォームに応じて CPU モデル名を返します。

- **戻り値**

  - **str**：CPU モデル名。見つからない場合は `"N/A"`。

- **例**

  ```python
  from capybara.utils.system_info import get_cpu_info

  cpu_info = get_cpu_info()
  print(cpu_info)
  ```

## get_external_ip

> [get_external_ip() -> str](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/system_info.py)

- **説明**：`https://httpbin.org/ip` から external IP を取得します。失敗した場合は `"Error obtaining IP: ..."` のような文字列を返します。

- **例**

  ```python
  from capybara.utils.system_info import get_external_ip

  ip = get_external_ip()
  print(ip)
  ```

## get_system_info

> [get_system_info() -> dict](https://github.com/DocsaidLab/Capybara/blob/main/capybara/utils/system_info.py)

- **説明**：OS バージョン、CPU 情報、メモリ、ディスク使用量などのシステム情報を取得します。

- **備考**

  - NIC 名は `enp5s0` に固定して `IPV4 Address` / `MAC Address` を取得します。環境で NIC 名が異なる場合、これらのフィールドは空リストになることがあります。
  - GPU 情報は `nvidia-smi` から取得します。利用できない場合は `"N/A or Error"` を返します。

- **戻り値**

  - **dict**：システム情報の辞書。

- **例**

  ```python
  from capybara.utils.system_info import get_system_info

  system_info = get_system_info()
  print(system_info)
  ```

