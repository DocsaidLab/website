---
sidebar_position: 3
---

# Environment

## Overview

We have designed a Docker image specifically for machine learning and deep learning model training based on the NVIDIA PyTorch image. Combined with our in-house developed toolkit, it provides a foundational training environment.

- **Related Resources**

    - For details on each version, please consult: [**PyTorch Release Notes**](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

    - For NVIDIA runtime preparation, refer to: [**Installation (Native GPU Support)**](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#usage)

    - For NVIDIA Toolkit installation, refer to: [**Installing the NVIDIA Container Toolkit**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

---

When choosing a PyTorch image, it's crucial to consider the version of ONNXRuntime to ensure compatibility.

- **For example:**

    When opting for pytorch:23.11, the corresponding CUDA version is 12.3.0. Therefore, it is impossible to use the onnxruntime-gpu version in this image, as even the latest v1.16 version requires CUDA version 11.8. If onnxruntime-gpu is desired, one must choose the pytorch:22.12 version, which corresponds to CUDA version 11.8.0.

:::tip
For more on ONNXRuntime, refer to: [ONNX Runtime Release Notes](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
:::

## Building the Docker Image

### Prerequisites

- Ensure your system has [**Docker**](https://docs.docker.com/engine/install/) installed.
- Ensure your system supports NVIDIA Docker and has [**NVIDIA Container Toolkit**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

### Base Image

We use `nvcr.io/nvidia/pytorch:24.03-py3` provided by NVIDIA as the base.

### Build Instructions

- **Reference Build File**: [**Dockerfile**](https://github.com/DocsaidLab/DocsaidKit/blob/main/docker/Dockerfile)
- **Environment Variables**: Several environment variables are set to optimize the image operation.
- **Installed Packages**: Includes libraries and tools related to audio, video, and image processing, along with necessary Python packages.
- **Python Packages**: Includes tools and libraries for training, such as `tqdm`, `Pillow`, `tensorboard`, etc.
- **Working Directory**: Sets `/code` as the default working directory.

### Build Commands

In the DocsaidKit directory, execute the following command to build the Docker image:

```bash
cd DocsaidKit
bash docker/build.bash
```

## Running the Docker Image

After a successful build, you can use the following commands to run the image:

### Basic Run Command

```bash
#!/bin/bash
docker run \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -it --rm docsaid_training_base_image bash
```

### Command Explanation

- `--gpus all`: Assigns all available GPUs to the Docker container.
- `--shm-size=64g`: Sets the shared memory size, suitable for large-scale deep learning tasks.
- `--ipc=host --net=host`: The container will use the host's IPC and network settings.
- `--cpuset-cpus="0-31"`: Restricts CPU usage, can be adjusted based on requirements.

### Considerations

- Ensure that the host has sufficient resources (such as memory and storage space) when running the Docker image.
- If there are version conflicts or specific requirements, adjust the installation packages and versions in the Dockerfile as needed.