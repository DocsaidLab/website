---
sidebar_position: 3
---

# Advanced

## Common References

- For detailed information on each version of the PyTorch image built by NVIDIA, refer to: [**PyTorch Release Notes**](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

- For preparations required for NVIDIA runtime, see: [**Installation (Native GPU Support)**](<https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#usage>)

- For instructions on installing the NVIDIA Toolkit, refer to: [**Installing the NVIDIA Container Toolkit**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

- For information on ONNXRuntime, see: [**ONNX Runtime Release Notes**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

## Environment Setup

Our working environment, though not overly complex, can encounter compatibility issues with some packages.

Generally speaking, we can categorize our setup into:

- **Training Environment**: Requires PyTorch, OpenCV, CUDA, and cuDNN to be compatible.
- **Deployment Environment**: Requires ONNXRuntime, OpenCV, and CUDA to be compatible.

The most common conflict occurs between the versions of PyTorch-CUDA and ONNXRuntime-CUDA.

:::tip
Why do they always mismatch? 💢 💢 💢
:::

## Use Docker!

We always use Docker for installation to ensure consistency across environments, with no exceptions.

Using Docker saves a lot of time adjusting the environment and avoids many unnecessary issues.

We continuously test related environments during development, and you can use the following commands:

### Installing the Inference Environment

```bash
cd Capybara
bash docker/build.bash
```

In the "inference environment", we use `nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` as the base image.

This image is specifically designed for deploying models, so it doesn't include training environment packages like PyTorch.

Users can replace this according to their needs, and the versions will change with updates to ONNXRuntime.

For inference-related images, refer to: [**NVIDIA NGC**](https://ngc.nvidia.com/catalog/containers/nvidia:cuda)

## Usage

Generally, we apply this module in projects like `DocAligner`.

### Daily Use

Here's an example. Suppose you have a `your_scripts.py` file that you need to run with Python.

Assuming you've completed the installation of the inference environment, write a `Dockerfile`:

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

# Set working directory, replace as needed
WORKDIR /code

# Example: Install DocAligner
RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

ENTRYPOINT ["python"]
```

Then build the image:

```bash
docker build -f your_Dockerfile -t your_image_name .
```

After completion, run the command wrapped in Docker each time:

```bash
#!/bin/bash
docker run \
    --gpus all \
    -v ${PWD}:/code \
    -it --rm your_image_name your_scripts.py
```

This is equivalent to directly calling the packaged Python environment, ensuring consistency.

:::tip
If you want to enter the container instead of starting Python, change the entry point to `/bin/bash`.

```Dockerfile
ENTRYPOINT ["/bin/bash"]
```

:::

### Introducing `gosu` Configuration

If you encounter permission issues when running Docker:

- **For example: Output files or images in the container have root:root permissions, making them difficult to modify or delete!**

We recommend considering the `gosu` tool.

Modify the original Dockerfile for `gosu` usage as follows:

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

# Set working directory, replace as needed
WORKDIR /code

# Example: Install DocAligner
RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

# Set entry point script path
ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

# Install gosu
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# Create entry point script
RUN printf '#!/bin/bash\n\
    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
    groupadd -g "$GROUP_ID" -o usergroup\n\
    useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
    export HOME=/home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /code\n\
    fi\n\
    \n\
    # Check if there are arguments\n\
    if [ $# -gt 0 ]; then\n\
    exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\n\
    else\n\
    exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\n\
    fi' > "$ENTRYPOINT_SCRIPT"

# Grant permissions
RUN chmod +x "$ENTRYPOINT_SCRIPT"

# Entry point
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
```

Then build the image:

```bash
docker build -f your_Dockerfile -t your_image_name .
```

Run the command wrapped in Docker each time:

```bash
#!/bin/bash
docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    -v ${PWD}:/code
    -it --rm your_image_name your_scripts.py
```

### Installing Internal Packages

If you need to install some internal packages while building the image, include environment variables.

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

# Set working directory, replace as needed
WORKDIR /code

# Example: Install DocAligner (assuming it's an internal package)

# Introduce environment variables
ARG PYPI_ACCOUNT
ARG PYPI_PASSWORD

# Change to your internal package source
ENV SERVER_IP=192.168.100.100:28080/simple/

# Install DocAligner
# Remember to change to your package server address
RUN python -m pip install \
    --trusted-host 192.168.100.100 \
    --index-url http://${PYPI_ACCOUNT}:${PYPI_PASSWORD}@192.168.100.100:16000/simple docaligner

ENTRYPOINT ["python"]
```

Then build the image:

```bash
docker build \
    -f your_Dockerfile \
    --build-arg PYPI_ACCOUNT=your_account \
    --build-arg PYPI_PASSWORD=your_password \
    -t your_image_name .
```

If your account and password are stored elsewhere, such as in a `pip.conf` file, you can parse the string to import them:

```bash
docker build \
    -f your_Dockerfile \
    --build-arg PYPI_PASSWORD=$(awk -F '://|@' '/index-url/{print $2}' your/config/path/pip.conf | cut -d: -f2) \
    -t your_image_name .
```

After completion, run the command wrapped in Docker each time. The usage is the same as above.

## FAQs

### Permission Denied

After switching users with `gosu`, your permissions will be restricted. If you need to read/write files in the container, you might encounter permission issues.

For example, if you installed the `DocAligner` package, it automatically downloads model files when starting the model, placing them in a Python-related folder.

In this example, the default path for model files is:

- `/usr/local/lib/python3.10/dist-packages/docaligner/heatmap_reg/ckpt`

This path is beyond the user's permission range!

To grant this path to the user, modify the Dockerfile as follows:

```Dockerfile title="your_Dockerfile" {28}
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

# Set working directory, replace as needed
WORKDIR /code

# Example: Install DocAligner
RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

# Set entry point script path
ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

# Install gosu
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# Create entry point script
RUN printf '#!/bin/bash\n\
    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
    groupadd -g "$GROUP_ID" -o usergroup\n\
    useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
    export HOME=/home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /code\n\
    chmod -R 777 /usr/local/lib/python3.10/dist-packages\n\
    fi\n\
    \n\
    # Check if there are arguments\n\
    if [ $# -gt 0 ]; then\n\
    exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\n\
    else\n\
    exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\n\
    fi' > "$ENTRYPOINT_SCRIPT"

# Grant permissions
RUN chmod +x "$ENTRYPOINT_SCRIPT"

# Entry point
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
```

If you encounter similar issues, you can resolve them this way.
