---
sidebar_position: 3
---

# Advanced

## Common References

Before setting up the environment, here are a few official documents worth referencing:

- **PyTorch Release Notes**

  NVIDIA provides the [PyTorch release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) to help you understand the specific versions of PyTorch, CUDA, and cuDNN built into each image, reducing potential dependency conflicts.

---

- **NVIDIA Runtime Setup**:

  To use GPU with Docker, refer to the [Installation (Native GPU Support)](<https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#usage>) guide to ensure that NVIDIA drivers and container tools are correctly installed on your system.

---

- **NVIDIA Container Toolkit Installation**

  The official guide on [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is highly recommended to read carefully as it is crucial for GPU acceleration with Docker.

---

- **ONNXRuntime Release Notes**

  When using ONNXRuntime for inference, if GPU acceleration is needed, refer to the official [CUDA Execution Provider guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) to ensure version compatibility.

## Environment Installation

Most deep learning projects will encounter dependency issues. The typical division of work is as follows:

- **Training Environment**: PyTorch, OpenCV, CUDA, and cuDNN versions need to be compatible, or you’ll often face issues like "a library cannot be loaded correctly."
- **Deployment Environment**: ONNXRuntime, OpenCV, and CUDA also need to match the correct versions, especially for GPU acceleration, where ONNXRuntime-CUDA requires specific CUDA versions.

One of the most common pitfalls is version mismatch between **PyTorch-CUDA** and **ONNXRuntime-CUDA**. When this happens, it’s usually recommended to revert to the official tested combinations or carefully check their dependencies on CUDA and cuDNN versions.

:::tip
Why do they never match? 💢 💢 💢
:::

## Use Docker!

To ensure consistency and portability, we **strongly recommend** using Docker. While it's feasible to set up the environment locally, in the long run, during collaborative development and deployment phases, more time will be spent dealing with unnecessary conflicts.

### Install Environment

```bash
cd Capybara
bash docker/build.bash
```

The `Dockerfile` used for building is also included in the project. If you're interested, you can refer to the [**Capybara Dockerfile**](https://github.com/DocsaidLab/Capybara/blob/main/docker/Dockerfile).

In the "inference environment," we use the base image `nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`.

This image is specifically designed for model deployment, so it doesn't include training environment packages, and you won't find libraries like PyTorch in it.

Users can change the base image according to their needs, and related versions will be updated alongside ONNXRuntime.

For more information on inference images, refer to [**NVIDIA NGC**](https://ngc.nvidia.com/catalog/containers/nvidia:cuda).

## Usage

The following demonstrates a common use case: running an external script via Docker and mounting the current directory inside the container.

### Daily Use

Suppose you have a script `your_scripts.py` that you want to run using Python inside the inference container. The steps are as follows:

1. Create a new `Dockerfile` (named `your_Dockerfile`):

   ```Dockerfile title="your_Dockerfile"
   # syntax=docker/dockerfile:experimental
   FROM capybara_infer_image:latest

   # Set working directory, users can change this based on their needs
   WORKDIR /code

   # Example: Install DocAligner
   RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
       cd DocAligner && \
       python setup.py bdist_wheel && \
       pip install dist/*.whl && \
       cd .. && rm -rf DocAligner

   ENTRYPOINT ["python"]
   ```

2. Build the image:

   ```bash
   docker build -f your_Dockerfile -t your_image_name .
   ```

3. Write the execution script (for example, `run_in_docker.sh`):

   ```bash
   #!/bin/bash
   docker run \
       --gpus all \
       -v ${PWD}:/code \
       -it --rm your_image_name your_scripts.py
   ```

4. Run the script `run_in_docker.sh` to perform inference.

:::tip
If you want to enter the container and start bash instead of directly running Python, change `ENTRYPOINT ["python"]` to `ENTRYPOINT ["/bin/bash"]`.
:::

### Integrating gosu Configuration

In practice, you may encounter the issue of "output files inside the container being owned by root."

If multiple engineers share the same directory, it could lead to permission issues in the future.

This can be resolved with `gosu`. Here's how we can modify the Dockerfile example:

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

WORKDIR /code

# Example: Install DocAligner
RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

# Install gosu
RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

# Create the entrypoint script
RUN printf '#!/bin/bash\n\
    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
        groupadd -g "$GROUP_ID" -o usergroup\n\
        useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
        export HOME=/home/user\n\
        chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
        chown -R "$USER_ID":"$GROUP_ID" /code\n\
    fi\n\
    \n\
    # Check for parameters\n\
    if [ $# -gt 0 ]; then\n\
        exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\n\
    else\n\
        exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\n\
    fi' > "$ENTRYPOINT_SCRIPT"

RUN chmod +x "$ENTRYPOINT_SCRIPT"

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
```

### Image Build and Execution

1. Build the image:

   ```bash
   docker build -f your_Dockerfile -t your_image_name .
   ```

2. Run the image (GPU acceleration example):

   ```bash
   #!/bin/bash
   docker run \
       -e USER_ID=$(id -u) \
       -e GROUP_ID=$(id -g) \
       --gpus all \
       -v ${PWD}:/code \
       -it --rm your_image_name your_scripts.py
   ```

This way, the output files will automatically have the current user's permissions, making subsequent read/write operations easier.

### Installing Internal Packages

If you need to install **private packages** or **internal tools** (such as those hosted on a private PyPI), you can provide authentication credentials during the build process:

```Dockerfile title="your_Dockerfile"
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

WORKDIR /code

ARG PYPI_ACCOUNT
ARG PYPI_PASSWORD

# Specify your internal package source
ENV SERVER_IP=192.168.100.100:28080/simple/

RUN python -m pip install \
    --trusted-host 192.168.100.100 \
    --index-url http://${PYPI_ACCOUNT}:${PYPI_PASSWORD}@192.168.100.100:16000/simple docaligner

ENTRYPOINT ["python"]
```

Then, pass the credentials during the build:

```bash
docker build \
    -f your_Dockerfile \
    --build-arg PYPI_ACCOUNT=your_account \
    --build-arg PYPI_PASSWORD=your_password \
    -t your_image_name .
```

If your credentials are stored in `pip.conf`, you can also parse the string to inject them, for example:

```bash
docker build \
    -f your_Dockerfile \
    --build-arg PYPI_PASSWORD=$(awk -F '://|@' '/index-url/{print $2}' your/config/path/pip.conf | cut -d: -f2) \
    -t your_image_name .
```

After building, every time you use it, simply execute the command within Docker as shown above.

## Common Issue: Permission Denied

If you encounter the error `Permission denied` when running commands, it's a significant issue.

After switching users with `gosu`, your permissions will be restricted within certain boundaries. If you need to read/write files inside the container, you may encounter permission issues.

For example: If you installed the `DocAligner` package, it will automatically download model files during model initialization and place them in Python-related directories.

In this example, the model files are stored by default in:

- `/usr/local/lib/python3.10/dist-packages/docaligner/heatmap_reg/ckpt`

This path is clearly outside the user's permission scope!

So, you will need to grant the user access to this directory when starting the container. Modify the Dockerfile as follows:

```Dockerfile title="your_Dockerfile" {23}
# syntax=docker/dockerfile:experimental
FROM capybara_infer_image:latest

WORKDIR /code

RUN git clone https://github.com/DocsaidLab/DocAligner.git && \
    cd DocAligner && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd .. && rm -rf DocAligner

ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

RUN apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*

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
    if [ $# -gt 0 ]; then\n\
        exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\n\
    else\n\
        exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\n\
    fi' > "$ENTRYPOINT_SCRIPT"

RUN chmod +x "$ENTRYPOINT_SCRIPT"

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
```

If only specific directories need permission changes, you can modify the corresponding paths to avoid overexposing permissions.

## Summary

Although using Docker requires more learning, it ensures environment consistency and significantly reduces unnecessary complications during deployment and collaborative development.

This investment is definitely worth it, and we hope you enjoy the convenience it brings!
