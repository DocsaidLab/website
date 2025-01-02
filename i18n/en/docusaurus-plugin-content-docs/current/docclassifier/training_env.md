---
sidebar_position: 8
---

# Training

:::warning
We are currently transitioning `DocsaidKit`, with the content of this page being overhauled and related training modules being moved to the `Chameleon` project.

Once the transition is complete, we will rewrite this page. Please don't worry.
:::

Please ensure that you've built the foundational image `docsaid_training_base_image` from `DocsaidKit`.

If you haven't done so yet, please refer to the documentation of `DocsaidKit`.

```bash
# Build base image from docsaidkit at first
git clone https://github.com/DocsaidLab/DocsaidKit.git
cd DocsaidKit
bash docker/build.bash
```

Next, use the following command to build the Docker image for DocClassifier:

```bash
# Then build DocClassifier image
git clone https://github.com/DocsaidLab/DocClassifier.git
cd DocClassifier
bash docker/build.bash
```

## Building the Environment

Below is our default [Dockerfile](https://github.com/DocsaidLab/DocClassifier/blob/main/docker/Dockerfile) designed specifically for model training. We provide a brief explanation of this file, which you can modify according to your needs:

1. **Base Image**

   - `FROM docsaid_training_base_image:latest`
   - This line specifies the base image for the container, which is the latest version of `docsaid_training_base_image`. The base image serves as the starting point for building your Docker container, containing pre-configured operating systems and basic tools, which you can find in the `DocsaidKit` project.

2. **Working Directory Setup**

   - `WORKDIR /code`
   - Here, the working directory inside the container is set to `/code`. The working directory is a directory in the Docker container where your application and all commands will operate.

3. **Environment Variables**

   - `ENV ENTRYPOINT_SCRIPT=/entrypoint.sh`
   - This line defines an environment variable `ENTRYPOINT_SCRIPT` with a value set to `/entrypoint.sh`. Environment variables are used to store common configurations accessible anywhere within the container.

4. **Installing gosu**

   - The `RUN` command installs `gosu`. `gosu` is a lightweight tool that allows running commands as a specific user, similar to `sudo`, but more suitable for Docker containers.
   - `apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*` This command first updates the package lists, then installs `gosu`, and finally cleans up unnecessary files to reduce the image size.

5. **Creating the Entry Point Script**

   - A series of `RUN` commands create the entry point script `/entrypoint.sh`.
   - This script first checks if the environment variables `USER_ID` and `GROUP_ID` are set. If set, the script creates a new user and user group and executes commands as that user.
   - This is useful for handling file permission issues both inside and outside the container, especially when the container needs to access files on the host machine.

6. **Granting Permissions**

   - `RUN chmod +x "$ENTRYPOINT_SCRIPT"` This command makes the entry point script executable.

7. **Setting the Container's Entry Point and Default Command**
   - `ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]` and `CMD ["bash"]`
   - These commands specify the default command to run when the container starts. When the container starts, it will execute the `/entrypoint.sh` script.

## Running Training

This section explains how to perform model training using the Docker image you've built.

First, take a look at the contents of the `train.bash` file:

```bash
#!/bin/bash

cat > trainer.py <<EOF
from fire import Fire
from DocClassifier.model import main_docclassifier_train

if __name__ == '__main__':
    Fire(main_docclassifier_train)
EOF

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocClassifier:/code/DocClassifier \
    -v $PWD/trainer.py:/code/trainer.py \
    -v /data/Dataset:/data/Dataset \ # Replace this with your dataset directory
    -it --rm doc_classifier_train python trainer.py --cfg_name $1
```

Here's an explanation of the above file. Feel free to modify it as needed:

1. **Creating the Training Script**

   - `cat > trainer.py <<EOF ... EOF`
   - This command creates a Python script `trainer.py`. The script imports necessary modules and functions and calls the `main_docalign_train` function in the main part of the script. Google's Python Fire library is used to parse command-line arguments, making command-line interface generation easier.

2. **Running the Docker Container**

   - `docker run ... doc_classifier_train python trainer.py --cfg_name $1`
   - This command starts a Docker container and runs the `trainer.py` script inside it.
   - `-e USER_ID=$(id -u) -e GROUP_ID=$(id -g)`: These parameters pass the current user's user ID and group ID to the container to create a user with corresponding permissions inside the container.
   - `--gpus all`: Specifies that the container can use all GPUs.
   - `--shm-size=64g`: Sets the size of shared memory, which is useful for large-scale data processing.
   - `--ipc=host --net=host`: These settings allow the container to use the host's IPC namespace and network stack, improving performance.
   - `--cpuset-cpus="0-31"`: Specifies which CPU cores the container can use.
   - `-v $PWD/DocClassifier:/code/DocClassifier` and others: These are mounting parameters that map directories from the host to directories inside the container, facilitating access to training data and scripts.
   - `--cfg_name $1`: This is a parameter passed to `trainer.py`, specifying the name of the configuration file.

3. **Dataset Path**
   - Pay special attention to `/data/Dataset`, which is the path for storing training data. You'll need to adjust `-v /data/Dataset:/data/Dataset` to point to your dataset directory.

Finally, navigate to the parent directory of `DocClassifier` and execute the following command to start training:

```bash
bash DocClassifier/docker/train.bash lcnet050_cosface_96 # Replace this with your configuration file name
```

Through these steps, you can safely perform model training tasks within a Docker container while ensuring consistency and reproducibility using Docker's isolated environment. This approach makes deployment and scaling of the project more convenient and flexible.

## Converting to ONNX

This section explains how to convert your model to ONNX format.

First, take a look at the contents of the `to_onnx.bash` file:

```bash
#!/bin/bash

cat > torch2onnx.py <<EOF
from fire import Fire
from DocClassifier.model import main_docclassifier_torch2onnx

if __name__ == '__main__':
    Fire(main_docclassifier_torch2onnx)
EOF

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --shm-size=64g \
    --ipc=host --net=host \
    -v $PWD/DocClassifier:/code/DocClassifier \
    -v $PWD/torch2onnx.py:/code/torch2onnx.py \
    -it --rm doc_classifier_train python torch2onnx.py --cfg_name $1
```

Start by examining this file, but you don't need to modify it. You'll need to modify the corresponding file: `model/to_onnx.py`.

During training, you may use many branches to supervise the training of your model. However, during inference, you may only need one of these branches. Therefore, we need to convert the model to ONNX format and retain only the branch needed for inference.

For example:

```python
class WarpFeatureLearning(nn.Module):

    def __init__(self, model: L.LightningModule):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, img: torch.Tensor):
        xs = self.backbone(img)
        features = self.head(xs)
        return features
```

In this example, we extract only the branch needed for inference and encapsulate it as a new model `WarpFeatureLearning`. Then, make corresponding parameter settings in the yaml config:

```yaml
onnx:
  name: WarpFeatureLearning
  input_shape:
    img:
      shape: [1, 3, 128, 128]
      dtype: float32
  input_names: ["img"]
  output_names:
    - feats
  dynamic_axes:
    img:
      "0": batch_size
    output:
      "0": batch_size
  options:
    opset_version: 16
    verbose: False
    do_constant_folding: True
```

This specifies the input size, input name, output name, and ONNX version number.

The conversion part has already been written for you. After completing the modifications mentioned above and confirming that `model/to_onnx.py` points to your model, navigate to the parent directory of `DocClassifier` and execute the following command to start the conversion:

```bash
bash DocClassifier/docker/to_onnx.bash lcnet050_cosface_96 # Replace this with your configuration file name
```

## Conclusion

You should now see a new ONNX model in the `DocClassifier/model` directory.

Move this model to the corresponding directory `docclassifier/xxx`, update the model path parameter, and you're ready to perform inference.
