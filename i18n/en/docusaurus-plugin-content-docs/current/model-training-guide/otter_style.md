---
slug: otter-style
title: Otter Style
authors: Z. Yuan
---

## Based on Pytorch-Lightning

This chapter introduces the construction method of the project managed by Z. Yuan, primarily based on the Pytorch-Lightning training framework.

:::info
For detailed implementation details, please refer to: [**DocsaidLab/Otter**](https://github.com/DocsaidLab/Otter).

As for why it is called Otter... there is no special meaning, just a name chosen for differentiation purposes. 😅
:::

## Setting Up the Environment

The following sections use the `DocClassifier` project as an example to explain how to set up the model training environment. The content can also be applied to other projects such as `DocAligner` and `MRZScanner`, which are managed by Z. Yuan.

:::info
Most deep learning projects only offer the inference module. Currently, only the `DocClassifier` project offers the training module. If you need training modules for other projects, you can refer to the training methods in this chapter and implement them on your own.

For the `DocClassifier` project, refer to: [**DocClassifier github**](https://github.com/DocsaidLab/DocClassifier)
:::

First, use git to download the [**Otter**](https://github.com/DocsaidLab/Otter) module and create the Docker image:

```bash
git clone https://github.com/DocsaidLab/Otter.git
cd Otter
bash docker/build.bash
```

The build file contents are as follows:

```bash title="Otter/docker/build.bash"
docker build \
    -f docker/Dockerfile \
    -t otter_base_image .
```

In the file, you can replace `otter_base_image` with your preferred name, which will be used later during training.

:::info
PyTorch Lightning is a lightweight deep learning framework based on PyTorch, designed to simplify the model training process. It separates research code (model definition, forward/backward propagation, optimizer settings, etc.) from engineering code (training loops, logging, checkpoint saving, etc.), allowing researchers to focus on the model itself without dealing with cumbersome engineering details.

Interested readers can refer to the following resources:

- [**PyTorch Lightning Official Website**](https://lightning.ai/)
- [**PyTorch Lightning GitHub**](https://github.com/Lightning-AI/pytorch-lightning)
  :::

:::tip
A brief introduction to the `Otter` module:

It includes several basic modules for building models, such as:

1. `BaseMixin`: Basic training model, containing basic training settings.
2. `BorderValueMixin` and `FillValueMixin`: Padding modes for image augmentation.
3. `build_callback`: Used to build callback functions.
4. `build_dataset`: Used to build datasets.
5. `build_logger`: Used to build logging.
6. `build_trainer`: Used to build the trainer.
7. `load_model_from_config`: Used to load models from configuration files.

It also includes some system information recording features, which require a specific configuration file format to function correctly.

There's no need to focus on learning this part. Based on experience, each engineer will develop their own model training methods, and this is just one of many possible approaches, provided as a reference.
:::

Here is the default [**Dockerfile**](https://github.com/DocsaidLab/Otter/blob/main/docker/Dockerfile) we use, specially designed for model training:

```dockerfile title="Otter/docker/Dockerfile"
# syntax=docker/dockerfile:experimental
FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONWARNINGS="ignore" \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    tzdata wget git libturbojpeg exiftool ffmpeg poppler-utils libpng-dev \
    libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev gcc \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    python3-pip libharfbuzz-dev libfribidi-dev libxcb1-dev libfftw3-dev \
    libpq-dev python3-dev gosu && \
    ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir -U pip setuptools wheel

COPY . /usr/local/otter
RUN cd /usr/local/otter && \
    python setup.py bdist_wheel && \
    python -m pip install dist/*.whl && \
    cd ~ && rm -rf /usr/local/otter

RUN python -m pip install --no-cache-dir -U \
    tqdm colored ipython tabulate tensorboard scikit-learn fire \
    albumentations "Pillow>=10.0.0" fitsne opencv-fixer prettytable

RUN python -c "from opencv_fixer import AutoFix; AutoFix()"
RUN python -c "import capybara; import chameleon"

WORKDIR /code

ENV ENTRYPOINT_SCRIPT=/entrypoint.sh

RUN printf '#!/bin/bash\n\
    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
    groupadd -g "$GROUP_ID" -o usergroup\n\
    useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
    export HOME=/home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /code\n\
    exec gosu "$USER_ID":"$GROUP_ID" "$@"\n\
    else\n\
    exec "$@"\n\
    fi' > "$ENTRYPOINT_SCRIPT" && \
    chmod +x "$ENTRYPOINT_SCRIPT"

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]

CMD ["bash"]
```

Based on the Dockerfile above, we can create a deep learning container that includes multiple tools and libraries, suitable for image processing and machine learning tasks.

Here are explanations of several important parts:

---

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONWARNINGS="ignore" \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei
```

Setting environment variables:

- **`PYTHONDONTWRITEBYTECODE=1`**: Prevents the generation of `.pyc` compilation files, reducing unnecessary file generation.
- **`PYTHONWARNINGS="ignore"`**: Ignores Python warnings.
- **`DEBIAN_FRONTEND=noninteractive`**: Disables interactive prompts for automated deployment.
- **`TZ=Asia/Taipei`**: Sets the timezone to Taipei.

:::tip
You can change it to your preferred timezone or add other environment variables.
:::

---

```dockerfile
COPY . /usr/local/otter
RUN cd /usr/local/otter && \
    python setup.py bdist_wheel && \
    python -m pip install dist/*.whl && \
    cd ~ && rm -rf /usr/local/otter
```

1. Copies all content from the current directory to the container's `/usr/local/otter` path.
2. Navigates to this directory, generates a wheel package using `setup.py`.
3. Installs the generated wheel package and then deletes the build directory to clean up the environment.

---

```dockerfile
RUN python -m pip install --no-cache-dir -U \
    tqdm colored ipython tabulate tensorboard scikit-learn fire \
    albumentations "Pillow>=10.0.0" fitsne opencv-fixer prettytable
```

Installs required third-party Python libraries, including:

- **`tqdm`**: Progress bar tool.
- **`colored`**: Terminal output coloring.
- **`ipython`**: Interactive Python interface.
- **`tabulate`**: Table formatting tool.
- **`tensorboard`**: Deep learning visualization tool.
- **`scikit-learn`**: Machine learning library.
- **`fire`**: Command-line interface generation tool.
- **`albumentations`**: Image augmentation tool.
- **`Pillow`**: Image processing library, version 10.0 or higher required.
- **`fitsne`**: Efficient t-SNE implementation.
- **`opencv-fixer`**: OpenCV fix tool.
- **`prettytable`**: Table output tool.

:::tip
If you need additional tools, you can add the corresponding libraries here.
:::

---

```dockerfile
RUN python -c "from opencv_fixer import AutoFix; AutoFix()"
RUN python -c "import capybara; import chameleon"
```

These two lines execute simple Python commands to test the installation:

1. Automatically fixes OpenCV configuration issues.
2. Tests whether the `capybara` and `chameleon` modules are available.

:::tip
Since OpenCV often has version-related issues, we use `opencv-fixer` to automatically fix these.

In addition, the `capbybara` module has the function of automatically downloading font files. By calling the module here, the font files can be downloaded to the container in advance to avoid problems during subsequent use.
:::

---

```dockerfile
RUN printf '#!/bin/bash\n\
    if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
    groupadd -g "$GROUP_ID" -o usergroup\n\
    useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
    export HOME=/home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
    chown -R "$USER_ID":"$GROUP_ID" /code\n\
    exec gosu "$USER_ID":"$GROUP_ID" "$@"\n\
    else\n\
    exec "$@"\n\
    fi' > "$ENTRYPOINT_SCRIPT" && \
    chmod +x "$ENTRYPOINT_SCRIPT"
```

This script generates a Bash script that implements the following functionalities:

1. If the environment variables `USER_ID` and `GROUP_ID` are set, it dynamically creates a user and group with the corresponding IDs and assigns the appropriate permissions.
2. It uses `gosu` to switch to the created user and execute commands, ensuring that the operations inside the container are performed with the correct identity.
3. If these variables are not set, it directly executes the passed command.

:::tip
`gosu` is a tool used to switch user identities inside a container, avoiding permission issues caused by using `sudo`.
:::

## Execute Training

We have already built the Docker image. Next, we will use this image to run the model training.

Then, we will enter the `DocClassifier` project. First, please take a look at the content of the `train.bash` file:

- [**DocClassifier/docker/train.bash**](https://github.com/DocsaidLab/DocClassifier/blob/main/docker/train.bash)

```bash title="DocClassifier/docker/train.bash"
#!/bin/bash

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocClassifier:/code/DocClassifier \
    -v /data/Dataset:/data/Dataset \ # Replace this with your dataset directory
    -it --rm otter_base_image bash -c "
echo '
from fire import Fire
from DocClassifier.model import main_classifier_train

if __name__ == \"__main__\":
    Fire(main_classifier_train)
' > /code/trainer.py && python /code/trainer.py --cfg_name $1
"
```

Here’s an explanation of the above file. If you want to make modifications, you can refer to the related information:

1. **`-e USER_ID=$(id -u)` and `-e GROUP_ID=$(id -g)`**: Passes the current user’s UID and GID into the container to ensure that file operation permissions inside the container match those of the host system.
2. **`--gpus all`**: Enables GPU support and allocates all available GPU resources to the container.
3. **`--shm-size=64g`**: Sets the shared memory size to 64GB, suitable for deep learning tasks that require large amounts of memory.
4. **`--ipc=host` and `--net=host`**: Shares the host’s inter-process communication and network resources to improve performance and compatibility.
5. **`--cpuset-cpus="0-31"`**: Limits the container to use only CPU cores 0-31 to avoid affecting other processes.
6. **`-v $PWD/DocClassifier:/code/DocClassifier`**: Mounts the host’s current directory `DocClassifier` folder to `/code/DocClassifier` inside the container.
7. **`-v /data/Dataset:/data/Dataset`**: Mounts the host’s dataset directory to `/data/Dataset` inside the container. Modify it as needed based on the actual path.
8. **`-it`**: Runs the container in interactive mode.
9. **`--rm`**: Automatically removes the container when it stops, to avoid accumulating temporary containers.
10. **`otter_base_image`**: The name of the Docker image previously built. If you have made modifications, replace it with your custom name.

:::tip
Here are a few common issues:

1. `--gpus` error: Check if Docker is installed correctly, refer to [**Advanced Installation**](../capybara/advance.md).
2. `--cpuset-cpus`: Don’t exceed the number of CPU cores on your machine.
3. The working directory is set in the Dockerfile: `WORKDIR /code`. If you don’t like it, you can modify it.
4. `-v`: Be sure to check the directories you are mounting, otherwise, files might not be found.
5. In the `DocClassifier` project, we need to mount the ImageNet dataset from outside. If you don’t need it, you can delete that part.
   :::

---

After the container starts, we will execute the training command.

Here, we will directly write a Python script:

```bash
bash -c "
echo '
from fire import Fire
from DocClassifier.model import main_classifier_train

if __name__ == \"__main__\":
    Fire(main_classifier_train)
' > /code/trainer.py && python /code/trainer.py --cfg_name $1
"
```

1. **`echo`**: Writes a Python script into the `/code/trainer.py` file. The functionality of the script is as follows:

   - **`from fire import Fire`**: Imports the `fire` library, which is used to generate command-line interfaces.
   - **`from DocClassifier.model import main_classifier_train`**: Imports the main training function from the `DocClassifier.model` module.
   - **`if __name__ == "__main__":`**: When the script is executed, it starts `Fire(main_classifier_train)`, binding the command-line arguments to the function.

2. **`python /code/trainer.py --cfg_name $1`**: Executes the generated Python script and uses the parameter `$1` passed in as the value for `--cfg_name`. This parameter is typically used to specify a configuration file.

### Parameter Configuration

In the directory where the model is trained, there is usually a dedicated directory for storing configuration files, typically named `config`.

Within this directory, we can define different configuration files for training various models. Here's an example:

```yaml title="config/lcnet050_cosface_f256_r128_squeeze_lbn_imagenet.yaml"
common:
  batch_size: 1024
  image_size: [128, 128]
  is_restore: False
  restore_ind: ""
  restore_ckpt: ""
  preview_batch: 1000
  use_imagenet: True
  use_clip: False

global_settings:
  image_size: [128, 128]

trainer:
  max_epochs: 40
  precision: 32
  val_check_interval: 1.0
  gradient_clip_val: 5
  accumulate_grad_batches: 1
  accelerator: gpu
  devices: [0]

model:
  name: ClassifierModel
  backbone:
    name: Backbone
    options:
      name: timm_lcnet_050
      pretrained: True
      features_only: True
  head:
    name: FeatureLearningSqueezeLBNHead
    options:
      in_dim: 256
      embed_dim: 256
      feature_map_size: 4
  loss:
    name: CosFace
    options:
      s: 64
      m: 0.4
    num_classes: -1
    embed_dim: 256

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

dataset:
  train_options:
    name: SynthDataset
    options:
      aug_ratio: 1
      length_of_dataset: 2560000
      use_imagenet: True
      use_clip: False
  valid_options:
    name: RealDataset
    options:
      return_tensor: True

dataloader:
  train_options:
    batch_size: -1
    num_workers: 24
    shuffle: False
    drop_last: False
  valid_options:
    batch_size: -1
    num_workers: 16
    shuffle: False
    drop_last: False

optimizer:
  name: AdamW
  options:
    lr: 0.001
    betas: [0.9, 0.999]
    weight_decay: 0.001
    amsgrad: False

lr_scheduler:
  name: PolynomialLRWarmup
  options:
    warmup_iters: -1
    total_iters: -1
  pl_options:
    monitor: loss
    interval: step

callbacks:
  - name: ModelCheckpoint
    options:
      monitor: valid_fpr@4
      mode: max
      verbose: True
      save_last: True
      save_top_k: 5
  - name: LearningRateMonitor
    options:
      logging_interval: step
  - name: RichModelSummary
    options:
      max_depth: 3
  - name: CustomTQDMProgressBar
    options:
      unit_scale: -1

logger:
  name: TensorBoardLogger
  options:
    save_dir: logger
```

Each field's key-value pair is predefined in the `Otter` module. As long as you follow this naming convention, everything will work as expected.

By now, you should understand why we mentioned earlier that there’s no need to specifically learn the `Otter` module!

Although there is a certain level of abstraction and encapsulation here, it is still a highly customizable framework.

Ultimately, you will need to find the approach that works best for you, so there's no need to be overly strict about this particular format.

### Start Training

Finally, navigate to the parent directory of `DocClassifier` and run the following command to start the training:

```bash
# Replace with your configuration file name
bash DocClassifier/docker/train.bash lcnet050_cosface_f256_r128_squeeze_lbn_imagenet
```

With these steps, you can safely execute the model training task inside the Docker container, while leveraging Docker's isolated environment to ensure consistency and reproducibility. This approach makes project deployment and scaling more convenient and flexible.

## Convert to ONNX

This section explains how to convert the model to ONNX format.

First, take a look at the content of the `to_onnx.bash` file:

```bash title="DocClassifier/docker/to_onnx.bash"
#!/bin/bash

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocClassifier:/code/DocClassifier \
    -it --rm otter_base_image bash -c "
echo '
from fire import Fire
from DocClassifier.model import main_classifier_torch2onnx

if __name__ == \"__main__\":
    Fire(main_classifier_torch2onnx)
' > /code/torch2onnx.py && python /code/torch2onnx.py --cfg_name $1
"
```

Start by looking at this file, but there's no need to modify it. You need to modify the corresponding file: `model/to_onnx.py`.

During the training process, you may use multiple branches to supervise the model’s training. However, during the inference phase, you might only need one of these branches. Therefore, we need to convert the model to the ONNX format while retaining only the branches required for inference.

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

In the example above, we extract only the branch used for inference and encapsulate it into a new model called `WarpFeatureLearning`. Then, we make the corresponding parameter settings in the YAML config:

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

Describe the model's input dimensions, input names, output names, and the ONNX version.

The conversion part has already been written for you. After completing the modifications, make sure the `model/to_onnx.py` file points to your model. Then, navigate to the parent directory of `DocClassifier` and execute the following command to start the conversion:

```bash
# Replace with your configuration file name
bash DocClassifier/docker/to_onnx.bash lcnet050_cosface_f256_r128_squeeze_lbn_imagenet
```

## Final Notes

We still strongly recommend completing all tasks inside Docker to ensure your environment is consistent and to avoid many unnecessary issues.

After the above explanation, you should have a general understanding of the model training process. Although in actual applications, you may encounter more challenges, such as dataset preparation, model tuning, monitoring the training process, etc., these issues are too detailed to list individually. This article is meant to provide some basic guidance.

In any case, we wish you success in obtaining a great model!
