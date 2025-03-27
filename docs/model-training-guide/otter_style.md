---
slug: otter-style
title: Otter Style
authors: Z. Yuan
---

## 基於 Pytorch-Lightning

本章介紹由 Z. Yuan 所負責的專案的構建方式，主要基於 Pytorch-Lightning 訓練框架。

:::info
詳細實作細節，請參考：[**DocsaidLab/Otter**](https://github.com/DocsaidLab/Otter)。

至於為什麼要叫做 Otter......其實沒有特別的意義，取個名字方便區分而已。😅
:::

## 構建環境

以下章節使用 `DocClassifier` 專案為範例，說明該如何建立模型訓練環境，其內容可以應用到 `DocAligner` 和 `MRZScanner` 等，由 Z. Yuan 負責開發專案上。

:::info
深度學習專案大多只開放推論模組，目前只有 `DocClassifier` 開放訓練模組。若你有需要其他專案的訓練模組，可以參考本章節的訓練方式自行實作。

`DocClassifier` 專案請參考：[**DocClassifier github**](https://github.com/DocsaidLab/DocClassifier)
:::

請先使用 git 下載 [**Otter**](https://github.com/DocsaidLab/Otter) 模組，並且建立 Docker 映像：

```bash
git clone https://github.com/DocsaidLab/Otter.git
cd Otter
bash docker/build.bash
```

建置檔案內容如下：

```bash title="Otter/docker/build.bash"
docker build \
    -f docker/Dockerfile \
    -t otter_base_image .
```

在檔案中，你可以替換 `otter_base_image` 成你喜歡的名字，後續訓練時會用到。

:::info
PyTorch Lightning 是基於 PyTorch 的輕量級深度學習框架，旨在簡化模型訓練過程。它透過將研究程式碼（模型定義、前向/反向傳播、優化器設定等）與工程程式碼（訓練循環、日誌記錄、檢查點保存等）分離，使研究人員能夠專注於模型本身，而無需處理繁瑣的工程細節。

有興趣的讀者可以參考以下資源：

- [**PyTorch Lightning 官方網站**](https://lightning.ai/)
- [**PyTorch Lightning GitHub**](https://github.com/Lightning-AI/pytorch-lightning)
  :::

:::tip
簡單介紹一下 `Otter` 模組：

裡面有幾個構建模型的基本模組，包括：

1. `BaseMixin`：基本的訓練模型，包含了訓練的基本設定。
2. `BorderValueMixin` 和 `FillValueMixin`：用於圖像增強的填充模式。
3. `build_callback`：用於構建回呼函數。
4. `build_dataset`：用於構建數據集。
5. `build_logger`：用於構建日誌記錄。
6. `build_trainer`：用於構建訓練器。
7. `load_model_from_config`：用於從配置文件中讀取模型。

其中有加入一些系統資訊紀錄的功能，要搭配特定的配置文件格式才能正常運作。

不用特別去學習這個部分，根據經驗，每個工程師都會演化出一套自己的模型訓練方式，這只是無數種可能中的其中一種而已，僅供參考。
:::

這是我們預設採用的 [**Dockerfile**](https://github.com/DocsaidLab/Otter/blob/main/docker/Dockerfile)，專門為模型訓練設計：

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

基於上面的 Dockerfile，我們可以建立一個包含多種工具與庫的深度學習容器，適合進行圖像處理和機器學習相關任務。

以下說明幾個重要部分：

---

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONWARNINGS="ignore" \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei
```

設定環境變數：

- **`PYTHONDONTWRITEBYTECODE=1`**：防止生成 `.pyc` 編譯文件，減少不必要的文件產生。
- **`PYTHONWARNINGS="ignore"`**：忽略 Python 警告。
- **`DEBIAN_FRONTEND=noninteractive`**：禁用交互式提示，自動化部署。
- **`TZ=Asia/Taipei`**：設定時區為台北。

:::tip
你可以換成你喜歡的時區，或新增其他環境變數。
:::

---

```dockerfile
COPY . /usr/local/otter
RUN cd /usr/local/otter && \
    python setup.py bdist_wheel && \
    python -m pip install dist/*.whl && \
    cd ~ && rm -rf /usr/local/otter
```

1. 複製當前目錄的所有內容到容器的 `/usr/local/otter` 路徑下。
2. 進入該目錄，使用 `setup.py` 生成 wheel 格式的安裝包。
3. 安裝生成的 wheel 包，然後刪除構建目錄清理環境。

---

```dockerfile
RUN python -m pip install --no-cache-dir -U \
    tqdm colored ipython tabulate tensorboard scikit-learn fire \
    albumentations "Pillow>=10.0.0" fitsne opencv-fixer prettytable
```

安裝所需的 Python 第三方庫，包括：

- **`tqdm`**：進度條工具。
- **`colored`**：終端輸出著色。
- **`ipython`**：互動式 Python 介面。
- **`tabulate`**：表格格式化工具。
- **`tensorboard`**：深度學習可視化工具。
- **`scikit-learn`**：機器學習庫。
- **`fire`**：命令行介面生成工具。
- **`albumentations`**：影像增強工具。
- **`Pillow`**：影像處理庫，版本需為 10.0 或更高。
- **`fitsne`**：t-SNE 高效實現。
- **`opencv-fixer`**：OpenCV 修復工具。
- **`prettytable`**：表格輸出工具。

:::tip
如果你需要使用其他工具，可以在這裡添加相應的庫。
:::

---

```dockerfile
RUN python -c "from opencv_fixer import AutoFix; AutoFix()"
RUN python -c "import capybara; import chameleon"
```

這兩行用於執行簡單的 Python 命令來測試安裝：

1. 自動修復 OpenCV 配置問題。
2. 測試 `capybara` 和 `chameleon` 模組是否正常可用。

:::tip
因為 OpenCV 常有版本故障的問題，這裡使用 `opencv-fixer` 來自動修復。

此外，在 `capbybara` 模組中，有自動下載字型檔案的功能，在這裡先呼叫一次模組，可以將字型檔案預先下載到容器內，避免後續使用時出現問題。
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

這段程式碼生成一個 Bash 腳本，實現以下功能：

1. 如果環境變數 `USER_ID` 和 `GROUP_ID` 被設置，則動態創建一個用戶與用戶組，並設置相應的權限。
2. 使用 `gosu` 切換到該用戶執行命令，確保容器內的操作具有正確的身份標識。
3. 如果未設置這些變數，直接執行傳入的命令。

:::tip
gosu 是用來切換容器內的用戶身份的工具，可以避免使用 `sudo` 造成的權限問題。
:::

## 執行訓練

我們已經構建好了 Docker 映像，接下來我們將使用這個映像來執行模型訓練。

接著，我們將進入 `DocClassifier` 專案，首先，請你看到 `train.bash` 檔案內容：

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
    -v /data/Dataset:/data/Dataset \ # 這裡替換成你的資料集目錄
    -it --rm otter_base_image bash -c "
echo '
from fire import Fire
from DocClassifier.model import main_classifier_train

if __name__ == \"__main__\":
    Fire(main_classifier_train)
' > /code/trainer.py && python /code/trainer.py --cfg_name $1
"
```

針對上述檔案的說明如下，如果你想要動手修改的話，可以參考相關資訊：

1. **`-e USER_ID=$(id -u)` 和 `-e GROUP_ID=$(id -g)`**：將當前用戶的 UID 和 GID 傳遞到容器內，確保容器內的文件操作權限與主機一致。
2. **`--gpus all`**：啟用 GPU 支援，將所有可用的 GPU 資源分配給容器。
3. **`--shm-size=64g`**：設置共享記憶體大小為 64GB，適合需要大量記憶體的深度學習任務。
4. **`--ipc=host` 和 `--net=host`**：共享主機的進程間通訊和網絡資源，以提高性能和兼容性。
5. **`--cpuset-cpus="0-31"`**：限制容器僅使用 CPU 0-31 核心，避免影響其他進程。
6. **`-v $PWD/DocClassifier:/code/DocClassifier`**：將主機當前目錄的 `DocClassifier` 資料夾掛載到容器內 `/code/DocClassifier`。
7. **`-v /data/Dataset:/data/Dataset`**：將主機的資料集目錄掛載到容器內 `/data/Dataset`，需根據實際情況修改。
8. **`-it`**：以互動模式運行容器。
9. **`--rm`**：在容器結束時自動刪除容器，避免累積臨時容器。
10. **`otter_base_image`**：使用先前構建的 Docker 映像名稱，如果你有修改，請替換成你的名稱。

:::tip
這裡會有幾個常見的問題：

1. `--gpus` 壞了：看一下是不是 docker 沒有裝好，請參考：[**進階安裝**](../capybara/advance.md)。
2. `--cpuset-cpus`：不要超過你的 CPU 核心數量。
3. 工作目錄設定在 Dockerfile 內：`WORKDIR /code`，如果你不喜歡，可以自己修改。
4. `-v`：請一定要確認你到底掛載了什麼工作目錄，不然會找不到檔案。
5. 在 `DocClassifier` 的專案中，我們必須從外部掛載 ImageNet 資料集，如果你不需要，可以刪除這一段。

   :::

---

容器啟動後，我們會執行訓練指令，這裡直接寫入一個 Python 腳本：

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

1. **`echo`**：將一段 Python 程式碼寫入 `/code/trainer.py` 文件。該程式碼的功能如下：

   - **`from fire import Fire`**：引入 `fire` 庫，用於生成命令行介面。
   - **`from DocClassifier.model import main_classifier_train`**：從 `DocClassifier.model` 模組中引入訓練主函數。
   - **`if __name__ == \"__main__\":`**：當執行該腳本時，啟動 `Fire(main_classifier_train)`，將命令行參數綁定至函數。

2. **`python /code/trainer.py --cfg_name $1`**：執行生成的 Python 腳本，並使用 `$1` 傳入的參數作為 `--cfg_name` 的值，該參數通常用於指定配置檔案。

### 參數配置

在訓練模型的目錄中，會有一個專門用於放置配置文件的目錄，通常命名為 `config`。

在這個目錄中，我們可以定義不同的配置文件，用於訓練不同的模型，看個例子：

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

每個欄位的鍵值已經預先定義在 `Otter` 模組中，只要按照這個命名方式，就可以正常運作。

看到這裡，你應該也可以理解為什麼我們在一開始就提到說：不用特別去學習 `Otter` 模組！

雖然這裡已經有一定程度的抽象和封裝，但仍然是個高度客製化的架構。

最終你還是得找到最適合自己的方式，因此不必太拘泥於這個形式。

### 開始訓練

最後，請退到 `DocClassifier` 的上層目錄，並執行以下指令來啟動訓練：

```bash
# 後面要替換成你的配置文件名稱
bash DocClassifier/docker/train.bash lcnet050_cosface_f256_r128_squeeze_lbn_imagenet
```

通過這些步驟，你可以在 Docker 容器內安全地執行模型訓練的任務，同時利用 Docker 的隔離環境來確保一致性和可重現性。這種方法使得項目的部署和擴展變得更加方便和靈活。

## 轉換 ONNX

這部分說明如何將模型轉換為 ONNX 格式。

首先，請你看到 `to_onnx.bash` 檔案內容：

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

從這個檔案開始看起，但不需要修改它，你需要去修改對應的檔案：`model/to_onnx.py`

在訓練過程中，你可能會使用許多分支來監督模型的訓練，但是在推論階段，你可能只需要其中的一個分支。因此，我們需要將模型轉換為 ONNX 格式，並且只保留推論階段所需要的分支。

例如：

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

在上面這個範例中，我們只取出推論用的分支，並且將其封裝為一個新的模型 `WarpFeatureLearning`。接著，在 yaml config 上進行相對應的參數設定：

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

說明模型的輸入尺寸，輸入名稱，輸出名稱，以及 ONNX 的版本號。

轉換的部分我們已經幫你寫好了，完成上面的修改後，確認 `model/to_onnx.py` 檔案有指向你的模型，並且退到 `DocClassifier` 的上層目錄，並執行以下指令來啟動轉換：

```bash
# 後面要替換成你的配置文件名稱
bash DocClassifier/docker/to_onnx.bash lcnet050_cosface_f256_r128_squeeze_lbn_imagenet
```

## 最後

還是要再次強調，我們一律推薦在 docker 內做完所有的事情，這樣可以確保你的環境是一致的，還可以避免許多不必要的問題。

經過以上說明，相信你已經大概掌握模型訓練的流程了。雖然在實際應用中，你可能會遇到更多的問題，例如數據集的準備、模型的調參、訓練過程的監控等等。只是這些問題過於瑣碎，我們無法逐項列舉，只能透過這篇文章提供一點基本的指引。

總之，祝你也能得到一個好模型！
