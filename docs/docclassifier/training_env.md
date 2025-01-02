---
sidebar_position: 8
---

# 模型訓練指南

:::warning
我們已經在進行 `DocsaidKit` 的轉移，這個頁面的內容進行翻修，把相關訓練模組轉移到 `Chameleon` 專案中。

完成後我們會重新改寫本頁面，請不用擔心。
:::

請你確保已經從 `DocsaidKit` 內建置了基礎映像 `docsaid_training_base_image`。

如果你還沒有這樣做，請先參考 `DocsaidKit` 的說明文件。

```bash
# Build base image from docsaidkit at first
git clone https://github.com/DocsaidLab/DocsaidKit.git
cd DocsaidKit
bash docker/build.bash
```

接著，請使用以下指令來建置 DocClassifier 工作的 Docker 映像：

```bash
# Then build DocClassifier image
git clone https://github.com/DocsaidLab/DocClassifier.git
cd DocClassifier
bash docker/build.bash
```

## 構建環境

這是我們預設採用的 [Dockerfile](https://github.com/DocsaidLab/DocClassifier/blob/main/docker/Dockerfile)，專門為模型訓練設計，我們為該文件附上簡短的說明，你可以根據自己的需求進行修改：

1. **基礎鏡像**

   - `FROM docsaid_training_base_image:latest`
   - 這行指定了容器的基礎鏡像，即 `docsaid_training_base_image` 的最新版本。基礎映像像是建立你的 Docker 容器的起點，它包含了預先配置好的作業系統和一些基本的工具，你可以在 `DocsaidKit` 的專案中找到它。

2. **工作目錄設定**

   - `WORKDIR /code`
   - 這裡設定了容器內的工作目錄為 `/code`。 工作目錄是 Docker 容器中的一個目錄，你的應用程式和所有的命令都會在這個目錄下運作。

3. **環境變數**

   - `ENV ENTRYPOINT_SCRIPT=/entrypoint.sh`
   - 這行定義了一個環境變數 `ENTRYPOINT_SCRIPT`，其值設定為 `/entrypoint.sh`。 環境變數用於儲存常用配置，可以在容器的任何地方存取。

4. **安裝 gosu**

   - 透過 `RUN` 指令安裝了 `gosu`。 `gosu` 是一個輕量級的工具，允許使用者以特定的使用者身分執行命令，類似於 `sudo`，但更適合 Docker 容器。
   - `apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*` 這行指令首先更新了套件列表，然後安裝`gosu`，最後清理了不再需要 的檔案以減小鏡像大小。

5. **建立入口點腳本**

   - 透過一系列 `RUN` 指令建立了入口點腳本 `/entrypoint.sh`。
   - 此腳本首先檢查環境變數 `USER_ID` 和 `GROUP_ID` 是否被設定。 如果設定了，腳本會建立一個新的使用者和使用者群組，並以該使用者身分執行命令。
   - 這對於處理容器內外檔案權限問題非常有用，特別是當容器需要存取宿主機上的檔案時。

6. **賦予權限**

   - `RUN chmod +x "$ENTRYPOINT_SCRIPT"` 這行指令使入口點腳本成為可執行檔。

7. **設定容器的入口點和預設指令**
   - `ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]` 和 `CMD ["bash"]`
   - 這些命令指定了容器啟動時執行的預設命令。 當容器啟動時，它將執行 `/entrypoint.sh` 腳本。

## 執行訓練

這部分的說明如何利用你已經構建的 Docker 映像來進行模型訓練。

首先，請你看到 `train.bash` 檔案內容：

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
    -v /data/Dataset:/data/Dataset \ # 這裡替換成你的資料集目錄
    -it --rm doc_classifier_train python trainer.py --cfg_name $1
```

針對上述檔案的說明如下，如果你想要動手修改的話，可以參考相關資訊：

1. **創建訓練腳本**

   - `cat > trainer.py <<EOF ... EOF`
   - 這段命令創建了一個 Python 腳本 `trainer.py`。這個腳本導入了必要的模塊和函數，並在腳本的主部分中調用 `main_docalign_train` 函數。使用 Google's Python Fire 庫來解析命令行參數，使得命令行界面的生成更加容易。

2. **運行 Docker 容器**

   - `docker run ... doc_classifier_train python trainer.py --cfg_name $1`
   - 這行命令啟動了一個 Docker 容器，並在其中運行 `trainer.py` 腳本。
   - `-e USER_ID=$(id -u) -e GROUP_ID=$(id -g)`：這些參數將當前用戶的用戶 ID 和組 ID 傳遞給容器，以便在容器內創建具有相應權限的用戶。
   - `--gpus all`：指定容器可以使用所有 GPU。
   - `--shm-size=64g`：設置共享記憶體的大小，這在大規模數據處理時很有用。
   - `--ipc=host --net=host`：這些設置允許容器使用主機的 IPC 命名空間和網路堆棧，有助於性能提升。
   - `--cpuset-cpus="0-31"`：指定容器使用哪些 CPU 核心。
   - `-v $PWD/DocClassifier:/code/DocClassifier` 等：這些是掛載參數，將主機的目錄映射到容器內部的目錄，以便於訓練數據和腳本的訪問。
   - `--cfg_name $1`：這是傳遞給 `trainer.py` 的參數，指定了配置文件的名稱。

3. **數據集路徑**
   - 特別注意 `/data/Dataset` 是用於存放訓練數據的路徑，你會需要調整 `-v /data/Dataset:/data/Dataset` 這段指令，把 `/data/Dataset` 替換成你的資料集目錄。

最後，請退到 `DocClassifier` 的上層目錄，並執行以下指令來啟動訓練：

```bash
bash DocClassifier/docker/train.bash lcnet050_cosface_96 # 這裡替換成你的配置文件名稱
```

通過這些步驟，你可以在 Docker 容器內安全地執行模型訓練的任務，同時利用 Docker 的隔離環境來確保一致性和可重現性。這種方法使得項目的部署和擴展變得更加方便和靈活。

## 轉換 ONNX

這部分的說明如何利用你的模型轉換為 ONNX 格式。

首先，請你看到 `to_onnx.bash` 檔案內容：

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
bash DocClassifier/docker/to_onnx.bash lcnet050_cosface_96 # 這裡替換成你的配置文件名稱
```

## 最後

你應該已經在 `DocClassifier/model` 目錄下看到一個新的 ONNX 模型。

把這個模型搬到 `docclassifier/xxx` 對應的目錄下，改一下模型路徑參數，就可以進行推論了。
