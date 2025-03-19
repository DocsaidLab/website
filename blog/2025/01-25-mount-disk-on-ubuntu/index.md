---
slug: mount-disk-on-ubuntu
title: 在 Ubuntu 上掛載隨身硬碟
authors: Z. Yuan
image: /img/2025/0125.webp
tags: [ubuntu, mount]
description: 紀錄掛載硬碟流程
---

在 Ubuntu 系統插入隨身硬碟後，不自覺地就開始發呆。

然後才會想起 Ubuntu 是個要「自己掛載」硬碟的作業系統。

<!-- truncate -->

## 先確認硬碟在哪

在掛載硬碟之前，首先需要確認硬碟是否已被系統偵測。

1. 打開終端機，輸入以下指令：

   ```bash
   sudo fdisk -l
   ```

   此指令會列出系統中所有可用的硬碟與分割區。

   找到你需要掛載的硬碟裝置，通常名稱類似於 `/dev/sdb` 或 `/dev/sdc`，其分割區名稱可能是 `/dev/sda1` 或 `/dev/sdc1`。

   :::tip
   改用 `lsblk` 指令也可以，就是看個人使用習慣。
   :::

2. 查看硬碟的檔案系統類型：

   假設我們剛才找到的硬碟是 `/dev/sda1`，可以使用以下指令查看檔案系統類型：

   ```bash
   sudo blkid /dev/sda1 | grep TYPE
   ```

   此指令會顯示分割區的檔案系統類型，例如 `ext4`、`ntfs` 或 `exfat`，幫助你決定使用何種掛載方式。

## 建立掛載目錄

掛載硬碟前，需要有一個目錄作為掛載點。

這裡我先直接假設這個路徑叫做 `/mnt/mydisk`，可以執行以下命令：

```bash
sudo mkdir -p /mnt/mydisk
```

:::tip
你可以選擇任何你喜歡的目錄名稱，只要確保目錄存在並且沒有其他檔案。
:::

## 掛載硬碟

根據硬碟的檔案系統類型，選擇適當的掛載方式。

### ext4

執行以下指令將硬碟掛載到目標目錄：

```bash
sudo mount /dev/sda1 /mnt/mydisk
```

確認掛載是否成功：

```bash
df -h
```

若掛載成功，你應該會在輸出中看到 `/mnt/mydisk`。

### NTFS 或 exFAT

若硬碟使用的是 NTFS 或 exFAT 檔案系統，你可能需要安裝相關工具。

1. 安裝必要工具：

   ```bash
   sudo apt update
   sudo apt install ntfs-3g exfat-fuse exfat-utils
   ```

2. 掛載 NTFS 或 exFAT 分割區（以 exFAT 為例）：

   ```bash
   sudo mount -t exfat /dev/sda1 /mnt/mydisk
   ```

## 常見問題

1. **掛載後權限不足**：

   有些檔案系統（如 NTFS、exFAT）不支援 Linux 的原生權限修改指令（如 `chmod` 或 `chown`）。若遇到此問題，可以在掛載時指定適當的權限。

   先解除掛載：

   ```bash
   sudo umount /mnt/mydisk
   ```

   重新掛載並指定權限：

   ```bash
   sudo mount -t exfat -o uid=1000,gid=1000,fmask=0022,dmask=0022 /dev/sda1 /mnt/mydisk
   ```

   其中，各項參數的意義如下：

   - `-t exfat`：指定檔案系統類型。
   - `uid=1000`：指定檔案擁有者的 UID。
   - `gid=1000`：指定檔案群組的 GID。
   - `fmask=0022` 與 `dmask=0022`：設定檔案與目錄的預設權限。

   確認掛載後的權限是否正確：

   ```bash
   ls -l /mnt/mydisk
   ```

   :::tip
   所謂的 `0022` 指的是八進位數字，對應到的權限是 `755`。
   :::

---

2. **未建立掛載目錄**：

   若掛載目錄不存在，掛載指令會失敗，請確保目錄已建立：

   ```bash
   sudo mkdir -p /mnt/mydisk
   ```

---

3. **不知道 uid 和 gid**：

   可以使用以下指令查詢當前使用者的 UID 和 GID：

   ```bash
   id
   ```

   輸出範例：

   ```
   uid=1000(username) gid=1000(username)
   ```

   其中，`uid` 是使用者的 UID，`gid` 是使用者的 GID。

---

4. **自動掛載硬碟**：

   若要在每次開機時自動掛載硬碟，可以編輯 `/etc/fstab` 文件進行配置。

   使用文字編輯器打開 `/etc/fstab`：

   ```bash
   sudo vim /etc/fstab
   ```

   在文件末尾新增以下內容（根據實際硬碟資訊修改）：

   ```bash
   /dev/sda1 /mnt/mydisk ntfs-3g defaults,uid=1000,gid=1000 0 0
   ```

   存檔並退出後，執行以下指令驗證配置：

   ```bash
   sudo mount -a
   ```

   若無錯誤訊息，表示配置成功。

---

5. **卸載硬碟**：

   若要卸載硬碟，可以執行以下指令：

   ```bash
   sudo umount /mnt/mydisk
   ```

   若硬碟正在使用中，可能會出現錯誤訊息，此時可以使用 `-l` 選項強制卸載：

   ```bash
   sudo umount -l /mnt/mydisk
   ```

   若硬碟已被卸載，可以使用以下指令確認：

   ```bash
   df -h
   ```

---

6. **未格式化硬碟**：

   全新的硬碟可能需要先格式化才能使用，使用 mkfs 指令格式化硬碟：

   ```bash
   sudo mkfs -t ext4 /dev/sda1
   ```

   這條指令會將 `/dev/sda1` 格式化為 ext4 檔案系統。如果要使用其他檔案系統，可以更換 `-t` 參數。

   :::warning
   格式化硬碟會刪除所有資料，請務必提前備份重要資料。
   :::

## 小結

總之，Ubuntu 系統掛載硬碟的流程並不複雜，只要掌握了基本的指令和注意事項，就能輕鬆完成。

以上是我簡單記錄一下掛載硬碟的指令，希望對你有所幫助。
