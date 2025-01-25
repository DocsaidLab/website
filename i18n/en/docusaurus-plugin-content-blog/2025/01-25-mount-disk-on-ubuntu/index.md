---
slug: mount-disk-on-ubuntu
title: Mounting a USB Drive on Ubuntu
authors: Zephyr
image: /en/img/2025/0125.webp
tags: [ubuntu, mount]
description: A guide to mounting a disk on Ubuntu
---

After inserting the USB drive into an Ubuntu system, one may unintentionally start staring blankly at the screen.

Then it hits you that Ubuntu is an operating system that requires you to "mount" disks manually.

<!-- truncate -->

## Check Where the Disk Is

Before mounting the disk, you first need to confirm whether the system has detected it.

1. Open the terminal and run the following command:

   ```bash
   sudo fdisk -l
   ```

   This command will list all the available disks and partitions in the system.

   Locate the disk you want to mount, usually named something like `/dev/sdb` or `/dev/sdc`, with partition names such as `/dev/sda1` or `/dev/sdc1`.

   :::tip
   You can also use the `lsblk` command, depending on your personal preference.
   :::

2. Check the file system type of the disk:

   Assuming the disk found is `/dev/sda1`, you can check its file system type with the following command:

   ```bash
   sudo blkid /dev/sda1 | grep TYPE
   ```

   This command will display the file system type of the partition, such as `ext4`, `ntfs`, or `exfat`, helping you decide the appropriate mounting method.

## Create a Mount Directory

Before mounting the disk, you need a directory to serve as the mount point.

Here, we'll assume the path is `/mnt/mydisk` and run the following command:

```bash
sudo mkdir -p /mnt/mydisk
```

:::tip
You can choose any directory name you like, as long as the directory exists and is empty.
:::

## Mount the Disk

Choose the appropriate mount method based on the disk's file system type.

### ext4

Run the following command to mount the disk to the target directory:

```bash
sudo mount /dev/sda1 /mnt/mydisk
```

Check if the mount was successful:

```bash
df -h
```

If successful, you should see `/mnt/mydisk` in the output.

### NTFS or exFAT

If the disk uses NTFS or exFAT file systems, you may need to install the necessary tools.

1. Install the required tools:

   ```bash
   sudo apt update
   sudo apt install ntfs-3g exfat-fuse exfat-utils
   ```

2. Mount the NTFS or exFAT partition (using exFAT as an example):

   ```bash
   sudo mount -t exfat /dev/sda1 /mnt/mydisk
   ```

## Common Issues

1. **Insufficient Permissions After Mounting**:

   Some file systems (like NTFS or exFAT) do not support native Linux permission modification commands (such as `chmod` or `chown`). If you encounter this issue, you can specify the appropriate permissions when mounting.

   Unmount first:

   ```bash
   sudo umount /mnt/mydisk
   ```

   Remount and specify permissions:

   ```bash
   sudo mount -t exfat -o uid=1000,gid=1000,fmask=0022,dmask=0022 /dev/sda1 /mnt/mydisk
   ```

   The meaning of each parameter is as follows:

   - `-t exfat`: Specifies the file system type.
   - `uid=1000`: Specifies the UID of the file owner.
   - `gid=1000`: Specifies the GID of the file group.
   - `fmask=0022` and `dmask=0022`: Set the default permissions for files and directories.

   Verify if the permissions are correct after mounting:

   ```bash
   ls -l /mnt/mydisk
   ```

   :::tip
   The `0022` is an octal number, corresponding to `755` permissions.
   :::

---

2. **Mount Directory Not Created**:

   If the mount directory does not exist, the mount command will fail. Make sure the directory is created:

   ```bash
   sudo mkdir -p /mnt/mydisk
   ```

---

3. **Don't Know the UID and GID**:

   You can use the following command to find the UID and GID of the current user:

   ```bash
   id
   ```

   Example output:

   ```
   uid=1000(username) gid=1000(username)
   ```

   Here, `uid` is the user ID, and `gid` is the group ID.

---

4. **Automatically Mount the Disk**:

   To automatically mount the disk on every boot, you can configure it in the `/etc/fstab` file.

   Open the `/etc/fstab` file with a text editor:

   ```bash
   sudo vim /etc/fstab
   ```

   Add the following line at the end of the file (modify it according to the actual disk information):

   ```bash
   /dev/sda1 /mnt/mydisk ntfs-3g defaults,uid=1000,gid=1000 0 0
   ```

   After saving and exiting, verify the configuration with:

   ```bash
   sudo mount -a
   ```

   If no error message appears, the configuration is successful.

---

5. **Unmount the Disk**:

   To unmount the disk, use the following command:

   ```bash
   sudo umount /mnt/mydisk
   ```

   If the disk is in use, you may encounter an error. In this case, use the `-l` option to force the unmount:

   ```bash
   sudo umount -l /mnt/mydisk
   ```

   After unmounting, you can confirm with the following command:

   ```bash
   df -h
   ```

---

6. **Disk Not Formatted**:

   A new disk may need to be formatted before use. Use the `mkfs` command to format the disk:

   ```bash
   sudo mkfs -t ext4 /dev/sda1
   ```

   This command will format `/dev/sda1` as the ext4 file system. To use a different file system, change the `-t` option.

   :::warning
   Formatting the disk will erase all data, so make sure to back up important files in advance.
   :::

## Conclusion

In summary, mounting a disk on Ubuntu is not complicated. Once you master the basic commands and keep the necessary precautions in mind, you can easily complete the process.

This is a simple record of the disk mounting process, and we hope it helps you.
