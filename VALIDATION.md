# NVMe 租机验收清单(M3+M4 一次性)

目标:在真实本地 NVMe 上复测 WSL 已验证的全部结论,拿到可对外引用的数字。
预算:按量实例 3-4 小时,几十元。

## 1. 开机

阿里云控制台 → 按量付费 ECS:

- **规格**:`ecs.i3.xlarge`(4C32G + ~894GB 本地 NVMe)或更大的 i3/i4;
  关键是"本地盘"字样,**不要**用 ESSD 云盘替代
- **镜像**:Ubuntu 24.04 64 位
- 设置 root 密码,记下公网 IP

## 2. 环境(把 IP 告诉 Claude 自动执行,或手动)

```bash
apt-get update && apt-get install -y build-essential curl fio
# rustup(国内用 TUNA 镜像)+ cargo tuna 源,同 WSL 配置
# git clone https://github.com/pepopu/pixeldrant nvvec 或 scp 源码
# 数据:scp sift.tar.gz + vamana graph 缓存(或重新建图,4C 约 8-10 分钟)
```

本地盘通常是 /dev/nvme1n1(未格式化):`mkfs.ext4 /dev/nvme1n1 && mount /dev/nvme1n1 /mnt/nvme`,数据和索引放 `/mnt/nvme`。

## 3. 盘基线(先知道天花板)

```bash
fio --name=qd1  --filename=/mnt/nvme/fio.test --size=4G --rw=randread --bs=4k \
    --direct=1 --ioengine=io_uring --iodepth=1 --runtime=20 --time_based
fio --name=qd256 --filename=/mnt/nvme/fio.test --size=4G --rw=randread --bs=4k \
    --direct=1 --ioengine=io_uring --iodepth=256 --numjobs=4 --runtime=20 --time_based --group_reporting
```

记录:QD1 延迟(预期 ~60-100μs)、饱和 IOPS(i3 标称 ~30 万,i4 更高)。

## 4. 验收矩阵(全部 O_DIRECT,SIFT1M,ef=80,w=4)

| # | 命令要点 | 验收线 |
|---|---|---|
| A | `disk --backend pread --ws 1`(单线程基准) | 记录基线 QPS |
| B | `pipeline --routing exact --concurrency 1,8,32,64,128` | c 扫描饱和 IOPS ≥ fio 值的 **60%**;饱和 QPS ≥ A 的 **5×** |
| C | `pipeline --routing sq8`(同 B 的最优 c) | recall 损失 ≤0.5pt,QPS 与 B 持平 |
| D | (可选)多调度线程:开 2-4 个 pipeline 进程分摊查询 | 合计 IOPS 逼近 fio 饱和值 |

预期头条数字:盘若有 30 万 IOPS → **~3,000 QPS @ recall 0.95**(96 读/查询),
路由内存 128MB。同硬件内存版 HNSW 预计 30-50k QPS——即 **~1/10 QPS @ 1/4 内存**
(M4b RaBitQ 后内存再降 4-8×)。

## 5. 收尾

结果 CSV(results/)拷回本地 → **释放实例**(本地盘数据随实例销毁,别忘拷数据)。
