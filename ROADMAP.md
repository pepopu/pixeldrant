# Roadmap

定位:单机 NVMe 原生向量检索引擎。目标:**~1/10 内存达到内存版 HNSW ~90% 的查询性能**,用可复现 benchmark 证明。

## 核心假设(实验可证伪)

1. io_uring + 高队列深度能吃满 NVMe 的百万级随机读 IOPS,而 mmap 路径不能;
2. DiskANN 布局(向量+邻居表同一 4KB 块)使图遍历每跳一次 I/O;
3. 内存中只驻留量化码(RaBitQ/SQ8,~1/10 原始大小)做路由与剪枝,全精度只在盘上。

## 里程碑

- [x] **M0 基准骨架**:workspace + I/O trait(批量读接口)+ fvecs/ivecs 加载 +
  recall/QPS/latency harness + SIFT1M + 暴力基线(recall 校验 ~1.0)+ usearch 基线
  (Windows 上 hnswlib 无 wheel 且本机缺 Windows SDK,改用同算法的 usearch;
  Linux 环境再补 hnswlib 行)

  基线参考数字(2026-07-27,本机 16 线程,SIFT1M,`results/all.csv`):
  | method | recall@10 | QPS(批量) | p50 单线程 |
  |---|---|---|---|
  | brute(精确) | 0.9994 | 89 | — |
  | usearch ef=40 | 0.9270 | 29,930 | 396us |
  | usearch ef=80 | 0.9745 | 16,322 | 587us |
  | usearch ef=200 | 0.9953 | 5,640 | 1,809us |

  → M1 的 Go/No-Go 线:recall@10≥0.95 时 QPS ≥ ~13k(usearch 同召回的 70%)
- [x] **M1 内存版 Vamana**:建图(RobustPrune + 两轮并行插入,129s)+ beam search。
  Go/No-Go 通过:recall@10=0.955 时 **24.6k QPS**(目标 13k 的 189%),
  召回-QPS 曲线与 usearch 重合(0.97 召回档:16.7k vs 16.3k)。
  M2 关键数据:ef=80 时平均 86 跳 → 磁盘版每查询 ~86 次块读
- [ ] **M2 磁盘布局 + 同步 I/O**(2 周):4KB 节点块格式定稿,pread 版磁盘搜索。
  验证:I/O 次数 ≈ beam 步数 × W,recall 与内存版一致
- [ ] **M3 io_uring 异步流水线**(3-4 周,核心):O_DIRECT + 注册 buffer +
  跨查询共享队列。Go/No-Go:IOPS 利用率 ≥60%,QPS ≥ 同步版 5×。**需要 Linux 裸机**
- [ ] **M4 量化粗筛**(2-3 周):SQ8 → RaBitQ 码本驻留内存。
  验证:内存 ≤ 原始 1/10,recall 损失 ≤1pt
- [ ] **M5 工程化**(3-4 周):热点缓存、超内存建图、只读索引原子发布、最小 HTTP API
- [ ] **M6 头条 benchmark**:Deep100M 上 vs DiskANN/Qdrant/hnswlib,可复现报告

二期:在线写入(SPFresh 式)、过滤、f16、多向量。

## 参考

DiskANN (NeurIPS'19) / RaBitQ (SIGMOD'24) / Starling (SIGMOD'24) / SPANN (NeurIPS'21);
工程参考:microsoft/DiskANN、Qdrant `lib/segment`(spaces/ 与 hnsw_index/)。

## 环境备注

- 开发与正确性:Windows/任意平台均可(pread 后端);
- M3 起的性能数字:必须 Linux 裸机 NVMe(WSL2 虚拟盘数字无效),届时装双系统或租
  Hetzner/AWS i4i。
