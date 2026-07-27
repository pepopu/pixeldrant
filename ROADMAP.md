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
- [x] **M2 磁盘布局 + 同步 I/O**:4KB 块格式定稿(节点=向量+度数+邻居,SIFT 644B/节点、
  6 节点/块,索引 683MB/3s 写出),pread 版分轮束搜索(束宽 W,批量读接口)。
  验收通过:**W=1 与内存版逐位等价**(recall 全档一致),**reads == hops 精确成立**
  (ef=80:86 次读);W=4 多 ~10 读换小幅召回提升,是 M3 流水线甜点位。
  同步 syscall 代价:QPS 24.6k(内存)→ 1.3k(页缓存热磁盘,ef=80)——M3 的攻击目标。
  Linux 验证(2026-07-27,阿里云 2C/3.6G/高效云盘):11 测试全绿;冷盘 29 QPS
  = 盘 2.2k IOPS ÷ 86 读,**QPS = IOPS ÷ reads 公式精确成立** → NVMe 1M IOPS
  推算 ~11.6k QPS @ recall 0.95,这就是 M3/M6 的靶子
- [ ] **M3 io_uring 异步流水线**(核心):
  - [x] **M3a 单查询批量后端**:`UringReader`(每线程独立 ring、整批 frontier 一次
    提交、4KB 对齐缓冲池、可选 O_DIRECT)。WSL2 验证(2026-07-27):正确性与 pread
    逐位一致;O_DIRECT 下 w=1→8 吞吐 **5.8×**(295→1705 QPS),W 摊薄 I/O 延迟的
    机制成立;w=8→16 触及 WSL 虚拟盘并行度上限
  - [x] **M3b 跨查询流水线**(功能完成,WSL 验证 2026-07-27):`UringExecutor`
    (stage/flush/drain)+ 单线程查询状态机调度器,与单查询版**逐查询等价**
    (结果与读次数完全相同,测试断言)。O_DIRECT 单线程:c=1→64 得 **5.5×**
    (151→830 QPS),饱和于 WSL 虚拟盘单 ring ~80k IOPS;缓存热同线程 2648 QPS
    证明调度器非瓶颈。每线程效率 = M3a 16 线程模式的 8×
  - [x] **M3 验收**(2026-07-27,租用 4C/894GB 本地 NVMe,fio 500k IOPS):
    每线程效率 **5.4×** ✅(单线程流水线 1378 QPS vs 同步单线程份额 256);
    IOPS 利用率 48%(242k/500k)⚠️——瓶颈为 4 核 CPU 饱和而非 I/O 架构,
    多核机器上随核数增长。详见 BENCHMARK.md
- [x] **M4a 量化粗筛(SQ8)**(2026-07-27):`RouteScorer` 抽象 + SQ8 码本路由 +
  **块内免费精排**(展开节点的块里本来就有全精度向量;返回距离永远精确,测试断言
  与内存打分逐位一致——注意求和顺序必须用同一个 l2 核,浮点不结合)。
  WSL 实测(O_DIRECT,c=32,w=4):路由内存 512→**128MB(1/4)**,ef=80 召回
  0.9506→0.9496(**−0.1pt**),QPS 持平(898→902),IOPS 恒定 86-88k。
  训练 ~300ms/1M 向量
- [ ] **M4b RaBitQ(1-2bit)**:冲 1/10-1/32 内存目标;精排架构已证明对路由误差
  高度耐受,预期可行。优先级低于租机验收
- [x] **M5 核心**(2026-07-27):HTTP 服务(axum,/search + /health,scratch 池,
  spawn_blocking,SQ8 码本持久化,端到端测试);`DiskIndex::search` 与流水线统一为
  scorer 泛型 + 块内精排单一路径。遗留 M5b:热点缓存、超内存建图
- [x] **M6 第一代 benchmark**(2026-07-27 完成):SIFT1M + BIGANN-10M 真 NVMe
  全套数字入 BENCHMARK.md。关键规模结论:10× 数据量下每查询读数仅 +8%(96→104),
  磁盘版 sq8 单线程 = 内存版 4 线程的 50% QPS @ 1/5 内存;HTTP 服务在 10M 上
  单查询 4.7ms。Deep100M/1B 级别留给二代(依赖 M5b 超内存建图)

**第一代版本(v0.1)至此完成**:NVMe 原生磁盘向量索引,完整证据链,可服务化。

二期:在线写入(SPFresh 式)、过滤、f16、多向量。

## 参考

DiskANN (NeurIPS'19) / RaBitQ (SIGMOD'24) / Starling (SIGMOD'24) / SPANN (NeurIPS'21);
工程参考:microsoft/DiskANN、Qdrant `lib/segment`(spaces/ 与 hnsw_index/)。

## 环境备注

- 开发与正确性:Windows/任意平台均可(pread 后端);
- 云服务器 118.31.47.25(阿里云,CentOS 7 内核 3.10,2C/3.6G,高效云盘 2.2k IOPS
  封顶):已配免密 SSH + Rust 工具链,代码在 /root/nvvec。定位:Linux 正确性平台。
  **不能跑 M3**——无 io_uring(内核太老)且盘 2 线程即饱和;
- M3 起的性能数字:必须本地 NVMe 的 Linux(WSL2 虚拟盘数字无效),推荐阿里云
  i 系列按量实例(跑基准时开机)或 Hetzner/AWS i4i。
