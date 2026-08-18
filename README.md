# OpenVLA-AlignFlow_v1: 纯算法多模态具身大模型决策系统与代码库

> **项目名称**：OpenVLA-AlignFlow: 融合细粒度图文对齐与连续流匹配的具身大模型决策系统  
> **核心技术栈**：PyTorch / Qwen3-VL / SigLIP / Conditional Flow Matching (CFM) / SOTA Trajectory-DPO / Action Chunking (k=16) / OpenX (BridgeData v2)  
> **设计特点**：纯算法离线驱动、数学物理指标严谨、支持一键 Demo 秒级闭环自测与真实数据全流程训练。

---

## ⚡ 快速上手与运行指令

### 1. 一键端到端 Demo 快速自测（内置合成轨迹，0 门槛秒级闭环跑通）

```bash
# CPU 模式秒级测试
python run_pipeline.py --mode demo --device cpu

# GPU 模式运行
python run_pipeline.py --mode demo --device cuda
```

### 2. 真实 OpenX / BridgeData v2 数据全流程执行

#### Step 1: 预训练模型准备
```bash
python vla/data/download_models.py --all --target_dir ./pretrained_models
```

#### Step 2: 数据集 ETL 预处理与动力学清洗
```bash
python vla/data/dataset_downloader.py --data_root ./data/openx --output_dir ./data/processed
```

#### Step 3: 分阶段或端到端训练
```bash
# 方式 A: 一键端到端全量流水线 (Stage 1 -> Stage 2 -> Stage 3 -> Benchmark)
python run_pipeline.py --mode full --device cuda --output_dir ./data/processed

# 方式 B: 分步独立训练
python vla/training/train_vl_align.py --device cuda --epochs 25
python vla/training/train_flow_vla.py --device cuda --epochs 45
python vla/training/train_offline_rl_dpo.py --device cuda --epochs 20
```

#### Step 4: 多维度离线 Benchmark 评测
```bash
python vla/evaluation/offline_benchmark.py
```
