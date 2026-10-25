# IMIP 实验一：医学图像三分类

本仓库是智能医学图像处理课程实验一的代码与结果记录。数据由任课教师提供，不随仓库分发；预训练权重、训练产物、图像和实验报告也不入库。仓库只保留可公开的原创代码、MIT 许可的 ConvNeXt 适配代码，以及三个模型的最终结果说明。

> 仅用于课程实验与方法复现，不构成临床诊断依据。

## 结果总览

| 模型 | 初始化 | 输入 | test 集 | acc | 宏 F1 | 类 2 F1 | 说明 |
|---|---|---|---:|---:|---:|---:|---|
| MedMamba VSSM-tiny | DermaMNIST 迁移 | 128×128 | raw，n=143 | 0.6993 | 0.6277 | 0.444 | 上游无 LICENSE，代码不随仓库分发 |
| ConvNeXt-T | ImageNet-1k 迁移 | 224×224 | clean，n=142 | **0.8662** | **0.7683** | 0.485 | MIT；freeze5 |
| CSM-Net center | 从零训练 | 224×224，4 通道 | clean，n=142 | 0.7254 | 0.6649 | **0.5333** | 自建模型；最终主候选 |
| CSM-Net B | 从零训练 | 224×224，4 通道 | clean，n=142 | 0.7113 | 0.6508 | 0.4615 | 简化结构；保守备选 |
| CSM-Net D-120 | 从零训练 | 224×224，4 通道 | clean，n=142 | 0.7113 | 0.5532 | 0.1538 | 原始完整结构；对照 |

注意：

- MedMamba 使用未去重的 raw test（143 张），ConvNeXt 与 CSM-Net 使用去重后的 clean test（142 张），两者不是同一测试集，不能直接横向比较。
- 宏 F1 是主指标。类 2 在 clean test 中只有 18 张，单点波动很大。
- CSM-Net 的 center 与 B 都是 90 epoch 强增强配方；D-120 是 120 epoch 的原始完整配方。CSM-Net 的 test 对比不是严格的“只换结构”对照，详见 [`CSMNet/README.md`](CSMNet/README.md)。

## 数据与评估协议

### 数据

教师提供的原图为 RGB 2592×1728，按类别整理为 `ImageFolder` 结构：

| 划分 | 总数 | 类 0 | 类 1 | 类 2 |
|---|---:|---:|---:|---:|
| clean train | 568 | 286 | 209 | 73 |
| clean test | 142 | 72 | 52 | 18 |
| raw train（未去重） | 569 | 286 | 210 | 73 |
| raw test（未去重） | 143 | 72 | 53 | 18 |

- 数据不随仓库分发。复现时需自行把数据放到 `data_clean/train/{0,1,2}` 与 `data_clean/test/{0,1,2}`。
- raw 数据中有一对跨集重复（`test/1/494.jpg` 与 `train/1/552.jpg`）和一对训练集内重复（`train/1/366.jpg` 与 `train/1/426.jpg`）。clean 版本删除跨集重复与训练集内重复，原图未改。
- 没有病人 ID，无法做病人级分组。现有 3 折/5 折都是随机分层划分；dHash 近邻审计显示每折约 23–28% 的验证图在训练集中存在近邻，不能证明同一只眼泄漏，但这是明确的残留局限。

### 评估

- 主指标：宏 F1；同时报告准确率、逐类 F1、混淆矩阵、加权 F1 与 Cohen's κ。
- 3 折：`StratifiedKFold(n_splits=3, shuffle=True, random_state=0)`；OOF 为每折 best 在各自验证折预测后拼接。
- CSM-Net 另有 5 折 center 复跑，只作为稳定性检查，不能与 3 折数字直接比。
- 配对 bootstrap 只在同一 test 集上比较；置信区间跨 0 时不宣称稳定胜出。
- test 读取纪律：MedMamba 在阶段一收尾与 stage2 report 读取；ConvNeXt clean test 在 freeze5 定稿后只读一次；CSM-Net 累计读取 9 次（D 主集成 1 + D 单折 3 + B 主集成 1 + B 单折 3 + center 主集成 1），后续不再用 test 调参。

## 三个模型

### 1. MedMamba VSSM-tiny

- 上游：[YubiaoYue/MedMamba](https://github.com/YubiaoYue/MedMamba)，同步基线 `104beb3d641d67cef6f09b94ac299083d91bc417`。
- 上游未附 LICENSE。为避免未授权再分发，本仓库**不包含** MedMamba 的模型代码；只保留本实验的结果、方法说明与引用。
- 本实验用纯 PyTorch 的 `selective_scan_ref` 替换 CUDA 融合扫描，输入 128×128，batch 4，10 epoch。
- 提交模型为阶段一 DermaMNIST 迁移权重：raw test acc 0.6993、宏 F1 0.6277、类 2 F1 0.444。阶段二续训、Mixup、单周期 20 轮与 3 折集成都没有超过它。
- 完整结果与局限见 [`MedMamba/README.md`](MedMamba/README.md)。

### 2. ConvNeXt-T

- 上游：[facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt)，MIT License，同步基线 `048efcea897d999aed302f2639b6270aedf8d4c8`。
- 只保留上游 `models/convnext.py` 与 `LICENSE`；训练/验证/测试流程按本实验协议重写。
- 27.82M 参数，224×224，ImageNet-1k 预训练，先冻结 backbone 5 轮再全量微调（freeze5）。
- clean 3 折 best-val 宏 F1 0.7918 ± 0.0280；clean test acc 0.8662、宏 F1 0.7683、类 2 F1 0.485。
- 完整结果、raw 对照与复现命令见 [`ConvNeXt/README.md`](ConvNeXt/README.md)。

### 3. CSM-Net（自建 CNN）

- 从零训练，不加载 ImageNet、DermaMNIST 或任何预训练权重；不引用 MedMamba/ConvNeXt 的模型定义。
- 针对本数据的设计：
  - 先检测并擦除图像中的红色轮廓叠加，避免模型把标注当病灶；
  - 输入 `[RGB, S]`，其中 `S = ReLU(G - 0.5(R + B)) × 4`，显式提供荧光素响应；
  - DRSB 并行 depthwise 3×3 / 5×5，分别覆盖点状与片状染色；
  - stage2/3/4 多尺度 avg+max 池化，主头 3 类。
- 最终主候选为简化版 B + center crop：2.036M 参数，90 epoch，强增强，head128，无 Stain Gate、无辅助头。
- clean 3 折 best-val 宏 F1 0.6811 ± 0.0080，OOF 宏 F1 0.6808；clean test acc 0.7254、宏 F1 0.6649、类 2 F1 0.5333。
- 5 折 center 为 0.6969 ± 0.0324；B 为 0.6592 ± 0.0353；D-120 为 0.6176 ± 0.0022。
- center 对 B 的 test 宏 F1 差 +0.0145，95% CI [−0.0688, +0.0993]，跨 0；center 是 CV 明确赢家、test 点估计最好，但不是统计上稳定的胜出。
- 2 类仍是瓶颈：focal loss、OOF logit bias、1-vs-2 逻辑回归专家都没有改善 OOF 宏 F1。
- 设计、消融、失败尝试与复现命令见 [`CSMNet/README.md`](CSMNet/README.md)。

## 目录

```text
Experiment1/code/
├─ README.md                 本文件
├─ LICENSE                   原创部分 MIT
├─ NOTICE                    第三方组件版权与许可
├─ requirements.txt          公共依赖
├─ .gitignore
├─ .gitattributes
├─ CSMNet/
│  ├─ README.md
│  ├─ data.py                红线擦除、S 通道、几何预处理
│  ├─ model.py               DRSB、Stain Gate、辅助头（均可消融）
│  └─ train.py               smoke / fold / cv / eval
├─ ConvNeXt/
│  ├─ README.md
│  ├─ LICENSE                上游 MIT
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ convnext.py         上游模型定义，原样保留
│  └─ train.py               本实验重写的训练/评估
└─ MedMamba/
   └─ README.md              结果与方法；上游代码不随仓库分发
```

仓库外（不入 Git）：

| 路径 | 内容 |
|---|---|
| `Experiment1/data/` | 教师提供的 raw 数据 |
| `Experiment1/data_clean/` | 去重后的数据 |
| `Experiment1/weights/` | 预训练权重 |
| `Experiment1/code/run/` | 训练产物与 checkpoint |
| `Experiment1/code/logs/` | 运行日志 |
| `Experiment1/submission/` | 课程提交固化包 |

## 环境

| 项 | 值 |
|---|---|
| 系统 | Windows |
| Python | 3.10.22 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| scikit-learn | 1.7.2 |
| Pillow | 12.3.0 |
| GPU | NVIDIA RTX 4060 Laptop 8GB（sm_89） |

MedMamba 的本地复现还需要 `timm==0.9.16`、`einops==0.8.0` 与上游模型代码；这些不在本仓库分发。

## 快速开始

### 1. 安装依赖

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 `
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

### 2. 准备数据

```text
Experiment1/
├─ code/                本仓库
├─ data_clean/
│  ├─ train/0/...
│  ├─ train/1/...
│  ├─ train/2/...
│  ├─ test/0/...
│  ├─ test/1/...
│  └─ test/2/...
└─ weights/
   └─ ConvNeXt/convnext_tiny_1k_224_ema.pth
```

数据不随仓库分发。`ImageFolder` 只要求每个类别一个子目录；文件名和标签格式不影响训练。

### 3. ConvNeXt-T

```powershell
cd Experiment1/code

# 3 折 CV；权重默认从 ../weights/ConvNeXt/ 读取，也可用 --pretrained 指定
python ConvNeXt/train.py cv --folds 3 --epochs 30 --freeze-epochs 5 `
  --data-root ../data_clean `
  --out-dir run/convnext/cv_clean_freeze5

# clean test 一次性评估（3 折 best 的 softmax 概率平均）
python ConvNeXt/train.py eval --checkpoints `
  run/convnext/cv_clean_freeze5/fold0/best.pth `
  run/convnext/cv_clean_freeze5/fold1/best.pth `
  run/convnext/cv_clean_freeze5/fold2/best.pth `
  --data-root ../data_clean `
  --out run/convnext/eval_clean_freeze5/eval_test_ensemble.csv
```

### 4. CSM-Net

```powershell
cd Experiment1/code

# 最终主候选：B 结构 + center crop，3 折
python CSMNet/train.py cv --folds 3 --epochs 90 --warmup 3 --bs 32 --workers 4 `
  --strong-aug --head-dim 128 --head-dropout 0.4 --wd 0.1 --ls 0.1 `
  --ratio-mode center_crop --no-stain-gate --no-aux `
  --data-root ../data_clean `
  --out-dir run/csmnet/center_cv_90_strong

# clean test 一次性评估
python CSMNet/train.py eval --checkpoints `
  run/csmnet/center_cv_90_strong/fold0/best.pth `
  run/csmnet/center_cv_90_strong/fold1/best.pth `
  run/csmnet/center_cv_90_strong/fold2/best.pth `
  --ratio-mode center_crop --no-stain-gate --no-aux `
  --data-root ../data_clean `
  --out-dir run/csmnet/eval_center_final
```

未给 `--folds-dir` 时，CSM-Net 按 `StratifiedKFold` 与 `--folds-seed` 现场生成折；同一数据顺序与 seed 下与历史折文件一致。也可以复用 ConvNeXt 的 `run/convnext/cv_clean_freeze5/folds`。

### 5. MedMamba

本仓库不包含其模型代码。复现请从上游获取代码，并按 [`NOTICE`](NOTICE) 与 [`MedMamba/README.md`](MedMamba/README.md) 的说明替换扫描算子、准备 DermaMNIST 权重。

## 结果可信度与局限

- 所有数字来自实际训练产生的 `epochs.csv`、`cv_summary.csv`、OOF 与 test 预测 CSV，没有模拟数据。
- 训练集只有 568 张，clean test 只有 142 张，类 2 只有 73/18 张；单次宏 F1 的置信区间较宽，0.01 量级的差异不能当作方法收益。
- 没有病人 ID，随机划分可能把同一只眼的不同帧分到两边；dHash 审计只能提示风险，不能完成 patient-level split。
- CSM-Net 的 test 读取次数已披露；center 与 B 的 test 差异不显著，不能把 0.6649 说成稳定超过 0.6508。
- MedMamba 使用 raw test、CSM-Net 使用 clean test，跨模型数字只作背景，不做严格排名。

## 许可与致谢

- 本仓库原创部分：MIT License，版权归 Yui-0818，见 [`LICENSE`](LICENSE)。
- ConvNeXt：MIT License，版权归 Meta Platforms, Inc. and affiliates，见 [`ConvNeXt/LICENSE`](ConvNeXt/LICENSE) 与 [`NOTICE`](NOTICE)。
- MedMamba：上游未附 LICENSE。本仓库不包含其模型代码，仅保留结果、方法说明与引用；不对该部分重新授权，也不声称拥有其版权。任何再分发或商业使用请先联系原作者。
- 医学图像数据由任课教师提供，版权归原数据提供方；本仓库不分发任何图像。
