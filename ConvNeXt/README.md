# ConvNeXt-T 本地化适配（实验一，第二种模型）

状态：原始 569/143 数据上的 3 折 CV 与测试集一次评估已完成；去重后的 `data_clean`（568/142）也已完成 3 折 CV、正则对照与最终 test 一次评估。最终 clean 配方为 freeze5（先冻结 backbone 5 轮），test acc 0.8662、宏 F1 0.7683。旧数字与新数字分开记录，不混用。

本次实验只使用预训练迁移，不跑随机初始化（scratch），也不做初始化方式对照；第二种模型的定位是与 MedMamba 的迁移基线比较。

## 1. 选型理由

第二种模型要在同一份三分类数据上充当 MedMamba 的纯 CNN 对照。选 ConvNeXt-T 的原因：

- 结构上是标准 CNN（depthwise conv + 1×1 conv + LayerNorm），没有注意力或 SSM，和 MedMamba 的 SSM+Conv 形成明确对照；
- ImageNet-1K top-1 82.1%，28M 参数、4.5G FLOPs，RTX 4060 Laptop 8GB 可以全量微调；
- 上游以 MIT 发布，代码量小，`models/convnext.py` 可以原样保留，适配成本低；
- 预训练权重有公开直链，不依赖额外账号或网盘。

## 2. 上游与保留内容

| 项 | 值 |
|---|---|
| 上游 | `facebookresearch/ConvNeXt` |
| 地址 | https://github.com/facebookresearch/ConvNeXt |
| 基线 commit | `048efcea897d999aed302f2639b6270aedf8d4c8` |
| 许可证 | MIT，见 `LICENSE` |
| 论文 | Liu et al., *A ConvNet for the 2020s*, CVPR 2022 |

只保留 `models/convnext.py`（模型定义，原样未改）和上游 `LICENSE`。删除的内容包括：ImageNet 训练脚本、目标检测、语义分割、isotropic 变体、submitit 分布式启动、wandb 集成。训练/验证/测试流程没有沿用上游 `main.py`，而是按本实验的统一协议重写，原因见第 3 节。

## 3. 本地适配

### 3.1 与 MedMamba 对齐的部分

- 数据：默认 `--data-root Experiment1/data_clean`（train 568 / test 142，已去重）；数据不随仓库分发，需按 `ImageFolder` 结构自行放置。旧的 raw 数据在 `Experiment1/data`（569/143），旧结果只作参考。
- 阶段一划分：`random_split`，`split_seed=0`，`val_frac=0.2`；raw 数据为 fit 456 / val 113，clean 数据为 fit 455 / val 113。
- CV 划分：`StratifiedKFold(n_splits=3, shuffle=True, random_state=0)`；raw 为 fit 379/379/380、val 190/190/189；clean 为 fit 378/379/379、val 190/189/189。折索引由数据决定，clean 结果不与 raw 逐折配对。
- 损失：按 fit 集类别计数加权的交叉熵，标签平滑 0.1。
- 指标：准确率、宏 F1、逐类 F1、混淆矩阵；宏 F1 为主指标。
- 测试集：训练与 CV 默认不读 `data/test`；只有显式执行 `stage1 --test` 或 `eval` 子命令才评估。主流程只用 CV 做配方选择，最终模型再读一次 test。

### 3.2 ConvNeXt 的训练配方

| 项 | 默认值 |
|---|---|
| 输入 | 224×224，RGB，ImageNet mean/std |
| batch size | 16 |
| epochs | 30 |
| optimizer | AdamW，lr 1e-4，weight decay 0.05 |
| schedule | cosine，warmup 0 |
| label smoothing | 0.1 |
| drop path | 0.1 |
| AMP | 开启 |
| grad clip | 1.0 |
| 训练增强 | RandomResizedCrop(224, scale 0.7–1.0)、水平翻转、ColorJitter(0.2,0.2,0.2,0.05) |
| 验证/测试 | 直接 Resize 到 224×224 |

最终 clean 配方在默认值上只改 `--freeze-epochs 5`，即前 5 轮只训分类头，之后全量微调；其余超参不变。

说明：

- 原图 2592×1728（3:2），验证/测试直接拉伸到 224×224；MedMamba 因显存限制使用 128×128，两者都采用同一套原图直接缩放流程，但分辨率不同，报告中需注明这一差异，不能把性能差全部归因于网络结构。训练时的 RandomResizedCrop 比例范围是 0.75–1.333。后续如果要做分辨率消融，再单独加 288/320 的对照，不混进主表。
- 本次只跑 `--arm transfer`；`--arm scratch` 仅保留为代码能力，不纳入本次对照。`--freeze-epochs N` 可以先冻结 backbone 只训分类头，再解冻全量微调；`--head-lr-mult` 控制分类头学习率倍数。默认都是关闭/1.0，先跑全量微调基线。
- 每个 epoch 保存 `epN.pth` 与 `epN_opt.pth`，支持 `epochs.csv` 断点续跑与 `STOP` 文件优雅暂停；`best.pth` 按验证宏 F1 选，`last.pth` 供集成/复查。
- 训练不写 `data/test`；`stage1 --test` 或 `eval` 才会读取，且 `eval` 只接受显式 checkpoint，避免调参期间反复读测试集。

### 3.3 代码自检

已完成：`py_compile` 语法检查；CPU 上用随机张量做前向，输出 `(2, 3)`，参数量 27.82M（三分类头）；用实际下载的 109.1 MB 权重核对：182 个张量、`model` 字段、1000 类分类头被剥离，180 个 body 参数加载成功，三分类 head 重新初始化。raw 数据 3 折 CV 末轮 val 宏 F1 0.7359±0.0354，best-val 0.7619±0.0262；3 折 `last.pth` 概率平均的 test acc 0.8811、宏 F1 0.8225，类 2 F1 0.649。clean 数据又完成基线、drop-path 0.2、freeze5 三组 3 折 CV，最终 freeze5 的 test 一次评估见 3.5。

### 3.4 当前结果（raw 数据，未去重）

| 项 | 值 |
|---|---|
| 3 折 CV 末轮 val 宏 F1 | 0.7359 ± 0.0354 |
| 3 折 CV best-val 宏 F1 | 0.7619 ± 0.0262 |
| 3 折集成 test acc | 0.8811 |
| 3 折集成 test 宏 F1 | 0.8225 |
| test 类 0 / 1 / 2 F1 | 0.951 / 0.868 / 0.649 |

注意：raw 测试集含已知跨集重复 `test/1/494.jpg` ↔ `train/1/552.jpg`，且 `train/1/552.jpg` 在 fold1/fold2 的 fit 中。上表是未去重结果，可能略偏高；后续优化与最终评估以 `data_clean` 为准。

### 3.5 最终结果（data_clean，去重）

3 折 CV（`StratifiedKFold`，seed 0；fit 378/379/379，val 190/189/189）：

| 配方 | 末轮 val 宏 F1 | best-val 宏 F1 | 末轮 val acc |
|---|---:|---:|---:|
| clean 基线 | 0.7463 ± 0.0302 | 0.7716 ± 0.0440 | 0.8398 |
| drop-path 0.2 | 0.7585 ± 0.0309 | 0.7856 ± 0.0376 | 0.8362 |
| **freeze5（最终）** | **0.7839 ± 0.0213** | **0.7918 ± 0.0280** | **0.8503** |

freeze5 相对基线：best-val 均值 +0.0201，逐折 +0.0647 / +0.0176 / −0.0219，2/3 折提升；末轮 val acc 高于 0.8398。按预设规则（≥2/3 折 best 提升、均值 > 0.7716、末轮 val acc ≥ 0.8398），freeze5 被采纳为最终 clean 配方；fold2 单折 best 下降 0.0219，增益并非全折一致。

最终 clean test（n=142，3 折 `best.pth` 的 softmax 概率平均，只评估一次）：

| 指标 | 值 |
|---|---:|
| acc | **0.8662**（95% CI 0.8005–0.9126） |
| 宏 F1 | **0.7683**（95% CI 0.6803–0.8487） |
| 加权 F1 | 0.8629 |
| Cohen's κ | 0.7731 |
| 类 0 / 1 / 2 F1 | 0.958 / 0.862 / 0.485 |

混淆矩阵（行=真实）：

```text
[[68  2  2]
 [ 0 47  5]
 [ 2  8  8]]
```

19 个错误中 10 个为类 2 漏判，类 2 仍是主要短板；clean test 的宏 F1 比 CV best 低 0.0235，落在 142 张样本的波动范围内。clean 结果低于 raw 的 0.8811/0.8225，但 raw 含跨集重复，主表以 clean 为准。结果 CSV 在本地 `eval_clean_freeze5/`，图像保存在课程实验报告目录，均不随仓库分发。

## 4. 预训练权重

已下载 109.1 MB 的 ImageNet-1K 权重，SHA-256 为：

```text
14F3164E3EA6AC32AB3F574F528CE817696C9176FAD4221E0A77A905A7360595
```

文件位置（相对 `Experiment1` 根目录）：

```text
weights/ConvNeXt/convnext_tiny_1k_224_ema.pth
```

`train.py` 默认按 `<code>/../weights/ConvNeXt/convnext_tiny_1k_224_ema.pth` 解析；放在别处时用 `--pretrained` 指定，不需要改代码。来源为：

```text
https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
```

可选的 ImageNet-22K 预训练版本是 `convnext_tiny_22k_1k_224.pth`，本次不使用。

权重不入 Git；`Experiment1/weights/` 位于仓库外。

## 5. 文件说明

| 文件 | 内容 |
|---|---|
| `models/convnext.py` | 上游 ConvNeXt 模型定义，原样保留；含 tiny/small/base/large/xlarge 五个工厂函数 |
| `models/__init__.py` | 只导出上述五个工厂函数 |
| `train.py` | 本实验重写的训练/验证/测试脚本，子命令 `stage1`、`cv`、`eval` |
| `LICENSE` | 上游 MIT 许可证 |
| `README.md` | 本文件 |

`train.py` 的三个入口：

- `stage1`：单次 fit/val 训练；默认 `--arm transfer`，读预训练权重；`--test` 才会在训练结束后读 `data/test`。
- `cv`：3 折分层 CV，逐折训练并写 `folds/foldK.json`、`cv_summary.csv`；不读 `data/test`。
- `eval`：对单个 checkpoint，或多个 checkpoint 做 softmax 概率平均后，在 `data/test` 上评估一次，写 `eval_test.csv` / `eval_test_ensemble.csv`，并写逐样本 `test_predictions*.csv`（路径、标签、预测、三类概率）。

## 6. 实验状态

- [x] 上游模型文件与 LICENSE 入库
- [x] 统一协议的 `train.py`（stage1 / cv / eval）
- [x] 语法、CPU 前向、合成权重加载自检
- [x] 下载 `convnext_tiny_1k_224_ema.pth` 并用实际权重核对键名
- [x] raw 数据 smoke（1 个 step，峰值 2.44GB）
- [x] raw 数据 3 折 CV（末轮 0.7359±0.0354，best 0.7619±0.0262）
- [x] raw 数据 3 折集成 test（acc 0.8811，宏 F1 0.8225）
- [x] 去重数据 `data_clean`（train 568 / test 142）
- [x] `data_clean` 基线 CV（含 val_loss 记录）
- [x] 过拟合/正则对照（drop-path 0.2、freeze5）
- [x] 最终 clean 模型冻结（freeze5）并用 3 折 `best.pth` 读一次 clean test
- [x] 课程提交固化包已生成（本模型 3 个 fold 的 `best.pth` 已纳入；包内共 4 个权重、51 个 CSV；不随仓库分发）

结果表见 3.4 与 3.5；clean 数据结果不与 raw 数字混用。本模型已完成，不再调参。

## 7. 引用与许可

```bibtex
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022},
}
```

`models/convnext.py` 与 `LICENSE` 的版权归 Meta Platforms, Inc. and affiliates，按 MIT 许可证使用。`train.py` 与本文档由本实验编写，按仓库根目录 MIT 许可证发布。上游模型结构与论文贡献归原作者。
