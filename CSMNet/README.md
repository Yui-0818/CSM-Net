# CSM-Net（实验一，第三种模型，自建 CNN）

CSM-Net 是从零训练的纯卷积三分类网络，不加载 ImageNet、DermaMNIST 或任何预训练权重，也不引用 MedMamba/ConvNeXt 的模型定义。最终主候选是简化版 B + center crop：clean 3 折 best-val 宏 F1 0.6811 ± 0.0080，OOF 宏 F1 0.6808，clean test acc 0.7254、宏 F1 0.6649、类 2 F1 0.5333。clean test 只有 142 张，center 对 B 的 test 宏 F1 差 +0.0145，95% CI [−0.0688, +0.0993]，跨 0；center 是 CV 明确赢家、test 点估计最好，但不是统计上稳定的胜出。

## 1. 设计依据

用 ConvNeXt clean freeze5 的三个 `best.pth` 在各自 val 折上做 OOF（不读 test）：

- OOF acc 0.8574，宏 F1 0.7926；混淆矩阵（行=真值，列=预测）`[[271,9,6],[12,170,27],[2,25,46]]`。
- 2 类 F1 0.6053：25 张判成 1、2 张判成 0；1 类有 27 张判成 2。**1↔2 共 52 个错误，是主战场。**
- 0 类召回 271/286 = 94.8%，已经很高。

clean train 的只读统计：

- 红色轮廓叠加：0 类 29/286（10%），1 类 204/209（98%），2 类 71/73（97%）。红线来源未标注，不能当 ground truth；但“有没有红线”对 0 vs 病的 AUC≈0.98。直接用原图，模型很可能学红线而不是病灶。
- 绿色高响应面积均值：0 类 3.8%、1 类 6.2%、2 类 7.7%，单调但 1 vs 2 的简单形态单特征 AUC 大多只有 0.56–0.59。1↔2 不能靠面积解决，必须靠纹理、边界和多尺度。

设计纪律：以上全部来自 clean train；新模型的 test 只在配方冻结后评估。此前核对上一版结果时看过 clean test 的既有预测，但没有用 test 选新模型的结构或超参。

## 2. 结构

```text
原图 → 去红线 → [RGB, S] → Stem → S1 → S2 → S3 → S4
                                   ↑S门  ↑S门  ↑S门  ↑S门（可选）
S2/S3/S4 → avg+max (+stain/boundary) → 主头 3类
                                      ├─ 辅助头 0 vs 病（可选）
                                      └─ 辅助头 1 vs 2（可选）
```

- 输入：4 通道，RGB + `S = ReLU(G - 0.5(R+B)) × 4`。
- 主干：4 个 stage 的 DRSB。每个 block 并行 depthwise 3×3 / 5×5，1×1 融合，SE 通道重标定，残差 + drop-path；Stain Gate 为可关的空间门控。
- 池化：stage2/3/4 的 avg+max；stage4 可再加 stain 加权与 boundary 加权（`∇S × S`，抑制眼睑/睫毛边缘）。
- 分类头：主头 3 类 CE；可选辅助头 A 做 0 vs 病，辅助头 B 做 1 vs 2（只在病样本上算，`pos_weight` 偏向 2 类）。推理只用主头，不做级联，避免 0 类假阳性被放大。
- **最终结构 B**：保留去红线、S 输入、DRSB、多尺度池化与 3 类主头；关闭 Stain Gate 与辅助头，head 维度 128，参数量 2.036M。
- **完整结构 D**：打开 Stain Gate 与辅助头；head128 时 2.118M，原始 120 epoch 版本 head256 时为 2.365M。

## 3. 数据管线

- 红线擦除：全分辨率检测 `R>110, R-G>45, R-B>45, G/B<140`，短边缩到 512，dilate 3，最近邻填充。原始文件不改；不引入 cv2 或新依赖。clean train 568 张中，擦除后残留红色像素 >1e-5 的只有 12 张，最大 2.5e-5。
- RGB 归一化：clean train 568 张、红线擦除后、224×224 上的统计，`mean=[0.097106,0.270839,0.410090]`，`std=[0.126559,0.237889,0.309578]`；不用 ImageNet 的 mean/std。
- 训练增强（最终强增强）：RandomResizedCrop(224, scale 0.65–1.0, ratio 0.85–1.18)、水平翻转、ColorJitter(0.2,0.2,0.2,0.02)、RandomRotation(12°, p=0.5)、RandomAffine(translate 0.05, scale 0.9–1.1)、GaussianBlur(p=0.2)、RandomErasing(p=0.2)。
- 验证/测试：最终 center crop 口径为 `Resize(短边 256) + CenterCrop(224)`；旧口径 `stretch` 直接 Resize 到 224×224，`letterbox` 保比例 pad 到 224×224。
- center crop 在 train 上平均保留约 81.4% 的绿色高响应面积，72.9% 的图丢失 >10%；它用少量边缘信息换比例不变，CV 上优于 stretch 与 letterbox。
- 折划分：`StratifiedKFold(shuffle=True, random_state=0)`；3 折与 ConvNeXt clean 3 折索引一致；5 折为训练集内重划。脚本未给 `--folds-dir` 时按同一规则现场生成，同一数据顺序与 seed 下与历史折 JSON 逐索引一致。

## 4. 训练协议

最终 center 配方的 3 折参数：

| 项 | 值 |
|---|---|
| 输入 | 224×224，4 通道 |
| batch size | 32 |
| epochs | 90 |
| optimizer | AdamW，lr 3e-4，weight decay 0.1 |
| schedule | 线性 warmup 3 + cosine |
| label smoothing | 0.1 |
| drop-path | 0.1 |
| head | 128 维，dropout 0.4 |
| 结构 | B：无 Stain Gate、无辅助头 |
| 增强 | 强增强（见第 3 节） |
| EMA | 0.999；raw/EMA 同时评估，按 val macro F1 取较高者 |
| AMP / grad clip | 开启 / 1.0 |
| 主指标 | macro F1 |

对照配方：

- B（stretch）：同上，但几何预处理用 `stretch`。
- D-90：同上，但打开 Stain Gate 与辅助头。
- D-120（原始完整模型）：120 epoch，warmup 5，wd 0.05，ls 0.05，drop-path 0.05，head256，stretch，Stain Gate + 辅助头全开，2.365M 参数。

所有模型共用同一 clean 训练集与同一评估口径；没有加载预训练权重。

## 5. 结果

### 5.1 3 折

| 模型 | fold0 | fold1 | fold2 | mean ± std | 末轮 mean ± std |
|---|---:|---:|---:|---:|---:|
| D-120（原始完整） | 0.6159 | 0.6167 | 0.6201 | 0.6176 ± 0.0022 | 0.5644 ± 0.0211 |
| D-90（强配方完整） | 0.6463 | 0.6080 | 0.6186 | 0.6243 ± 0.0197 | 0.5615 ± 0.0129 |
| B（简化，stretch） | 0.6786 | 0.6184 | 0.6805 | 0.6592 ± 0.0353 | 0.6362 ± 0.0368 |
| **center（简化，center crop）** | 0.6901 | 0.6783 | 0.6748 | **0.6811 ± 0.0080** | **0.6454 ± 0.0410** |

- center 相对 B：均值 +0.0219，方差明显更小；主要提升类 0/类 1，类 2 基本持平。
- D-120 的 best 出现得早、last 明显更低；强增强 + 90 epoch 把过拟合间隙压小，但 D-90 的 3 折 best 仍低于 B。
- 所有 best 都是 raw 或 EMA 中 val macro F1 较高者；P1 短跑里 EMA 未收敛，正式训练后两者接近。

### 5.2 5 折 center

| fold0 | fold1 | fold2 | fold3 | fold4 | mean ± std |
|---:|---:|---:|---:|---:|---:|
| 0.6599 | 0.7371 | 0.6854 | 0.6781 | 0.7241 | **0.6969 ± 0.0324** |

last 均值 0.6424 ± 0.0305。5 折每折 fit 454–455（约 80%），3 折每折 fit 378–379；训练数据量不同，不能把 0.6969 直接当成 3 折的替代。

### 5.3 OOF

| 模型 | acc | macro F1 | 逐类 F1 | 混淆矩阵（行=真值） |
|---|---:|---:|---|---|
| D-120 | 0.6884 | 0.6202 | 0.7868 / 0.6606 / 0.4133 | `[[214,53,19],[36,146,27],[8,34,31]]` |
| D-90 | 0.6919 | 0.6255 | 0.7957 / 0.6483 / 0.4324 | `[[220,55,11],[36,141,32],[11,30,32]]` |
| B | 0.7165 | 0.6587 | 0.815 / 0.6833 / 0.4780 | `[[218,51,17],[27,151,31],[4,31,38]]` |
| **center** | **0.7535** | **0.6808** | **0.8663 / 0.6945 / 0.4815** | `[[256,16,14],[40,133,36],[9,25,39]]` |

center 的 OOF 宏 F1 比 B 高 0.0221，比 D-120 高 0.0606；类 2 F1 与 B 基本持平（0.4815 对 0.4780），提升主要来自类 0/类 1。

### 5.4 clean test

3 折 best 的 softmax 概率平均，n=142：

| 模型 | acc | macro F1 | 加权 F1 | κ | 类 0 / 1 / 2 F1 |
|---|---:|---:|---:|---:|---|
| D-120 | 0.7113 | 0.5532 | 0.6909 | 0.4951 | 0.8514 / 0.6545 / 0.1538 |
| B | 0.7113 | 0.6508 | 0.7196 | 0.5264 | 0.8182 / 0.6726 / 0.4615 |
| **center** | **0.7254** | **0.6649** | **0.7221** | **0.5430** | **0.8477 / 0.6136 / 0.5333** |

混淆矩阵：

```text
D-120:  [[63  9  0]      B:  [[54 15  3]      center:  [[64  5  3]
         [10 36  6]           [ 5 38  9]                [13 27 12]
         [ 3 13  2]]          [ 1  8  9]]               [ 2  4 12]]
```

配对 bootstrap（同一 clean test，20k）：

| 差值 | 点估计 | 95% CI |
|---|---:|---|
| center − B，macro F1 | +0.0145 | [−0.0688, +0.0993] |
| center − B，类 2 F1 | +0.0732 | [−0.0957, +0.2535] |
| center − D-120，macro F1 | +0.1115 | [−0.0054, +0.2174] |
| center − D-120，类 2 F1 | +0.3799 | [+0.0957, +0.6216] |
| B − D-120，macro F1 | +0.0975 | [−0.0166, +0.2027] |
| B − D-120，类 2 F1 | +0.3062 | [+0.0278, +0.5571] |

**注意：D 的 test 结果来自 D-120，而 B/center 是 90 epoch 强增强配方。** B−D、center−D 的 test 差异同时包含结构、轮数、正则与几何预处理的变化，不能全部归因于结构。center−B 是同一 90 epoch 配方下只改几何预处理/门的对比，更干净，但 CI 仍跨 0。

### 5.5 过拟合对照（fold0）

统一用“强增强 + 强正则”做对照；H2/H3 各 3 seeds，H0/H1 为单 seed。

| 配置 | best val macro F1 | last val macro F1 | 过拟合间隙（best−last） |
|---|---:|---:|---:|
| H0：60ep，弱增强 | 0.6186 | 0.5288 | 0.0898 |
| H1：60ep，强增强，head256 | 0.6498 | 0.5959 | 0.0539 |
| H2：60ep，强增强，head128 | 0.6356 ± 0.0108 | 0.5970 ± 0.0201 | 0.0385 |
| H3：90ep，强增强，head128 | 0.6566 ± 0.0186 | 0.6236 ± 0.0196 | 0.0330 |

强增强把 last 拉回、间隙缩小；90ep 的 fold0 单折均值最高。但 H3 的 3 折 CV 只有 0.6243 ± 0.0197（last 0.5615），对比原 D-120 的 0.6176 ± 0.0022（last 0.5644）：best 只高约 0.007，last 还略低。**“强增强 + 90ep”本身没有实质提升 3 折 best，只是把过拟合间隙压小；单 fold/单 seed 的 0.65–0.67 是乐观估计。**

### 5.6 A/B/C/D 消融（fold0，统一 90ep 强增强）

| 模型 | 输入 | 擦红线 | S 门 | 辅助头 | 参数量 | best macro F1 | last macro F1 | 类 1 F1 | 类 2 F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | RGB | 否 | 否 | 否 | 2.035 M | 0.6595 | 0.6299 | 0.7299 | 0.3043 |
| B | RGB+S | 是 | 否 | 否 | 2.036 M | **0.6674** | **0.6371** | 0.7333 | 0.4286 |
| C | RGB+S | 是 | 是 | 否 | 2.118 M | 0.6644 | 0.6349 | 0.7105 | 0.4400 |
| D | RGB+S | 是 | 是 | 是 | 2.118 M | 0.6463 | 0.5702 | 0.6126 | 0.4898 |

- **A→B**：去红线 + S 输入，macro F1 从 0.6595 到 0.6674，类 2 从 0.304 到 0.429；没有观察到“去掉红线就掉分”。
- **B→C**：加 Stain Gate/边界池化，macro F1 0.6674→0.6644，类 2 0.429→0.440；没有稳定的净收益，属于可删模块。
- **C→D**：加辅助头，macro F1 0.6644→0.6463，last 0.6349→0.5702，类 1 F1 0.711→0.613；类 2 单折 F1 0.440→0.490，但 OOF 类 2 反而 B>D（0.478 对 0.432）。辅助头把决策推向类 2，牺牲类 1 和整体宏 F1。
- A/C 只有单 seed fold0，B/D 有 3 折；上表是筛选证据，不是最终统计。

### 5.7 长宽比与 center crop

fold0 筛选（B 配方，90ep，强增强）：

| 模式 | best macro F1（fold0） | 说明 |
|---|---|---|
| stretch | 0.6674（1 seed） | 旧口径，全图，x 方向压 1.5× |
| letterbox | 0.6581（1 seed） | 全图不变形，但内容只占 224×149，有效分辨率降低 |
| center crop | 0.7106 / 0.6803 / 0.7057（3 seeds，均值 0.6989） | 保比例 + 裁边；train 平均只保留 81.4% 绿色面积，72.9% 的图丢 >10% |

3 折确认：

| 模型 | fold0 | fold1 | fold2 | mean ± std |
|---|---:|---:|---:|---:|
| stretch B | 0.6786 | 0.6184 | 0.6805 | 0.6592 ± 0.0353 |
| center crop | 0.6901 | 0.6783 | 0.6748 | **0.6811 ± 0.0080** |

center 的 3 折均值 +0.0219，方差小得多；last 均值 0.6454 对 B 0.6362。结论：**“保比例”本身不是收益来源；center crop 的裁边 + 更高有效分辨率才是。letterbox 保了全图但降低分辨率，没有优势。**

### 5.8 2 类特化尝试

- **focal loss**（center crop，fold0）：macro 0.7049、2 类 F1 0.5000；不如 center 普通 CE 的 0.7106/0.5581。它主要抬类 1，不是 2 类解药。
- **OOF class2 logit bias**：扫 −2…+2，最佳就是 0（不加偏置）；加正 bias 后 macro 从 0.6808 掉到 0.6592（+0.25）甚至更低。
- **1-vs-2 逻辑回归专家**（center OOF 128-D 特征，5 折）：cascade macro 0.6138、类 2 0.3709，低于 base 的 0.6808/0.4815；失败。
- 结论：当前数据和 B/center 结构下，**focal、阈值校准、1-vs-2 专家都没有解决 2 类**；2 类仍是瓶颈。

## 6. 局限

- clean test 只有 142 张，类 2 只有 18 张；center−B 的 test 宏 F1 CI 跨 0，不能宣称稳定胜出。
- 没有病人 ID，3 折/5 折都是随机分层划分。64-bit dHash 近邻审计中，每折 val 存在 train 邻居（Hamming ≤4）的数量为 32/114、28/114、26/114、26/113、28/113，约 23–28%；dHash 很粗，不能证明同一只眼泄漏，但也无法声称完成 patient-level split。
- test 累计读取 9 次：D 主集成 1、D 单折 3、B 主集成 1、B 单折 3、center 主集成 1。后续任何 test 数字都属于重复使用，报告中必须披露。
- D 的 test 结果来自 D-120；B/center 的 test 结果来自 90 epoch 强配方，跨配方比较有混杂。
- 5 折比 3 折每折多约 13% 训练数据，5 折数字不能直接替代 3 折。

## 7. 复现

数据布局：

```text
Experiment1/
├─ code/
├─ data_clean/
│  ├─ train/0/...
│  ├─ train/1/...
│  ├─ train/2/...
│  └─ test/0/...
└─ weights/            # CSM-Net 不需要预训练权重
```

最终 center 3 折：

```powershell
cd Experiment1/code

python CSMNet/train.py cv --folds 3 --folds-seed 0 --epochs 90 --warmup 3 `
  --bs 32 --workers 4 --strong-aug --head-dim 128 --head-dropout 0.4 `
  --wd 0.1 --ls 0.1 --ratio-mode center_crop --no-stain-gate --no-aux `
  --data-root ../data_clean `
  --out-dir run/csmnet/center_cv_90_strong
```

5 折复跑只需把 `--folds 3` 改为 `--folds 5`，输出换一个目录。B 的 stretch 版本把 `--ratio-mode center_crop` 改为 `--ratio-mode stretch`；D 打开 `--stain-gate --aux`。

clean test 一次性评估：

```powershell
python CSMNet/train.py eval --checkpoints `
  run/csmnet/center_cv_90_strong/fold0/best.pth `
  run/csmnet/center_cv_90_strong/fold1/best.pth `
  run/csmnet/center_cv_90_strong/fold2/best.pth `
  --ratio-mode center_crop --no-stain-gate --no-aux `
  --data-root ../data_clean `
  --out-dir run/csmnet/eval_center_final
```

未给 `--folds-dir` 时，脚本按 `StratifiedKFold(shuffle=True, random_state=--folds-seed)` 现场生成折；给了 `--folds-dir` 但文件不存在会直接报错，避免静默换折。

## 8. 文件与许可

| 文件 | 内容 |
|---|---|
| `data.py` | 红线擦除、S 通道、stretch/letterbox/center_crop、强增强 |
| `model.py` | DRSB、SE、可选 Stain Gate/辅助头、多尺度池化 |
| `train.py` | smoke / fold / cv / eval；默认只留 `best.pth` 与 CSV |
| `README.md` | 本文件 |

本目录代码与文档由本实验编写，按仓库根目录 MIT License 发布；版权归 Yui-0818。数据、权重、图像与逐样本预测 CSV 不随仓库分发。
