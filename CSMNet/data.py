"""CSM-Net 数据管线。

只做三件事：
1. 擦除图像自带的红色轮廓叠加（不修改原始文件）；
2. 生成荧光素绿色响应图 S，与 RGB 拼成四通道；
3. 训练/验证/测试的几何与颜色增强。

红线擦除的阈值与 512 短边工作分辨率，是在 clean train 上抽查后定的；
不要用 test 图像调这里的参数。
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# clean train 568 张、红线擦除后、224×224 上的逐通道统计（只算 train）
RGB_MEAN = (0.097106, 0.270839, 0.410090)
RGB_STD = (0.126559, 0.237889, 0.309578)

# S = ReLU(G - 0.5(R+B)) 的理论范围约 [0, 1]；乘 4 拉开动态范围
STAIN_SCALE = 4.0

# 红线擦除与“不擦红线”分支共用的短边工作分辨率；以后只改这一处
WORK_SHORT = 512

# 强增强的仿射缩放范围；letterbox 与 RandomResizedCrop 两条分支共用，
# 避免不同 ratio_mode 的对比被增强强度差异污染
STRONG_AFFINE_SCALE = (0.9, 1.1)


def erase_red_overlay(path, work_short: int = WORK_SHORT,
                      dilate: int = 3) -> Image.Image:
    """检测并擦除亮红色细线，返回 RGB PIL 图像。

    红色轮廓在原图中约 2 px 宽；先在全分辨率上检测，再缩到短边 WORK_SHORT 做
    最近邻填充。这样既不会把细线缩没，也不用在 2592×1728 上做距离变换。

    阈值用较宽的红色范围：纯红核心 R>110、R-G>45、R-B>45，同时限制
    G/B<140，避免误伤绿色荧光素。dilate=3 覆盖抗锯齿边缘；在 clean train
    的 359/282 两张图上抽查，擦除后残留红色像素降到 0～1.5e-5。
    """
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = im.size
    scale = work_short / min(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    arr = np.asarray(im)
    # arr 是 uint8，直接相减会按 256 回绕（例如 120-139 得 237）；
    # 差值必须先提升到 int16，且 r-g 与 r-b 两个差值都要提升。
    rgb = arr.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red = ((r > 110) & ((r - g) > 45) & ((r - b) > 45)
           & (g < 140) & (b < 140))
    if not red.any():
        # 没有红线也要缩到统一工作分辨率，否则 cache 会保留 2592×1728 原图
        return im.resize((nw, nh), Image.BILINEAR)

    small = im.resize((nw, nh), Image.BILINEAR)
    m = Image.fromarray((red * 255).astype(np.uint8)).resize(
        (nw, nh), Image.NEAREST)
    mask = np.asarray(m) > 127
    if dilate > 0:
        mask = ndi.binary_dilation(mask, iterations=dilate)

    work = np.asarray(small).astype(np.float32)
    if mask.any():
        # 每个红线像素取最近的“非红线”像素；对细线足够稳，且不引入 cv2
        idx = ndi.distance_transform_edt(
            mask, return_distances=False, return_indices=True)
        fill = work[idx[0], idx[1]]
        work = np.where(mask[..., None], fill, work)
    return Image.fromarray(np.clip(work, 0, 255).astype(np.uint8))


class AddStainChannel:
    """在 ToTensor 之后，把 RGB 变成 [R, G, B, S]。"""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.clamp(x[1] - 0.5 * (x[0] + x[2]), 0.0, 1.0) * STAIN_SCALE
        return torch.cat([x, s.unsqueeze(0)], dim=0)


class NormalizeInput:
    """RGB 用 train 统计归一化；S 保持固定尺度，不再标准化。"""

    def __init__(self, in_ch: int):
        self.in_ch = in_ch

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rgb = TF.normalize(x[:3], RGB_MEAN, RGB_STD)
        if self.in_ch == 3:
            return rgb
        return torch.cat([rgb, x[3:4]], dim=0)


class Letterbox:
    """保比例缩放到方形画布，空白处填黑；全图保留，不变形。"""

    def __init__(self, size: int, fill=(0, 0, 0)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = img.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - nw) // 2,
                               (self.size - nh) // 2))
        return canvas


def build_transforms(res: int, train: bool, input_mode: str = "stain",
                     strong_aug: bool = False, ratio_mode: str = "stretch"):
    """训练增强；ratio_mode 控制几何预处理。

    - stretch：直接拉成方形；全图保留但几何失真（旧口径）。
    - letterbox：最长边缩到 res，再 pad 成方形；全图、不变形，有效分辨率降低。
    - center_crop：验证/测试 Resize(短边 256)+CenterCrop(res)；保比例但裁边。
      这是最终 B/center 配方使用的口径。
    """
    if ratio_mode not in ("stretch", "letterbox", "center_crop"):
        raise ValueError("ratio_mode=%r" % (ratio_mode,))
    ops = []
    if train:
        if ratio_mode == "letterbox":
            # 先 pad 到 224，再做几何增强，避免在 512×768 上做 affine/rotation
            ops += [Letterbox(res), transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.02)]
            if strong_aug:
                ops += [
                    transforms.RandomApply(
                        [transforms.RandomRotation(
                            12, interpolation=InterpolationMode.BICUBIC)], p=0.5),
                    transforms.RandomAffine(
                        degrees=0, translate=(0.05, 0.05),
                        scale=STRONG_AFFINE_SCALE,
                        interpolation=InterpolationMode.BICUBIC, fill=0),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
                ]
        else:
            ops += [
                transforms.RandomResizedCrop(
                    res, scale=(0.65, 1.0) if strong_aug else (0.75, 1.0),
                    ratio=(0.85, 1.18) if strong_aug else (0.9, 1.11),
                    interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            ]
            if strong_aug:
                ops += [
                    transforms.RandomApply(
                        [transforms.RandomRotation(
                            12, interpolation=InterpolationMode.BICUBIC)], p=0.5),
                    transforms.RandomAffine(
                        degrees=0, translate=(0.05, 0.05),
                        scale=STRONG_AFFINE_SCALE,
                        interpolation=InterpolationMode.BICUBIC),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
                ]
    else:
        if ratio_mode == "letterbox":
            ops += [Letterbox(res)]
        elif ratio_mode == "center_crop":
            short = int(round(res * 256 / 224))
            ops += [transforms.Resize(short,
                                      interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(res)]
        else:
            ops += [transforms.Resize(
                (res, res), interpolation=InterpolationMode.BICUBIC)]
    ops += [transforms.ToTensor()]
    if input_mode == "stain":
        ops += [AddStainChannel()]
    ops += [NormalizeInput(4 if input_mode == "stain" else 3)]
    if train and strong_aug:
        ops += [transforms.RandomErasing(
            p=0.2, scale=(0.02, 0.08), ratio=(0.5, 2.0), value=0.0)]
    return transforms.Compose(ops)


class CSMImageFolder(datasets.ImageFolder):
    """按 ImageFolder 顺序读取；可选是否擦除红线。

    ImageFolder 的 samples/targets 顺序与 ConvNeXt 的 clean 折索引一致，
    因此可以直接复用 cv_clean_freeze5/folds/fold*.json。
    """

    def __init__(self, root, transform=None, erase_red: bool = True):
        super().__init__(root, transform=transform)
        self.erase_red = erase_red
        # 缓存擦除后的短边图（WORK_SHORT）；568 张约 0.7 GB，换掉每轮重复检测/填充
        self._cache = {}

    def _load(self, path: str) -> Image.Image:
        if not self.erase_red:
            # 不擦红线时也不要每次读 2592×1728 原图；统一缓存到 WORK_SHORT 短边
            if path not in self._cache:
                im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
                w, h = im.size
                scale = WORK_SHORT / min(w, h)
                self._cache[path] = im.resize(
                    (max(1, int(round(w * scale))),
                     max(1, int(round(h * scale)))), Image.BILINEAR)
            return self._cache[path]
        if path not in self._cache:
            self._cache[path] = erase_red_overlay(path)
        return self._cache[path]

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self._load(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def make_dataset(root, split: str, res: int, train: bool,
                 input_mode: str = "stain", erase_red: bool = True,
                 strong_aug: bool = False, ratio_mode: str = "stretch"):
    transform = build_transforms(
        res, train, input_mode, strong_aug, ratio_mode)
    return CSMImageFolder(
        str(root / split), transform=transform, erase_red=erase_red)
