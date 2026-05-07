# Content-preserving Warps for 3D Video Stabilization

Python/PyTorch 复现 SIGGRAPH 2009 论文 [Content-Preserving Warps for 3D Video Stabilization](https://people.csail.mit.edu/kapu/papers/Liu_3DvideoStab_2009.pdf)（Liu et al.）。

## 项目结构

```
.
├── cp4stabilizer/          # 核心稳像库
│   ├── colmap.py           # 读取 COLMAP 稀疏重建（cameras/images/points3D）
│   ├── geometry.py         # 投影、单应矩阵等几何运算
│   ├── warp.py             # content-preserving mesh warp 求解器
│   └── paths.py            # 理想相机路径拟合
├── input/                  # 输入数据
│   ├── video/              # 原始视频（如 video8_mini.mp4）
│   ├── frames/             # 抽帧后的帧图像
│   └── sfm/                # COLMAP 稀疏重建导出（cameras.txt / images.txt / points3D.txt）
├── run_stabilization.py    # 主稳像脚本
├── bake_and_stabilize.py   # 将 3D 点投影可视化到帧上，再执行稳像
├── make_mask.py            # 用 Mask R-CNN 生成运动物体遮罩
└── trim_to_minimal.py      # 裁剪 video8 生成最小示例的工具
```

## 环境配置

```bash
conda create -n cp python=3.10
conda activate cp
pip install torch opencv-python numpy scipy imageio tqdm
```

如需使用 `make_mask.py` 的 mask 生成功能，还需安装：

```bash
pip install torchvision
# 可选：SAM 细化遮罩
pip install segment-anything
```

## 输入数据准备

稳像需要三类输入，均放在 `input/` 下：

### 1. 原始视频

```bash
input/video/<name>.mp4
```

### 2. 帧图像

从视频抽帧（JPEG 节省空间）：

```bash
mkdir -p input/frames/<name>
ffmpeg -i input/video/<name>.mp4 -start_number 0 -q:v 92 input/frames/<name>/%04d.jpg
```

### 3. COLMAP 稀疏重建

在 `input/frames/<name>/` 上运行 COLMAP SfM，将稀疏模型导出为文本格式：

```bash
input/sfm/<name>/
├── cameras.txt
├── images.txt
└── points3D.txt
```

> 本项目已附带 video8 的最小示例（90 帧），位于 `input/*/video8_mini/`。

## run_stabilization.py

主稳像脚本，按论文流程执行：加载 SfM → 拟合理想路径 → 生成稀疏约束 → pre-warp → content-preserving mesh warp → 裁剪输出。

### 基础用法

```bash
conda activate cp

python run_stabilization.py \
  --sfm input/sfm/video8_mini \
  --frames input/frames/video8_mini \
  --output output/stabilized \
  --video output/stabilized.mp4
```

### 重要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sfm` | `input/sfm/video8` | COLMAP 导出目录 |
| `--frames` | `input/frames/video8` | 帧图像目录 |
| `--output` | `output/stabilized` | 输出帧目录 |
| `--video` | `None` | 输出视频路径（可选） |
| `--path-mode` | `smooth` | 理想路径模式：`smooth` / `linear` / `quadratic` / `constant` |
| `--rotation-mode` | `None` | 旋转路径模式（默认与 path-mode 一致） |
| `--smooth-sigma` | `24.0` | 高斯平滑 σ（帧数） |
| `--prewarp` | `general` | 预变换类型：`general` / `infinite` / `homography` / `none` |
| `--grid-cols` | `64` | 网格列数 |
| `--grid-rows` | `36` | 网格行数 |
| `--alpha` | `20.0` | 相似性项权重 |
| `--min-track` | `20` | 最小观测次数阈值 |
| `--fade` | `50` | 时序权重渐变帧数 |
| `--crop` | `common` | 裁剪模式：`common`（公共有效区域）/ `none` |
| `--fps` | `30.0` | 输出视频帧率 |
| `--start` / `--end` | `0` / `None` | 处理帧范围（`--end` 为不包含的索引） |
| `--rectify-domain` | `auto` | 是否将畸变帧矫正到 pinhole 域 |
| `--source-points` | `observed` | 约束源点：`observed`（图像观测）/ `projected`（源相机重投影） |
| `--device` | `auto` | 计算设备：`auto` / `cuda` / `cpu` |

### 快速冒烟测试

```bash
python run_stabilization.py \
  --sfm input/sfm/video8_mini \
  --frames input/frames/video8_mini \
  --output output/smoke \
  --start 0 --end 5 \
  --grid-cols 32 --grid-rows 18 \
  --max-points 1200
```

### 最小示例参数

对于较短的视频（如 90 帧），建议调低 `--min-track`、`--fade` 和 `--smooth-sigma`：

```bash
python run_stabilization.py \
  --sfm input/sfm/video8_mini \
  --frames input/frames/video8_mini \
  --output output/stabilized \
  --video output/stabilized.mp4 \
  --min-track 5 --smooth-sigma 8 --fade 15
```

## bake_and_stabilize.py

将 COLMAP 稀疏 3D 点的投影绘制为绿色圆点叠加到原始帧上，然后对带点的帧执行稳像。**用于可视化 3D 点在稳像前后的运动轨迹是否合理。**

### 工作流程

1. 从 COLMAP 中选取 track 最长的 top-N 个 3D 点
2. 将每帧中可见的点投影到图像上，画绿色实心圆
3. 对叠加了绿点的帧调用 `run_stabilization.py` 执行稳像
4. 合成稳像后的视频（带绿点轨迹）

### 用法

```bash
conda activate cp

python bake_and_stabilize.py \
  --sfm input/sfm/video8_mini \
  --frames input/frames/video8_mini \
  --output-frames input/frames/video8_mini_dotted \
  --stabilized-output output/stabilized_dotted \
  --video-output output/video8_stabilized_with_points.mp4 \
  --min-track 90 \
  --max-points 50 \
  --radius 3
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sfm` | `input/sfm/video8` | COLMAP 导出目录 |
| `--frames` | `input/frames/video8` | 原始帧目录 |
| `--output-frames` | `input/frames/video8_dotted` | 叠加绿点后的帧输出目录 |
| `--stabilized-output` | `output/stabilized_dotted` | 稳像结果帧目录 |
| `--video-output` | `output/video8_stabilized_with_points.mp4` | 最终视频路径 |
| `--min-track` | `90` | 点的最小 track 长度（只选取稳定的长 track 点） |
| `--max-points` | `50` | 全局选取的 top-N 点数量 |
| `--radius` | `3` | 绿点半径（像素） |
| `--fps` | `30.0` | 输出视频帧率 |

## make_mask.py

使用 Mask R-CNN（ResNet50-FPN-V2）自动检测视频帧中的**人物**，生成 COLMAP 格式的遮罩（白色=背景有效区域，黑色=需要忽略的运动物体）。可选 SAM 进行遮罩细化。

### 用途

在 COLMAP 特征提取/匹配时加载遮罩，使运动物体（如行人）不参与 SfM 重建和稳像约束，避免运动物体干扰相机位姿估计和 warp 计算。

### 用法

```bash
conda activate cp

python make_mask.py \
  --frames input/frames/video8_mini \
  --output input/masks/video8_mini \
  --device auto
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--frames` | （必填） | 帧图像目录 |
| `--output` | `input/masks/<name>` | 遮罩输出目录 |
| `--device` | `auto` | 推理设备 |
| `--score-threshold` | `0.55` | 人物检测最低置信度 |
| `--fill-box` | `False` | 用整个检测框作为遮罩（更保守） |
| `--dilate` | `13` | 遮罩膨胀核大小（填补漏检） |
| `--close` | `11` | 遮罩闭运算核大小（填补空洞） |
| `--sam-checkpoint` | `""` | SAM 模型权重路径（可选，启用 SAM 细化） |
| `--sam-model` | `vit_h` | SAM 模型类型 |
| `--preview-dir` | （有默认值） | 遮罩预览叠加图输出目录 |
| `--extract-frames` | `False` | 先从视频抽帧再生成遮罩 |
| `--inpaint-dir` | `""` | 输出人物区域 inpainting 后的帧（可选） |

### 遮罩格式说明

- 生成的遮罩为单通道 PNG
- **白色（255）** = COLMAP 会提取特征的有效区域
- **黑色（0）** = COLMAP 忽略的区域（检测到的运动人物）
- `--name-mode both` 会同时生成 `0001.png` 和 `0001.png.png` 两种命名（兼容不同 COLMAP 版本）

## 参考

- Liu, F., Gleicher, M., Jin, H., & Agarwala, A. (2009). Content-preserving warps for 3D video stabilization. *ACM Transactions on Graphics (SIGGRAPH 2009)*.
- 论文中所有默认参数均可在 `run_stabilization.py` 的 `--help` 中查看
