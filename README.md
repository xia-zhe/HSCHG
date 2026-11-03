# HSCHG — V999-repair-v1

本仓库保存当前完成训练的 **V999-repair-v1**。模型以冻结 ImageBind 特征为输入，结合视听异构图和 Lorentz 蕴含约束，并通过受限残差保留 T=0.1 基线的表示。这里发布的是修复版，不是论文原始公式的完整复现。

## 模型

```text
冻结 ImageBind 视听特征
       ├── 冻结的 T=0.1 基线时序层 ──────────── 参照表示
       └── 可训练时序层 → 四层视听图 ──────── 受限修正
                         ↓                     ↓
                    视频节点           参照表示 + 受限修正
                         └── 双曲蕴含监督 ──────┤
                                        官方文本匹配头
                                              ↓
                                    每个片段的事件类别/other
```

图保留三方向时序聚合、双阈值跨模态门控，以及片段—视频双向更新。音频和视觉各使用一层可训练 Transformer；另有一份冻结的基线时序参照。时序修正与图修正的范数各不超过参照范数的 0.1。图输出映射零初始化，初始 eval 预测等于基线。

双曲分支使用共享映射，文本、视频、片段半径分别固定为 0.5、1、1.5，曲率为 −1。共享映射满足 `||W-I||₂ < 0.1`。分类仍使用原官方余弦乘积、ReLU、平方根和类别 min–max 预测头，训练温度为 0.1。蕴含权重为 0.1；主网络与几何映射分别按 0.8 裁剪梯度。

## 当前权重的验证性能

同一次 seed123 实验，validation 使用完整 68 候选；seen/unseen 使用相同完整词表预测，再按原顺序分组。F1 沿用官方 batch32 聚合规则。

| 验证集合 | Accuracy | Segment F1 | Event F1 | Avg |
|---|---:|---:|---:|---:|
| Seen | 72.9800 | 62.6688 | 55.7255 | 63.7914 |
| Unseen | 67.4343 | 57.3437 | 49.4426 | 58.0735 |
| Total | 69.0135 | 58.9995 | 51.4110 | **59.8080** |

数值单位为百分比。T=0.1 基线 validation total avg 为 59.8487%，当前版低 0.0407 个百分点。当前权重未做 test 评估。

训练以 T=0.1 基线第 1 轮权重初始化，保留冻结参照，再训练 1 epoch（412 次更新）。基线先前也训练 412 次更新，因此条件训练预算为 824 次更新。不能把本结果视为与 fresh-init 模型同预算的从头训练比较。

## 文件结构

```text
hschg/
  model.py             受限修正、共享几何映射、蕴含损失与梯度裁剪
  graph.py             四层异构图及消息传递
  graph_config.py      图的固定参数
  geometry.py          Lorentz 映射与锥损失
  temporal.py          与 OV-AVEL 权重兼容的时序模块
  heads.py             官方分类头
  network.py           完整模型与独立权重加载
  backbone.py          外部 ImageBind 在线音频编码适配
  data.py              训练/验证缓存清单读取与转换
  train.py             一个 epoch 的训练与断点恢复
  evaluate.py          完整缓存评估
  predict.py           单批缓存特征推理
  metrics.py           原顺序 seen/unseen/total 聚合
  official_metrics.py  官方 Segment/Event F1 定义
```

仓库不包含数据、预缓存、训练日志、实验报告或模型二进制。权重和运行输出请放在仓库之外。

## 安装

验证环境：Python 3.10、PyTorch 2.0.1+cu118、NumPy 1.24.4。缓存推理只需 PyTorch 和 NumPy；在线音频训练还需要官方依赖。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24.4
```

在线音频编码另外安装 `timm==0.4.5`、`einops==0.6.1`、`iopath==0.1.10`、`fvcore==0.1.5.post20221221`、`ftfy==6.1.1`、`regex==2023.6.3`、`pytorchvideo==0.1.5`、`scipy==1.10.1` 和 `Pillow==9.5.0`。运行前用 `CUDA_VISIBLE_DEVICES` 选择空闲物理 GPU；程序不会替用户挑选机器，也不包含原工作区的 Windows/服务器管理脚本。

## 预训练权重

当前版本单独保留两个文件：

- `hschg-v999-repair-v1.pt`：约 252.6 MB，包含完整推理所需的冻结参照、可训练时序层、图、几何映射、类别文本和配置，不依赖历史基线 checkpoint。
- `training-state.pt`：约 698.8 MB，原始训练状态的逐字节备份，包括优化器、调度器、随机状态与身份记录；用于完整归档。它保留原研究工作区格式，独立仓库使用前一个文件加载模型。

这些文件由仓库所有者单独提供；当前源码提交不捆绑权重下载。可移植权重 SHA-256：

```text
9bc7bdc69d4e7eb65ca7ae21de5e1e87b90a2930a7a47cb4efe7a32943a73f7d
```

只加载可信的 PyTorch 文件。缓存推理不需要 ImageBind 骨干权重，原始媒体特征提取和在线训练音频编码需要官方 `imagebind_huge.pth`。本轮所用骨干文件 SHA-256：

```text
d6f6c22bedcc90708448d5d2fbb7b2db9c73f505dc89bd0b2e09b23af1b62157
```

## 推理

```python
import torch
from hschg import load_model

model, checkpoint = load_model('/path/to/hschg-v999-repair-v1.pt', device='cpu')
# audio, vision: FP32 [B,10,1024]，位于冻结 ImageBind 后、时序层前。
# text: FP32 [C,1024]；candidate_ids 与 text 行顺序一致，包含 'other'。
with torch.no_grad():
    output = model(audio, vision, text, candidate_ids, candidate_ids.index('other'))
    prediction = output['scores'].argmax(-1)  # [B,10]
```

`text` 必须来自同一个冻结 ImageBind 文本编码器。训练使用 46 seen + other；标准验证与测试使用 67 前景 + other。候选顺序可以改变，背景索引必须随之改变。标签不进入推理。

对于包含 `audio`、`vision`、`text`、`candidate_ids` 的单批 `.pt` 文件：

```bash
python -m hschg.predict --checkpoint /path/to/hschg-v999-repair-v1.pt \
  --features /path/to/features.pt --device cpu
```

## 复用现有缓存

训练继续使用视觉特征、文本与 Mel 输入缓存，音频特征在线计算。验证复用冻结 AV/text 特征，时序和图输出实时计算。不得把训练音频编码改成固定音频特征而声称训练路径相同。

对于包含当前研究版缓存索引的 OVAVEL 工作目录，可生成轻量清单，不复制或修改缓存：

```bash
python -m hschg.data --project /path/to/OVAVEL --output /path/to/manifests
```

普通用户也可以按下述格式自行准备清单，所有路径相对于清单文件或使用绝对路径。

训练 `train.json`：

```json
{
  "format": "hschg-training-v1", "split": "train",
  "candidate_ids": ["seen class 1", "...", "other"],
  "entries": [{
    "video": "video_id", "mel": "mel.npy", "vision": "vision.pt",
    "mel_sha256": "file checksum", "vision_sha256": "file checksum",
    "target": [0, 0, 46, 46, 46, 46, 46, 46, 46, 46]
  }]
}
```

`mel.npy` 是 FP32 `[10,1,128,204]`（也接受外包一维的 `[1,10,1,128,204]`）；`vision.pt` 是 FP32 `[10,1024]`。标签列对应完整 seen 词表。

评估 `val.json`：

```json
{
  "format": "hschg-evaluation-v1", "split": "val",
  "candidate_ids": ["class 1", "...", "other"],
  "batches": [{
    "path": "batch_0000.pt", "sha256": "file checksum",
    "videos": ["video_id"], "types": ["close"]
  }]
}
```

每批 `.pt` 包含 `audio[B,10,1024]`、`vision[B,10,1024]`、`text[C,1024]`、`labels`（首项为 `[B,10]` 类别标签）、`videos`。按原验证视频顺序划分 batch32，末批保留；`types` 使用 `close`/`open`。标准实验要求原始完整 5798 个验证视频、68 候选，转换器默认只读取 val。

```bash
python -m hschg.evaluate --checkpoint /path/to/hschg-v999-repair-v1.pt \
  --manifest /path/to/manifests/val.json --device cpu
```

## 训练

在线音频适配读取外部 [OV-AVEL 官方源码](https://github.com/jasongief/OV-AVEL)。本轮固定上游 commit 为 `b5fe1d685d0c6d0d6fd80312b5ccde79f9b73ea6`，适配器检查关键源文件哈希。上游代码、数据和 ImageBind 骨干放在本仓库之外；下载与使用遵循上游项目说明。

```bash
git clone https://github.com/jasongief/OV-AVEL.git /path/to/OV-AVEL-upstream
git -C /path/to/OV-AVEL-upstream checkout b5fe1d685d0c6d0d6fd80312b5ccde79f9b73ea6

CUDA_VISIBLE_DEVICES=0 python -m hschg.train \
  --checkpoint /path/to/hschg-v999-repair-v1.pt \
  --train-manifest /path/to/manifests/train.json \
  --val-manifest /path/to/manifests/val.json \
  --imagebind-source /path/to/OV-AVEL-upstream/proposed_method/ImageBind-main \
  --imagebind-weights /path/to/imagebind_huge.pth \
  --output /path/to/experiment --device cuda:0 --workers 4
```

该命令从权重包内的冻结 T=0.1 参照重新初始化修复模块，并非接续已训练修复权重。固定 seed123、Adam `5e-5`、batch32、1 epoch/412 更新、原时序 dropout0.1、FP32、几何标量 FP64、独立梯度裁剪0.8、MultiStepLR `[10,20,30]`/0.5。4 worker、每 worker 1 线程、prefetch1。首次正式运行还包含冻结音频骨干加载时间。

每 10 批打印进度，每 25 批保存 `last.pt`；停止信号在本批提交后生效。使用相同命令追加 `--resume` 恢复下一批；验证中断后重新执行完整无随机性的缓存验证。恢复不会重放已提交的训练批次。程序不会自动重试失败训练。完成后保存独立 `model.pt`，所有输出均在指定的外部目录。

## 来源与许可

本版本基于 [OV-AVEL](https://github.com/jasongief/OV-AVEL) 的时序模块、分类头与指标，并使用 [ImageBind](https://github.com/facebookresearch/ImageBind) 冻结表示。时序模块的上游代码还注明借鉴 `yujiangpu20/PEL4VAD`。仓库保留所使用上游源码目录附带的 `LICENSE`；上游代码和模型的原有许可与归属保持有效。本 README 不额外授予第三方数据或权重的使用权限。

仓库只实现当前修复版，不包含历史版本、日志或其他实验方法。
