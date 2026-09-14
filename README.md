# Qwen2.5-VL-7B LoRA Object Detection

[简体中文](README.md) | [English](README_EN.md)

使用 LoRA 将 `Qwen/Qwen2.5-VL-7B-Instruct` 微调为自定义类别目标检测器。项目提供从数据校验、训练、断点续训、最佳 checkpoint 选择，到测试集评估、预测可视化和单图推理的完整流程。

类别没有写死在代码中。修改一个 JSON 配置即可训练任意类别组合，无需改 Python 文件。

![Qwen2.5-VL LoRA detection workflow](assets/workflow.svg)

## 效果示意

`evaluate.py` 会输出带类别名称和边界框的检测可视化，实际效果与下图形式相同。下图来源于网络，仅用于说明预期的可视化形式，不是本项目模型在公开测试集上的实测结果，也不代表本项目的检测精度。

![网络目标检测效果参考图](assets/example.jpg)


## 功能

- Qwen2.5-VL-7B-Instruct + PEFT LoRA 微调
- 任意数量、任意名称的自定义类别
- 像素坐标格式 `[x1, y1, x2, y2]`
- 仅对 assistant 输出计算训练损失
- 按验证集 `eval_loss` 自动选择最佳 checkpoint
- Early stopping 与自动断点续训
- IoU 匹配、每类 Precision/Recall/F1、Micro F1 和 Macro F1
- Ground Truth 与 Prediction 并排可视化
- Windows/Linux NVIDIA GPU 和 Google Colab 均可运行

## 项目结构

```text
.
├── assets/
│   ├── detection_demo.svg
│   ├── example.jpg
│   └── workflow.svg
├── configs/
│   └── example.json
├── data/
│   ├── .gitkeep
│   └── example_train.json
├── outputs/
│   └── .gitkeep
├── scripts/
│   ├── common.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── predict.py
│   ├── train.py
│   └── validate_dataset.py
├── .gitignore
├── LICENSE
├── README.md
├── README_EN.md
└── requirements.txt
```

`train.py` 内聚训练专用的数据集和 collator；`common.py` 统一处理配置、路径、随机种子及推理模型加载；`metrics.py` 集中管理预测解析、IoU 匹配、指标统计和可视化；其余文件是职责单一的命令行入口。

## 环境要求

- Python 3.10 或 3.11
- Windows 或 Linux
- 支持 CUDA 的 NVIDIA GPU
- 推荐 24 GB 以上显存；默认配置面向高显存 GPU

显存不足时优先把 `train_batch_size` 调为 `1`，减小 `model_max_length`，并保持或提高 `gradient_accumulation_steps`。本项目默认使用 BF16；不支持 BF16 的 GPU 需在配置中将 `bf16` 改为 `false`、`fp16` 改为 `true`。

macOS 用户可以在本机完成代码编辑、配置修改和数据集校验，但当前训练代码依赖 CUDA，不能直接使用 Apple Silicon 或 Intel Mac 的 GPU 训练 7B 模型。Mac 用户可以连接 Linux/Windows NVIDIA 工作站，或使用 Google Colab 等云端 GPU 运行训练。

## 使用 Anaconda 安装

先安装 Anaconda 或 Miniconda，然后克隆仓库：

```bash
git clone https://github.com/LeonBytes/Qwen-LoRA.git
cd Qwen-LoRA
```

在 Windows 的 Anaconda Prompt、PowerShell，或者 macOS/Linux 的终端中创建独立环境：

```bash
conda create -n qwen-detection python=3.10 -y
conda activate qwen-detection
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows 和 macOS/Linux 使用相同的 `conda create` 与 `conda activate` 命令，不需要执行 `source .venv/bin/activate`。以后每次打开新的终端，只需进入项目目录并运行 `conda activate qwen-detection`。

Windows 或 Linux 用户在开始训练前，应确认安装的 PyTorch 能识别 NVIDIA GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA GPU not found')"
```

第一项应输出 `True`。如果输出 `False`，需要根据电脑的 NVIDIA 驱动和 CUDA 环境安装对应的 PyTorch CUDA 版本，然后再运行训练。

首次运行时，Transformers 会从 Hugging Face 下载基础模型。也可以提前下载模型，并在命令中通过 `--model-path /path/to/Qwen2.5-VL-7B-Instruct` 指定本地目录。

## 数据集准备

### 1. 目录布局

创建以下结构：

```text
data/
├── train.json
├── validation.json
├── test.json
└── images/
    ├── train/
    ├── validation/
    └── test/
```

三个 JSON 文件分别引用各自图片目录中的文件。建议按约 80% / 10% / 10% 划分训练、验证和测试集。同一来源的近似图片不要跨集合分配，以免数据泄漏。

#### 图片尺寸建议

建议在制作标注前，将训练集、验证集和测试集的所有图片统一处理为 `1280 × 1280`。对于非正方形原图，推荐保持原始宽高比缩放，再在空白区域进行填充（letterbox），避免直接拉伸造成目标形状失真。

边界框坐标必须对应处理后的 `1280 × 1280` 图片。如果图片和标注已经完成，之后又改变了图片尺寸，必须同步换算或重新生成所有边界框。训练、验证、测试和实际推理应采用相同的图片预处理方式。本项目不会在读取数据时自动把图片改成 `1280 × 1280`，请在创建数据集时提前完成处理。

### 2. 标注格式

每个 JSON 文件的根节点是数组，每个样本包含：

- `image`：图片文件名或相对路径
- `conversations`：固定包含一条 `human` 消息和一条 `gpt` 消息
- `human.value`：以 `<image>\n` 开头，后面是目标检测指令
- `gpt.value`：JSON 字符串，内容是类别到边界框列表的映射

下面的两张示例图片都包含多个类别，并且每个类别都有多个实例：

```json
[
  {
    "image": "example_001.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nLocate all class_a and class_b objects in this image and output their bbox coordinates in JSON format."
      },
      {
        "from": "gpt",
        "value": "{\"class_a\":[[35,48,180,240],[225,60,390,275],[430,95,610,320]],\"class_b\":[[70,310,245,465],[330,345,560,510]]}"
      }
    ]
  },
  {
    "image": "example_002.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nLocate all class_a and class_b objects in this image and output their bbox coordinates in JSON format."
      },
      {
        "from": "gpt",
        "value": "{\"class_a\":[[22,30,160,190],[190,45,350,225]],\"class_b\":[[55,250,210,410],[260,270,420,440],[455,300,620,500]]}"
      }
    ]
  }
]
```

边界框必须使用原图像素坐标：

```text
(x1, y1) ---------
   |              |
   |              |
   --------- (x2, y2)
```

必须满足 `0 <= x1 < x2 <= 图片宽度` 和 `0 <= y1 < y2 <= 图片高度`。`gpt.value` 本身是字符串，因此内部 JSON 的双引号需要转义为 `\"`。某个类别在图片中不存在时仍建议保留对应键，并使用空列表 `[]`。

### 3. 修改类别配置

复制配置文件：

```bash
cp configs/example.json configs/my_dataset.json
```

编辑以下字段：

```json
{
  "class_names": ["class_a", "class_b"],
  "prompt": "Locate all class_a and class_b objects in this image and output their bbox coordinates in JSON format."
}
```

`class_names`、所有 `gpt.value` 内部 JSON 的键、配置文件的 `prompt`，以及每条 `human.value` 中的类别名称必须一致。训练读取各样本的 `human.value`；评估和单图推理读取配置文件的 `prompt`，因此两处指令应保持一致。

### 4. 校验数据

```bash
python scripts/validate_dataset.py \
  --config configs/my_dataset.json \
  --data-dir data
```

校验器会检查 JSON 结构、类别名称、图片是否存在、图片是否可读取以及边界框是否越界。出现任何错误时返回非零退出码；建议在训练前始终运行。

## 训练

```bash
python scripts/train.py \
  --config configs/my_dataset.json \
  --data-dir data \
  --output-dir outputs/my_experiment
```

输出结构：

```text
outputs/my_experiment/
├── checkpoint-*/
├── best_model/
├── logs/
├── run_config.json
├── trainer_state.json
└── training_results.json
```

`checkpoint-*` 是可续训的完整训练状态，`best_model` 是根据最低验证损失选出的 LoRA adapter 和 processor。测试集不参与模型选择。

训练中断后，重新执行完全相同的命令会自动从最新完整 checkpoint 恢复。若希望忽略已有 checkpoint 并从头开始，应换一个新的输出目录；也可以显式传入 `--no-resume`，但不要在包含旧结果的目录中混合不同实验。

临时覆盖训练步数：

```bash
python scripts/train.py \
  --config configs/my_dataset.json \
  --data-dir data \
  --output-dir outputs/my_experiment \
  --max-steps 500
```

查看 TensorBoard：

```bash
tensorboard --logdir outputs/my_experiment/logs
```

## 测试集评估

```bash
python scripts/evaluate.py \
  --config configs/my_dataset.json \
  --data-dir data \
  --output-dir outputs/my_experiment
```

结果写入：

```text
outputs/my_experiment/test_evaluation/
├── evaluation_report.json
├── predictions.json
└── visualizations/
```

评估采用每个类别内的一对一 IoU 匹配。默认 IoU 阈值为 `0.5`，可在配置中修改。这里的 F1 是固定 IoU 阈值下基于生成框计算的指标，不等同于 COCO mAP。

如果只需要数值结果、不需要生成图片：

```bash
python scripts/evaluate.py \
  --config configs/my_dataset.json \
  --data-dir data \
  --output-dir outputs/my_experiment \
  --skip-visualizations
```

## 单图推理

```bash
python scripts/predict.py path/to/image.jpg \
  --config configs/my_dataset.json \
  --output-dir outputs/my_experiment
```

输出示例：

```json
{
  "cat": [[34.0, 52.0, 280.0, 410.0]],
  "dog": []
}
```

使用其他 adapter 或本地基础模型：

```bash
python scripts/predict.py path/to/image.jpg \
  --config configs/my_dataset.json \
  --adapter-path path/to/adapter \
  --model-path path/to/base-model
```

## Google Colab

将仓库和数据放到 Google Drive，例如 `/content/drive/MyDrive/qwen-lora-detection`，然后依次运行：

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content/drive/MyDrive/qwen-lora-detection
!pip install -r requirements.txt
!python scripts/validate_dataset.py --config configs/my_dataset.json --data-dir data
!python scripts/train.py --config configs/my_dataset.json --data-dir data --output-dir outputs/my_experiment
!python scripts/evaluate.py --config configs/my_dataset.json --data-dir data --output-dir outputs/my_experiment
```

安装依赖后若 Colab 提示重启运行时，请重启、重新挂载 Drive，再执行训练。输出保存在 Drive 中，因此 runtime 断开后 checkpoint 不会丢失。

## 配置说明

| 字段 | 含义 |
|---|---|
| `model_id` | Hugging Face 基础模型 ID |
| `class_names` | 检测类别列表 |
| `prompt` | 训练与推理使用的统一指令 |
| `iou_threshold` | 评估时判定匹配的 IoU 阈值 |
| `max_new_tokens` | 推理最大生成 token 数 |
| `lora.target_modules` | 注入 LoRA 的线性层 |
| `lora.r` | LoRA rank |
| `lora.alpha` | LoRA scaling 参数 |
| `lora.dropout` | LoRA dropout |
| `training.max_steps` | optimizer step 总数 |
| `training.train_batch_size` | 单设备训练 micro-batch |
| `training.gradient_accumulation_steps` | 梯度累积次数 |
| `training.eval_steps` | 验证间隔 |
| `training.save_steps` | checkpoint 保存间隔 |
| `training.early_stopping_patience` | 连续多少次验证无改善后早停 |

有效 batch size 为 `GPU 数量 × train_batch_size × gradient_accumulation_steps`。

## 常见问题

### CUDA out of memory

将 `train_batch_size` 降为 `1`，减小 `model_max_length`，或缩小输入图片。降低 LoRA rank 主要减少可训练参数和 optimizer 状态，但基础 7B 模型本身仍需显存。

### 模型输出不是合法 JSON

确保训练和推理使用同一 prompt，标注答案只含 JSON，且不同样本的键顺序和格式尽量一致。增加干净样本通常比单纯增加训练步数更有效。

### 预测坐标越界

数据校验器会拒绝训练标注中的越界框。生成结果仍可能越界；评估可视化会裁剪绘制范围，但模型输出文件保留原始预测，便于排查。

### 想增加或删除类别

修改 `class_names`、`prompt` 和三个标注 JSON，然后使用新的输出目录重新训练。旧 adapter 的输出空间已经由旧数据决定，不建议直接复用。

## 开源与模型许可

本仓库代码使用 MIT License。Qwen2.5-VL 模型权重不包含在本仓库中，使用者需另外遵守模型发布方的许可证和使用条款。训练数据也不包含在仓库中，使用者应确保自己拥有数据及标注的使用和发布权限。
