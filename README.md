# Audio-LLM 情感识别项目

本项目实现了一个基于 Qwen2.5 (0.5B/3B) 和 Emotion2Vec 的多模态 Audio-LLM，用于音频情感识别任务。

## 技术原理：从 LLM 到多模态模型

本项目展示了如何将一个纯文本的大语言模型 (LLM) 扩展为能够理解音频输入的多模态模型 (Audio-LLM)。核心思想是将预训练音频编码器的表示空间对齐到 LLM 的输入嵌入空间。

### 1. 架构设计
代码实现见 [`src/model.py`](src/model.py) 中的 `AudioQwenModel` 类。

*   **LLM Backbone (大脑)**: 我们使用 `Qwen2.5` 作为推理核心。它在第一阶段保持冻结，或在第二阶段使用 LoRA 进行微调，以在适应新任务的同时保留其语言能力。
    *   *代码引用*: `src/model.py` 中 `self.llm = AutoModelForCausalLM.from_pretrained(...)`
*   **Audio Encoder (耳/眼)**: 我们使用 `Emotion2Vec` 从音频中提取高层语义特征。该模型在大规模音频数据上进行了预训练，并在本项目中保持冻结。
    *   *代码引用*: 特征提取逻辑位于 [`scripts/process_chiser5.py`](scripts/process_chiser5.py)
*   **Projector (连接器)**: 一个简单的多层感知机 (MLP) 充当桥梁。它将音频嵌入（维度 $D_{audio}$）投影到 LLM 的输入嵌入空间（维度 $D_{llm}$）。
    *   *代码引用*: `src/model.py` 中的 `self.projector = nn.Sequential(...)`

### 2. 两阶段训练策略
为了有效地融合模态，我们采用了两阶段训练流程。训练逻辑主要位于 [`scripts/train.py`](scripts/train.py)。

*   **Stage 1: 预对齐 (Feature Alignment)**
    *   **目标**: 教会 LLM 将音频特征"看作"是文本 token。
    *   **可训练参数**: 仅 **Projector**。LLM 和 Audio Encoder 被冻结。
    *   **数据**: 音频-文本对。模型学习将音频特征映射到相应的文本嵌入。
    *   **代码实现**: 见 `scripts/train.py` 中 `if args.stage == 1:` 分支，设置 `model.llm.requires_grad_(False)`。

*   **Stage 2: 指令微调 (Instruction Tuning)**
    *   **目标**: 基于对齐后的特征，微调模型以执行特定任务（如情感识别）。
    *   **可训练参数**: **Projector** + **LLM (LoRA)**。
    *   **数据**: 指令跟随数据（例如："分析这段音频的情感..."）。模型学习推理音频内容并生成结构化回复。
    *   **代码实现**: 见 `scripts/train.py` 中 `if args.stage == 2:` 分支，加载 Stage 1 的 Projector 权重，并应用 LoRA (`PeftModel`)。

## 项目结构

```
Audio-LLM/
├── README.md               # 项目说明文档
├── src/                    # 核心源码包
│   ├── __init__.py
│   ├── config.py           # 配置文件 (路径等)
│   ├── model.py            # AudioQwenModel 模型定义
│   └── dataset.py          # 数据集加载逻辑
├── scripts/                # 执行脚本
│   ├── process_chiser5.py  # ChiSER5 数据预处理脚本
│   ├── train.py            # 训练脚本 (支持 Stage 1 & 2)
│   ├── merge.py            # 模型合并脚本
│   ├── inference.py        # 推理脚本
│   └── evaluate.py         # 评估脚本
├── data/                   # 数据目录
│   └── processed/          # 预处理后的特征和 JSON 数据
├── output/                 # 输出目录
    ├── results/        # 训练检查点
    └── audio_qwen_merged/  # 最终合并的模型
```

## 环境安装

1. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
   (注意: 确保安装了 `torch`, `transformers`, `peft`, `modelscope`, `datasets`, `soundfile`, `librosa`)

2. 配置路径:
   在 `src/config.py` 中修改模型和数据路径（如果需要）。

## 使用指南

### 1. 数据预处理
使用 Emotion2Vec 提取音频特征并准备数据集。
```bash
python scripts/process_chiser5.py
```
这将生成 `data/processed/train.json` 和 `data/processed/test.json`。

### 2. 训练 (Training)

**Stage 1: 预对齐**
训练 Projector 以将音频特征与冻结的 LLM 对齐。
```bash
python scripts/train.py --stage 1 --epochs 5
```
检查点将保存到 `output/checkpoints/stage1`。

**Stage 2: 指令微调**
微调 LLM (LoRA) 和 Projector。
```bash
python scripts/train.py --stage 2 --checkpoint output/checkpoints/stage1/checkpoint-XXX --epochs 5
```
检查点将保存到 `output/checkpoints/stage2`。

### 3. 模型合并 (Merging)
将 LoRA 权重和 Projector 合并到一个模型中，以便于推理。
```bash
python scripts/merge.py --checkpoint output/checkpoints/stage2/checkpoint-XXX --output_dir output/audio_qwen_merged
```

### 4. 推理 (Inference)
对单个音频文件进行推理。
```bash
python scripts/inference.py --model_path output/audio_qwen_merged --audio_path path/to/audio.wav
```

### 5. 评估 (Evaluation)
在测试集上评估模型准确率。
```bash
python scripts/evaluate.py --model_path output/audio_qwen_merged
```

## 模型架构
- **Audio Encoder**: Emotion2Vec (冻结)
- **Projector**: MLP (可训练)
- **LLM**: Qwen2.5 (LoRA 微调)

简单来说，**并没有变成一个单一的“黑盒”模型文件**。目前的架构是 **“逻辑上一体，物理上分离”** 的状态。

具体来说，最终的系统由以下 **三个部分** 组成，它们的存储和加载方式如下：

### 1. 语音编码器 (Audio Encoder) - **完全独立**
*   **模型**: `Emotion2Vec`
*   **状态**: **独立存储，未合并**。
*   **原因**: 这个模型在训练过程中是**冻结 (Frozen)** 的，参数没有发生任何变化。因此，我们不需要把它保存到新的模型文件夹中，推理时直接加载原始的预训练模型即可。这样做可以节省存储空间（不需要复制一份没变过的参数）。
*   **代码体现**: inference.py 中单独调用 `pipeline(..., model=EMOTION2VEC_MODEL_PATH)`。

### 2. 大语言模型 (LLM) - **已合并 (Base + LoRA)**
*   **模型**: `Qwen2.5` (Base) + `LoRA` (Adapter)
*   **状态**: **已合并**。
*   **说明**: 在 merge.py 中，我们将训练好的 LoRA 权重（原本是外挂的补丁）**永久合并**到了 Qwen 的基础权重中。
*   **结果**: audio_qwen_merged 文件夹中包含的是一套完整的、参数已更新的 LLM 权重。加载时不需要再挂载 LoRA。

### 3. 投影层 (Projector) - **跟随 LLM 存储**
*   **模型**: 一个简单的 MLP (多层感知机)
*   **状态**: **独立文件，但与 LLM 存放在一起**。
*   **说明**: 这是一个连接层，它不是 LLM 原生结构的一部分，无法直接“合并”进 Transformer 的权重矩阵中。
*   **结果**: 它被保存为 `projector.pt` 文件，放置在 audio_qwen_merged 目录下。
*   **代码体现**: inference.py 会检测模型目录下是否有 `projector.pt` 并加载它。

---

### 总结：最终的 audio_qwen_merged 文件夹里有什么？

它包含了两部分核心资产：
1.  **LLM 权重**: (合并了 LoRA 后的 Qwen，可以直接用 `AutoModelForCausalLM` 加载)
2.  **Projector 权重**: (`projector.pt`，需要用代码显式加载到 `model.projector`)

**推理流程图解：**

```mermaid
graph LR
    A[音频文件] --> B(Emotion2Vec <br/> 独立模型/不在此文件夹)
    B --> C[音频特征]
    C --> D(Projector <br/> projector.pt)
    D --> E[对齐后的特征]
    E --> F(Qwen LLM <br/> merged weights)
    F --> G[文本输出]
```

所以，虽然物理上不是一个文件，但在使用时，您只需要指向 audio_qwen_merged 这个目录，脚本会自动加载 LLM 和 Projector，而 Audio Encoder 则指向通用的预训练路径。