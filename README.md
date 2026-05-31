# baby_gpt

从零实现一个可以运行的小型语言模型，目标是循序渐进掌握底层原理。

当前阶段先实现 NumPy 版字符级 bigram 语言模型：

- 字符级 tokenizer
- 文本切分与 batch 采样
- bigram 概率模型
- 负对数似然训练
- 自回归文本生成

## 快速运行

```bash
python scripts/check_env.py
python scripts/train_bigram.py
python scripts/sample.py --prompt "模型"
uv --cache-dir /tmp/uv-cache run --extra dev pytest
```

如果要进入 PyTorch/CUDA 阶段，先安装带 CUDA 支持的 PyTorch，再运行：

```bash
python scripts/check_env.py
```

在 WSL2 中，`nvidia-smi` 可用只说明系统能看到 GPU；Python 训练代码还需要安装 CUDA 版 PyTorch，`torch.cuda.is_available()` 才会返回 `True`。

## 计划路线

1. 字符级 bigram 模型，跑通语言模型闭环
2. NumPy 版 embedding、softmax、优化器
3. 单独实现 causal self-attention
4. 组合最小 GPT block
5. 切换到 PyTorch，加速训练并支持 CUDA
6. 实现 byte-level / BPE tokenizer
