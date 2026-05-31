# 当前工程学习导览：从 bigram 到 GPT

这个项目的目标不是一开始就堆出完整 GPT，而是先把语言模型最核心的闭环拆到足够小：

1. 文本如何变成 token id
2. 模型如何根据当前 token 预测下一个 token
3. loss 如何衡量预测错误
4. 梯度下降如何更新参数
5. 生成时如何把预测结果再喂回模型

当前代码实现的是字符级 bigram 语言模型。它还没有 attention、Transformer block、position embedding，也还不是严格意义上的 GPT；但它已经包含了 GPT 训练和生成的骨架。

## 先跑通

```bash
python scripts/train_bigram.py
python scripts/sample.py --prompt "模型"
uv --cache-dir /tmp/uv-cache run --extra dev pytest
```

你看到的训练日志里：

- `train_loss` 是模型在训练 batch 上的平均负对数似然。
- `val_loss` 是模型在验证 batch 上的同类指标。
- loss 下降代表模型越来越会猜“某个字符后面常出现什么字符”。

## 文件地图

建议按这个顺序读：

1. `data/raw/tiny.txt`
   训练语料。现在只有几行中文，所以模型只能学到很局部的字符转移规律。

2. `src/baby_gpt/tokenizer.py`
   字符级 tokenizer。
   它把语料里出现过的每个不同字符映射成整数 id。

3. `src/baby_gpt/dataset.py`
   数据加载、编码、训练/验证切分、随机 batch 采样。
   对 bigram 来说，一个训练样本就是 `(当前字符, 下一个字符)`。

4. `src/baby_gpt/model/bigram.py`
   当前最核心的模型文件。
   参数是一个 `logits[vocab_size, vocab_size]` 矩阵。
   第 `i` 行表示“当前 token 是 i 时，下一个 token 是每个词表 token 的未归一化分数”。

5. `src/baby_gpt/train.py`
   训练循环。
   它不断采样 batch，调用 `model.train_step(...)`，定期算验证 loss。

6. `scripts/sample.py`
   生成入口。
   它先训练一个 bigram 模型，再从 prompt 最后一个字符开始持续采样新字符。

## 一条样本如何流过系统

假设文本里有一句：

```text
语言模型
```

tokenizer 可能把它编码成：

```text
语 -> 10
言 -> 8
模 -> 5
型 -> 3
```

bigram 训练样本就是相邻字符对：

```text
(语, 言)
(言, 模)
(模, 型)
```

在代码里，`random_bigram_batch` 做的就是随机选一些位置：

```python
x = ids[starts]
y = ids[starts + 1]
```

这里 `x` 是当前 token，`y` 是正确的下一个 token。

## 模型到底学了什么

`BigramLanguageModel` 的全部可训练参数是：

```python
self.logits = 0.01 * rng.standard_normal((vocab_size, vocab_size))
```

如果 `vocab_size = 100`，那么参数矩阵就是 `100 x 100`。

第 `i` 行只负责一种情况：

```text
当前 token = i 时，下一个 token 的分布
```

例如：

```text
logits[语_id] = [下一个是 A 的分数, 下一个是 B 的分数, ..., 下一个是 言 的分数, ...]
```

`probabilities` 用 softmax 把分数变成概率：

```python
exp = np.exp(shifted)
return exp / exp.sum(axis=-1, keepdims=True)
```

softmax 的作用是：

- 输入可以是任意实数分数。
- 输出每一项都大于 0。
- 所有输出加起来等于 1。
- 分数越高，概率越大。

## loss 是什么

训练时模型会给每个可能的下一个 token 一个概率。

如果真实答案是 `言`，模型给 `言` 的概率越高，loss 越低；概率越低，loss 越高。

代码：

```python
return float(-np.log(probs[np.arange(n), y] + 1e-12).mean())
```

这就是平均负对数似然：

```text
loss = -log(模型给正确答案的概率)
```

几个直觉数值：

- 正确 token 概率是 `1.0`，loss 是 `0`
- 正确 token 概率是 `0.5`，loss 约 `0.69`
- 正确 token 概率是 `0.1`，loss 约 `2.30`
- 正确 token 概率越接近 `0`，loss 越大

## train_step 在做什么

`train_step` 是这个项目最值得反复读的函数。

它做四件事：

1. 前向计算概率：

   ```python
   probs = self.probabilities(x)
   ```

2. 计算 loss：

   ```python
   loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())
   ```

3. 计算 softmax + cross entropy 的梯度：

   ```python
   grad = probs
   grad[np.arange(n), y] -= 1.0
   grad /= n
   ```

   这行的含义很重要：模型当前预测的概率分布 `probs`，减去真实答案的 one-hot 分布。

   如果真实答案位置概率太低，那个位置的梯度会是负数，参数更新后对应 logit 会升高。
   如果错误答案位置概率太高，那个位置的梯度会是正数，参数更新后对应 logit 会降低。

4. 把 batch 梯度累积回整张 `logits` 表：

   ```python
   np.add.at(dlogits, x, grad)
   self.logits -= learning_rate * dlogits
   ```

`np.add.at` 是因为一个 batch 里可能多次出现同一个当前 token。它会把这些样本对同一行 logits 的梯度累加起来。

## 生成为什么叫自回归

生成代码在 `generate`：

```python
ids = [start_id]
current = start_id
for _ in range(max_new_tokens):
    probs = self.probabilities(np.array([current]))[0]
    current = int(rng.choice(self.vocab_size, p=probs))
    ids.append(current)
```

流程是：

1. 已知当前 token
2. 预测下一个 token 的概率分布
3. 按概率随机采样一个 token
4. 把采样出的 token 当成新的当前 token
5. 重复

GPT 生成也是这个思路，只是 GPT 看的不是“当前一个 token”，而是“一段上下文 token”。

## bigram 和 GPT 的对应关系

当前 bigram：

```text
P(next_token | current_token)
```

真正 GPT：

```text
P(next_token | previous_tokens)
```

也就是说，bigram 只看 1 个 token 的历史；GPT 会看一个上下文窗口，比如 128、1024、8192 个 token。

当前工程未来路线可以这样理解：

1. bigram logits 表
   只学习“一个字符后面常接什么字符”。

2. embedding
   把 token id 映射成向量，而不是直接用表格行。

3. position embedding
   让模型知道 token 在序列里的位置。

4. causal self-attention
   让每个位置能读取它前面的 token 信息，但不能偷看未来。

5. MLP
   对每个位置的表示做非线性变换。

6. Transformer block
   attention + MLP + residual + normalization。

7. GPT
   多层 Transformer block 后面接一个 vocab 投影，输出下一个 token 的 logits。

## 你应该重点掌握的 5 个问题

读完当前代码后，能回答下面 5 个问题，就说明你已经掌握当前阶段：

1. 为什么 tokenizer 必须同时支持 `encode` 和 `decode`？
2. 为什么 bigram 训练样本是 `ids[i] -> ids[i + 1]`？
3. `logits[x]` 为什么能拿到当前 batch 的预测分数？
4. 为什么 loss 只取 `probs[np.arange(n), y]` 这些位置？
5. 生成时为什么只需要一个 `current`，而 GPT 需要一个上下文序列？

## 推荐实验

按顺序做这些小实验，比单纯读代码更容易掌握。

### 实验 1：观察 tokenizer

在 Python 里运行：

```python
from baby_gpt.dataset import encode_file

ids, tokenizer = encode_file("data/raw/tiny.txt")
print(tokenizer.chars)
print(tokenizer.vocab_size)
print(ids[:20])
print(tokenizer.decode(ids[:20].tolist()))
```

目标：确认文本和 token id 可以互相转换。

### 实验 2：看一个 batch

```python
import numpy as np
from baby_gpt.dataset import encode_file, random_bigram_batch

ids, tokenizer = encode_file("data/raw/tiny.txt")
x, y = random_bigram_batch(ids, batch_size=8, rng=np.random.default_rng(0))
print(x)
print(y)
print([tokenizer.decode([int(a), int(b)]) for a, b in zip(x, y)])
```

目标：确认 batch 中每个样本都是相邻字符对。

### 实验 3：看 loss 下降

把 `src/baby_gpt/config.py` 里的：

```python
steps: int = 500
```

临时改成 `20`、`100`、`1000` 分别跑：

```bash
python scripts/train_bigram.py
```

目标：观察训练步数对 loss 和生成文本的影响。

### 实验 4：改变学习率

把 `learning_rate` 分别改成：

```python
0.01
0.1
0.8
5.0
```

目标：理解学习率太小会慢，太大可能不稳定。

### 实验 5：扩大语料

往 `data/raw/tiny.txt` 里加入更多同风格文本，再跑：

```bash
python scripts/sample.py --prompt "模型"
```

目标：理解语言模型不是凭空懂语言，而是从数据统计里学规律。

## 当前阶段的边界

这个模型生成效果会很有限，这是预期结果。

原因是：

- 语料非常小。
- tokenizer 是字符级，不是 BPE 或 byte-level。
- bigram 只能看当前一个字符。
- 没有 attention，所以不能建模长距离依赖。
- 没有深层网络，所以表达能力很弱。

但这些限制正好适合学习。你现在应该先掌握闭环，而不是急着追求生成质量。

