import numpy as np


class BigramLanguageModel:
    """可训练的 bigram 语言模型，参数是 logits[vocab, vocab]。"""

    def __init__(self, vocab_size: int, rng: np.random.Generator):
        self.vocab_size = vocab_size
        # logits[i, j] 表示：当前 token 是 i 时，下一个 token 是 j 的未归一化分数。
        # 初始值很小且随机，让每个转移概率一开始接近均匀分布，但又不是完全相同。
        self.logits = 0.01 * rng.standard_normal((vocab_size, vocab_size))

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        # x 是一批当前 token id，形状通常是 [batch]。
        # self.logits[x] 会取出每个当前 token 对应的那一行，得到 [batch, vocab]。
        logits = self.logits[x]

        # softmax 前先减去每行最大值，避免 exp(很大的数) 导致数值溢出。
        # 这不会改变 softmax 的最终概率，只是让计算更稳定。
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        probs = self.probabilities(x)
        n = len(x)

        # probs[np.arange(n), y] 取出每个样本中“正确下一个 token”的预测概率。
        # loss 是平均负对数似然：正确 token 概率越高，loss 越低。
        # 1e-12 防止概率极小时 log(0)。
        return float(-np.log(probs[np.arange(n), y] + 1e-12).mean())

    def train_step(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        # 前向传播：根据当前 token x，得到对下一个 token 的概率分布。
        probs = self.probabilities(x)
        n = len(x)
        loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())

        # softmax + cross entropy 的梯度有一个简洁形式：
        # grad = 预测概率分布 - 真实 one-hot 分布。
        # 这里直接复用 probs 作为梯度矩阵，形状是 [batch, vocab]。
        grad = probs
        grad[np.arange(n), y] -= 1.0

        # loss 取的是 batch 平均值，所以梯度也要除以样本数。
        grad /= n

        # dlogits 是整张参数表的梯度，形状和 self.logits 一样。
        # 每个样本只会更新“当前 token 对应的那一行”。
        dlogits = np.zeros_like(self.logits)

        # 一个 batch 里可能多次出现同一个当前 token。
        # np.add.at 会把这些样本的梯度安全地累加到同一行上。
        np.add.at(dlogits, x, grad)

        # 梯度下降：沿着让 loss 变小的方向更新参数。
        self.logits -= learning_rate * dlogits
        return loss

    def generate(self, start_id: int, max_new_tokens: int, rng: np.random.Generator) -> list[int]:
        ids = [start_id]
        current = start_id
        for _ in range(max_new_tokens):
            # bigram 只看当前一个 token，然后预测下一个 token 的概率分布。
            probs = self.probabilities(np.array([current]))[0]

            # 按模型给出的概率随机采样下一个 token，再把它作为新的 current。
            current = int(rng.choice(self.vocab_size, p=probs))
            ids.append(current)
        return ids
