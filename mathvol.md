```{=latex}
\makemathcoverpage
\tableofcontents
\clearpage
```

## 前言：为什么需要这本附册

《算法的低语》主本用 7 章篇幅勾勒了大语言模型背后的数学骨架与产业现实——上半场从信息论的香农熵、嵌入空间的高维几何、Transformer 的矩阵微分、非凸优化的鞍点逃逸，讲到涌现现象的渗流相变隐喻；下半场则把镜头拉远，看这套数学如何级联出万亿美元的算力供应链（第六章）、以及 Transformer 之后下一代架构正在从哪里凿开缺口（第七章）。主本的目标是把这条逻辑链讲清楚、讲完整；而本附册的目标是把每一处「为什么是这个公式」「这个公式怎么推出来」补齐。

本册 **7 章与主本逐章镜像**：前 5 章是数学引擎本身的严格推导；第 6 章《硬件感知算法的数学》覆盖 Roofline 模型、Hong-Kung I/O 下界、FlashAttention 在线 softmax 推导、RingAttention 通信复杂度；第 7 章《后 Transformer 时代的数学》覆盖 HiPPO LegS 矩阵推导、S4 零阶保持离散化、Mamba 选择性扫描的并行关联律、线性注意力的核函数分解、扩散语言模型与 AR 的语义差别。

本册不重复主本的叙事。每个小节默认读者已经读完主本对应章节，因此直接进入数学：

- **直觉版**：用一两段话回顾概念由来与几何含义，铺出数学动机；不含公式。
- **严格版**：给出严格定义、关键公式、必要的推导步骤；所有矩阵显式标注维度，符号尽量与原始论文一致。

读者无需逐节深入。建议在读完主本某一章后，对感兴趣的小节回看附册——把直觉补齐、把公式补全、然后回到主本继续读下一章。

公式约定：行内公式用 \(...\)，块公式用 $$...$$，矩阵以 \( \mathbb{R}^{m \times n} \) 形式标注维度。底数缺省时 log 按自然对数处理；信息论场景下若涉及 bit 单位会显式注明 \( \log_2 \)。

\clearpage

# 数学卷·第 1 章：信息论与语言的概率结构

## §1.1 香农的赌局：语言是对抗熵的工具

**直觉**

原书以香农的"猜字母实验"引入，指出英语文本的熵率约为 1 bit/字符，远低于 26 个字母均匀分布下的理论上限。这里的关键问题是：用什么数学量度量"不确定性"，以及当一个概率分布与另一个分布不一致时，如何量化其代价？

香农熵（Shannon entropy）给出了不确定性的绝对标尺；条件熵刻画了"已知部分信息后剩余的不确定性"；互信息衡量两个变量之间共享了多少信息；交叉熵和 KL 散度则是模型分布偏离真实分布时付出的"额外代价"。大模型的训练目标函数——交叉熵损失——正是这条链条的终点。

**严格**

**1. 香农熵（Shannon entropy）**

设离散随机变量 $X$ 取值于有限集合 $\mathcal{X}$，概率质量函数为 $p(x) = P(X = x)$。**香农熵**定义为：

$$
H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x)
$$

其中约定 $0 \log 0 = 0$（取极限值）。

**对数底数约定**：当底数为 2 时，单位为比特（bit）；当底数为 $e$ 时，单位为奈特（nat）；当底数为 10 时，单位为哈特（hart）。大模型训练中几乎统一使用自然对数（底数 $e$），以便与梯度计算衔接；信息论教材常用 $\log_2$。本节公式默认底数为 $e$，在需要与原书对应的地方显式标注。

熵的基本性质：

- $H(X) \geq 0$，等号成立当且仅当 $X$ 为退化分布（某一值概率为 1）；
- 均匀分布时熵最大，$H(X) \leq \log |\mathcal{X}|$；
- $H(X)$ 只依赖于分布 $p$，而不依赖于具体取值。

**2. 条件熵（conditional entropy）**

设 $(X, Y)$ 的联合分布为 $p(x, y)$，**条件熵**定义为：

$$
H(Y \mid X) = -\sum_{x \in \mathcal{X}} p(x) \sum_{y \in \mathcal{Y}} p(y \mid x) \log p(y \mid x)
$$

等价地，$H(Y \mid X) = H(X, Y) - H(X)$，即联合不确定性减去 $X$ 本身的不确定性。链式法则由此推广为：

$$
H(X_1, X_2, \dots, X_n) = \sum_{i=1}^{n} H(X_i \mid X_1, \dots, X_{i-1})
$$

**3. 互信息（mutual information）**

**互信息**衡量 $X$ 与 $Y$ 共享的信息量：

$$
I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y)
$$

等价定义为联合分布与边缘分布之积的 KL 散度（见下文）：

$$
I(X; Y) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x)\, p(y)}
$$

$I(X; Y) \geq 0$，等号成立当且仅当 $X$ 与 $Y$ 相互独立。

**4. KL 散度（Kullback-Leibler divergence）**

设 $p$ 为真实分布，$q$ 为近似分布，二者定义在同一有限集合上且 $q(x) > 0$ 只要 $p(x) > 0$。**KL 散度**（又称相对熵）定义为：

$$
D_{\mathrm{KL}}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}
$$

性质：$D_{\mathrm{KL}}(p \| q) \geq 0$（由 Jensen 不等式和 $\log$ 的凹性得到），等号成立当且仅当 $p = q$；一般情况下 $D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p)$，即 KL 散度不是度量（metric）。

**5. 交叉熵（cross-entropy）**

**交叉熵**定义为：

$$
H(p, q) = -\sum_{x} p(x) \log q(x)
$$

与 KL 散度的关系：

$$
H(p, q) = H(p) + D_{\mathrm{KL}}(p \| q)
$$

因为真实数据分布 $p$ 的熵 $H(p)$ 是常数（不依赖于模型参数），最小化交叉熵等价于最小化 KL 散度，等价于让模型分布 $q_\theta$ 最大化真实数据的对数似然。大模型的训练损失为：

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log q_\theta(w_i \mid w_{<i})
$$

其中 $w_{<i} = (w_1, \dots, w_{i-1})$ 为前缀，$N$ 为序列长度，$q_\theta$ 为模型输出的条件概率。

**6. 困惑度（perplexity）**

**困惑度**是评估语言模型的标准指标，定义为交叉熵的指数：

$$
\mathrm{PPL} = \exp\!\left(-\frac{1}{N} \sum_{i=1}^{N} \log q_\theta(w_i \mid w_{<i})\right) = \exp\bigl(H(p, q)\bigr)
$$

直观解释：困惑度等于模型在每一步预测时"等效面临的均匀选择数"。若 $\mathrm{PPL} = k$，模型的表现等价于从 $k$ 个等可能选项中均匀猜测。困惑度越低，模型越确定。注意困惑度的数值大小与 $\log$ 底数直接绑定——若改用 $\log_2$，则 $\mathrm{PPL} = 2^{H_2(p,q)}$，其中 $H_2$ 表示以 2 为底的交叉熵（单位为 bit）。

**与原书呼应**：原书 §1.1 指出大模型的训练目标是最小化交叉熵损失 $\mathcal{L} = -\frac{1}{N}\sum_i \log P_\theta(w_i \mid w_{<i})$，对应本节交叉熵与 KL 散度的定义，以及困惑度作为其指数化形式。

---

## §1.2 从马尔可夫链到 N-Gram：第一代尝试的局限

**直觉**

原书从马尔可夫 1913 年对俄语字母的手工统计出发，介绍了 N-Gram 模型的思路与两个致命缺陷：组合爆炸和无法建模长距离依赖。

N-Gram 的数学核心是一个严格的条件独立假设：下一个词只依赖于前 $k$ 个词。这在数学上对应 $k$ 阶马尔可夫链。它的参数通过最大似然估计（频率计数）获得，但稀疏性问题使得大量条件概率为零，必须依靠平滑技术修补。

**严格**

**1. $k$ 阶马尔可夫性（$k$-th order Markov property）**

设词序列 $(w_1, w_2, \dots, w_T)$，每个 $w_t$ 取值于词表 $\mathcal{V}$（$|\mathcal{V}| = V$）。**$k$ 阶马尔可夫假设**为：

$$
P(w_t \mid w_1, w_2, \dots, w_{t-1}) = P(w_t \mid w_{t-k}, w_{t-k+1}, \dots, w_{t-1})
$$

即给定最近 $k$ 个词后，$w_t$ 与更早的历史条件独立。

**N-Gram 的对应关系**：$k$ 阶马尔可夫链等价于 $(k+1)$-gram 模型。

| $k$（马尔可夫阶） | N-Gram 名称 | 条件窗口大小 |
|:-:|:-:|:-:|
| 0 | unigram（1-gram） | 无条件（词独立） |
| 1 | bigram（2-gram） | 前 1 个词 |
| 2 | trigram（3-gram） | 前 2 个词 |
| $k$ | $(k+1)$-gram | 前 $k$ 个词 |

整条序列的概率在 $k$ 阶马尔可夫假设下分解为：

$$
P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_{t-k}, \dots, w_{t-1})
$$

（对 $t \leq k$ 的情形，以序列开头作为边界条件处理。）

**2. 最大似然估计（maximum likelihood estimation, MLE）**

给定训练语料，$(k+1)$-gram 模型的 MLE 参数为计数之比：

$$
\hat{P}(w_n \mid w_{n-k}^{n-1}) = \frac{\mathrm{count}(w_{n-k}, \dots, w_{n-1}, w_n)}{\mathrm{count}(w_{n-k}, \dots, w_{n-1})}
$$

其中 $w_{n-k}^{n-1}$ 是从 $w_{n-k}$ 到 $w_{n-1}$ 的子序列，$\mathrm{count}(\cdot)$ 表示该 $n$-gram 在训练语料中出现的次数。这一估计是无偏的，且在语料趋于无穷时收敛至真实分布。

**3. 稀疏性与零概率问题**

MLE 的根本缺陷：若某 $n$-gram 在训练语料中从未出现，则 $\hat{P} = 0$，导致任何包含该 $n$-gram 的句子概率为零。词表大小 $V$ 时，$k$ 阶 N-Gram 的参数空间为 $O(V^{k+1})$。当 $V = 50000$，$k = 4$ 时，参数量级为 $V^5 \approx 3.1 \times 10^{23}$，远超可观测语料的大小，绝大多数组合计数为零。

**4. 平滑（smoothing）技术直觉**

**Add-one（Laplace）平滑**：对每个可能的 $n$-gram 计数加 1，等价于在参数的先验（均匀 Dirichlet）下取后验均值：

$$
\hat{P}_{\text{add-1}}(w_n \mid w_{n-k}^{n-1}) = \frac{\mathrm{count}(w_{n-k}^{n-1}, w_n) + 1}{\mathrm{count}(w_{n-k}^{n-1}) + V}
$$

这保证了所有概率非零，但对低频 $n$-gram 的估计偏差较大，因为它对稀有和常见组合施加了同等的绝对调整量。

**Kneser-Ney 平滑**的核心直觉（不做完整推导）：观察到"出现在多样上下文中的词"在未见 $n$-gram 中应有更高的延续概率。具体地，用**延续计数**（continuation count）代替频率计数：

$$
P_{\text{KN}}(w \mid \text{context}) \propto \max\bigl(\mathrm{count}(\text{context}, w) - d,\ 0\bigr) + \lambda(\text{context})\, P_{\text{KN}}(w)
$$

其中 $d \in (0, 1)$ 为折扣参数，$\lambda(\text{context})$ 为归一化常数，$P_{\text{KN}}(w)$ 为低阶（回退）分布，定义为 $w$ 出现在其左侧上下文种类数与总上下文种类数之比。Kneser-Ney 是 N-Gram 时代效果最好的平滑方法，至今仍是基线评估的参照点。

**与原书呼应**：原书 §1.2 指出 $k$ 阶马尔可夫假设是 N-Gram 的数学本质，并以词表大小 $V = 50000$、5-gram 参数量 $V^5 \approx 3.1 \times 10^{23}$ 说明组合爆炸，对应本节 MLE 定义和稀疏性分析。

---

## §1.3 跨界映射：语言序列与 DNA 碱基对

**直觉**

原书将语言和 DNA 视为"有限符号集 + 组合规则 + 信息传递"的共同框架，两者都可以用信息论分析。这一节的数学内核是**序列的熵率**：当序列长度趋于无穷时，每个符号平均携带的信息量。Shannon-McMillan-Breiman 定理（典型序列定理）说明，对平稳遍历过程，长序列的概率几乎必然由熵率决定。

**严格**

**1. 信息源与符号序列**

设符号序列 $(X_1, X_2, \dots)$ 为定义在有限字母表 $\mathcal{A}$（$|\mathcal{A}| = M$）上的平稳随机过程（stochastic process）。**平稳性**（stationarity）要求联合分布在时间平移下不变：

$$
P(X_1 = a_1, \dots, X_n = a_n) = P(X_{1+k} = a_1, \dots, X_{n+k} = a_n), \quad \forall k \geq 0
$$

**2. 熵率（entropy rate）**

序列的**熵率**（entropy rate）定义为：

$$
\mathcal{H} = \lim_{n \to \infty} \frac{1}{n} H(X_1, X_2, \dots, X_n)
$$

对平稳过程，该极限存在。等价地（由链式法则），熵率也等于条件熵的极限：

$$
\mathcal{H} = \lim_{n \to \infty} H(X_n \mid X_1, \dots, X_{n-1})
$$

对语言序列而言，$\mathcal{H}$ 刻画了"在已知所有历史的条件下，下一个 token 还剩多少不确定性"。香农估计英语文本的熵率约为 1 bit/字符（$\log_2$ 底数），这正是他猜字母实验的结论。

**3. Shannon-McMillan-Breiman（SMB）定理**

对有限字母表上的**平稳遍历**（stationary ergodic）过程，SMB 定理（亦称渐近等分性，Asymptotic Equipartition Property，AEP）成立：

$$
-\frac{1}{n} \log P(X_1, X_2, \dots, X_n) \xrightarrow{a.s.} \mathcal{H}, \quad n \to \infty
$$

即序列的经验对数概率以概率 1 收敛至熵率 $\mathcal{H}$。其核心推论是：对充分大的 $n$，长度为 $n$ 的序列几乎全部集中在约 $2^{n\mathcal{H}}$（$\log_2$ 底数）个"典型序列"上，每个典型序列的概率约为 $2^{-n\mathcal{H}}$。非典型序列的概率之和趋于零。

对语言模型的意义：SMB 定理说明，一个足够好的模型只需要"覆盖"典型序列集合，就可以捕获几乎全部的概率质量。训练语料的有效信息容量由熵率决定，而不是由字母表大小的幂次决定。

**与原书呼应**：原书 §1.3 指出语言和 DNA 在"有限符号集 + 组合规则"的信息论框架下具有结构相似性，对应本节序列熵率的定义和 SMB 定理对典型序列的描述。

---

## §1.4 反类比：当 DNA 的隐喻失效

**直觉**

原书在承认语言与 DNA 的结构相似后，立即指出三堵"铁墙"。从数学角度，最直接的一堵墙是量级差异：DNA 的字母表只有 4 个符号，而自然语言的词表有数万个 token。这一差异不是细节，而是导致熵率、稀疏性、建模难度在量级上截然不同。

**严格**

**1. 字母表大小与熵率上限**

对大小为 $M$ 的均匀字母表，熵率上限为 $\log M$（自然对数，单位 nat）：

$$
\mathcal{H} \leq \log M
$$

- DNA：$M = 4$，熵率上限 $\log 4 \approx 1.39$ nat/符号（约 2 bit/符号）；
- 大模型的 subword 词表（如 GPT-4 的 BPE 词表）：$M \approx 50000$，熵率上限 $\log 50000 \approx 10.82$ nat/token（约 15.6 bit/token）。

两者的熵率上限之比约为 $\log 50000 / \log 4 \approx 7.8$，即语言 token 级别的理论信息密度是 DNA 碱基的近 8 倍。

**2. 稀疏性的量级差异**

对 $k$ 阶 N-Gram，参数空间大小为 $M^{k+1}$。

- DNA，$k = 2$（三联体密码子）：$4^3 = 64$ 个组合，全部可在有限实验数据中覆盖。
- 语言，$k = 4$（5-gram）：$50000^5 \approx 3.1 \times 10^{23}$ 个组合，任何有限语料都无法覆盖其中绝大多数。

因此，DNA 序列在三联体层级上不存在稀疏性问题，可以对密码表做穷举统计；而语言序列在 $k \geq 3$ 时稀疏性已经不可避免，必须依赖平滑或参数共享（如词嵌入）。

**3. 遗传码的信息冗余与语言冗余的本质差异**

遗传密码存在**简并性**（degeneracy）：64 个密码子仅编码 20 种氨基酸（加 3 个终止密码子），多个密码子映射到同一氨基酸。这是物理化学约束（tRNA 反密码子的摆动配对）决定的，与统计规律无关。

语言的冗余则来自两个来源：句法约束（词类共现规则）和语义约束（上下文限制词义）。这两类约束没有物理化学的必然性，它们是使用者群体统计行为的涌现，因此可以随时间和社群漂移。数学上，语言的冗余度体现为熵率 $\mathcal{H}$ 远低于 $\log M$：香农估计英语的实际熵率约为 1 bit/字符，而字母表（26 字母）的上限为 $\log_2 26 \approx 4.7$ bit/字符，冗余度约为 $1 - 1/4.7 \approx 79\%$。DNA 的情况则不同：以碱基为单位，其熵率接近理论上限（基因组中存在大量非编码区，编码区本身因密码子偏好性略低于上限），冗余度低得多。

这一量级差异说明，把 DNA 的序列建模技术直接迁移到语言，不仅需要应对更大的字母表，还需要捕获更深层、更长程的统计依赖——这正是 N-Gram 失败、变换器（Transformer）崛起的数学根源。

**与原书呼应**：原书 §1.4 指出 DNA 字母表（4 符号）与语言词表（数万 token）的差异是类比失效的起点之一，对应本节熵率上限的量级比较和稀疏性的参数空间计算。


# 数学卷·第 2 章：嵌入空间与高维几何

## §2.1 词嵌入：为什么字典救不了机器

**直觉**

N-Gram 模型把每个词当作原子符号——词与词之间没有任何相似性信息。将"cat"替换为"dog"，模型毫无察觉，因为两者在符号层面是完全不同的对象。字典式的递归定义同样无济于事：用文字解释文字，最终仍是符号游戏。真正的突破是放弃符号表示，转而用**实数向量**承载语义。词在高维空间中的位置——而非名字——成为其含义的载体。向量间的几何关系（距离、角度）自然地编码语义相似性，使得"cat"与"dog"天然相邻，而无需任何显式的规则。

**严格**

**One-hot 编码（one-hot encoding）**是最朴素的词表示。设词表大小为 $V$，词 $w$ 对应词表中第 $k$ 个位置，则其 one-hot 向量 $\mathbf{e}_w \in \{0,1\}^V$ 满足：

$$
(\mathbf{e}_w)_i = \begin{cases} 1 & i = k \\ 0 & \text{otherwise} \end{cases}
$$

该表示有两个根本缺陷。第一，**稀疏性（sparsity）**：向量中只有一个非零分量，储存和计算效率极低，在 $V \sim 10^5$ 量级时尤为突出。第二，**维度灾难（curse of dimensionality）**：任意两个不同词的 one-hot 向量正交，$\mathbf{e}_u^T \mathbf{e}_v = 0$，无法表达语义相关性。

**嵌入矩阵（embedding matrix）** $E \in \mathbb{R}^{V \times d}$ 将上述问题一并解决，其中 $d \ll V$（典型地 $d \in [256, 12288]$）。矩阵 $E$ 的第 $k$ 行即为词 $w_k$ 的稠密低维嵌入向量。**查表（lookup）**操作等价于矩阵-向量乘法：

$$
\mathbf{v}_w = E^T \mathbf{e}_w \in \mathbb{R}^d
$$

由于 $\mathbf{e}_w$ 是 one-hot 向量，该乘法实际上是直接取 $E$ 的第 $k$ 行，计算复杂度为 $O(d)$ 而非 $O(Vd)$。

**余弦相似度（cosine similarity）**是衡量两嵌入向量语义接近程度的标准度量：

$$
\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\| \, \|\mathbf{v}\|} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \cdot \sqrt{\sum_{i=1}^d v_i^2}}
$$

选择余弦而非欧氏距离的动机在于：词频导致高频词的嵌入向量模长系统性偏大（§2.5 将详述），归一化后的角度更稳健地反映语义方向。

嵌入矩阵 $E$ 的各分量在训练中通过反向传播学习，初始化通常为小的随机值（如 $\mathcal{N}(0, 1/d)$）。最终每一行对应一个词在语义几何空间中的坐标。

**与原书呼应**：原书 §2.1 指出 N-Gram 无法泛化的根本原因在于缺乏词间相似性信息，对应本节 one-hot 编码正交性缺陷的分析。

---

## §2.2 语义的坐标系：那个被讲烂了的公式

**直觉**


```{=latex}
\begin{center}
\includegraphics[width=0.92\linewidth]{assets/figs/fig_math_03_01_skipgram.png}\\[0.3em]
\small\itshape 图 M2.1 · Skip-gram 训练范式 · 用中心词预测上下文 + 负采样对比
\end{center}
```
Word2Vec 训练目标极为简单：用一个词预测其上下文中的词（或反过来）。统计上出现在相似上下文中的词，被"推到"向量空间中相邻的位置。这一训练压力的副产品是令人惊讶的语义几何：性别、国籍、时态等关系被编码为空间中固定的方向向量，使得向量加减能近似类比推理。但要理解这个现象的边界，必须先看清训练目标的数学形式。

**严格**

**Skip-gram 目标函数**给定词 $w$，最大化其上下文词 $c$（在窗口 $[-L, L]$ 内，排除自身）的条件概率：

$$
\mathcal{L}_{\text{SG}} = \frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-L \le j \le L \\ j \ne 0}} \log P(w_{t+j} \mid w_t)
$$

条件概率用 softmax 定义，引入两套向量：中心词向量 $\mathbf{v}_w \in \mathbb{R}^d$（来自矩阵 $W \in \mathbb{R}^{V \times d}$）和上下文词向量 $\mathbf{u}_c \in \mathbb{R}^d$（来自矩阵 $W' \in \mathbb{R}^{V \times d}$）：

$$
P(c \mid w) = \frac{\exp(\mathbf{u}_c^T \mathbf{v}_w)}{\sum_{c'=1}^{V} \exp(\mathbf{u}_{c'}^T \mathbf{v}_w)}
$$

分母对全词表求和，计算开销为 $O(V)$，在大词表下不可接受。**负采样（negative sampling, NEG）**以局部二分类替代全局 softmax：对每个正样本对 $(w, c)$，从噪声分布 $P_n(c) \propto f(c)^{3/4}$（$f(c)$ 为词频）采 $k$ 个负样本 $c_1', \ldots, c_k'$，目标函数变为：

$$
\mathcal{L}_{\text{NEG}} = \log \sigma(\mathbf{u}_c^T \mathbf{v}_w) + \sum_{i=1}^{k} \mathbb{E}_{c_i' \sim P_n}\left[\log \sigma(-\mathbf{u}_{c_i'}^T \mathbf{v}_w)\right]
$$

其中 $\sigma(x) = 1/(1 + e^{-x})$ 为 sigmoid 函数。典型地取 $k \in [5, 20]$，计算复杂度降至 $O(k)$。

**类比关系的几何解释。** 设 $\delta_{\text{gender}} = \mathbf{v}_{\text{man}} - \mathbf{v}_{\text{woman}}$。若训练数据中"king"和"queen"出现在几乎相同的上下文中，除性别相关词汇外，则梯度更新会将两者沿性别方向对齐，形成：

$$
\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{queen}} \approx \mathbf{v}_{\text{man}} - \mathbf{v}_{\text{woman}} = \delta_{\text{gender}}
$$

即：

$$
\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}
$$

这一近似成立的充分条件是：语义关系（如性别）在空间中近似平行且等长地分布，即嵌入空间对该关系具有**平移等变性（translational equivariance）**。然而，此条件并不总是满足：原始 Word2Vec 论文报告的类比准确率约为 60–70%，且依赖具体词对；在上下文相关嵌入（BERT、LLaMA 隐藏状态）中，简单向量加减更频繁失败。

**线性表示假说（Linear Representation Hypothesis）**将上述观察形式化：语义概念对应残差流（residual stream）中的线性方向，沿该方向干预可控制模型输出对应属性（Park, Choe & Veitch, ICML 2024）。

**与原书呼应**：原书 §2.2 给出 king−man+woman≈queen 等若干类比示例，并引入线性表示假说，对应本节 Skip-gram 目标函数及类比几何成立条件的分析。

---

## §2.3 高维几何的诡异世界

**直觉**

词嵌入的维度通常在数百到万量级。在这样的空间里，我们关于"近""远""随机点分布"的三维直觉几乎全部失效。理解高维几何的反直觉性质——质量集中于球壳、随机向量近似正交、内积的统计行为——是理解为何余弦相似度仍然有效、为何超位置现象能够存在的数学基础。

**严格**

**球壳质量集中（measure concentration）。** 设 $\mathbf{x} = (x_1, \ldots, x_d)^T$，各分量独立同分布 $x_i \sim \mathcal{N}(0,1)$。则 $\|\mathbf{x}\|^2 = \sum_{i=1}^d x_i^2 \sim \chi^2(d)$，故：

$$
\mathbb{E}[\|\mathbf{x}\|^2] = d, \quad \operatorname{Var}[\|\mathbf{x}\|^2] = 2d
$$

由集中不等式，$\|\mathbf{x}\|$ 以高概率集中在 $\sqrt{d}$ 附近，偏离幅度为 $O(d^{1/4})$。更精确地，对任意 $\varepsilon > 0$：

$$
P\!\left(\left|\|\mathbf{x}\| - \sqrt{d}\right| > \varepsilon\right) \le 2\exp\!\left(-\frac{\varepsilon^2}{4}\right)
$$

直观含义：在高维下，高斯向量的模长以极高概率位于半径为 $\sqrt{d}$ 的球壳薄层内，球的"内部"几乎是空的。

**高维内积分布。** 设 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ 独立同分布 $\mathcal{N}(\mathbf{0}, I_d)$。则内积 $\mathbf{x}^T \mathbf{y} = \sum_i x_i y_i$，各项独立，方差为 1，由中心极限定理：

$$
\frac{\mathbf{x}^T \mathbf{y}}{\sqrt{d}} \xrightarrow{d \to \infty} \mathcal{N}(0,1)
$$

即 $\mathbf{x}^T \mathbf{y} \sim \mathcal{N}(0, d)$，标准差为 $\sqrt{d}$，而两向量的模长均约为 $\sqrt{d}$，故余弦值：

$$
\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x}^T \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|} \approx \frac{\mathcal{N}(0,d)}{d} = \mathcal{N}(0, 1/d) \xrightarrow{d \to \infty} 0
$$

随机高维向量几乎正交。这意味着：高维空间能容纳指数级多的"几乎正交"向量，这正是超位置（superposition）现象的数学基础。

**Johnson-Lindenstrauss 引理（Johnson-Lindenstrauss lemma）。** 设 $n$ 个点 $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^D$，$\varepsilon \in (0, 1/2)$。存在线性映射 $f: \mathbb{R}^D \to \mathbb{R}^k$，其中：

$$
k = O\!\left(\frac{\log n}{\varepsilon^2}\right)
$$

使得对所有点对 $(i, j)$：

$$
(1 - \varepsilon)\|\mathbf{x}_i - \mathbf{x}_j\|^2 \le \|f(\mathbf{x}_i) - f(\mathbf{x}_j)\|^2 \le (1 + \varepsilon)\|\mathbf{x}_i - \mathbf{x}_j\|^2
$$

该映射可由随机高斯矩阵 $\Phi \in \mathbb{R}^{k \times D}$（各元素独立 $\sim \mathcal{N}(0, 1/k)$）构造，以高概率满足上述距离保持性质。引理的关键推论：**降维所需的目标维度仅与点的数量对数相关，与原始维度 $D$ 无关**。这解释了为何在高维嵌入中进行近似最近邻搜索是可行的。

**余弦相似度在高维的稳定性。** 虽然随机向量趋于正交，但经过训练的嵌入向量并非随机——其方向被语义信息约束。余弦相似度的优势在于：它对向量模长不敏感，只关注方向差异。在球壳质量集中的高维空间中，模长携带的信息主要是词频偏置（高频词嵌入模长系统偏大），归一化后恰好消除这一噪声，保留语义方向信号。

**与原书呼应**：原书 §2.3 列举了高维诡异性的三条现象，并引出超位置与稀疏自编码器（SAE），对应本节球壳集中、内积分布与 JL 引理的数学推导。

---

## §2.4 跨界映射：从罗马路网到曼哈顿距离

**直觉**

原书用"罗马路网"类比语义空间中的推理路径，这一类比暗示了空间是均匀欧氏的。但更精确的图像是曼哈顿网格：不同语义维度上的"一个单位"代表的语义差异强度各不相同，距离的计算需要一个加权度量。理解 $L^p$ 范数族和内积空间与度量空间的区别，有助于把握嵌入空间几何的实际复杂度。

**严格**

**$L^p$ 范数（$L^p$ norm）**定义为：

$$
\|\mathbf{x}\|_p = \left(\sum_{i=1}^{d} |x_i|^p\right)^{1/p}, \quad p \ge 1
$$

几个特殊情形：

- $p = 1$（**曼哈顿距离**，Manhattan distance / taxicab norm）：$\|\mathbf{x}\|_1 = \sum_i |x_i|$。单位球为超立方体的对角截面（$d=2$ 时为菱形）。
- $p = 2$（**欧氏范数**，Euclidean norm）：$\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$。单位球为超球面。
- $p \to \infty$（**Chebyshev 距离**，Chebyshev distance）：$\|\mathbf{x}\|_\infty = \max_i |x_i|$。单位球为超立方体。证明：$\|\mathbf{x}\|_p = \left(\sum_i |x_i|^p\right)^{1/p} \le \left(d \cdot \max_i |x_i|^p\right)^{1/p} = d^{1/p} \max_i |x_i| \to \max_i |x_i|$（$p \to \infty$）；下界同理可得。

**加权内积与度量矩阵。** 若嵌入空间的不同维度具有不同的"语义单位长度"，可引入正定矩阵 $G \in \mathbb{R}^{d \times d}$（$G \succ 0$），定义加权内积：

$$
\langle \mathbf{u}, \mathbf{v} \rangle_G = \mathbf{u}^T G \mathbf{v}
$$

对应的马氏（Mahalanobis）距离（Mahalanobis distance）为：

$$
d_G(\mathbf{u}, \mathbf{v}) = \sqrt{(\mathbf{u} - \mathbf{v})^T G (\mathbf{u} - \mathbf{v})}
$$

当 $G = I$ 时退化为欧氏距离；当 $G = \Sigma^{-1}$（数据协方差矩阵的逆）时，Mahalanobis 距离消除各向异性，等价于在白化（whitening）后的空间中计算欧氏距离。

**内积空间（inner product space）vs 度量空间（metric space）。** 内积空间配备双线性、对称、正定的内积，自然诱导范数 $\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$ 及度量 $d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$；反过来，并非所有度量都来自内积（满足平行四边形恒等式 $\|\mathbf{u}+\mathbf{v}\|^2 + \|\mathbf{u}-\mathbf{v}\|^2 = 2\|\mathbf{u}\|^2 + 2\|\mathbf{v}\|^2$ 是内积范数的充要条件）。词嵌入通常假设在内积空间中工作，但其"自然度量"实为加权内积，而非普通欧氏内积。

**与原书呼应**：原书 §2.4 以罗马路网 vs 曼哈顿网格为比喻，指出词向量空间的"自然内积"为加权内积 $\mathbf{u}^T G \mathbf{v}$，对应本节 $L^p$ 范数几何含义与度量矩阵的数学形式化。

---

## §2.5 反类比：当几何隐喻失效

**直觉**

嵌入空间并非理想的欧氏空间。词频统计在空间中留下了系统性的几何扭曲：高频词的嵌入向量聚集在一个狭窄的椎体（cone）中，而非均匀分布在整个空间。这种**各向异性（anisotropy）**使得简单的向量算术和余弦相似度都面临失真，并且解释了为何 king−man+woman≈queen 在现代嵌入中的成功率远低于早期论文所暗示的水平。

**严格**

**各向异性（anisotropy）。** 理想的嵌入空间应满足各向同性：嵌入向量均匀分布在各方向，任意随机向量与所有嵌入向量的平均余弦相似度为零。实践中，Ethayarajh（2019）发现，BERT 的上下文嵌入高度各向异性——大多数嵌入向量与"平均方向"高度对齐，投影到低维子空间后几乎共线。形式化：设嵌入集合 $\{\mathbf{v}_w\}_{w=1}^V$，定义各向异性指标：

$$
\bar{\cos} = \frac{1}{\binom{V}{2}} \sum_{i < j} \cos(\mathbf{v}_i, \mathbf{v}_j)
$$

各向同性时 $\bar{\cos} \approx 0$；实际 GPT-2 的层级嵌入中 $\bar{\cos}$ 可达 0.99，即几乎所有嵌入向量指向同一方向。

**词频导致的椎体效应（frequency-induced cone effect）。** 负采样目标函数（§2.2）中，高频词作为负样本被采到的概率高，其上下文向量受到更多"排斥"梯度，导致高频词向量被推向向量空间的边缘，聚集在一个狭窄的锥形区域内。Mimno & Thompson（2017）将此称为"椎体效应"：频率最高的词（如 "the"、"of"）形成一个低维锥，其余词分布在锥外，使得整个嵌入空间的有效维度远低于名义维度 $d$。

**白化（whitening）作为修正。** 白化变换通过对嵌入集合进行 PCA 并按特征值归一化，消除各向异性偏置：

设嵌入矩阵（去中心化后）的协方差为 $\Sigma = \frac{1}{V} E^T E \in \mathbb{R}^{d \times d}$，其特征分解为 $\Sigma = U \Lambda U^T$（$\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_d)$，$\lambda_1 \ge \cdots \ge \lambda_d > 0$）。白化变换为：

$$
\tilde{\mathbf{v}}_w = \Lambda^{-1/2} U^T \mathbf{v}_w
$$

变换后 $\operatorname{Cov}(\tilde{\mathbf{v}}) = I_d$，每个方向的方差相等，空间各向同性恢复。Su 等人（2021）的 BERT-whitening 实验表明，白化后用余弦相似度评估语义相似性的效果优于原始 BERT 表示，且在多项语义文本相似度基准上超过更复杂的对比学习方法。

**后果总结。** 各向异性意味着：（1）余弦相似度在未白化的嵌入空间中存在系统性偏置，高频词对之间的余弦值虚高；（2）向量加减（如类比算术）在高频词参与时误差更大；（3）嵌入空间的"有效维度"显著低于 $d$，大量方向携带的是词频噪声而非语义信号。

**与原书呼应**：原书 §2.5 指出几何隐喻失效的三个层次（连续 vs 离散、概率 vs 确定、构造 vs 发现），与本节各向异性和椎体效应共同构成对"嵌入空间是理想欧氏空间"这一假设的否定。

---

## §2.6 跨越欧氏：当表示空间不再"平直"

**直觉**

树形层级结构（动物→哺乳动物→狗→金毛）的节点数随深度指数增长，而欧氏球的体积仅按多项式（$\sim r^d$）增长。这种指数 vs 多项式的不匹配意味着：在欧氏空间中嵌入深层树需要巨大的维度才能保持层级距离关系。双曲空间正好相反——其"体积"随半径指数增长，天然匹配树的拓扑。Poincaré 球模型给出了一个具体且数学上自洽的框架。

**严格**

**欧氏嵌入的维度瓶颈。** 深度为 $L$、分支数为 $b$ 的完全 $b$-叉树有 $N = \frac{b^{L+1}-1}{b-1} = O(b^L)$ 个节点。在欧氏空间 $\mathbb{R}^d$ 中，半径为 $r$ 的球最多包含 $O(r^d)$ 个互相距离 $\ge 1$ 的点（packing argument）。要将 $N$ 个节点嵌入使得树上距离被 $(1+\varepsilon)$-近似保持，所需维度满足 $r^d \ge N$，即：

$$
d \ge \Omega\!\left(\frac{L \log b}{\log r}\right)
$$

对大 $L$ 此下界线性增长，不可避免。

**Poincaré 球模型（Poincaré ball model）。** $d$ 维双曲空间 $\mathbb{H}^d$ 的 Poincaré 球模型定义在开单位球 $\mathbb{B}^d = \{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x}\| < 1\}$ 上，配以黎曼度量：

$$
ds^2 = \frac{4\,\|\mathrm{d}\mathbf{x}\|^2}{\left(1 - \|\mathbf{x}\|^2\right)^2}
$$

即欧氏度量乘以保角因子 $\lambda(\mathbf{x})^2 = \frac{4}{(1-\|\mathbf{x}\|^2)^2}$（共形因子）。当 $\|\mathbf{x}\| \to 1$ 时 $\lambda \to \infty$，空间被无限"拉伸"——有限欧氏距离对应无限双曲距离，边界不可达。

**双曲距离公式。** 对 $\mathbf{u}, \mathbf{v} \in \mathbb{B}^d$，双曲距离为：

$$
d_{\mathbb{H}}(\mathbf{u}, \mathbf{v}) = \mathrm{arcosh}\!\left(1 + 2\,\frac{\|\mathbf{u} - \mathbf{v}\|^2}{\left(1 - \|\mathbf{u}\|^2\right)\left(1 - \|\mathbf{v}\|^2\right)}\right)
$$

其中 $\mathrm{arcosh}(t) = \ln(t + \sqrt{t^2-1})$，$t \ge 1$。

**推导梗概。** 从 $\mathbf{u}$ 到 $\mathbf{v}$ 的双曲测地线长度为 $\int_\gamma ds$，利用 Möbius 变换将 $\mathbf{u}$ 映射到原点（原点到 $\mathbf{r}$ 的距离简化为 $\int_0^{\|\mathbf{r}\|} \frac{2\,dt}{1-t^2} = 2\,\mathrm{arctanh}(\|\mathbf{r}\|)$），再利用 $\mathrm{arcosh}$ 与 $\mathrm{arctanh}$ 的关系 $\mathrm{arcosh}(1 + 2t^2/(1-t^2)) = 2\,\mathrm{arctanh}(t)$，得到上述闭式表达。

**指数体积增长。** 在 Poincaré 球中，以 $\mathbf{0}$ 为中心、双曲半径为 $r$ 的球的（欧氏）体积为：

$$
\mathrm{Vol}_{\mathbb{H}}(B(r)) \propto \sinh^{d-1}(r) \sim \frac{1}{2^{d-1}} e^{(d-1)r}, \quad r \to \infty
$$

体积随 $r$ **指数增长**，与 $b$-叉树节点数的指数增长 $O(b^r)$（以深度 $r$ 衡量）匹配，只需令 $b = e^{d-1}$。因此，在 $d=5$ 维双曲空间中，已能容纳 WordNet 约 $8 \times 10^4$ 个名词节点，且层级距离重建精度超过 200 维欧氏嵌入（Nickel & Kiela, 2017）。

**层级位置编码的几何含义。** 在 Poincaré 球中，节点在层级结构中的深度与其到球心的双曲距离直接对应：

- **球心**（$\|\mathbf{x}\| \approx 0$）：抽象概念、树根（如"实体"）；
- **球边界**（$\|\mathbf{x}\| \to 1$）：具体叶节点（如"金毛犬"），到边界的距离趋于无穷。

父节点到所有子节点的距离均等且较短，体现层级关系；兄弟节点之间的距离则由双曲距离公式的分母 $(1-\|\mathbf{u}\|^2)(1-\|\mathbf{v}\|^2)$ 调节，靠近边界时兄弟之间距离迅速增大，有效隔离不同子树。

**为何主流 LLM 仍用欧氏空间。** 自然语言的语义关系同时包含层级（is-a）和网状联想（associate-with）结构。欧氏空间对混合拓扑最为稳健；双曲空间在强层级数据（本体树、分子图）上有明确收益，但在通用语言建模中的优势尚不稳定。选择几何空间本质上是对**数据内在拓扑**的假设，没有普适最优解。

**与原书呼应**：原书 §2.6 介绍 Poincaré 嵌入、双曲距离公式及主流 LLM 仍用欧氏空间的原因，对应本节度量推导、指数体积增长的严格分析，以及几何选择的元原则。


# 数学卷·第 3 章：注意力机制的矩阵微分

---

## §3.1 从 RNN 到变换器（Transformer）：序列建模的范式革命

**直觉**


```{=latex}
\begin{center}
\includegraphics[width=0.92\linewidth]{assets/figs/fig_math_05_01_transformer_block.png}\\[0.3em]
\small\itshape 图 M3.1 · Transformer Block 一层内部的前向数据流
\end{center}
```
循环神经网络（Recurrent Neural Network，RNN）把序列理解为一条因果链：每一步的计算依赖前一步的结果。这种设计在直觉上符合"阅读"的体验，但同时带来了致命的时序依赖——序列无法并行处理，梯度信号在长链上传播时会随步数指数式放大或衰减。Transformer 的根本突破在于：彻底切断时间依赖，让每个位置的表示直接从全局上下文中"读取"，把序列问题还原为一组独立的矩阵运算，从而让 GPU 的大规模并行硬件得以充分发挥。

**严格**

RNN 的隐状态（hidden state）递推公式为

$$
h_t = \sigma(W h_{t-1} + U x_t + b)
$$

其中 $h_t \in \mathbb{R}^{d_h}$ 是第 $t$ 步隐状态，$W \in \mathbb{R}^{d_h \times d_h}$，$U \in \mathbb{R}^{d_h \times d_x}$，$\sigma$ 为逐元素非线性激活（如 $\tanh$）。

**梯度消失/爆炸的数学根源**

在时刻 $T$ 的损失 $\mathcal{L}$ 对时刻 $t$ 的隐状态求梯度，需通过链式法则展开

$$
\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}
$$

每一步的雅可比矩阵（Jacobian matrix）为

$$
\frac{\partial h_{k+1}}{\partial h_k} = \mathrm{diag}\!\left(\sigma'(W h_k + U x_{k+1})\right) W
$$

连乘项的谱范数满足

$$
\left\| \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k} \right\|_2 \leq \prod_{k=t}^{T-1} \left\| \mathrm{diag}(\sigma') \right\|_2 \cdot \|W\|_2
$$

设 $\rho = \|W\|_2 \cdot \max|\sigma'|$（即该乘积谱半径的上界），则梯度范数以 $\rho^{T-t}$ 的速率随距离 $T-t$ 指数变化：

- 若 $\rho < 1$：梯度指数衰减（梯度消失，gradient vanishing）；
- 若 $\rho > 1$：梯度指数增长（梯度爆炸，gradient explosion）。

LSTM 通过门控机制（gating mechanism）使部分梯度路径的有效谱半径接近 1，但无法从根本上解除连乘结构，长序列仍受限于此。

**为什么 Transformer 可并行**

Transformer 完全去掉了时序递推：给定输入矩阵 $X \in \mathbb{R}^{n \times d}$（$n$ 个 token 同时存在于矩阵中），每个位置的输出仅通过矩阵乘法和 softmax 计算，不依赖任何"前序状态"。具体而言，注意力层的输入输出关系为

$$
O = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中 $Q, K, V$ 均由 $X$ 一次性线性投影得到，整个计算路径无时间轴上的串行依赖，因此 $n$ 个位置的输出可完全并行计算。

**与原书呼应**：原书 §3.1 提到 RNN「不能并行」和「长距离依赖丢失」两个缺陷，对应本节雅可比连乘的谱半径分析与并行化条件。

---

## §3.2 注意力公式的拆解：QKV 的真实含义

**直觉**


```{=latex}
\begin{center}
\includegraphics[width=0.92\linewidth]{assets/figs/fig_math_02_01_softmax.png}\\[0.3em]
\small\itshape 图 M3.2 · Softmax · 把任意实数变成概率(放大差距 → 归一化)
\end{center}
```
把注意力机制想象成一种"软索引"：给定一个查询向量，在一组键值对中按相似度加权检索。与硬索引（哈希表）不同，这里每个"键"都会有或多或少的匹配权重，最终返回的是所有"值"的加权混合。$\sqrt{d_k}$ 缩放的作用是压制高维内积随维度增大而自然增大的方差，防止 softmax 因输入过大而在某一项上接近饱和、梯度接近零。多头设计则让模型同时在多个线性子空间中做这种检索，捕捉不同粒度的语义关系。

**严格**

设输入序列嵌入矩阵 $X \in \mathbb{R}^{n \times d}$，可学习投影矩阵

$$
W^Q, W^K \in \mathbb{R}^{d \times d_k}, \quad W^V \in \mathbb{R}^{d \times d_v}
$$

三个投影矩阵给出

$$
Q = X W^Q \in \mathbb{R}^{n \times d_k}, \quad K = X W^K \in \mathbb{R}^{n \times d_k}, \quad V = X W^V \in \mathbb{R}^{n \times d_v}
$$

**缩放因子 $\sqrt{d_k}$ 的来源**

设 $q_i, k_j \in \mathbb{R}^{d_k}$ 各分量独立同分布，均值为 0、方差为 1，则

$$
\mathrm{Var}(q_i \cdot k_j) = \mathrm{Var}\!\left(\sum_{l=1}^{d_k} q_{il} k_{jl}\right) = \sum_{l=1}^{d_k} \mathrm{Var}(q_{il})\mathrm{Var}(k_{jl}) = d_k
$$

即内积 $q_i \cdot k_j$ 的标准差为 $\sqrt{d_k}$。不作缩放时，当 $d_k$ 较大（典型值 64–128），内积分布的量级随 $\sqrt{d_k}$ 线性增大，进入 softmax 后所有概率质量集中于极少数最大项，形成近似 one-hot 分布，梯度趋近于零。除以 $\sqrt{d_k}$ 将方差还原为 1，保持 softmax 输入在合理量级。

**注意力分数与归一化**

注意力分数矩阵

$$
A = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}, \quad A_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}
$$

softmax **沿第二维（行内）** 归一化，即对第 $i$ 行独立计算

$$
P_{ij} = \frac{\exp(A_{ij})}{\sum_{l=1}^{n} \exp(A_{il})}, \quad \sum_{j=1}^{n} P_{ij} = 1
$$

这使得 $P \in \mathbb{R}^{n \times n}$ 的每行构成一个概率分布：第 $i$ 个 token 对所有位置的关注权重之和为 1。输出为

$$
\mathrm{Attention}(Q, K, V) = P V \in \mathbb{R}^{n \times d_v}
$$

**多头注意力的参数化**

设头数为 $h$，每头的子空间维度通常取 $d_k = d_v = d/h$。第 $i$ 头（$i = 1, \ldots, h$）独立维护一套投影矩阵

$$
W^Q_i \in \mathbb{R}^{d \times d_k},\quad W^K_i \in \mathbb{R}^{d \times d_k},\quad W^V_i \in \mathbb{R}^{d \times d_v}
$$

各头输出拼接后再经输出投影 $W^O \in \mathbb{R}^{hd_v \times d}$：

$$
\mathrm{MultiHead}(X) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)\, W^O
$$

其中 $\mathrm{head}_i = \mathrm{Attention}(X W^Q_i,\, X W^K_i,\, X W^V_i)$。多头设计的参数总量与单头相同（$d^2$ 量级），但在 $h$ 个低维子空间中并行运行不同的"检索模式"。

**与原书呼应**：原书 §3.2 逐步拆解了 QKV 投影与缩放技巧，对应本节方差推导与 softmax 归一化维度分析。

---

## §3.3 矩阵乘法的大规模并发：为什么 AI 是 GPU 的盛宴

**直觉**


```{=latex}
\begin{center}
\includegraphics[width=0.92\linewidth]{assets/figs/fig_math_03_02_attention_matmul.png}\\[0.3em]
\small\itshape 图 M3.3 · Attention 矩阵乘法三步 · QKᵀ → 缩放 → softmax 权重
\end{center}
```
GPU 的设计哲学是"以宽度换速度"：数以千计的简单核心并发执行同一条指令，作用于不同数据。矩阵乘法天然符合这一模式——结果矩阵的每个元素是独立内积，可完全并行。反观 attention 的计算，其核心运算 $QK^T$、$PV$ 均为大型矩阵乘法；但真正的瓶颈往往不在浮点计算本身，而在芯片片上缓存（SRAM）与显存（HBM）之间的数据搬运速度——即算术强度（arithmetic intensity）是否高到足以让计算单元保持满载。

**严格**

**矩阵乘法的算术强度**

对两个矩阵相乘 $C = AB$，其中 $A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$，$C \in \mathbb{R}^{m \times n}$：

- 浮点运算量（FLOPs）：$2mnk$（每个输出元素做 $k$ 次乘加）
- 内存访问字节数（读 $A$、$B$，写 $C$）：$(mk + kn + mn) \times \text{sizeof}(\text{dtype})$

算术强度定义为

$$
I = \frac{2mnk}{mk + kn + mn} \quad (\text{FLOP/byte})
$$

当 $m, n, k$ 均很大（如 $m = n = k = N \gg 1$）时，$I \approx \frac{2N^3}{3N^2} = \frac{2N}{3}$，随矩阵尺寸线性增长。典型大矩阵乘法的算术强度远超 GPU 的"屋脊点"（ridge point，即峰值算力 / 峰值内存带宽），因此属于**计算受限**（compute-bound）操作，GPU 的算力可以充分利用。

**Roofline 模型简介**

屋脊线（Roofline）模型将核函数的实际性能 $P$（FLOP/s）建模为

$$
P = \min\!\bigl(\text{峰值算力},\; I \times \text{峰值内存带宽}\bigr)
$$

当 $I$ 低于屋脊点时，性能受内存带宽限制（memory-bound）；高于屋脊点时，性能受峰值算力限制（compute-bound）。对 A100 GPU，屋脊点约为 $312\,\text{TFLOP/s} / 2\,\text{TB/s} = 156\,\text{FLOP/byte}$；大矩阵乘法的算术强度通常超过此值，而逐元素操作（如 softmax 的 $\exp$）算术强度接近 1，远低于屋脊点。

**Attention 的 FLOPs 分析**

对单头注意力，设序列长度 $n$，键/查询维度 $d_k$，值维度 $d_v$：

| 步骤 | 计算 | FLOPs |
|------|------|-------|
| $S = QK^T$ | $\mathbb{R}^{n \times d_k} \times \mathbb{R}^{d_k \times n}$ | $2n^2 d_k$ |
| softmax | 逐行归一化 | $\Theta(n^2)$（低算术强度） |
| $O = PV$ | $\mathbb{R}^{n \times n} \times \mathbb{R}^{n \times d_v}$ | $2n^2 d_v$ |

忽略常数，总 FLOPs 为

$$
\mathrm{FLOPs}_{\text{attention}} = \Theta(n^2 d)
$$

其中 $d = d_k = d_v$。对 $h$ 头，乘以 $h$，但此时每头 $d_k = d/h$，总量不变。$O(n^2 d)$ 的二次复杂度使得长上下文（大 $n$）下计算量急剧增长，是 attention 的核心算力瓶颈。

**与原书呼应**：原书 §3.3 提到 LLaMA-3 70B 单层 attention 的算力估算及 GPU 的并行哲学，对应本节算术强度与 Roofline 模型的量化分析。

---

## §3.4 RoPE：把位置编码从加法变成旋转

**直觉**


```{=latex}
\begin{center}
\includegraphics[width=0.92\linewidth]{assets/figs/fig_math_05_02_rope.png}\\[0.3em]
\small\itshape 图 M3.4 · RoPE 旋转位置编码 · 把「位置」刻进向量的旋转角
\end{center}
```
原始 Transformer 的正弦位置编码（Sinusoidal Positional Encoding）是在嵌入向量上做加法：每个位置对应一个固定向量叠加到 token 表示上。这种绝对编码让模型难以直接利用"两个 token 相距多远"这一语言学上更本质的信号，且在训练长度之外几乎立即失效。旋转位置编码（Rotary Position Embedding，RoPE）改变思路：不加法，而是把位置信息编码为向量在复平面上的旋转角度，使得 Query 与 Key 的内积天然只依赖二者的相对位置差，绝对位置信息从内积中消去。

**严格**

**正弦位置编码的形式**

对位置 $\mathrm{pos}$ 和嵌入维度下标 $2i, 2i+1$：

$$
\mathrm{PE}(\mathrm{pos}, 2i) = \sin\!\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right), \quad \mathrm{PE}(\mathrm{pos}, 2i+1) = \cos\!\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right)
$$

编码后 $x'_{\mathrm{pos}} = x_{\mathrm{pos}} + \mathrm{PE}(\mathrm{pos})$，绝对位置以加法混入嵌入，无法分离相对位置信息。

**RoPE 的复数化表示**

RoPE 将 $d$ 维向量 $q \in \mathbb{R}^d$ 看成 $d/2$ 个复数，第 $i$ 对分量 $(q_{2i}, q_{2i+1})$ 对应复数 $z^q_i = q_{2i} + \mathrm{i}\, q_{2i+1}$。对位于位置 $m$ 的 token，每对分量乘以单位复数 $e^{\mathrm{i} m\theta_i}$：

$$
\tilde{z}^q_i = z^q_i \cdot e^{\mathrm{i} m\theta_i}
$$

频率参数

$$
\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, \frac{d}{2}-1
$$

低下标 $i$ 对应高频（旋转快），高下标对应低频（旋转慢），覆盖不同尺度的位置信息。

**等价实数旋转矩阵**

对每对分量，乘以 $e^{\mathrm{i} m\theta_i}$ 等价于施加二维旋转矩阵

$$
R_m^{(i)} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
$$

对整个 $d$ 维向量，RoPE 的变换可写为块对角旋转矩阵

$$
\mathcal{R}_m = \mathrm{blockdiag}\!\left(R_m^{(0)}, R_m^{(1)}, \ldots, R_m^{(d/2-1)}\right) \in \mathbb{R}^{d \times d}
$$

位置 $m$ 处的 query 和 key 在施加 RoPE 后分别为

$$
\tilde{q}_m = \mathcal{R}_m\, q_m, \quad \tilde{k}_n = \mathcal{R}_n\, k_n
$$

**核心性质：内积只依赖相对位置**

由于旋转矩阵是正交矩阵（$\mathcal{R}_m^T = \mathcal{R}_m^{-1} = \mathcal{R}_{-m}$），且旋转角度可加（$\mathcal{R}_m^T \mathcal{R}_n = \mathcal{R}_{n-m}$），attention score 为

$$
\langle \tilde{q}_m, \tilde{k}_n \rangle = (\mathcal{R}_m q_m)^T (\mathcal{R}_n k_n) = q_m^T \mathcal{R}_m^T \mathcal{R}_n k_n = q_m^T \mathcal{R}_{n-m} k_n = \langle q_m,\, \mathcal{R}_{n-m}\, k_n \rangle
$$

结果仅依赖位置差 $n - m$，而不依赖绝对位置 $m$ 或 $n$。这一性质使得模型在推理时天然感知相对距离，并且——通过调整频率基数 $\theta_i$ 的底数（将 10000 替换为更大的值）或对不同频率段做差异化插值——可以向训练长度以外外推，这是 YaRN 等长上下文扩展方法的数学基础。

与正弦编码的对比如下：

| 属性 | 正弦位置编码 | RoPE |
|------|------------|------|
| 编码方式 | 加法叠加 | 乘法旋转 |
| 位置信号 | 绝对位置 | 相对位置 |
| 内积形式 | 依赖绝对位置 | 仅依赖 $n - m$ |
| 训练外外推 | 直接失效 | 可通过频率调整延伸 |

**与原书呼应**：原书 §3.4 介绍了 RoPE 的旋转思想与「位置差从指数里掉出来」的性质，对应本节旋转矩阵正交性的严格推导。

---

## §3.5 FlashAttention：当注意力撞上内存墙

**直觉**


```{=latex}
\begin{center}
\includegraphics[width=0.92\linewidth]{assets/figs/fig_math_06_01_kv_cache.png}\\[0.3em]
\small\itshape 图 M3.5 · KV Cache 推理时复用过去 token 的 K/V → O(N²) → O(N)
\end{center}
```
标准 attention 实现的瓶颈不在浮点运算量，而在显存带宽：$n \times n$ 的注意力矩阵需要在 GPU 片上缓存（SRAM）和高带宽显存（HBM）之间反复搬运，而 SRAM 的带宽约是 HBM 的 10 倍但容量小四个数量级。FlashAttention 的思路是：把 $Q, K, V$ 切成能放入 SRAM 的小块，在芯片内部完成"矩阵乘 + softmax + 再乘"的全流程，只将最终输出写回 HBM，从而将 HBM 访问量从 $\Theta(n^2)$ 降至接近线性，而浮点运算量不变。难点在于 softmax 需要全局统计量，FlashAttention 通过在线 softmax 技巧解决了这一矛盾。

**严格**

**标准实现的内存访问代价**

设序列长度 $n$，维度 $d$，标准实现的 HBM 读写量（以元素数计）：

1. 计算 $S = QK^T \in \mathbb{R}^{n \times n}$，写回 HBM：$\Theta(n^2)$
2. 从 HBM 读 $S$，计算 softmax 得 $P$，写回 HBM：$\Theta(n^2)$
3. 从 HBM 读 $P$（$\Theta(n^2)$）和 $V$（$\Theta(nd)$），计算 $O = PV$，写回

总 HBM 访问量 $\Theta(n^2 + nd)$，长序列下 $n^2$ 项主导，使 attention 成为**内存受限**操作。

**在线 softmax（online softmax）的递推公式**

设将 $K, V$ 按行切成 $T$ 个块，第 $j$ 块对应局部注意力分数向量 $s_j \in \mathbb{R}^{B}$（$B$ 为块大小）。维护两个滚动统计量：

- $m_j \in \mathbb{R}$：前 $j$ 块中 $s$ 分量的最大值（用于数值稳定移位）
- $\ell_j \in \mathbb{R}$：前 $j$ 块经移位后的指数和

递推更新为

$$
m_j = \max(m_{j-1},\; \tilde{m}_j)
$$

$$
\ell_j = e^{m_{j-1} - m_j}\, \ell_{j-1} + e^{\tilde{m}_j - m_j}\, \tilde{\ell}_j
$$

其中 $\tilde{m}_j = \max(s_j)$，$\tilde{\ell}_j = \sum_{l} \exp(s_{jl} - \tilde{m}_j)$ 是当前块的局部指数和。

**输出的分块更新**

维护运行输出 $O_j \in \mathbb{R}^{d_v}$，加载第 $j$ 块时更新为

$$
O_j = \mathrm{diag}(\ell_j)^{-1} \!\left( \mathrm{diag}(\ell_{j-1})\, e^{m_{j-1} - m_j} O_{j-1} + e^{\tilde{m}_j - m_j}\, \tilde{P}_j V_j \right)
$$

其中 $\tilde{P}_j = \exp(s_j - \tilde{m}_j \mathbf{1})$ 是当前块的未归一化注意力权重，$V_j$ 是对应的 Value 块。可以验证，遍历所有块后 $O_T$ 恰好等于完整的归一化注意力输出。

**复杂度分析**

| 量 | 标准 Attention | FlashAttention |
|----|------------|----------------|
| FLOPs | $\Theta(n^2 d)$ | $\Theta(n^2 d)$（不变） |
| HBM 读写（元素数） | $\Theta(n^2 + nd)$ | $\Theta(n^2 d / M)$（$M$ 为 SRAM 大小） |
| 需显式存储 $n \times n$ 矩阵 | 是 | 否 |
| 额外显存开销（除 $O$ 外） | $\Theta(n^2)$ | $\Theta(n)$（仅存 $m_j, \ell_j$）|

FlashAttention 将内存复杂度从 $O(n^2)$ 降至 $O(n)$（按辅助统计量计），HBM 带宽使用减少 4–16 倍，使 attention 从内存受限转变为计算受限，GPU Tensor Core 的利用率大幅提升。FLOPs 仍为 $O(n^2 d)$，即注意力的算力复杂度没有改变，改变的只是其对硬件内存层级的友好度。

**与原书呼应**：原书 §3.5 给出了在线 softmax 的递推公式及 HBM 访问复杂度分析，对应本节的逐步推导与复杂度对比表。

---

## §3.6 注意力的真实身份：关联记忆，而非"注意力"

**直觉**

"注意力"这个名字容易让人联想到认知心理学中的注意力——有限资源的主动分配。但从数学上看，softmax attention 更接近一种**内容寻址的关联记忆**（content-addressable associative memory）检索：给定一个查询模式，在一个存储了大量模式的系统中检索最接近的内容。现代 Hopfield 网络（modern Hopfield network）的更新规则与 softmax attention 公式在数学上完全等价，而现代 Hopfield 网络的存储容量是维度的**指数级**，远超经典版本的线性容量。

**严格**

**经典 Hopfield 网络的能量函数**

经典 Hopfield 网络（1982）的能量函数为

$$
E_{\text{classic}} = -\frac{1}{2} \xi^T W \xi, \quad W = X^T X - N I
$$

其中 $\xi \in \{-1, +1\}^d$ 为查询状态，$X \in \mathbb{R}^{N \times d}$ 为存储模式，存储容量约为 $0.14 d$（线性于维度）。

**现代 Hopfield 网络的能量函数**

Ramsauer 等（ICLR 2021）将能量函数改为

$$
E = -\,\mathrm{lse}(\beta,\, X \xi) + \frac{1}{2}\,\xi^T \xi + \frac{1}{2\beta}\log N + C
$$

其中 $\mathrm{lse}(\beta, z) = \beta^{-1} \log \sum_i \exp(\beta z_i)$ 是 log-sum-exp 算子，$\beta > 0$ 是逆温度参数，$C$ 是与 $\xi$ 无关的常数。对此能量函数求不动点条件 $\partial E / \partial \xi = 0$，得到更新规则

$$
\xi^{\text{new}} = X^T \,\mathrm{softmax}(\beta X \xi)
$$

**与 softmax Attention 的等价性**

令 $\xi \leftarrow Q_i$（某行查询向量），$X \leftarrow K$（键矩阵的行向量集合），$\beta = 1/\sqrt{d_k}$，则

$$
\xi^{\text{new}} = K^T \,\mathrm{softmax}\!\left(\frac{K Q_i^T}{\sqrt{d_k}}\right)
$$

这恰好是对单个 query $Q_i$ 的 attention 输出（忽略 Key 与 Value 不同这一细节；原始论文中取 $K = V^T$ 的情形即精确等价）。因此，**一次 softmax attention 运算在数学上等价于现代 Hopfield 网络以 $\beta = 1/\sqrt{d_k}$ 做一步不动点迭代**。

**存储容量的提升**

现代 Hopfield 网络的存储容量从经典的 $O(d)$ 提升至

$$
C_{\text{modern}} = O\!\left(\exp\!\left(\frac{\alpha^2 d}{2}\right)\right)
$$

（$\alpha$ 为模式间最小分离距离的某个函数），即指数级于维度。这解释了为何大型 Transformer 能以有限参数量存储和检索海量知识模式：权重矩阵 $W^K$ 的行向量构成一个指数容量的关联记忆索引，推理时的"理解"本质是高维模式检索的级联。

**与原书呼应**：原书 §3.6 引用了 Ramsauer 等的等价性证明，对应本节能量函数推导与不动点更新规则的显式对应。

---

## §3.7 一个被忽略的争论：注意力权重 ≠ 解释

**直觉**

直觉上很容易把注意力权重 $\alpha_{ij}$（即 softmax 输出）理解为"模型因为关注了第 $j$ 个词，所以预测受到它影响"。但从数学上，注意力权重只是计算图中的一个中间量，它对最终输出的实际影响必须通过梯度来量化，而梯度归因与注意力权重并不等价，甚至可以构造出二者大相径庭的反例。

**严格**

设单层注意力的输出为

$$
y = \sum_{j=1}^{n} \alpha_j v_j, \quad \alpha_j = \frac{\exp(q \cdot k_j / \sqrt{d_k})}{\sum_{l} \exp(q \cdot k_l / \sqrt{d_k})}
$$

**注意力权重与梯度归因的不一致性**

基于梯度的特征重要性度量定义为

$$
g_j = \left\|\frac{\partial \mathcal{L}}{\partial x_j}\right\| \quad \text{或} \quad g_j = \left\|\frac{\partial y}{\partial x_j}\right\|
$$

$\alpha_j$ 衡量的是 $v_j$（Value 向量）对 $y$ 的**线性组合权重**，而 $g_j$ 衡量的是输入 $x_j$ 对输出 $y$（或损失 $\mathcal{L}$）的**微分敏感度**。二者通过不同路径传播：$\alpha_j$ 只经由 Value 分支，而 $g_j$ 还包含 $x_j$ 对 $q$、$k_j$ 乃至其他位置 $k_l$ 的影响，路径更复杂。

**可交换性的反例构造**

Jain 和 Wallace（NAACL 2019）指出，可以在保持输出近似不变的前提下构造替代注意力分布 $\alpha'$，使得 $\|\alpha' - \alpha\|_1$ 较大。形式化地：对给定输出 $y = \alpha^T V$，存在 $\alpha' \in \Delta^{n-1}$（概率单纯形）使得

$$
\|\alpha' - \alpha\|_1 \geq \epsilon \quad \text{而} \quad \|y' - y\| = \|\alpha'^T V - \alpha^T V\| \leq \delta
$$

对足够小的 $\delta$ 和足够大的 $\epsilon$，这说明注意力分布本身对输出不具有唯一决定性——当 Value 矩阵 $V$ 的行向量之间相关性高时，不同的权重混合可以得到接近相同的输出向量。因此，$\alpha_j$ 大不能直接推断"第 $j$ 个 token 对预测起决定性作用"；真正的因果归因必须通过梯度方法（gradient attribution）或激活路径分析（如 circuit analysis）来进行。

**与原书呼应**：原书 §3.7 引用了 Jain & Wallace 的核心发现及 induction head 电路分析，对应本节注意力权重与梯度归因不等价的形式化说明。

---

## §3.8 跨界映射：宏观工业供应链

**直觉**

注意力机制有一个简洁的工业类比：全球供应链的动态调度。每个生产环节（Query）发出需求，每个供应商（Key）公布能力标签，softmax 将匹配度归一化为调度权重，最终按权重汇总实际物料（Value）。多头注意力则对应多条并行运作的独立供应链，各自按不同逻辑（语法、语义、话题）调度，最后在总装车间合并。这个比喻把注意力从拟人化的"心理过程"还原为可量化的资源匹配，帮助建立对矩阵运算规模的直觉感受。

---

## §3.9 反类比：当供应链隐喻失效

**直觉**

供应链类比在两处根本性地失效。其一：供应链有硬性产能约束，而 attention 中的 Value 向量可被无限次、无竞争地"调用"——同一 token 可同时以最高权重响应序列中所有其他位置，不存在资源争抢，调度的实质只是相似度计算，而非真正的权衡分配。其二：供应链的优化目标（成本、时效）是可测量的单一指标，而单个注意力头的输出没有明确语义目标——它是反向传播在高维参数空间中塑造出的中间张量，其"功能"是涌现的、上下文相关的，且通常无法用自然语言直接描述。这两处失效共同说明：注意力是数学上透明的，但语义上不透明；可精确计算，但难以直接解释。


# 数学卷·第 4 章：非凸优化与超参数标度律

---

## §4.1 损失函数：如何用数学定义"机器犯错"

**直觉**

训练一个神经网络，本质是找到一组参数，使模型的输出尽可能贴近真实标签。要做到这一点，必须先把"犯错程度"量化成一个数——这个数就是损失函数（loss function）。交叉熵（cross-entropy）损失之所以被普遍采用，不是因为它直觉上最自然，而是因为它与最大似然估计等价，具有坚实的统计学基础。理解这条等价链，是理解为什么"预测概率 vs 正确标签"的对数差距就是最合理的犯错度量的关键。

**严格**

**经验风险（empirical risk）**的定义如下。给定数据集 $\{(x_i, y_i)\}_{i=1}^N$，参数化模型 $f_\theta$，以及单样本损失 $\ell$，经验风险为：

$$\hat{R}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell\bigl(f_\theta(x_i),\, y_i\bigr)$$

训练目标是 $\theta^* = \arg\min_\theta \hat{R}(\theta)$。

**交叉熵损失的来源。** 对 $K$ 分类问题，设真实标签以 one-hot 向量 $\mathbf{y} \in \{0,1\}^K$ 表示，模型输出概率分布 $\mathbf{p} = (p_1,\ldots,p_K)$，则交叉熵（cross-entropy）损失为：

$$L = -\sum_{k=1}^{K} y_k \log p_k$$

其来源可从三条等价路径理解：

1. **最大似然估计（Maximum Likelihood Estimation, MLE）。** 数据集的对数似然为 $\log \prod_i P_\theta(y_i \mid x_i) = \sum_i \log P_\theta(y_i \mid x_i)$。最大化似然即最小化负对数似然（negative log-likelihood）：

$$\min_\theta -\frac{1}{N}\sum_{i=1}^N \log P_\theta(y_i \mid x_i)$$

对 one-hot 标签，这恰好等于 $\frac{1}{N}\sum_i L_i$。

2. **最小化 KL 散度。** 设真实数据分布为 $q$，模型分布为 $p_\theta$，KL 散度为：

$$D_{\mathrm{KL}}(q \| p_\theta) = \sum_k q_k \log \frac{q_k}{p_k} = -\sum_k q_k \log p_k + \text{const}$$

其中常数项（$-\sum_k q_k \log q_k$，即真实分布的熵）不依赖 $\theta$，故最小化 KL 等价于最小化交叉熵。

三条路径（MLE ＝ 最小负对数似然 ＝ 最小 KL 散度）在数学上完全等价。

**Softmax + 交叉熵的联合梯度推导。** 设网络最后一层的 logit 向量为 $\mathbf{z} = (z_1,\ldots,z_K)$，归一化指数（Softmax）将其映射为：

$$p_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

对第 $i$ 个 logit，计算 $\partial p_j / \partial z_i$：

$$\frac{\partial p_j}{\partial z_i} = p_j(\delta_{ij} - p_i)$$

其中 $\delta_{ij}$ 是 Kronecker delta。损失对 $z_i$ 的梯度为：

$$\frac{\partial L}{\partial z_i} = -\sum_k y_k \frac{\partial \log p_k}{\partial z_i} = -\sum_k y_k \frac{1}{p_k} \cdot \frac{\partial p_k}{\partial z_i}$$

代入上式，利用 $\sum_k y_k = 1$（one-hot）：

$$\frac{\partial L}{\partial z_i} = -\sum_k y_k (\delta_{ki} - p_i) = -(y_i - p_i) = p_i - y_i$$

这是一个极其简洁的结果：**反向传播到最后一层 logit 的误差信号，恰好等于预测概率与真实标签之差**。不需要单独计算 Softmax 的雅可比（Jacobian），两者的链式法则自动约分。

**与原书呼应**：原书 §4.1 指出交叉熵"把训练目标和统计学中的最大似然估计无缝衔接"，本节给出了这条等价链的完整代数路径。

---

## §4.2 梯度下降：在黑暗中沿着最陡的方向下山

**直觉**


```{=latex}
\begin{center}
\includegraphics[width=0.92\linewidth]{assets/figs/fig_math_04_01_gradient_descent.png}\\[0.3em]
\small\itshape 图 M4.1 · 梯度下降 · 沿坡度最陡的方向下山,每步把损失变小一点
\end{center}
```
在一个高维空间中找函数的极小值，最自然的思路是：每一步都沿着函数下降最快的方向走一小步。梯度反向就是下降最快的方向，这来自一阶泰勒（Taylor）展开的几何含义。步长（学习率）不能任意大——太大会越过极小值，甚至发散；适当的上界由函数的曲率（李普希茨（Lipschitz）常数）决定。实践中用 mini-batch 近似梯度引入的噪声，在高维优化中反而有正则化效果。动量法和 Adam 是对纯梯度下降的两类重要改进，各有清晰的数学动机。

**严格**

**一阶 Taylor 展开导出更新规则。** 在 $\theta_t$ 处展开：

$$L(\theta_t + \Delta\theta) \approx L(\theta_t) + \nabla L(\theta_t)^\top \Delta\theta$$

在约束 $\|\Delta\theta\| = \epsilon$ 下，使 $L$ 下降最多的方向是 $\Delta\theta = -\epsilon \frac{\nabla L(\theta_t)}{\|\nabla L(\theta_t)\|}$。令步长 $\eta = \epsilon / \|\nabla L\|$，得到梯度下降（gradient descent）更新规则：

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

**步长上界与 Lipschitz 常数。** 若 $L$ 的梯度是 $L$-Lipschitz 连续的，即：

$$\|\nabla L(\theta) - \nabla L(\theta')\| \leq L \|\theta - \theta'\|, \quad \forall \theta, \theta'$$

则可证明当 $\eta < 2/L$ 时，梯度下降保证每步损失不增（对凸函数则保证单调收敛）。直觉：$L$ 是曲率的上界，步长过大会"跨过"极小值导致震荡。

**SGD 的方差与 mini-batch。** 随机梯度下降（Stochastic Gradient Descent, SGD）用 mini-batch $B \subset \{1,\ldots,N\}$ 估计梯度：

$$\tilde{g}_t = \frac{1}{|B|} \sum_{i \in B} \nabla \ell(f_\theta(x_i), y_i)$$

$\tilde{g}_t$ 是真实梯度 $\nabla L(\theta_t)$ 的无偏估计。其方差为：

$$\mathrm{Var}(\tilde{g}_t) = \frac{1}{|B|} \mathrm{Var}(\nabla \ell_i)$$

方差随 batch size $|B|$ 增大而线性减小。更大的 batch 更准确，但引入的随机噪声也更少——这一噪声在 §4.3 中将看到有正面意义。

**动量法（momentum）。** 动量法引入速度向量 $v_t$，模拟质点在损失景观中的惯性运动：

$$v_{t+1} = \mu v_t + \nabla L(\theta_t), \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}$$

其中 $\mu \in [0,1)$ 是动量系数。展开递推可得：

$$v_t = \sum_{k=0}^{t} \mu^k \nabla L(\theta_{t-k})$$

即当前速度是历史梯度的指数加权和。物理类比：质点在黏性介质中滚动——历史方向提供惯性，有助于穿越平坦区域、抑制振荡方向的震荡。

**Adam 的二阶矩估计与偏差修正。** Adam（Adaptive Moment Estimation）同时维护一阶矩 $m_t$ 和二阶矩 $v_t$：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

其中 $g_t = \nabla L(\theta_t)$，逐元素平方。初始时 $m_0 = v_0 = 0$，导致早期估计偏小（偏向零）。偏差修正（bias correction）为：

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

最终更新：

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

核心思想：用 $\hat{v}_t$ 自适应地缩放每个参数的有效学习率——历史梯度幅值大的参数步长缩小，历史梯度幅值小的参数步长放大，使各参数的更新幅度更均匀。典型超参数：$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$。

**与原书呼应**：原书 §4.2 给出了 Adam 的更新规则并描述其为"为每个参数自适应地调整学习率"，本节完整推导了一阶 Taylor 展开来源、步长上界约束，以及偏差修正的数学必要性。

---

## §4.3 高维非凸优化的反直觉：为什么 SGD 居然能工作

**直觉**

神经网络的损失函数是高维非凸的，经典优化理论预言梯度下降几乎必然陷入局部极小值。但现实中，从任意初始化出发训练大模型都能得到好解。这个矛盾的答案藏在高维几何里：高维空间中局部极小值极为稀少，鞍点才是主要障碍；而 SGD 的噪声恰好有能力逃离鞍点；此外，极小值并非孤立，而是构成连通的流形。

**严格**

**鞍点主导：随机矩阵理论的结论。** 设损失函数在临界点 $\theta^*$ 处的 Hessian 矩阵（海森矩阵）为 $H = \nabla^2 L(\theta^*) \in \mathbb{R}^{d \times d}$。局部极小值要求 $H$ 正定（所有特征值 $> 0$），鞍点（saddle point）则要求 $H$ 至少有一个负特征值。

Dauphin 等人（NIPS 2014）借助随机矩阵理论（random matrix theory）中的 Bray-Dean 计算，对高维随机非凸函数得出以下结论：在 $d$ 维空间中，临界点处 Hessian 的负特征值比例 $\rho \in [0,1]$ 决定其类型——$\rho = 0$ 对应局部极小，$\rho > 0$ 对应鞍点。纯局部极小（$\rho = 0$）的数量占总临界点数的比例随维度 $d$ **指数衰减**：

$$\Pr[\text{纯局部极小}] \propto e^{-\Omega(d)}$$

这意味着在 $d \sim 10^8$ 量级的参数空间中，SGD 几乎不会遇到局部极小值——它面对的几乎全是鞍点。

**Hessian 谱的半圆律（semicircle law）。** 在高维随机设定下，随机对称矩阵的特征值经验分布收敛到 Wigner 半圆分布：

$$\rho(\lambda) = \frac{1}{2\pi \sigma^2} \sqrt{4\sigma^2 - \lambda^2}, \quad |\lambda| \leq 2\sigma$$

深度网络损失 Hessian 的谱在过参数化区域近似符合此规律（带有一部分大特征值的"bulk + spike"结构），说明 Hessian 的负特征值占主体，纯局部极小在谱意义上极为罕见。

**SGD 噪声协方差与鞍点逃逸。** SGD 每步的梯度噪声向量 $\xi_t = \tilde{g}_t - \nabla L(\theta_t)$ 的协方差矩阵近似为：

$$\Sigma(\theta) \approx \frac{\eta}{B} \cdot C(\theta)$$

其中 $B$ 是 batch size，$C(\theta) = \frac{1}{N}\sum_i \nabla \ell_i \nabla \ell_i^\top - \nabla L \nabla L^\top$ 是样本梯度的协方差矩阵。在鞍点附近，$C(\theta)$ 的特征向量与 Hessian 负曲率方向有重叠，$\Sigma$ 提供了沿负曲率方向的随机扰动，使模型以 $O(\eta/B)$ 的速率逃离鞍点。减小 batch size 或增大学习率，均可加强这种逃逸能力。

**模式连通性（mode connectivity）。** Garipov、Izmailov 等人（NeurIPS 2018）实验表明：两个独立训练得到的神经网络极小值 $\theta_A$, $\theta_B$，可以用一条损失值始终低的曲线 $\phi: [0,1] \to \mathbb{R}^d$ 连接，即 $L(\phi(t)) \leq \max(L(\theta_A), L(\theta_B)) + \delta$ 对所有 $t$ 成立，$\delta$ 很小。这说明极小值不是孤立的点，而是连通流形（connected manifold）的一部分，SGD 从几乎任何初始化出发均可漂到其中某处。

**与原书呼应**：原书 §4.3 引用 Dauphin et al. 2014 的随机矩阵论据，并描述了 SGD 噪声的隐式正则化效应，本节给出了鞍点主导的指数估计、谱分布形式，以及噪声协方差 $\Sigma \approx \frac{\eta}{B} C(\theta)$ 的结构。

---

## §4.4 损失景观的可视化：黑暗中的地形图

**直觉**

高维损失函数无法直接可视化，但可以沿随机选取的二维切面观察其形状。这种切面图揭示了架构选择对"损失景观平坦度"的影响：有 skip connection 的网络损失景观宽阔平滑，而无 skip connection 的网络则陡峭混乱。平坦极小值（flat minima）不仅好训练，还与更好的泛化能力有数学上可证明的联系。

**严格**

**Filter-wise normalization 切片方法。** Li 等人（NeurIPS 2018）的可视化方案：选取两个随机方向 $d_1, d_2 \in \mathbb{R}^P$，对每个方向按 filter 归一化（filter-wise normalization）——将 $d_1$ 的每个 filter 缩放使其与 $\theta^*$ 中对应 filter 的 Frobenius 范数相等，消除参数尺度差异。随后在二维坐标 $(\alpha, \beta)$ 上绘制：

$$L(\alpha, \beta) = L\!\left(\theta^* + \alpha d_1 + \beta d_2\right)$$

不做 filter-wise normalization 时，方向尺度与参数尺度不匹配，可视化会被参数量级的差异所主导，无法反映真实曲率。

**平坦极小值 vs 尖锐极小值（flat vs sharp minima）。** 设极小值处 Hessian 的最大特征值（谱半径）为 $\lambda_{\max}$。Keskar 等人（ICLR 2017）的实验表明：

- 大 batch SGD 收敛到尖锐极小值（large $\lambda_{\max}$），测试集性能差；
- 小 batch SGD 收敛到平坦极小值（small $\lambda_{\max}$），测试集性能好。

**泛化界（generalization bound）。** 对平坦极小值，存在如下形式的泛化上界：

$$R(\theta) \leq \hat{R}(\theta) + O\!\left(\sqrt{\frac{\lambda_{\max} \cdot n}{N}}\right)$$

其中 $R(\theta)$ 是真实风险，$\hat{R}(\theta)$ 是经验风险，$n$ 是参数数量，$N$ 是样本数。该界说明：$\lambda_{\max}$ 越小（极小值越平坦），泛化误差上界越紧，模型在未见数据上的性能越有保障。直觉：平坦极小值意味着参数在其邻域内小扰动不改变模型输出，模型"不依赖特定参数值"，故泛化更好。

**SAM（Sharpness-Aware Minimization）。** Foret 等人（2021）将平坦性显式化为优化目标：

$$\min_\theta L^{\mathrm{SAM}}(\theta) := \max_{\|\epsilon\| \leq \rho} L(\theta + \epsilon)$$

即在参数邻域内最坏情况的损失最小化，强制解落在平坦区域。SAM 需要两次前向-反向传播（先求 $\epsilon^*$，再计算更新），计算开销加倍，但泛化通常改善显著。

**与原书呼应**：原书 §4.4 描述了 Li et al. 2018 的可视化工作及 skip connection 对损失景观的影响，本节补充了 filter-wise normalization 的数学必要性、泛化界的具体形式，以及 SAM 的优化目标。

---

## §4.5 μP：超参数迁移的"元定律"

**直觉**

在标准初始化下，当模型宽度增大时，各层激活的更新量随宽度发散，导致最优学习率随宽度漂移——在小模型上调好的超参数无法直接用于大模型。μP（Maximal Update Parametrization）通过精心设计初始化方差和学习率随宽度的缩放规则，保证所有层的激活更新量在宽度极限下保持 $O(1)$，从而使最优超参数在不同宽度下稳定不变。

**严格**

**标准参数化（Standard Parametrization, SP）的尺度失控。** 设隐藏层宽度为 $n$，权重 $W \in \mathbb{R}^{n \times n}$ 按 He 初始化 $W_{ij} \sim \mathcal{N}(0, 1/n)$，学习率为 $\eta$。考察单步更新对激活的影响：

- 前向激活：$h^{(l)} = W h^{(l-1)} \in \mathbb{R}^n$，方差 $\mathrm{Var}(h^{(l)}_j) = O(1)$（设计目的）；
- 反向梯度：$g^{(l)} = W^\top g^{(l+1)} \odot \sigma'(z^{(l)})$，尺度 $O(1/\sqrt{n})$（因 $W^\top$ 的谱范数 $\sim O(1)$ 但分量求和数目为 $n$，结果分量方差为 $O(1/n)$，尺度为 $O(1/\sqrt{n})$）；
- 参数更新：$\Delta W = -\eta \, g^{(l+1)} (h^{(l)})^\top$，尺度 $O(\eta / \sqrt{n})$；
- 激活变化：$\Delta h^{(l)} = \Delta W \cdot h^{(l-1)}$，尺度 $O(\eta \sqrt{n})$——**随宽度发散**。

为使 $\Delta h = O(1)$，需令 $\eta = O(1/n)$，即**最优学习率随宽度线性衰减**，迁移不成立。

**μP 的缩放规则。** Yang & Hu 在《Tensor Programs V》中证明，通过以下层依赖缩放可实现所有层 $\Delta h = O(1)$：

| 层类型 | 初始化方差 $\sigma^2$ | SGD 学习率 | Adam 学习率 |
|:---|:---|:---|:---|
| 输入嵌入 $V \to n$ | $O(1)$ | $O(1)$ | $O(1)$ |
| 隐藏 $n \to n$ | $O(1/n)$ | $O(1/n)$ | $O(1)$ |
| 输出 $n \to V$ | $O(1/n^2)$ | $O(1/n)$ | $O(1/n)$ |

其中 $V$ 为词表大小。关键点：在 Adam 下，隐藏层学习率无需随宽度衰减（因 Adam 的二阶矩归一化已吸收了梯度尺度的变化），仅输出层需要 $O(1/n)$ 的衰减。

**与 SP 和 NTK 参数化的对比。** 神经切线核（Neural Tangent Kernel, NTK）参数化是另一种极限：隐藏层初始化方差 $O(1/n)$，学习率 $O(1/n^2)$，使得整个训练过程中参数几乎不动，模型退化为固定核方法（"懒学习"，lazy regime），无法真正学习特征。μP 则保持了"特征学习极限"（feature learning limit）：参数在训练中有 $O(1)$ 的实质更新，模型确实在学习新的特征表示。

$$\text{NTK} \subset \text{SP} \subset \text{μP}$$

（从左到右：核极限 ⊂ 标准参数化 ⊂ 特征学习极限）

三者的本质区别在于激活变化 $\Delta h$ 的尺度：NTK 下为 $O(1/n)$，SP 下为 $O(\sqrt{n})$（随宽度发散），μP 下为 $O(1)$（稳定）。只有 μP 在宽度 $n \to \infty$ 时保持非平凡的特征动力学。

**与原书呼应**：原书 §4.5 描述了 μP 的超参数迁移实证效果，本节补充了尺度失控的完整推导链（$\Delta h \sim O(\eta\sqrt{n})$）以及 SP/NTK/μP 三者的对比表与特征学习含义。

---

## §4.6 Lion 与 Shampoo：Adam 不是终点

**直觉**

Adam 统治 LLM 训练近十年，但近年出现了两条值得关注的挑战路线。Lion 是通过符号优化搜索演化出的算法，只用梯度的符号而非幅值，内存需求减半。Shampoo 则将二阶方法用克罗内克（Kronecker）分解重新带入大规模训练，以可承受的计算代价换取更快的收敛。

**严格**

**Lion（EvoLved Sign Momentum）的更新规则。** Chen 等人（ICML 2023）通过符号优化搜索得到：

$$\begin{aligned}
c_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
\theta_{t+1} &= \theta_t - \eta \cdot \mathrm{sign}(c_t) - \eta \lambda \theta_t \\
m_t &= \beta_2 m_{t-1} + (1 - \beta_2) g_t
\end{aligned}$$

其中 $g_t = \nabla L(\theta_t)$，$\lambda$ 是权重衰减系数，$\mathrm{sign}(\cdot)$ 逐元素取符号。

核心区别：更新方向取 $c_t$ 的符号，每个参数每步只做 $\pm\eta$ 的更新，完全丢弃梯度幅值信息。内存开销：只需维护 $m_t$（一份动量），而 Adam 需要 $m_t$ 和 $v_t$ 两份，节省约 1/3 的优化器状态内存（在混合精度训练中，Adam 优化器状态约占总内存的 2/3，Lion 将其减半）。

理论动机：当损失景观高度各向异性（不同方向曲率差异巨大）时，梯度幅值包含大量尺度噪声，符号反而是更稳健的方向信号。Lion 可以看作带动量的符号随机梯度下降（SignSGD with momentum），其收敛性在一定条件下可证明。

**Shampoo 的 Kronecker 分解预条件。** 对权重矩阵 $W \in \mathbb{R}^{m \times n}$，梯度 $G_t \in \mathbb{R}^{m \times n}$，Shampoo 维护左右二阶矩统计：

$$L_t = \sum_{s=1}^{t} G_s G_s^\top \in \mathbb{R}^{m \times m}, \qquad R_t = \sum_{s=1}^{t} G_s^\top G_s \in \mathbb{R}^{n \times n}$$

更新规则为：

$$W_{t+1} = W_t - \eta \cdot L_t^{-1/4} \, G_t \, R_t^{-1/4}$$

预条件子 $P_L = L_t^{-1/4} \in \mathbb{R}^{m \times m}$，$P_R = R_t^{-1/4} \in \mathbb{R}^{n \times n}$ 近似了精确 高斯-牛顿（Gauss-Newton）预条件子 $(L_t \otimes R_t)^{-1/2}$ 的 Kronecker 分解。计算复杂度：矩阵根 $L_t^{-1/4}$ 需要 $O(m^3)$，$R_t^{-1/4}$ 需要 $O(n^3)$，远低于精确 Hessian 逆的 $O((mn)^3)$。对 $m, n \sim 10^4$ 量级（变换器（Transformer）典型隐藏维度），每步预条件计算是可承受的。

SOAP（2024）在 Shampoo 基础上融合 Adam：将梯度旋转到 Kronecker 矩的特征基中，再在特征基里运行 Adam，兼得二阶预条件与 Adam 自适应性，报告相比 AdamW 减少约 40% 训练步数。

**与原书呼应**：原书 §4.6 描述了 Lion 的符号更新与内存优势，以及 Shampoo 的 Kronecker 分解框架和 SOAP 的融合结果，本节给出了完整的更新方程和计算复杂度分析。

---

> **本节的关键论证**：(1) 在无穷宽极限下，前向激活方差应保持 \(\Theta(1)\)（**激活稳定性条件**）；(2) 反向梯度方差也应保持 \(\Theta(1)\)（**梯度稳定性条件**）；(3) 同时满足两条 → 输入层、隐藏层、输出层的学习率必须按 \(\Theta(1/n)\)、\(\Theta(1/n)\)、\(\Theta(1)\) 缩放（**μP 缩放律**）。**所有「凭空出现」的 \(1/n\) 因子都来自这两条平衡条件**。

## §4.7 μP 的完整面貌：超参数迁移背后的数学

**直觉**

§4.5 给出了 μP 的工程规则。这里深入到其理论基础：Tensor Programs 框架。其核心是：当宽度 $n \to \infty$ 时，网络各层激活的分布在某种弱收敛意义下有极限，而不同参数化方案决定了这个极限是平凡的（核极限）还是非平凡的（特征学习极限）。理解这一框架，揭示了为什么 μP 是唯一"正确"的宽度极限。

**严格**

**Tensor Programs 的核心思想。** Yang 的 Tensor Programs 系列论文（I–V）建立了一套分析框架，核心结论是：在宽度 $n \to \infty$ 的极限下，随机初始化神经网络的所有层间激活（以及梯度、参数更新）的经验联合分布弱收敛（converge weakly）到一个高斯过程（Gaussian process）。具体地，对输入 $x^{(1)}, x^{(2)} \in \mathbb{R}^V$，第 $l$ 层激活向量的内积 $\frac{1}{n} h^{(l)}(x^{(1)}) \cdot h^{(l)}(x^{(2)})$ 在 $n \to \infty$ 时依概率收敛到确定极限，极限的递推由核函数给出。

**abc 参数化（abc-parameterization）。** 对权重矩阵 $W \in \mathbb{R}^{n \times n}$，引入三参数族：

$$W = n^{-a} \tilde{W}, \quad \tilde{W}_{ij} \sim \mathcal{N}(0, 1)$$

学习率 $\eta = n^{-b} \tilde{\eta}$，输出层缩放 $n^{-c}$。其中 $a$（初始化指数）、$b$（学习率指数）、$c$（输出缩放指数）是三个可选参数。

Yang & Hu 证明，要使宽度极限下存在非平凡特征学习（即 $\Delta h = O(1)$），三参数必须满足约束：

$$a + b = 1 \qquad (\text{hidden layers, Adam})$$

具体而言，对隐藏—隐藏权重：$a = 1$ 对应初始化方差 $O(1/n)$，$b = 0$ 对应 Adam 学习率 $O(1)$（不随宽度缩放）。违反此约束会导致要么激活爆炸（$a + b < 1$），要么激活冻结退化为核极限（$a + b > 1$）。

**特征学习极限（feature learning limit）vs 核极限（kernel limit / NTK）。** 在 NTK 参数化下，$a = 1/2$（He 初始化），$b = 0$（固定学习率），$a + b = 1/2 < 1$（对 SGD），$\Delta W = O(\eta/\sqrt{n})$，$\Delta h = O(\eta) \to 0$（宽度极限下特征不变），网络退化为固定核方法，训练动力学由初始 NTK 完全决定，与宽度无关（不再学习新特征）。

μP 下，$a = 1$（隐藏层初始化方差 $O(1/n)$），$b = 0$（Adam），$\Delta h = O(1)$，网络在宽度极限下仍有 $O(1)$ 的特征更新量——这是"特征学习极限"。它是唯一使特征动力学在 $n \to \infty$ 时既非爆炸也非消失的参数化，也是使超参数在宽度极限下有稳定极值的唯一参数化。

**与原书呼应**：原书 §4.7 描述了 Tensor Programs 框架与 abc 参数化的核心思想，以及特征学习极限与核极限的本质区别，本节给出了弱收敛的形式化陈述、约束方程 $a + b = 1$ 的来源，以及两类极限的对比。

---

## §4.8 跨界类比：复杂系统的工程管理

**直觉**

原书将训练大模型类比为"指挥数千亿员工的超大型工程项目"：反向传播对应绩效分摊系统，学习率对应调整幅度，Adam 对应个性化绩效考核，μP 对应规模不变的管理制度设计。

**严格（简版）**

类比的数学核心在于链式法则的线性性：每个参数的梯度 $\partial L / \partial \theta_i$ 通过链式法则被精确分解为各层贡献之和，这与责任分摊（responsibility allocation）的线性叠加结构同构。Adam 的自适应步长 $\eta / (\sqrt{\hat{v}_t} + \epsilon)$ 对历史梯度方差大的参数自动降低有效步长，这与"不确定性越大，调整越保守"的管理逻辑一致。μP 的"规模不变性"对应于管理制度在员工数量变化时的结构不变性——是代数上的 $n$-均匀性（$n$-homogeneity），而非工程直觉。

---

## §4.9 反类比：工程管理隐喻的失效边界

**直觉**

类比在三点上失效：深度学习训练没有明确的"完成"状态；神经元的职责是涌现的，不是预先指定的；损失函数是高维非凸的，不像工程优化那样可分解为凸子问题。

**严格（简版）**

1. **无终止条件。** 工程完工有二值判断（done/not done），训练的停止准则是连续的（验证集 loss 曲线的斜率阈值或算力预算约束），本质上是一个截断（truncation）而非收敛（convergence）。

2. **功能涌现（emergent function）。** 每个参数的"语义"由训练动力学决定，事后才能通过可解释性方法（如激活补丁、稀疏自编码器）反推，不存在先验的功能指定。这与工程系统的模块化分工在信息论层面不同构。

3. **非凸 vs 凸。** 工程最优化问题（成本最小化、路径最短化）通常可转化为凸规划，有全局最优保证。深度学习损失景观的非凸性不可消除，训练成功依赖高维几何的"幸运"结构（鞍点主导、模式连通性），而非可控的算法保证。第四章的核心教训是：**我们能利用这些几何性质，但无法从工程意义上"控制"它们**。


# 数学卷·第 5 章：标度律、相变与涌现的数理基础

---

## §5.1 尺度律（Scaling Law）：算力即权力的冷酷物理感

**直觉**

原书 §5.1 揭示了一个令整个工业界沉默的事实：语言模型的损失值在七个数量级的范围内，对参数量 $N$、数据量 $D$、算力 $C$ 呈精确的幂律下降。这意味着模型性能可以被预测，AI 研究从"艺术"变成"重工业"。

为什么会是幂律而不是指数律？直觉是：指数律意味着每增加一个单位投入，收益以固定比率衰减——最终收益几乎为零；幂律则意味着收益以递减但永不截断的方式下降，在双对数坐标上表现为一条无限延伸的直线。幂律描述的是自相似（scale-free）结构，而语言的复杂性恰好在很多层次上具有自相似性（词→短语→段落→篇章）。这不是巧合，而是语言统计结构的深层反映。

**严格**

**Kaplan 单变量幂律。** 设 $N$ 为非嵌入参数量，$D$ 为训练 token 数，$C$ 为训练总 FLOPs，Kaplan 等（2020）拟合出：

$$
\mathcal{L}(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad
\mathcal{L}(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad
\mathcal{L}(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

其中 $\alpha_N \approx 0.076$，$\alpha_D \approx 0.095$，$\alpha_C \approx 0.050$，$N_c, D_c, C_c$ 为拟合常数。

**幂律 vs 指数律的渐近行为。** 设 $f(x) = x^{-\alpha}$（幂律），$g(x) = e^{-\lambda x}$（指数律）。对任意 $\alpha, \lambda > 0$，有：

$$
\lim_{x \to \infty} \frac{f(x)}{g(x)} = \lim_{x \to \infty} x^{-\alpha} e^{\lambda x} = +\infty
$$

即指数律衰减比任意幂律都快——对数坐标下，幂律是直线，指数律是向下弯曲的曲线。双对数坐标下，$\log \mathcal{L} = -\alpha \log N + \text{const}$，斜率即幂指数 $-\alpha$。

**Hoffmann 联合损失函数（为 §5.2 铺垫）。** 原书在 §5.1 引入 Kaplan，在 §5.2 引入 Chinchilla 的联合形式：

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

其中 $E \geq 0$ 是不可归约损失（irreducible loss），即语言本身的熵下限；$A/N^\alpha$ 表示参数不足导致的欠拟合；$B/D^\beta$ 表示数据不足导致的欠拟合。$\alpha, \beta > 0$ 为正数，意味着两项均单调递减。

**Compute-optimal 的拉格朗日推导。** 给定固定算力预算 $C$，变换器（Transformer）的标准 FLOPs 近似为：

$$
C \approx 6ND
$$

（每个参数在前向+反向传播中各贡献约 $2D$ 次乘加，共 $6ND$。）目标是在约束 $6ND = C$ 下最小化 $L(N, D)$。构造拉格朗日函数：

$$
\mathcal{L}(N, D, \lambda) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} - \lambda(6ND - C)
$$

对 $N$ 和 $D$ 分别求偏导并令其为零：

$$
\frac{\partial \mathcal{L}}{\partial N} = -\frac{\alpha A}{N^{\alpha+1}} - 6\lambda D = 0
\quad \Rightarrow \quad
\frac{\alpha A}{N^{\alpha+1}} = 6\lambda D \tag{1}
$$

$$
\frac{\partial \mathcal{L}}{\partial D} = -\frac{\beta B}{D^{\beta+1}} - 6\lambda N = 0
\quad \Rightarrow \quad
\frac{\beta B}{D^{\beta+1}} = 6\lambda N \tag{2}
$$

由 $(1)$ 和 $(2)$ 消去 $\lambda$：

$$
\frac{\alpha A}{N^{\alpha+1}} \cdot \frac{1}{D} = \frac{\beta B}{D^{\beta+1}} \cdot \frac{1}{N}
\quad \Rightarrow \quad
\frac{\alpha A}{N^\alpha D} = \frac{\beta B}{N D^\beta}
\quad \Rightarrow \quad
\frac{\alpha A}{N^\alpha} = \frac{\beta B}{D^\beta}
$$

即在最优点处，参数项与数据项的边际贡献之比满足：

$$
\frac{A/N^\alpha}{B/D^\beta} = \frac{\beta}{\alpha} \tag{3}
$$

结合约束 $6ND = C$，由 $(3)$ 解出：

$$
N^* \propto C^{\,\beta/(\alpha+\beta)}, \quad D^* \propto C^{\,\alpha/(\alpha+\beta)}
$$

当 $\alpha \approx \beta$（Hoffmann 拟合值 $\alpha \approx 0.34$，$\beta \approx 0.28$ 近似）时，$N^* \propto C^{0.5}$，$D^* \propto C^{0.5}$，即参数量与 token 数应等比例扩展。

更一般地，若记 $r = \alpha/\beta$，则：

$$
N^*_{\rm opt} \propto C^{1/(1+r)}, \quad D^*_{\rm opt} \propto C^{r/(1+r)}
$$

当 $r > 1$（参数的边际回报衰减更快）时，$D^*$ 的增长指数大于 $N^*$，表明应更多增加数据。

**与原书呼应**：原书 §5.1 提到"模型的性能可以被精确预测"，对应本节幂律拟合与拉格朗日推导给出的 compute-optimal 配比公式 $N^*_{\rm opt} \propto C^{\beta/(\alpha+\beta)}$。

---

## §5.2 Chinchilla 的颠覆与再颠覆

**直觉**

Hoffmann 等（2022）用同样的联合损失函数 $L(N,D)$，却得出了与 Kaplan 截然不同的结论：参数量与训练 token 数应以约 1:20 的比例同等扩展，而非"参数优先"。这一修正背后有三种不同的拟合方法，它们给出的结论方向一致，但置信区间宽窄差异悬殊。Epoch AI 的复现工作揭示了其中方法三的统计学问题，而 Sardana 等（ICML 2024）进一步指出，即使 Chinchilla 最优解在训练成本上正确，也忽略了推理成本——真实世界中推理调用数十亿次，应将推理成本纳入总优化目标。

**严格**

**Hoffmann 三种拟合方法。** 设损失模型为：

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} \tag{4}
$$

共有 $5$ 个参数 $\{E, A, \alpha, B, \beta\}$ 待拟合。Hoffmann 等使用了三种方法：

- **方法一（IsoFLOP）**：固定总算力 $C = 6ND$，在多组 $(N, D)$ 上测量损失，拟合每条等算力曲线的最优 $N^*$，再对 $N^*(C)$ 做幂律回归，得出 $N^* \propto C^{0.50}$。
- **方法二（IsoLoss）**：找出达到相同损失值所需的 $(N, D)$ 组合，沿等损失曲线回归，得出类似结论。
- **方法三（参数化拟合）**：直接对全部 $(N, D, L)$ 观测数据用最小二乘拟合方程 $(4)$ 的 $5$ 个参数，Hoffmann 报告的置信区间极窄（如 $\alpha = 0.34 \pm 0.02$）。

三种方法的方向性结论一致：最优 $N^*/D^*$ 比约为 $1/20$（即每个参数对应约 $20$ 个训练 token），与 Kaplan 建议的 $1/5$ 差异显著。

**Epoch AI 复现与置信区间统计学根源。** Epoch AI（2024）指出，方法三的置信区间过窄，在统计学上源于以下机制：

方法三对有限的训练点做非线性最小二乘，设观测数为 $n$，参数数为 $p = 5$，残差方差为 $\hat{\sigma}^2$，则参数 $\hat{\theta}$ 的渐近协方差矩阵为：

$$
\mathrm{Cov}(\hat{\theta}) \approx \hat{\sigma}^2 \left(J^\top J\right)^{-1}
$$

其中 $J \in \mathbb{R}^{n \times p}$ 是雅可比矩阵。置信区间宽度正比于 $\hat{\sigma} / \sqrt{n}$。Chinchilla 论文实际训练的模型数量约为 $400$ 个，但非线性拟合中信息矩阵 $J^\top J$ 的条件数往往很大（参数 $\alpha, \beta$ 与 $A, B$ 之间存在强相关性），导致 $\left(J^\top J\right)^{-1}$ 的对角元素被低估，最终报告的 CI 比实际参数不确定性窄得多。要支撑如此紧致的置信区间，理论上需要数十万量级的独立观测点。

**Sardana 推理时计算修正。** 设 $C_{\rm train}$ 为训练总算力，$C_{\rm inf}$ 为单次推理算力（约 $2N$ FLOPs），$n_{\rm inf}$ 为生命周期内推理调用次数，总算力预算为 $C_{\rm total}$。扩展后的优化问题为：

$$
\min_{N, D} \; L(N, D) \quad \text{s.t.} \quad C_{\rm train} + n_{\rm inf} \cdot C_{\rm inf} \leq C_{\rm total}
$$

其中 $C_{\rm train} \approx 6ND$，$C_{\rm inf} \approx 2N$，故约束变为：

$$
6ND + 2N \cdot n_{\rm inf} \leq C_{\rm total}
\quad \Rightarrow \quad
N\bigl(6D + 2n_{\rm inf}\bigr) \leq C_{\rm total} \tag{5}
$$

当 $n_{\rm inf} \gg 3D$（即推理调用次数远超训练 epoch 数的三倍）时，约束 $(5)$ 的主导项变为 $2N n_{\rm inf}$，即推理成本主导。此时拉格朗日最优解偏向使用**更小的 $N$**（配合更多训练 token $D$），因为减小 $N$ 既降低推理成本，又可将节省下来的预算投入更多训练数据。

这是 LLaMA-3 8B 使用 15T tokens（约为 Chinchilla 最优的 $100$ 倍）训练的理论依据：在推理量极大的部署场景下，小模型+超量训练比大模型+少量训练更经济。

**与原书呼应**：原书 §5.2 提到"Chinchilla 所指出的方向性结论仍然具有重要影响"及 Epoch AI 对"方法三置信区间过窄"的批评，对应本节对信息矩阵条件数问题的分析，以及 Sardana 修正公式 $\min L(N,D) \; \text{s.t.} \; C_{\rm train} + n_{\rm inf} \cdot C_{\rm inf} \leq C_{\rm total}$。

---

## §5.3 数据墙与 §5.4 推理时计算

**直觉**

原书 §5.3 提出数据墙（data wall）：公开人类文本总量约 $3 \times 10^{14}$ token，在当前 scaling 速率下将于 2026–2032 年耗尽。§5.4 描述了"第二曲线"——推理时计算缩放（inference-time scaling），通过在推理阶段投入更多算力（链式推理、N 选优（Best-of-N）采样）来补偿训练数据的不足。

数学上，数据墙是一个约束边界：$D \leq D_{\max} \approx 3 \times 10^{14}$。一旦 $D$ 到顶，若仍要降低 $L$，只能继续增大 $N$——但 $B/D^{\beta}$ 项固定，损失存在一个受数据量支配的下界。

**严格**

**模型崩溃（model collapse）的分布退化。** Shumailov 等（Nature 2024）证明，若模型递归训练于自身生成的数据，第 $n$ 代模型的输出分布 $\mathcal{D}_n$ 的方差满足：

$$
\Sigma_n \xrightarrow{a.s.} 0 \quad \text{as } n \to \infty
$$

直觉：每代生成数据相当于对真实分布做一次有噪采样后再拟合，低概率区域（尾部）在采样时被截断，经多代迭代后尾部彻底消失，分布收敛到退化点。Wasserstein-2 距离满足：

$$
\mathbb{E}\bigl[\mathbb{W}_2^2(\mathcal{N}(\mu_n, \Sigma_n),\, \mathcal{D}_0)\bigr] \to \infty
$$

即合成数据与原始真实分布的距离随代数增加而发散。

**推理时计算的等价 token 折算。** 设基础模型参数量为 $N_0$，推理时使用 Best-of-$K$ 采样：从 $K$ 个独立采样中选最优输出，验证成本约为 $K \cdot 2N_0$ FLOPs。如果以"等价训练 token"衡量推理投入，令 $C_{\rm test} = K \cdot 2N_0$，则其对应的等价训练 token 数为：

$$
D_{\rm equiv}(K) = \frac{C_{\rm test}}{6N_0} = \frac{K}{3}
$$

即每 Best-of-3 采样大约相当于额外提供 $1$ 个训练 token 的算力。实践中，推理时计算与训练时计算的"兑换率"取决于任务类型：可验证任务（数学、代码）兑换率高，因为有客观奖励信号指导采样选择；不可验证任务（创意写作）兑换率低。

**与原书呼应**：原书 §5.3 引用 Shumailov 等人的"模型崩溃"结果，对应本节 $\Sigma_n \to 0$ 的方差退化；§5.4 描述推理时计算缩放，对应本节 $D_{\rm equiv}(K) = K/3$ 的等价折算关系。

---

## §5.5 涌现：神话还是幻觉？

**直觉**

原书 §5.5 的核心争论：Wei 等（2022）记录了许多任务上随模型规模增大出现"突然跃升"的准确率——被称为涌现能力（emergent abilities）。Schaeffer、Miranda 与 Koyejo（NeurIPS 2023 杰出论文）反驳说，这种"涌现"是测量幻觉：研究者选择了分段不连续的评估指标，把底层平滑的能力增长映射成了视觉上的阶跃。

类比：用阶梯状的温度计（只有 0 和 1 两个刻度）去量一壶从室温缓慢升温的水，读数会从 0 突然跳到 1——但水温本身是连续上升的。换连续刻度的温度计，"突变"就消失了。

**严格**

**指标非线性导致的表观涌现。** 设模型规模为 $N$，token 级正确率（token-level accuracy）为 $p(N)$，假设其随 $N$ 连续、单调递增（例如随 $\log N$ 线性增长）。

对长度为 $\ell$ 的序列，"全序列正确"（exact match）指标为：

$$
M_{\rm exact}(N) = \prod_{i=1}^{\ell} p_i(N) \approx p(N)^\ell
$$

其中假设各 token 独立且每位正确率相同。当 $p(N) < 1/\ell$ 时，$M_{\rm exact} \approx 0$；当 $p(N)$ 略超过 $1/\ell$ 时，$M_{\rm exact}$ 迅速从 $0$ 跳向正值。具体地，在 $p(N^*) = 1/\ell^{1/\ell}$ 附近，$M_{\rm exact}$ 对 $N$ 的导数为：

$$
\frac{dM_{\rm exact}}{dN} = \ell \cdot p(N)^{\ell-1} \cdot \frac{dp}{dN}
$$

当 $\ell$ 大时（如 $\ell = 10$ 的三位数加法），$\ell \cdot p^{\ell-1}$ 在过渡区附近极大，导致 $M_{\rm exact}$ 对 $N$ 呈近似阶跃响应。Schaeffer 将此一般化为：若评估指标 $m(p)$ 是关于 $p$ 的非线性函数（如 $m(p) = \mathbf{1}[p \geq \theta]$），则：

$$
\frac{dm}{dN} = m'(p) \cdot \frac{dp}{dN}
$$

对阶梯函数 $m(p) = \mathbf{1}[p \geq \theta]$，其导数为 Dirac delta：$m'(p) = \delta(p - \theta)$，故：

$$
\frac{dm}{dN} = \delta\bigl(p(N) - \theta\bigr) \cdot \frac{dp}{dN}
$$

这是一个在 $p(N^*) = \theta$ 处集中的脉冲——视觉效果即"涌现"。而换用连续指标 $m(p) = p$（token-level log-likelihood）时，$m'(p) = 1$，曲线即还原为 $dp/dN$ 的平滑形状。

**连续度量下的平滑性验证。** Schaeffer 等在实验中替换指标后验证：原本呈现"涌现"的任务，在 token-level log-likelihood 下均表现为平滑的幂律增长，与预训练损失曲线吻合。这一结果的可证伪推论是：**若某任务的"涌现"在所有连续可微指标下均持续存在，则该涌现更可能是真实能力阈值，而非指标伪影**。

反驳路线保留的空间：Olsson 等（2022）的 induction head 工作从机制层面证明，in-context learning 能力的出现与模型内部 induction head 电路的形成在训练步骤上高度同步，呈现明确的阶跃特征——即使换用连续指标，该阶跃仍然可见。这说明存在少数真实的能力阈值，但它们与物理意义上的"相变"仍有本质区别（见 §5.10）。

**与原书呼应**：原书 §5.5 引用 Schaeffer 等（NeurIPS 2023 杰出论文），对应本节 $dm/dN = \delta(p(N)-\theta) \cdot dp/dN$ 的形式分析；原书提及换用 token-level log-likelihood 曲线变平滑，对应本节连续指标下的 $m'(p)=1$ 情形。

---

## §5.6–§5.7 Grokking 的数学解剖：从死记到顿悟

**直觉**

原书 §5.6–§5.7 描述了 Grokking 现象（Power 等，2022）：在模算术任务 $a + b \bmod p$ 上，模型先 100% 记忆训练集（测试准确率约 0%），再经数万步训练后测试准确率突然跃升至 100%。这被称为"延迟泛化"。

Nanda 等（ICLR 2023）的机制可解释性研究揭示，模型并非在"顿悟"——它悄悄学到了离散傅里叶基表示，并用三角恒等式实现了模加法。权重衰减（weight decay）是驱动这一转变的关键正则化力量：它持续压缩稠密记忆权重，直到更紧凑的傅里叶电路"胜出"。

**严格**

**任务设置。** 取素数 $p = 113$，输入为 $(a, b) \in \{0, \ldots, p-1\}^2$，目标为 $(a+b) \bmod p$。训练集取所有 $p^2$ 对中的 30%，测试集为剩余 70%。

**Embedding 矩阵的傅里叶基收敛。** Nanda 等逆向工程发现，训练后模型的 embedding 矩阵 $E \in \mathbb{R}^{p \times d}$（$d$ 为隐藏维度）在若干频率 $k$ 处形成了如下结构：

$$
E[a, 2k] = \cos\!\left(\frac{2\pi k a}{p}\right), \quad
E[a, 2k+1] = \sin\!\left(\frac{2\pi k a}{p}\right), \quad k \in \mathcal{K}
$$

其中 $\mathcal{K}$ 是模型自发选择的少数几个频率（实验中约 5 个）。这等价于将整数 $a$ 映射为其在特定频率下的复数相位 $e^{2\pi i k a/p}$。

**注意力层实现频域加法。** 设 attention 层在频率 $k$ 处的操作将 $a$ 和 $b$ 的嵌入合并，实现：

$$
\cos\!\left(\frac{2\pi k (a+b)}{p}\right)
= \cos\!\left(\frac{2\pi k a}{p}\right)\cos\!\left(\frac{2\pi k b}{p}\right)
- \sin\!\left(\frac{2\pi k a}{p}\right)\sin\!\left(\frac{2\pi k b}{p}\right)
$$

这正是余弦的加法定理。模型利用 bilinear 注意力结构（query 与 key 的内积）自然地实现了这一乘积——两个 embedding 向量的内积提取了跨频率的 $\cos(\theta_a)\cos(\theta_b)$ 和 $\sin(\theta_a)\sin(\theta_b)$ 项。

**输出层的逆傅里叶解码。** 网络最终输出 logit 向量 $\hat{y} \in \mathbb{R}^p$，其中：

$$
\hat{y}_c \propto \sum_{k \in \mathcal{K}} w_k \cos\!\left(\frac{2\pi k (a+b-c)}{p}\right)
$$

对 $c$ 取 argmax，即等价于用若干傅里叶频率的叠加来定位 $(a+b) \bmod p$。

**两阶段动力学：权重衰减驱动的延迟泛化。** 定义两个进度指标：

$$
L_{\rm Fourier}(t) = L\bigl(\text{仅保留傅里叶频率分量的模型}\bigr), \quad
L_{\rm mem}(t) = L\bigl(\text{去除傅里叶分量后的模型}\bigr)
$$

训练动力学分三阶段：

| 阶段 | 步数（$p=113$） | 训练 loss | 测试 loss | $L_{\rm Fourier}$ | $L_{\rm mem}$ |
|---|---|---|---|---|---|
| 记忆 | $0 - 1\text{k}$ | 快速下降 | 不动 | 高 | 快速下降 |
| 电路竞争 | $1\text{k} - 9\text{k}$ | 接近 0 | 不动 | 下降 | 下降 |
| 清理 | $9\text{k} - 14\text{k}$ | $\approx 0$ | 突然下降至 0 | $\approx 0$ | $\approx 0$ |

权重衰减的数学角色：设权重为 $\theta$，$\ell_2$ 正则化强度为 $\lambda_{\rm wd}$，则训练目标为：

$$
\mathcal{J}(\theta) = L_{\rm CE}(\theta) + \frac{\lambda_{\rm wd}}{2}\|\theta\|_2^2
$$

稠密记忆电路需要大量权重（大 $\|\theta\|_2^2$），而稀疏傅里叶电路用少数频率分量表示同样的功能，权重范数更小。正则化项持续惩罚记忆电路，使其在竞争中逐渐处于劣势；当记忆电路权重衰减至低于傅里叶电路的表达阈值时，后者独占输出，测试 loss 才突然降至 0。

这解释了 Grokking 的关键实验观察：增大 $\lambda_{\rm wd}$ 可以提前 Grokking 时刻，减小 $\lambda_{\rm wd}$ 则推迟甚至阻止 Grokking——表明这是正则化驱动的竞争过程，而非任何形式的真实相变。

**与原书呼应**：原书 §5.7 引用 Nanda 等（ICLR 2023），对应本节 embedding 的傅里叶基收敛公式与三角恒等式推导；§5.6 提及"权重衰减驱动延迟泛化"，对应本节正则化目标 $\mathcal{J}(\theta)$ 及两阶段动力学表格。

---

## §5.8 渗流相变：把 Scaling Law 焊到统计物理上

**直觉**

原书 §5.8 将"涌现"与统计物理中的渗流理论（percolation theory）相联系：把模型掌握的"概念"想象成图上的节点，复合能力对应图上的连通路径。当单概念掌握概率越过临界值 $p_c$，最大连通簇尺寸从对数级跳到线性级——这给"涌现"一个可证伪的渗流解释。

渗流为什么是合适的类比？因为它恰好是"单元素平滑增长 $\to$ 整体性质急剧变化"的典型模型，与"底层能力平滑增长、表观性能阶跃"的结构吻合。

**严格**

**二维 site percolation 的临界现象。** 考虑二维正方形格点 $\mathbb{Z}^2$ 上的伯努利渗流（Bernoulli site percolation）：每个格点以概率 $p$ 被"占据"，彼此独立。临界概率 $p_c \approx 0.5927$（数值精确值）。

设 $P_\infty(p)$ 为无穷大格点中随机选取的占据格点属于无穷大连通簇的概率（序参量）。在 $p_c$ 附近：

$$
P_\infty(p) \sim (p - p_c)^\beta, \quad p \to p_c^+
$$

其中临界指数 $\beta = 5/36$。关联长度（correlation length）$\xi$，即连通簇线性尺度的特征值，满足：

$$
\xi(p) \sim |p - p_c|^{-\nu}, \quad \nu = 4/3
$$

$\xi \to \infty$ 意味着在临界点附近，任意尺度的涨落均存在——这是连续二阶相变的标志。序参量 $P_\infty$ 在 $p_c$ 处连续地从 $0$ 上升（不发生不连续跳变），故为二阶（连续）相变。

这两个精确临界指数（$\beta = 5/36$，$\nu = 4/3$）来自二维伯努利渗流的精确结果，是严格数学意义上（通过共形场论和 SLE 理论）已经证明的，并非数值近似。

**能力涌现的渗流映射。** 将单概念 $c$ 的"掌握概率"建模为规模的函数：

$$
p(N) = \sigma(\alpha \log N + \beta_0) = \frac{1}{1 + N^{-\alpha} e^{-\beta_0}}
$$

其中 $\sigma$ 为 sigmoid 函数，$p(N)$ 关于 $N$ 连续单调递增。

定义复合能力函数 $\mathcal{C}(N)$ 为"存在长度 $\geq k$ 的全连通路径"的概率（即多跳推理成功的概率）。在渗流框架下，当 $p(N)$ 越过 $p_c$ 时：

$$
\frac{d\mathcal{C}}{dN} \approx \delta\bigl(p(N) - p_c\bigr) \cdot \frac{dp}{dN}
$$

单概念能力 $p(N)$ 平滑，但复合能力 $\mathcal{C}(N)$ 在 $p$ 过 $p_c$ 时呈现类 delta 脉冲。可证伪预测：**能力涌现的陡峭程度应正比于推理链长度 $k$**——单步任务平滑，多步任务陡峭。

**神经网络剪枝的相变行为。** Pesce、He 与 Caldarelli（2026）研究神经网络剪枝时发现：随着删除边的比例 $q$ 从 0 增大，网络功能（以某任务准确率衡量）在临界比例 $q_c$ 处发生从"基本保持"到"完全失能"的急剧转变，且临界指数与渗流的标度律一致。实验表明，最高可移除约 98% 的边而性能损失很小，越过 $q_c$ 后才发生崩塌。

设网络功能 $F$ 为边保留率 $\bar{q} = 1-q$ 的函数，则：

$$
F(\bar{q}) \sim (\bar{q} - \bar{q}_c)^\beta \cdot \mathbf{1}[\bar{q} > \bar{q}_c]
$$

其中 $\bar{q}_c \approx 0.02$（约 2% 保留）对应 $q_c \approx 98\%$。这与渗流的 $P_\infty \sim (p-p_c)^\beta$ 在形式上同构，指向同一物理机制：**网络功能是连通性的函数**。

**与原书呼应**：原书 §5.8 提到二维 site percolation 的临界指数 $\beta = 5/36$、$\nu = 4/3$ 及 Pesce 等（2026）剪枝临界比例约 98%，对应本节严格推导；渗流序参量连续上升（二阶相变）的说明对应原书"标准二维伯努利渗流是连续（二阶）相变"的表述。

---

## §5.9 跨界映射与 §5.10 反类比：相变叙事的边界

**直觉**

原书 §5.9 列举了农业革命、城邦兴起、工业革命、互联网等历史拐点，将它们类比为文明的"相变"，并指出 AI 涌现被许多业界领袖以"phase transition"的语言包装。§5.10 则给出三条反驳：物理相变要求热力学极限；文明"相变"是后视镜叙事，不是预测工具；物理相变（除特殊情形外）可逆，文明涌现不可逆。

**严格**

**热力学极限的必要性。** 设系统有 $N$ 个粒子（或参数）。自由能密度为 $f(N, T) = F(N,T)/N$，其中 $T$ 为温度（或某控制参数）。严格意义上的相变（即 $f$ 在某 $T_c$ 处的非解析性）仅在热力学极限下存在：

$$
f(T) = \lim_{N \to \infty} \frac{F(N, T)}{N}
$$

对有限 $N$，$f(N,T)$ 是有限个指数函数的对数之和，解析函数的有限和仍是解析函数，因此严格奇异点不存在。有限系统观察到的只是平滑的**交叉（crossover）**行为，而非奇异点。

大语言模型即使有 $10^{12}$ 个参数，仍是有限系统；其"涌现"行为即使在某些指标下看似陡峭，在严格数学意义上也只是 crossover，而非相变。这一区别不仅是字面上的：crossover 没有普适临界指数，其"陡峭程度"依赖模型细节，无法从统计物理的普适类理论中推导出任何定量预测。

**后视镜叙事的不可预测性。** 设文明状态变量为 $\mathbf{x}(t) \in \mathbb{R}^d$，"相变时刻"为 $t^*$ 使得某序参量 $\phi(\mathbf{x}(t))$ 在 $t^*$ 附近急剧变化。历史学家能在 $t > t^*$ 时用 $\phi$ 描述转变，但事前预测 $t^*$ 需要知道 $\phi$ 的具体形式——而这在转变发生前往往未知。AI 语境中，"下一次涌现何时发生"同样无法从现有理论推导，Schaeffer 等的工作进一步表明，若涌现是指标伪影，则"预测涌现"更无理论基础。

**不可逆性。** 一阶相变（如水的液固转变）满足 Clausius-Clapeyron 方程，可沿相图逆向路径恢复原相；二阶相变（如铁磁相变）在翻转控制参数后同样可逆。文明相变（如工业化）和 AI 能力涌现均不满足时间反演对称性——已习得的能力无法通过逆向训练消除。这一不可逆性意味着，物理相变与 AI/文明涌现之间的类比缺乏动力学层面的同构，只是在序参量行为的表观形式上相似。

**特别说明：有限 $N$ 系统只能有 crossover。** 以上分析的核心结论值得单独强调：热力学极限 $N \to \infty$ 是严格相变的必要条件，任何有限参数量的神经网络，无论其涌现行为看起来多么陡峭，数学上均为 crossover，而非奇异点。这一限制不依赖于模型大小的具体数值，而是有限系统的普遍性质。

**与原书呼应**：原书 §5.10 明确指出"热力学极限是严格相变的必要条件，有限 $N$ 系统只能有 crossover"，对应本节自由能密度对 $N$ 极限的分析；"物理相变背后有严格统计力学理论；AI 涌现背后只有经验观察"，对应本节后视镜叙事与不可逆性的讨论。

---

*本章对应原书第五章「量变引发的智能飞跃——统计物理与复杂系统涌现」（§5.1–§5.10）。主要参考文献：Kaplan et al. (2020)；Hoffmann et al. (2022)；Epoch AI (2024)；Sardana et al. (ICML 2024)；Schaeffer, Miranda & Koyejo (NeurIPS 2023)；Nanda et al. (ICLR 2023)；Power et al. (2022)；Pesce, He & Caldarelli (2026)；Shumailov et al. (Nature 2024)。*


# 数学卷·第 6 章:硬件感知算法的数学——屋脊线（Roofline）、I/O 复杂度与 FlashAttention 推导

> **导读**：主本第六章以产业链视角讲述为什么 HBM 已成为万亿美元生意；本章在此基础上，逐一补全那些被主本略去的不等式推导。读者需要掌握矩阵分析、基础算法复杂度与变换器（Transformer）结构的预备知识（可参照数学卷第三章 §3.5 关于 FlashAttention 直觉版的铺垫）。本章的核心线索是：**带宽是常数，算力在增长，因此算法必须在数学层面不断"提升算术强度"，才能不被内存墙钉死**。

---

## §6.1 Roofline 模型的形式化

### 6.1.1 算术强度的精确定义

设某计算核（kernel）执行 \(W\) 次浮点运算，在此过程中从最慢存储层级（对现代 GPU 即 HBM）读写共计 \(Q\) 字节数据。定义**算术强度**（arithmetic intensity）为

\[
I \;=\; \frac{W}{Q} \quad [\text{FLOP/byte}].
\]

Williams、Waterman 与 Patterson 在 2009 年的原始论文中使用术语"operational intensity"，本章与数学卷其余各章统一改用更通行的"算术强度"，两者同义。

若存储层级选取不同（如 L2 而非 HBM），则得到**层次化 Roofline**（hierarchical Roofline），但本章专注于 HBM 层级，以与主本第六章的产业链叙事对应。

### 6.1.2 性能上界与 Roofline 公式

设硬件的浮点峰值算力为 \(P_\text{peak}\)（FLOP/s），HBM 带宽为 \(\beta\)（byte/s）。对一个算术强度为 \(I\) 的 kernel，其可达性能上界为

\[
P \;\leq\; \min\!\bigl(P_\text{peak},\; I \cdot \beta\bigr).
\]

直觉：若 kernel 是**内存受限**（memory-bound），则每秒可喂入算力单元的数据量为 \(\beta\) byte/s，携带算力仅 \(I \cdot \beta\) FLOP/s；若 kernel 是**计算受限**（compute-bound），则上限即为 \(P_\text{peak}\)。两者取 min 给出严格上界。

### 6.1.3 脊点（Ridge Point）的推导

两个上界相交的算术强度称为**脊点** \(I^*\)，由

\[
I^* \cdot \beta = P_\text{peak}
\]

解得

\[
\boxed{I^* = \frac{P_\text{peak}}{\beta}.}
\]

当 \(I < I^*\) 时 kernel 内存受限；当 \(I \geq I^*\) 时计算受限。

**H100 SXM5 的具体数值**：依据 NVIDIA 官方规格，H100 的 FP16 张量核（Tensor Core）峰值算力 \(P_\text{peak} = 989\) TFLOPS（TF32，dense）或 \(1{,}979\) TFLOPS（FP16，dense）。主本与本章使用 TF32/FP16 混合精度训练场景中最常被引用的 989 TFLOPS 作为代表值，HBM3 带宽 \(\beta = 3.35\) TB/s，因此

\[
I^*_{\text{H100}} = \frac{989 \times 10^{12}\ \text{FLOP/s}}{3.35 \times 10^{12}\ \text{byte/s}} \approx 295\ \text{FLOP/byte}.
\]

这意味着：一个算术强度低于 295 FLOP/byte 的 kernel 在 H100 上永远是内存受限的，无论如何优化 CUDA 核心的利用率都无济于事——**瓶颈在带宽，不在算力**。

### 6.1.4 GEMM 的算术强度分析

考虑矩阵乘 \(C = AB\)，其中 \(A \in \mathbb{R}^{m \times k}\)，\(B \in \mathbb{R}^{k \times n}\)，\(C \in \mathbb{R}^{m \times n}\)。

**浮点运算数**：每个输出元素 \(C_{ij}\) 需做 \(k\) 次乘加，共 \(2mnk\) FLOP。

**数据搬运量**（假设矩阵在 HBM 中各读一次、\(C\) 写一次）：

\[
Q_{\text{GEMM}} = (mk + kn + mn) \times \text{sizeof(element)}.
\]

忽略数据精度系数，定义算术强度为

\[
I_{\text{GEMM}} = \frac{2mnk}{mk + kn + mn}.
\]

**大 batch 极限**（\(m, n, k \to \infty\) 且同阶）：分子 \(\sim 2n^3\)，分母 \(\sim 3n^2\)，故 \(I_{\text{GEMM}} \to \frac{2n}{3} \to \infty\)——大矩阵乘是计算受限的，不受内存墙威胁。

**小 batch 极限**（\(m = n = 1\)，即矩阵-向量乘）：

\[
I_{\text{GEMM}}\big|_{m=n=1} = \frac{2k}{k + k + 1} = \frac{2k}{2k+1} \underset{k \to \infty}{\longrightarrow} 1 \approx 1\ \text{FLOP/byte (FP32)}.
\]

若采用 FP16（每元素 2 byte），则

\[
I_{\text{GEMM}}\big|_{m=n=1,\,\text{FP16}} \approx \frac{2k}{2(2k+1)} \to 1\ \text{FLOP/byte}.
\]

这一数值与 H100 脊点 295 相差 **295 倍**——LLM 推理中的 token-by-token（batch size = 1）解码阶段，矩阵乘完全内存受限，这正是 HBM 带宽对 LLM 推理速度如此关键的根本原因。

---

## §6.2 I/O 复杂度的下界（Hong-Kung 1981）

本节给出香港中文大学洪钧和 H. T. Kung 于 1981 年发表的 I/O 下界定理的严格陈述与证明梗概。原论文通过 STOC 1981 发表，是现代 I/O 感知算法设计的奠基石。

### 6.2.1 计算模型与红蓝卵石游戏

**两级存储模型**：系统有容量无限的**慢速存储**（对应 HBM）和容量为 \(S\) 个字的**快速存储**（对应 SRAM/shared memory）。在任意时刻，快速存储中最多驻留 \(S\) 个值。一次 I/O 操作将一个值在两级存储之间搬移。

**红蓝卵石游戏（Red-Blue Pebble Game）**：将算法表示为有向无环图（DAG）\(G = (V, E)\)，节点对应计算值，边对应数据依赖。两种卵石的语义如下：

- **红色卵石**：该值当前在快速存储（SRAM）中；最多同时存在 \(S\) 颗。
- **蓝色卵石**：该值在慢速存储（HBM）中持久化。

博弈规则：
1. **计算**：若某节点的所有前驱节点均有红色卵石，可在该节点放置红色卵石（计算一步）。
2. **读（HBM→SRAM）**：将蓝色卵石替换为红色卵石，计 1 次 I/O。
3. **写（SRAM→HBM）**：将红色卵石替换为蓝色卵石，计 1 次 I/O。
4. **驱逐**：若快速存储已满，可无代价地移除某红色卵石（丢弃，若以后需要则须重新读入）。

目标：对 DAG 的所有输出节点放置红色卵石，同时使 I/O 操作总数 \(T_{I/O}\) 最小。

### 6.2.2 关键引理：S-划分与路径数

**定义（\(H(S,G)\)——S-碎石数）**：在 DAG \(G\) 中，使用 \(S\) 颗红色卵石可以覆盖的最大节点数称为 \(H(S,G)\)——即红蓝游戏的"路径复杂度"。更形式化地，\(H(2S,G)\) 是 DAG 的**2S-块流量（2S-dominator set size）**：每个"流"至多穿过 \(2S\) 个节点。

**引理 6.1（Hong-Kung 基本不等式）**：设 DAG \(G\) 有 \(|V|\) 个非输入节点，输入节点数为 \(|\text{Inputs}(G)|\)。则在 \(S\) 个红色卵石约束下，任意正确博弈策略满足

\[
\Bigl\lceil \frac{T_{I/O}}{S} \Bigr\rceil \cdot H(2S,\, G) \;\geq\; |V| - |\text{Inputs}(G)|.
\]

**证明梗概（S-划分论证）**：

将整个博弈过程按 I/O 操作划分为**阶段（phases）**，每个阶段包含至多 \(S\) 次 I/O。在阶段 \(t\) 开始与结束时，快速存储至多各含 \(S\) 个值，故阶段 \(t\) 的"边界"上至多有 \(2S\) 个值（进入 + 离开）。

在阶段 \(t\) 内，所有被计算的节点形成一个子图；其所有外部输入（来自 HBM 读入或上一阶段留存）至多 \(2S\) 个。由 \(H(2S,G)\) 的定义，该子图中最多包含 \(H(2S,G)\) 个节点。

阶段总数为 \(\lceil T_{I/O}/S \rceil\)，故

\[
\text{（阶段数）} \times H(2S,G) \;\geq\; |V| - |\text{Inputs}(G)|,
\]

即

\[
\Bigl\lceil \frac{T_{I/O}}{S} \Bigr\rceil \;\geq\; \frac{|V| - |\text{Inputs}(G)|}{H(2S,G)}.
\]

将两边乘以 \(S\) 即得引理。\(\square\)

### 6.2.3 矩阵乘的 I/O 下界

**定理 6.2（矩阵乘 I/O 下界）**：对 \(n \times n\) 矩阵乘 \(C = AB\)，在快速存储大小为 \(M\) 的两级存储模型下，任意正确算法的 I/O 复杂度满足

\[
T_{I/O} = \Omega\!\left(\frac{n^3}{\sqrt{M}}\right).
\]

**证明梗概**：

矩阵乘的标准算法可由以下 DAG 表示：节点集为 \(\{(i,j,k) : 1 \leq i,j,k \leq n\}\)（即 \(n^3\) 个部分乘积节点），共 \(|V| = n^3\) 个内部节点（忽略 \(n^2\) 个累加节点的低阶项），输入节点为 \(A\) 与 \(B\) 的 \(2n^2\) 个元素。

**关键一步 · 估计 \(H(2S,G)\)**：若快速存储中驻留 \(N_A\) 个 \(A\) 的元素、\(N_B\) 个 \(B\) 的元素、\(N_C\) 个 \(C\) 的部分和，则在这批元素上能完成的 FMA 数至多为 \(\sqrt{N_A N_B N_C}\)（此为 Irony、Toledo 与 Tiskin 于 2004 年给出的精确估计，Hong-Kung 原文以略松的论证给出相同渐近结果）。**由 AM-GM 不等式**，

\[
N_A + N_B + N_C \leq 2S \;\Longrightarrow\; \sqrt{N_A N_B N_C} \leq \left(\frac{2S}{3}\right)^{3/2} = O(S^{3/2}).
\]

故 \(H(2S, G) = O(S^{3/2})\)。代入引理 6.1：

\[
T_{I/O} \;\geq\; S \cdot \frac{n^3 - 2n^2}{H(2S,G)} \;=\; S \cdot \frac{\Theta(n^3)}{O(S^{3/2})} \;=\; \Omega\!\left(\frac{n^3}{S^{1/2}}\right) = \Omega\!\left(\frac{n^3}{\sqrt{M}}\right). \quad \square
\]

**可达性**：分块矩阵乘（block size \(b = \lfloor\sqrt{M/3}\rfloor\)）实现 \(\Theta(n^3/\sqrt{M})\) 次 I/O，与下界匹配，故该下界是**紧的（tight）**。

**硬件含义**：H100 的 SRAM（每 SM 约 256 KB shared memory，108 个 SM）总计 \(\approx 27\) MB。对 \(n = 4096\)（典型大语言模型的注意力维度量级）：

\[
T_{I/O} = \Omega\!\left(\frac{4096^3}{\sqrt{27 \times 10^6}}\right) \approx \Omega\!\left(\frac{6.87 \times 10^{10}}{5196}\right) \approx \Omega(1.3 \times 10^7)\ \text{word transfers}.
\]

即使算法最优，大矩阵也必须反复在 HBM 和 SRAM 之间搬移数据——这不是实现缺陷，而是**信息论意义上的必然**。

---

## §6.3 FlashAttention 的 I/O 分析（Dao et al. 2022）

### 6.3.1 标准注意力（Attention）的 I/O 复杂度

设序列长度 \(N\)，head 维度 \(d\)，查询/键/值矩阵 \(Q, K, V \in \mathbb{R}^{N \times d}\)。标准 attention 的前向计算分三步：

1. \(S = QK^\top \in \mathbb{R}^{N \times N}\)；写入 HBM。
2. \(P = \text{softmax}(S) \in \mathbb{R}^{N \times N}\)；从 HBM 读 \(S\)，写回 \(P\)。
3. \(O = PV \in \mathbb{R}^{N \times d}\)；从 HBM 读 \(P\) 和 \(V\)。

各步主要 I/O：

| 步骤 | 读 (byte) | 写 (byte) |
|------|-----------|-----------|
| \(S = QK^\top\) | \(2Nd\) | \(N^2\) |
| \(\text{softmax}(S)\) | \(N^2\) | \(N^2\) |
| \(O = PV\) | \(N^2 + Nd\) | \(Nd\) |

**总 I/O** \(= \Theta(Nd + N^2)\)。

**极限假设 \(N \gg d\)**（例如 \(N = 8192, d = 128\)）下，**主导项为 \(N^2\)**；而 \(Q, K, V, O\) 的搬运仅为 \(Nd \ll N^2\)。这意味着标准 attention 的 I/O 瓶颈在于 \(N \times N\) 注意力矩阵在 HBM 上的多次读写，而非矩阵乘本身的计算。

### 6.3.2 在线归一化指数（Softmax）的递推公式

Dao 等人的核心技巧是**分块 tiling + 在线 softmax**，使得 \(N \times N\) 矩阵永远无需完整写入 HBM。设将 \(K, V\) 沿序列维度分成 \(T_c = \lceil N / B_c \rceil\) 块，每块大小 \(B_c \times d\)；将 \(Q, O\) 分成 \(T_r = \lceil N / B_r \rceil\) 块，每块大小 \(B_r \times d\)。对 \(Q\) 的第 \(i\) 行块，算法遍历 \(K,V\) 的所有列块，维护以下三个**在线统计量**：

\[
m_i \;=\; \max\!\bigl(m_{i-1},\; \text{rowmax}(S_i)\bigr),
\]

\[
\ell_i \;=\; e^{m_{i-1} - m_i}\,\ell_{i-1} \;+\; \text{rowsum}\!\bigl(e^{S_i - m_i}\bigr),
\]

\[
O_i \;=\; \text{diag}\!\bigl(e^{m_{i-1} - m_i}\bigr)\,O_{i-1} \;+\; e^{S_i - m_i}\,V_i.
\]

其中 \(S_i = Q \cdot K_i^\top \in \mathbb{R}^{B_r \times B_c}\) 是当前块的注意力分数，\(m_0 = -\infty\)，\(\ell_0 = 0\)，\(O_0 = 0\)。

**引理 6.3（在线 softmax 正确性）**：设总共有 \(T\) 个键块，则经过所有 \(T\) 次递推后，

\[
O_T = \text{diag}(\ell_T)^{-1} \cdot \sum_{j=1}^{T} e^{S_j - m_T} V_j = \text{softmax}(S_{\text{full}}) \cdot V,
\]

其中 \(S_{\text{full}} = [S_1, S_2, \ldots, S_T]\) 为完整注意力分数行向量。

**证明**：采用数学归纳法，对块数 \(t\) 归纳。

**基础**：\(t = 1\)：\(m_1 = \text{rowmax}(S_1)\)，\(\ell_1 = \text{rowsum}(e^{S_1 - m_1})\)，\(O_1 = e^{S_1 - m_1} V_1\)。注意

\[
\frac{O_1}{\ell_1} = \frac{e^{S_1 - m_1} V_1}{\text{rowsum}(e^{S_1 - m_1})} = \text{softmax}(S_1) V_1,
\]

归一化后正确。\(\checkmark\)

**归纳步**：设在处理 \(t-1\) 块后，归一化输出

\[
\hat{O}_{t-1} \;=\; \frac{O_{t-1}}{\ell_{t-1}} \;=\; \text{softmax}(S_{1:t-1}) V_{1:t-1}
\]

成立（归纳假设）。处理第 \(t\) 块时，令 \(m_t = \max(m_{t-1}, \text{rowmax}(S_t))\)，则

\[
\ell_t = e^{m_{t-1} - m_t} \ell_{t-1} + \text{rowsum}(e^{S_t - m_t}).
\]

注意 \(e^{m_{t-1} - m_t} \ell_{t-1} = \text{rowsum}(e^{S_{1:t-1} - m_t})\)（指数中的 \(m_{t-1}\) 被消去），故

\[
\ell_t = \text{rowsum}\bigl(e^{S_{1:t} - m_t}\bigr).
\]

同理，

\[
O_t = e^{m_{t-1} - m_t} O_{t-1} + e^{S_t - m_t} V_t
     = e^{S_{1:t-1} - m_t} \cdot \mathbf{1} \cdot \hat{O}_{t-1} \cdot \ell_{t-1} + e^{S_t - m_t} V_t.
\]

精确地，

\[
\frac{O_t}{\ell_t} = \frac{\sum_{j=1}^{t} e^{S_j - m_t} V_j}{\text{rowsum}(e^{S_{1:t} - m_t})} = \text{softmax}(S_{1:t}) V_{1:t}. \quad \square
\]

### 6.3.3 FlashAttention 的 I/O 复杂度

**命题 6.4（FlashAttention 前向 I/O 复杂度）**：设 SRAM 大小为 \(M\)，满足 \(d \leq M \leq Nd\)，块大小取 \(B_c = \Theta(M/d)\)，\(B_r = \Theta(M/d)\)。FlashAttention 前向传播的 HBM I/O 复杂度为

\[
T_{I/O}^{\text{FlashAttn}} = \Theta\!\left(\frac{N^2 d^2}{M}\right).
\]

**证明**：

外循环遍历 \(T_c = N/B_c\) 个键块，内循环遍历 \(T_r = N/B_r\) 个查询块。

- **内循环每次**：从 HBM 读入 \(Q\) 块（\(B_r \times d\)）+ \(K_j, V_j\) 块（各 \(B_c \times d\)），计算 \(S_i = Q K_j^\top\)（纯 SRAM 内计算，无额外 HBM I/O），更新 \(O_i\)（SRAM 内），写出 \(O\) 块（\(B_r \times d\)）。每次内迭代 HBM I/O 量 \(= O(B_r d + B_c d)\)。
- **总 I/O**：共 \(T_r \times T_c = (N/B_r)(N/B_c)\) 次内迭代，每次 \(O(B_r d + B_c d)\)，故

\[
T_{I/O} = O\!\left(\frac{N}{B_r} \cdot \frac{N}{B_c} \cdot (B_r + B_c) \cdot d\right).
\]

取 \(B_r = B_c = B\)，则

\[
T_{I/O} = O\!\left(\frac{N^2}{B^2} \cdot 2Bd\right) = O\!\left(\frac{N^2 d}{B}\right).
\]

**消元条件 · SRAM 容量约束**：SRAM 需同时容纳 \(Q, K, V\) 块及 \(O, S\) 块：\(O(Bd)\) 个元素，故约束 \(Bd = O(M)\)，**即 \(B = O(M/d)\)**。代入：

\[
T_{I/O}^{\text{FlashAttn}} = O\!\left(\frac{N^2 d}{M/d}\right) = O\!\left(\frac{N^2 d^2}{M}\right).
\]

下界同理可得 \(\Omega(N^2 d^2 / M)\)（见原文 Theorem 2 的最优性证明），故 **\(T_{I/O}^{\text{FlashAttn}} = \Theta(N^2 d^2 / M)\)**。\(\square\)

> **复杂度跃迁 · 这一步是全文的关键转折**：标准 attention 的 \(\Theta(N^2)\) 被换写为 \(\Theta(N^2 d^2 / M)\)——当 \(M \gg d^2\)（H100 上 \(M \approx 10^7\)、\(d^2 \approx 1.6 \times 10^4\)），**同一计算结果的 HBM I/O 几乎下降两个数量级**。I/O 复杂度不是被「优化」出来的，是被「重新推导」出来的。

### 6.3.4 FlashAttention vs. 标准 Attention 的复杂度比较

\[
\frac{T_{I/O}^{\text{standard}}}{T_{I/O}^{\text{FlashAttn}}} = \frac{\Theta(Nd + N^2)}{\Theta(N^2 d^2 / M)} = \Theta\!\left(\frac{M}{d^2} \cdot \frac{Nd + N^2}{N^2}\right) \underset{N \gg d}{\approx} \Theta\!\left(\frac{M}{d^2}\right).
\]

代入 A100 的 SRAM 大小 \(M \approx 20\) MB \(= 20 \times 10^6\) byte，head dimension \(d = 128\)，每 FP16 元素 2 byte，则 \(M \approx 10^7\) 元素，\(d^2 = 16384\)，故加速比约 \(\Theta(10^7 / 16384) \approx 610\) 倍（渐近）。实测加速约 7.6 倍（主要原因是序列长度有限，常数项不可忽略）。

### 6.3.5 FlashAttention-2 的工作划分（Work Partitioning）改进

Dao（2023）在 FlashAttention 的基础上提出三项改进，不改变 I/O 复杂度量级，但大幅提升硬件利用率：

1. **减少非矩阵乘 FLOP**：FlashAttention 原版的缩放与 rescaling 操作引入额外逐元素运算；FA-2 重新推导递推公式，将 \(\text{diag}(e^{m_{i-1}-m_i})\) 的显式展开推迟到最终输出时执行，减少中间 rescaling 次数。

2. **序列维度并行化**：FA-1 仅对 batch 和 head 维度并行调度 thread block，导致长序列小 batch 时 GPU 占用率（occupancy）低。FA-2 对外循环（\(T_r\) 个行块）额外并行，使 SM 利用率从约 40% 提升至 50–73%。

3. **Warp 内 Work Partitioning**：FA-1 将 \(K, V\) 切分至 4 个 warp（"split-K"方案），各 warp 需写中间结果至 shared memory 后同步累加；FA-2 改为将 \(Q\) 切分至 warp，\(K, V\) 共享，消除 warp 间同步，降低 shared memory 访问延迟。

这三项改进使 FA-2 在 A100 上达到理论峰值的 50–73%，在 H100 上可达 335 TFLOPS（FlashAttention-2 本身无需特殊 Hopper 指令），是截至 2023 年实现最高 GPU 利用率的 attention 实现之一。

---

## §6.4 环形注意力（RingAttention）的通信复杂度（Liu et al. 2024）

### 6.4.1 多卡场景设定

设总序列长度 \(N\)，分布在 \(P\) 张 GPU 上，每张 GPU 持有

\[
Q_p \in \mathbb{R}^{(N/P) \times d}, \quad K_p, V_p \in \mathbb{R}^{(N/P) \times d}, \quad p = 1, 2, \ldots, P.
\]

完整注意力需要每张 GPU 的 \(Q_p\) 对全局 \(K, V\) 做 attention，即 \(O_p = \text{softmax}(Q_p K^\top / \sqrt{d}) V\)。

朴素做法：先用 AllGather 将完整 \(K, V\)（各 \(Nd\) 元素，\(2Nd \cdot \text{sizeof}\) byte）汇聚至每张卡，通信量 \(\Theta(Nd)\) 字节。这与 \(P\) 无关，但实际上每张卡需存储完整 \(K, V\)，内存随 \(N\) 增长，无法扩展至超长序列。

### 6.4.2 环形通信方案

RingAttention 将 \(P\) 张 GPU 排列为**逻辑环**，在计算与通信之间实现**流水线重叠（pipeline overlap）**：

**第 \(t\) 步**（\(t = 1, 2, \ldots, P-1\)）：

- GPU \(p\) 将其持有的当前 \(K_j, V_j\)（大小各 \((N/P) \times d\)）**发送**至 GPU \(p+1 \pmod{P}\)，同时从 GPU \(p-1 \pmod{P}\) **接收**下一批 \(K, V\) 块。
- 发送/接收过程中，GPU \(p\) 同时用 FlashAttention 计算当前 \(K_j, V_j\) 对本地 \(Q_p\) 的贡献，更新在线 softmax 统计量。

每步通信量：发送 \(K_j\) 和 \(V_j\) 各 \((N/P) \times d \times \text{sizeof}\) = \(2 \cdot (N/P) \cdot d\) 元素。

总通信量（\(P-1\) 步）：

\[
T_{\text{comm}} = (P-1) \cdot 2 \cdot \frac{N}{P} \cdot d \cdot \text{sizeof} \;\approx\; 2Nd \cdot \text{sizeof},
\]

即总通信量 **\(\Theta(Nd)\) 字节，不依赖 \(P\)**——设备数越多，每步通信量越小，总量不变。

### 6.4.3 计算与通信重叠条件

每步**计算时间**：GPU \(p\) 需完成 \(Q_p\)（大小 \(N/P \times d\)）与 \(K_j\)（大小 \(N/P \times d\)）的矩阵乘，计算量为 \(2(N/P)^2 d\) FLOP，耗时

\[
t_{\text{comp}} = \frac{2(N/P)^2 d}{F},
\]

其中 \(F\) 为单卡浮点算力（FLOP/s）。

每步**通信时间**：传输 \(2 \times (N/P) \times d\) 个元素（FP16，每元素 2 byte），耗时

\[
t_{\text{comm}} = \frac{4(N/P)d}{\beta_{\text{NVLink}}},
\]

其中 \(\beta_{\text{NVLink}}\) 为单向 NVLink 带宽（H100 SXM: \(\approx 450\) GB/s 单向）。

**完全重叠条件**（计算时间 \(\geq\) 通信时间）：

\[
\frac{2(N/P)^2 d}{F} \;\geq\; \frac{4(N/P)d}{\beta_{\text{NVLink}}},
\]

化简得

\[
\frac{N}{P} \;\geq\; \frac{2F}{\beta_{\text{NVLink}}}.
\]

即每卡持有的局部序列长度 \(N/P\) 不得小于 \(2F/\beta_{\text{NVLink}}\)（类似一个"局部 ridge point"）。

### 6.4.4 环路最大规模推导

由上式解出 \(P\) 的上界：

\[
P \;\leq\; \frac{N \cdot \beta_{\text{NVLink}}}{2F} = \frac{N \cdot \beta_{\text{NVLink}}}{2F}.
\]

若希望通信完全被计算掩盖，GPU 数量不得超过

\[
P_{\max} = \left\lfloor \frac{N \cdot \beta_{\text{NVLink}}}{2F} \right\rfloor.
\]

**H100 SXM5 的具体数值**：\(F = 989 \times 10^{12}\) FLOP/s（TF32），\(\beta_{\text{NVLink}} = 450 \times 10^9\) byte/s（单向），序列长度 \(N = 1{,}000{,}000\)（百万 token）：

\[
P_{\max} = \frac{10^6 \times 450 \times 10^9}{2 \times 989 \times 10^{12}} \approx \frac{4.5 \times 10^{17}}{1.978 \times 10^{15}} \approx 227.
\]

即对百万 token 的序列，约 227 张 H100 可令 RingAttention 通信完全被计算掩盖。超过此数目时通信将成为瓶颈，需进一步的通信压缩或序列分片策略。

**与 Roofline 的类比**：\(P_{\max}\) 的公式 \(\sim \sqrt{N \cdot F / \beta}\)（若同时考虑计算量正比 \(N^2/P^2\)、通信量正比 \(N/P\)）提示了一个多卡版本的"ridge point"：设备数增多使局部计算强度下降，超过 \(P_{\max}\) 后即陷入"通信受限"状态，与单卡内存受限具有同构的数学结构。

---

## §6.5 算术强度的演化：从 V100 到 B200

### 6.5.1 各代 GPU 的 Roofline 参数

以下数据来源于 NVIDIA 官方规格表与公开基准测试，均为 dense（无稀疏加速）FP16 Tensor Core 性能：

| GPU | 架构 | \(P_\text{peak}\) (TFLOPS, FP16) | \(\beta\) (TB/s, HBM) | \(I^* = P_\text{peak}/\beta\) (FLOP/byte) |
|-----|------|-------------------------------|----------------------|------------------------------------------|
| V100 SXM2 | Volta (2018) | 125 | 0.90 | **139** |
| A100 SXM4 | Ampere (2020) | 312 | 2.00 | **156** |
| H100 SXM5 | Hopper (2022) | 989\(^\dagger\) | 3.35 | **295** |
| B200 SXM | Blackwell (2025) | 2,250\(^\ddagger\) | 8.00\(^§\) | **281** |

\(^\dagger\) 989 TFLOPS 为 TF32 dense；FP16 dense 为 1,979 TFLOPS，对应 \(I^* \approx 590\) FLOP/byte。本章与主本一致，使用 TF32 基准以便对比训练场景。

\(^\ddagger\) B200 FP16 Tensor Core dense 值；FP4 dense 可达 9,000 TFLOPS，仅用于推理量化场景，不计入此处 Roofline 对比。

\(^§\) B200 HBM3e 带宽官方数值因文献版本不同在 7.7–8.0 TB/s 之间；本章取 8.0 TB/s。

### 6.5.2 Ridge Point 的单调上行论证

**命题 6.5（Ridge Point 单调上行）**：设在相邻两代 GPU 之间，算力增长因子为 \(\alpha_F = P_\text{peak}^{(t+1)} / P_\text{peak}^{(t)}\)，带宽增长因子为 \(\alpha_\beta = \beta^{(t+1)} / \beta^{(t)}\)。若 \(\alpha_F > \alpha_\beta\)，则脊点单调上升：

\[
I^{*(t+1)} = \frac{P_\text{peak}^{(t+1)}}{\beta^{(t+1)}} = I^{*(t)} \cdot \frac{\alpha_F}{\alpha_\beta} > I^{*(t)}.
\]

**验证**：

- V100 \(\to\) A100：\(\alpha_F = 312/125 = 2.50\)，\(\alpha_\beta = 2.00/0.90 = 2.22\)，比值 \(1.12 > 1\)。\(I^*\) 从 139 升至 156。\(\checkmark\)
- A100 \(\to\) H100：\(\alpha_F = 989/312 = 3.17\)，\(\alpha_\beta = 3.35/2.00 = 1.68\)，比值 \(1.89 > 1\)。\(I^*\) 从 156 升至 295。\(\checkmark\)
- H100 \(\to\) B200：\(\alpha_F = 2250/989 = 2.27\)，\(\alpha_\beta = 8.00/3.35 = 2.39\)，比值 \(0.95 < 1\)。\(I^*\) 从 295 微降至 281。

注意 B200 的带宽增速（2.39×）**首次超过**算力增速（2.27×），脊点出现轻微下降。这并非 Blackwell 架构设计失误，而是 HBM3e 技术在本代实现了相对更大幅度的带宽跃升（叠片数增加、引脚速率 8 Gbps vs. 5.23 Gbps）。

**结构性含义**：从历史趋势看，V100→H100 三代间 \(\alpha_F/\alpha_\beta\) 均大于 1，脊点从 139 升至 295，**翻倍有余**。这意味着：

1. 曾经在 V100 上计算受限的 kernel，在 H100 上可能已变为内存受限（因为脊点右移，需要更高算术强度才能逃离内存墙）。
2. 算法设计必须不断"主动提升 \(I\)"——或通过更大 tile（如 FlashAttention-2 的更大 \(B_r, B_c\)），或通过算子融合（kernel fusion），或通过量化（FP8/FP4）降低数据搬运字节数。

**量化对 \(I\) 的影响**：若将 FP16（2 byte/元素）改为 FP8（1 byte/元素），相同计算量下字节搬运减半，算术强度 \(I \to 2I\)。结合 H100 FP8 下 3,958 TFLOPS 的峰值，脊点

\[
I^*_{\text{H100, FP8}} = \frac{3958}{3.35} \approx 1181\ \text{FLOP/byte}.
\]

在此精度下，即使是 batch size = 1 的矩阵-向量乘（\(I \approx 2\)）依然深度内存受限，带宽仍然是主要瓶颈。

---

## §6.6 反类比与边界

### 6.6.1 Roofline 模型的假设局限

**假设一：完美重叠（perfect overlap）**。Roofline 模型假设内存访问与计算可以完全流水，实际 GPU 中存在 **latency hiding** 限制：若 outstanding memory requests 不足以填满内存子系统的延迟（约 500–800 ns on HBM），则 kernel 会出现 latency-bound 而非纯 bandwidth-bound 的停顿。此时实际带宽利用率可能显著低于 \(\beta\)，Roofline 的预测偏于乐观。

**假设二：单一带宽值**。标准 Roofline 用单个 \(\beta\) 代表所有内存访问，忽视了 L1 cache、L2 cache、shared memory 与 HBM 之间 **3–4 个量级**的带宽差异。层次化 Roofline（hierarchical Roofline）可缓解此问题，但引入更多硬件参数，分析复杂度成倍增加。

**假设三：FP 运算同质**。峰值算力 \(P_\text{peak}\) 通常指 Tensor Core 的矩阵乘算力，而逐元素运算（softmax 中的 exp、LayerNorm 等）需走 CUDA Core，吞吐量低 4–8×。对于逐元素密集的 kernel，实际峰值远低于 \(P_\text{peak}\)。

### 6.6.2 Hong-Kung 下界的适用范围

**限制一：渐近下界，常数不可忽略**。定理 6.2 中的 \(\Omega(\cdot)\) 隐藏了常数因子。分块 GEMM 的实际 I/O 为 \(2\sqrt{3}\, n^3 / \sqrt{M}\)（Irony et al. 2004 给出），而 cuBLAS 的实现常数约为 2–4，因此对小矩阵（\(n \leq 512\)），香港下界的渐近优化意义有限，常数项的优化往往更关键。

**限制二：顺序存储层级假设**。Hong-Kung 定理假设单一快速存储与单一慢速存储，现代 GPU 的存储层级为 **Register → L1/Shared Memory → L2 Cache → HBM**，各级之间存在非均匀延迟与带宽。真实 I/O 下界应针对多级存储层级分别建立，见 Ballard 等（2012）对多级层次的扩展。

**限制三：仅适用于确定性算法**。对于随机化算法（如随机稀疏注意力），Hong-Kung 框架需要修正，访问模式的随机性可能允许绕过某些下界。

### 6.6.3 FlashAttention I/O 分析的隐含前提

FlashAttention 的 \(\Theta(N^2 d^2 / M)\) 复杂度依赖 \(d \leq M \leq Nd\) 的假设。若 \(M < d\)（极端 SRAM 受限），则整个 head 无法驻留 SRAM，分块策略失效，复杂度退化至 \(\Theta(N^2)\)。反之若 \(M \geq Nd\)（即整个 \(K, V\) 可放入 SRAM），则标准 attention 的第 1-3 步均可在 SRAM 内完成，I/O 降至 \(\Theta(Nd)\)，FlashAttention 的优势消失。

FlashAttention 的最优性定理（Dao et al. 2022, Theorem 2）同时给出了下界 \(\Omega(N^2 d^2 / M)\)，证明**不存在**在所有 \(M\) 范围内均优于 FlashAttention 的精确 attention 算法——它在渐近意义下已是最优的。

### 6.6.4 RingAttention 的通信模型假设

本章 §6.4 的推导假设：（1）NVLink 为**全双工、无竞争**的点对点通信；（2）计算与通信可以**完全重叠**（需要硬件对计算流与通信流的并发调度支持，如 CUDA Stream 与 NCCL 的集成）；（3）序列在设备间均匀分布（causal masking 下负载不均衡，Striped Attention 等变体通过重排 token 索引加以缓解）。

在实际部署中，由于 NCCL 通信延迟、InfiniBand 带宽共享等因素，可达 \(P_{\max}\) 可能远低于理论值，需通过 profiling 工具（如 NVIDIA Nsight Systems）实测通信/计算重叠率。

---

## §6.7 参考文献

1. **Hong, J.-W., & Kung, H. T.** (1981). I/O complexity: The red-blue pebble game. *Proceedings of the 13th Annual ACM Symposium on Theory of Computing (STOC)*, 326–333. https://doi.org/10.1145/800076.802486

2. **Williams, S., Waterman, A., & Patterson, D.** (2009). Roofline: An insightful visual performance model for multicore architectures. *Communications of the ACM*, 52(4), 65–76. https://doi.org/10.1145/1498765.1498785

3. **Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C.** (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *Advances in Neural Information Processing Systems (NeurIPS) 35*. https://arxiv.org/abs/2205.14135

4. **Dao, T.** (2023). FlashAttention-2: Faster attention with better parallelism and work partitioning. *International Conference on Learning Representations (ICLR 2024)*. https://arxiv.org/abs/2307.08691

5. **Liu, H., Zaharia, M., & Abbeel, P.** (2024). Ring attention with blockwise transformers for near-infinite context. *International Conference on Learning Representations (ICLR 2024)*. https://arxiv.org/abs/2310.01889

6. **Irony, D., Toledo, S., & Tiskin, A.** (2004). Communication lower bounds for distributed-memory matrix multiplication. *Journal of Parallel and Distributed Computing*, 64(9), 1017–1026. https://doi.org/10.1016/j.jpdc.2004.03.021

7. **Ballard, G., Demmel, J., Holtz, O., & Schwartz, O.** (2012). Graph expansion and communication costs of fast matrix multiplication. *Journal of the ACM*, 59(5), 1–23. https://doi.org/10.1145/2344422.2344424

8. **Aggarwal, A., & Vitter, J. S.** (1988). The input/output complexity of sorting and related problems. *Communications of the ACM*, 31(9), 1116–1127. https://doi.org/10.1145/48529.48535

9. **Milakov, M., & Gimelshein, N.** (2018). Online normalizer calculation for softmax. *arXiv preprint*. https://arxiv.org/abs/1805.02867

10. **NVIDIA Corporation.** (2022). NVIDIA H100 Tensor Core GPU Architecture White Paper. https://resources.nvidia.com/en-us-tensor-core

11. **NVIDIA Corporation.** (2025). NVIDIA B200 GPU Product Specifications. https://www.nvidia.com/en-us/data-center/b200/

12. **Brandon, W., Mishra, M., Nrusimha, A., Panda, R., & Ragan-Kelley, J.** (2023). Striped attention: Faster ring attention for causal transformers. *arXiv preprint*. https://arxiv.org/abs/2311.09431

13. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I.** (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS) 30*. https://arxiv.org/abs/1706.03762

14. **Demmel, J., Grigori, L., Hoemmen, M., & Langou, J.** (2012). Communication-optimal parallel and sequential QR and LU factorizations. *SIAM Journal on Scientific Computing*, 34(1), A206–A239. https://doi.org/10.1137/080731992

---

*本章与主本第六章相互镜像：主本讲产业逻辑（HBM 为何成为万亿生意、CoWoS 封装与超大规模云厂商（Hyperscaler）采购），本章专注数学骨干——不等式如何推、下界如何建立、算法如何在严格意义上最优。两章合读方能得到完整的"硬件感知"图景。*


# 数学卷·第 7 章:后变换器（Transformer）时代的数学——状态空间模型、线性注意力与扩散语言模型的数理基础

> **说明**:本章为《算法的低语·数学卷》的第七章,专注严格数学推导。与主本第七章的架构评述相互补充,读者需具备线性代数、常微分方程与概率论的基础知识。本章中所有内联公式使用 \(\cdot\) 记号,展示公式使用独立段落。

---

## §7.1 状态空间模型的连续形式


### 7.1.0 工程直觉引子：从「滑动窗口 + 缓存状态」走到连续 ODE

> 在跳进连续时间、一阶 ODE 和勒让德多项式之前，我们先用一个工程老兵都熟悉的画面开场。这个引子并不严格，但能为后面所有那些「凭空出现」的记号预存一个心锚。

**设想这样一个年轻的服务。** 它要实时处理一条无限长的 token 流，但只允许维护一个 **固定大小的缓存状态** \(h \in \mathbb{R}^N\)。每来一个新输入 \(x_t\)，服务需要决定：旧状态保留多少，新输入注入多少。最自然的写法是一条**带遗忘因子的线性递推**：

\[
h_t = \alpha \, h_{t-1} + \beta \, x_t, \qquad y_t = c^\top h_t,
\tag{7.0}
\]

其中 \(\alpha \in (0, 1)\) 控制「旧状态衰减多快」，\(\beta\) 控制「新输入注入多重」。QA 老兵一眼就能认出：这就是监控系统里近 30 年被反复使用的 **指数加权滑动平均（EWMA）**——Prometheus 的 `rate()`、Kafka 的 consumer lag 告警、高频交易系统的 tick-by-tick 指标都用这条递推。它有三个友好的性质：

- **记忆有限**：远古输入的权重衰减为 \(\alpha^k\)，\(k\) 越大越被遗忘；
- **状态恒定**：不论流多长，\(h_t\) 始终为 \(N\) 维向量；
- **线性叠加**：两条输入流的状态可以独立计算后相加。

**连续化只需一步跳跃。** 如果我们不再假设 token 是按 \(\Delta t = 1\) 均匀到达，而是令 \(\Delta t \to 0\)，记 \(\alpha = 1 + A \Delta t\)、\(\beta = B \Delta t\)，那么

\[
\frac{h_t - h_{t-1}}{\Delta t} = A \, h_{t-1} + B \, x_t.
\]

取极限就得到

\[
h'(t) = A \, h(t) + B \, x(t), \qquad y(t) = C \, h(t),
\]

**这正是后面 §7.1.1 要介绍的一阶线性 ODE**。换句话，**连续状态空间模型不是什么新东西，它是「带衰减的滑动窗口」在 \(\Delta t \to 0\) 下的连续极限**。全章将反复使用的**连续信号、状态转移矩阵、双线性变换**，都只是在问一个问题：**应该选什么样的 \(A\)，才能让这个「无限长流 · 固定状态」的缓存尽可能多地保留历史信息？**§7.2 HiPPO 的答案会是：选一个与勒让德多项式正交的 \(A\)。

> **一句话总结**：接下来所有最奇怪的记号——\(e^{At}\)、勒让德多项式、双线性变换——都只是「滑动窗口 + 缓存状态」这个工程原型的**连续极限版本**。请带着这个心锚走进后面的推导。

---

### 7.1.1 线性时不变系统的基本方程

状态空间模型(State Space Model, SSM)的出发点是经典的线性时不变(Linear Time-Invariant, LTI)系统。给定一维输入信号 \(x(t) \in \mathbb{R}\) 和 \(N\) 维隐状态 \(h(t) \in \mathbb{R}^N\),系统方程为

\[
h'(t) = A h(t) + B x(t), \qquad y(t) = C h(t),
\tag{7.1}
\]

其中 \(A \in \mathbb{R}^{N \times N}\) 是状态转移矩阵,\(B \in \mathbb{R}^{N \times 1}\) 是输入投影向量,\(C \in \mathbb{R}^{1 \times N}\) 是输出投影向量。此方程描述了隐状态如何随时间演化以及如何从隐状态恢复输出。

在信号处理与控制理论中,式 \eqref{7.1} 已有数十年研究历史。将其引入序列建模的关键洞察在于:若选取合适的矩阵 \(A\),则 \(h(t)\) 能够以固定内存代价维护对任意长历史的压缩表示。LTI 性质保证了卷积结构,而卷积结构则保证了 \(O(L \log L)\) 的训练效率。

### 7.1.2 闭式解的推导

方程 \eqref{7.1} 是一阶线性常微分方程组。将其改写为

\[
h'(t) - A h(t) = B x(t).
\]

左乘积分因子 \(e^{-At}\),得

\[
\frac{d}{dt}\bigl(e^{-At} h(t)\bigr) = e^{-At} B x(t).
\]

在 \([0, t]\) 上积分,并利用初始条件 \(h(0)\),

\[
e^{-At} h(t) - h(0) = \int_0^t e^{-As} B x(s)\, ds.
\]

两边乘以 \(e^{At}\),得到**闭式解**:

\[
\boxed{h(t) = e^{At} h(0) + \int_0^t e^{A(t-s)} B x(s)\, ds.}
\tag{7.2}
\]

其中 \(e^{At}\) 是矩阵指数(matrix exponential),定义为 \(e^{At} = \sum_{k=0}^{\infty} \frac{(At)^k}{k!}\)。

**稳定性条件**:若 \(A\) 的所有特征值 \(\lambda_i\) 满足 \(\mathrm{Re}(\lambda_i) < 0\),则 \(\|e^{At}\| \to 0\) 当 \(t \to \infty\),系统渐近稳定。HiPPO 矩阵的特征值恰好位于左半复平面,保证了状态的有界演化。

### 7.1.3 卷积视角

将闭式解代入输出方程,并设 \(h(0) = 0\)(零初始状态),得

\[
y(t) = C h(t) = \int_0^t C e^{A(t-s)} B x(s)\, ds = (K * x)(t),
\tag{7.3}
\]

其中卷积核定义为

\[
K(t) = C e^{At} B.
\tag{7.4}
\]

这一卷积视角至关重要:在离散化之后,若参数 \(A, B, C\) 不依赖于输入(LTI 假设),则整个序列的输出可通过**单次全局卷积**在频域以 \(O(L \log L)\) 时间内计算,其中 \(L\) 是序列长度。这也是 S4 等模型实现高效训练的核心。

**注记**:式 \eqref{7.4} 中的卷积核 \(K(t)\) 完全由矩阵 \(A\) 的特征结构决定。若 \(A\) 具有负实部的特征值,则 \(K(t)\) 随时间指数衰减;若 \(A\) 有纯虚特征值,则 \(K(t)\) 表现为振荡记忆。HiPPO 框架(§7.2)将系统地回答:什么样的 \(A\) 能够最优地压缩历史信息?

### 7.1.4 频域分析与传递函数

对式 \eqref{7.1} 作单边拉普拉斯变换(设零初始条件),令 \(s\) 为复频率变量:

\[
s H(s) = A H(s) + B X(s), \quad Y(s) = C H(s),
\]

解出 \(H(s) = (sI - A)^{-1} B X(s)\),故传递函数为

\[
G(s) = \frac{Y(s)}{X(s)} = C(sI - A)^{-1} B.
\tag{7.5}
\]

这是有理分式,分母的零点即为矩阵 \(A\) 的特征值(系统极点)。S4 中利用此结构,在 \(z\) 域上用 DPLR 分解将 \((zI - \bar{A})^{-1}\) 的计算化为柯西（Cauchy）矩阵向量积(见 §7.3)。

---

## §7.2 HiPPO 框架的严格推导

> **本节的关键论证链**：(1) 问题陈述——压缩历史信号；(2) 选 Legendre 多项式做基函数（正交、闭式系数）；(3) 推导 LegT/LegS 矩阵 \(A\)；(4) 证明所得 \(A\) 正是「让有限维状态 \(h(t)\) 在每一时刻都对历史的最优最小二乘投影」的那一个。**关键转折点会在每个小节标题后用粗体提示**。

### 7.2.1 问题陈述:在线最优投影

**HiPPO**(High-order Polynomial Projection Operators)由 Gu et al. (NeurIPS 2020) 提出,其核心问题是:

> 给定连续信号 \(x : [0, t] \to \mathbb{R}\),如何在线地(online)将其历史压缩到固定维向量 \(h(t) \in \mathbb{R}^N\),使得 \(h(t)\) 以某种意义最优地近似 \(x\) 在 \([0,t]\) 上的全部历史?

**形式化**:设 \(\mu^{(t)}\) 是 \([0, t]\) 上的测度(measure),\(\{g_n\}_{n=0}^{N-1}\) 是关于 \(\mu^{(t)}\) 的正交多项式基。定义**投影算子**

\[
\mathrm{proj}_t : L^2(\mu^{(t)}) \to \mathrm{span}\{g_0, \ldots, g_{N-1}\},
\]

令 \(h_n(t) = \langle x, g_n \rangle_{\mu^{(t)}}\) 为 \(x(s)\)(视为 \([0,t]\) 上的函数,\(s \leq t\))在第 \(n\) 个基函数上的内积系数。则 \(h(t) = (h_0(t), \ldots, h_{N-1}(t))^\top\) 是最优 \(N\) 维近似的系数向量。

**关键结论**:当测度 \(\mu^{(t)}\) 随时间演化时,\(h(t)\) 满足线性微分方程 \(h'(t) = A h(t) + B x(t)\),矩阵 \(A\) 由测度的具体形式决定。

### 7.2.2 Legendre 多项式的正交性

**勒让德多项式** \(P_n : [-1, 1] \to \mathbb{R}\) 由 Rodrigues 公式定义:

\[
P_n(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n}\bigl[(x^2-1)^n\bigr].
\]

关于标准 Lebesgue 测度 \(dx\) 在 \([-1, 1]\) 上满足正交性:

\[
\int_{-1}^{1} P_n(x) P_m(x)\, dx = \frac{2}{2n+1} \delta_{nm}.
\tag{7.6}
\]

归一化后的正交基为 \(\tilde{P}_n(x) = \sqrt{\frac{2n+1}{2}} P_n(x)\),满足 \(\int_{-1}^1 \tilde{P}_n \tilde{P}_m\, dx = \delta_{nm}\)。

前几阶勒让德多项式为 \(P_0(x) = 1\),\(P_1(x) = x\),\(P_2(x) = \frac{1}{2}(3x^2-1)\),满足递推关系:

\[
(n+1)P_{n+1}(x) = (2n+1)x P_n(x) - n P_{n-1}(x).
\tag{7.7}
\]

这一递推关系是 HiPPO 矩阵推导中的核心工具。

### 7.2.3 HiPPO-LegT:平移 Legendre 测度

**LegT(Translated Legendre)**使用滑动窗口测度:对固定窗口大小 \(\theta > 0\),

\[
\mu^{(t)} = \frac{1}{\theta} \mathbf{1}_{[t-\theta, t]}\, ds \quad \text{(均匀分布在长度为 }\theta\text{ 的窗口上)}.
\]

通过变量替换将 \([t-\theta, t]\) 映射到 \([-1, 1]\),令 \(u = \frac{2(s - (t-\theta))}{\theta} - 1\),则系数 \(h_n(t) = \sqrt{\frac{2n+1}{\theta}} \int_{t-\theta}^{t} P_n\!\left(\frac{2(s-t)}{\theta}+1\right) x(s)\, ds\) 对时间 \(t\) 求导,利用莱布尼茨积分法则和勒让德多项式的递推关系,可以推导出:

\[
h'(t) = A_{\text{LegT}} h(t) + B_{\text{LegT}} x(t),
\]

其中**显式矩阵**为

\[
(A_{\text{LegT}})_{nk} = -\frac{1}{\theta}
\begin{cases}
(2n+1)^{1/2}(2k+1)^{1/2} & \text{若 } n \geq k, \\
(-1)^{n-k}(2n+1)^{1/2}(2k+1)^{1/2} & \text{若 } n < k,
\end{cases}
\tag{7.8}
\]

以及 \((B_{\text{LegT}})_n = \frac{1}{\theta}(2n+1)^{1/2}(-1)^n\)。此处为简洁起见设 \(\theta = 1\)。

**物理意义**:矩阵 \(A_{\text{LegT}}\) 的下三角块使得低阶系数影响高阶系数,而上三角块(反号)体现了窗口滑动时旧信息的遗忘机制。

### 7.2.4 引理(HiPPO-LegS 矩阵的推导)

**引理 7.1 (HiPPO-LegS 矩阵的显式形式)**:设测度为累积(缩放 Legendre)测度:
\[
\mu^{(t)} = \frac{1}{t} \mathbf{1}_{[0, t]}\, ds,
\]
即在 \([0,t]\) 上的均匀测度,随时间整体缩放。则最优投影系数 \(h(t)\) 满足的 SSM 矩阵为

\[
(A_{\text{LegS}})_{nk} =
\begin{cases}
-\sqrt{(2n+1)(2k+1)} & \text{若 } n > k, \\
-(n+1) & \text{若 } n = k, \\
0 & \text{若 } n < k,
\end{cases}
\tag{7.9}
\]

以及 \((B_{\text{LegS}})_n = \sqrt{2n+1}\)。

**证明**:

设归一化基函数 \(\tilde{P}_n(x) = \sqrt{2n+1} P_n(2x-1)\),在 \([0,1]\) 上关于均匀测度正交,即

\[
\int_0^1 \tilde{P}_n(x) \tilde{P}_m(x)\, dx = \delta_{nm}.
\]

将 \([0, t]\) 通过变量替换 \(u = s/t\) 映射到 \([0,1]\),系数为

\[
h_n(t) = \int_0^1 \tilde{P}_n(u)\, x(tu)\, du = \frac{1}{t} \int_0^t \tilde{P}_n(s/t)\, x(s)\, ds.
\]

对 \(t\) 求导(应用莱布尼茨法则):

\[
h_n'(t) = -\frac{1}{t^2} \int_0^t \tilde{P}_n(s/t)\, x(s)\, ds + \frac{1}{t}\tilde{P}_n(1)\, x(t) - \frac{1}{t^2}\int_0^t \frac{s}{t} \tilde{P}_n'(s/t)\, x(s)\, ds.
\]

第一项化简为 \(-\frac{1}{t} h_n(t)\);第三项中利用勒让德多项式导数的展开公式:

\[
P_n'(x) = \sum_{\substack{k=0 \\ n-k \text{ 奇}}}^{n-1} (2k+1) P_k(x),
\]

因此 \(\tilde{P}_n'(u) = \sqrt{2n+1} \sum_{\substack{k < n \\ n-k \text{ 奇}}} (2k+1) P_k(2u-1) = \sqrt{2n+1} \sum_{k < n} \sqrt{2k+1} \tilde{P}_k(u) \cdot [(n-k \text{ 奇}) ? 1 : 0]\)。

将上式代入第三项并整理,对角项系数汇总为 \(-(n+1)/t\),次对角项系数为 \(-\sqrt{(2n+1)(2k+1)}/t\) (\(n > k\)),上三角项为 0。注意 \(\tilde{P}_n(1) = \sqrt{2n+1}\)(因为 \(P_n(1) = 1\)),故 \(B_n = \sqrt{2n+1}\)。

此系统通常写为 \(t h'(t) = \hat{A} h(t) + \hat{B} x(t)\),并通过时间变换 \(\tau = \log t\) 化为标准 LTI 形式。\(\square\)

**数值示例**:对 \(N = 4\) 的 LegS 矩阵:

\[
A_{\text{LegS}} = -\begin{pmatrix} 1 & 0 & 0 & 0 \\ \sqrt{3} & 2 & 0 & 0 \\ \sqrt{5} & \sqrt{15} & 3 & 0 \\ \sqrt{7} & \sqrt{21} & \sqrt{35} & 4 \end{pmatrix}, \quad B_{\text{LegS}} = \begin{pmatrix} 1 \\ \sqrt{3} \\ \sqrt{5} \\ \sqrt{7} \end{pmatrix}.
\]

矩阵是严格下三角占优的,这保证了在适当归一化后矩阵指数的有界性。

### 7.2.5 最优投影的精确陈述

**定理 7.2 (HiPPO 最优近似保证)**:设 \(\hat{x}_t(s) = \sum_{n=0}^{N-1} h_n(t) \tilde{P}_n(s/t)\) 为 \(x|_{[0,t]}\) 的 \(N\) 阶 Legendre 多项式近似。则

\[
\hat{x}_t = \arg\min_{p \in \mathcal{P}_{N-1}} \int_0^t \bigl|p(s) - x(s)\bigr|^2 \frac{ds}{t},
\]

其中 \(\mathcal{P}_{N-1}\) 是次数不超过 \(N-1\) 的多项式空间。即 \(h(t)\) 给出了 \(x|_{[0,t]}\) 在 \([0,t]\) 上的均方意义下的最优 \(N\) 维 Legendre 系数近似。

**注记**:不同的测度 \(\mu^{(t)}\) 导致不同的"最优性"——LegT 最优近似的是最近的窗口,LegS 最优近似的是全历史的加权(均匀)版本。§7.8 将讨论这一点对模型性质的影响。

---

## §7.3 S4 的离散化与卷积核

> **本节的关键骨架**：连续 ODE \(h'(t) = A h(t) + B x(t)\) → 零阶保持（ZOH）离散化 → 离散递推 \(h_k = \bar A h_{k-1} + \bar B x_k\) → 展开为卷积 \(y = \bar K * x\) → **关键消元** Cauchy 核 + Vandermonde 矩阵把卷积从 \(O(NL)\) 降至 \(O((N+L)\log^2(N+L))\)。

### 7.3.1 零阶保持离散化


```{=latex}
\begin{center}
\includegraphics[width=0.92\linewidth]{assets/figs/fig_math_07_01_moe.png}\\[0.3em]
\small\itshape 图 M7.1 · MoE 路由 · Router 给每个 token 选 Top-2 专家激活(稀疏计算节省推理算力)
\end{center}
```
在实际应用中,序列数据是离散的:时刻 \(k = 0, 1, \ldots, L-1\),步长(time step)为 \(\Delta > 0\)。最常用的离散化方案是**零阶保持(Zero-Order Hold, ZOH)**,假设输入在每个时间步内保持常数。

**引理 7.3 (ZOH 离散化的精确形式)**:在步长 \(\Delta\) 的 ZOH 假设下,连续 SSM \((A, B, C)\) 离散化为

\[
\bar{A} = e^{\Delta A}, \qquad
\bar{B} = (\Delta A)^{-1}(e^{\Delta A} - I) \cdot \Delta B, \qquad
\bar{C} = C.
\tag{7.10}
\]

离散递推为 \(h_k = \bar{A} h_{k-1} + \bar{B} x_k\),\(y_k = \bar{C} h_k\)。

**证明**:

将时间区间 \([k\Delta, (k+1)\Delta]\) 上的连续解 \eqref{7.2} 应用于 \(h((k+1)\Delta)\):

\[
h((k+1)\Delta) = e^{A \Delta} h(k\Delta) + \int_{k\Delta}^{(k+1)\Delta} e^{A((k+1)\Delta - s)} B x(s)\, ds.
\]

ZOH 假设 \(x(s) = x_{k+1}\) 为常数,令 \(\tau = (k+1)\Delta - s\),则

\[
\int_{k\Delta}^{(k+1)\Delta} e^{A((k+1)\Delta - s)} B x_{k+1}\, ds = \left(\int_0^\Delta e^{A\tau}\, d\tau\right) B x_{k+1}.
\]

利用矩阵指数的积分公式:

\[
\int_0^\Delta e^{A\tau}\, d\tau = A^{-1}(e^{A\Delta} - I),
\]

故

\[
\bar{B} = A^{-1}(e^{A\Delta} - I) \cdot B = (\Delta A)^{-1}(e^{\Delta A} - I) \cdot \Delta B.
\]

这正是式 \eqref{7.10} 中的 \(\bar{B}\)。\(\square\)

**与双线性方法的比较**:S4 原始论文中也使用双线性(Tustin/bilinear)离散化 \(\bar{A} = (I - \Delta A/2)^{-1}(I + \Delta A/2)\),其精度为 \(O(\Delta^2)\) 而 ZOH 同阶。两者实践差异较小,但 Mamba 重新引入 ZOH 以支持更直观的选择机制——因为 ZOH 下 \(\bar{A} = e^{\Delta A}\) 具有更清晰的"遗忘率"语义。

### 7.3.2 离散卷积核

在 LTI 设定下,离散 SSM 的输入-输出关系展开为卷积。将递推展开:

\[
y_k = \bar{C} \bar{A}^k \bar{B} x_0 + \bar{C} \bar{A}^{k-1} \bar{B} x_1 + \cdots + \bar{C} \bar{B} x_k,
\]

即 \(y_k = \sum_{j=0}^k \bar{K}_j x_{k-j}\),其中**离散卷积核向量**为

\[
\bar{\mathbf{K}} = (\bar{C}\bar{B},\; \bar{C}\bar{A}\bar{B},\; \bar{C}\bar{A}^2\bar{B},\; \ldots,\; \bar{C}\bar{A}^{L-1}\bar{B}).
\tag{7.11}
\]

向量 \(\bar{\mathbf{K}} \in \mathbb{R}^L\) 称为 SSM 的卷积核(SSM kernel)。一旦计算出 \(\bar{\mathbf{K}}\),整个序列的输出可由 \(\mathbf{y} = \bar{\mathbf{K}} * \mathbf{x}\)(离散卷积)在频域通过 FFT 以 \(O(L \log L)\) 计算。

### 7.3.3 计算复杂度的降低:Cauchy 核与范德蒙（Vandermonde）矩阵

**瓶颈**:直接计算 \(\bar{\mathbf{K}}\) 需要 \(L\) 次矩阵幂 \(\bar{A}^k B\),若 \(\bar{A}\) 是稠密 \(N \times N\) 矩阵,则复杂度为 \(O(N^2 L)\),在 \(N = L = 4096\) 的典型设定下为 \(\sim 7 \times 10^{10}\) 次运算,不可接受。

**DPLR 参数化**:S4 的核心技术是将 \(A\) 参数化为**对角加低秩(Diagonal Plus Low-Rank, DPLR)**形式:

\[
A = \Lambda - P Q^*,
\tag{7.12}
\]

其中 \(\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_N)\) 是对角矩阵,\(P, Q \in \mathbb{C}^{N \times r}\) 是低秩矩阵(\(r \ll N\))。HiPPO-LegS 矩阵恰好可以写成 DPLR 形式:其正规部分为斜对称矩阵 \(S_{nk} = -\frac{1}{2}[(2n+1)(2k+1)]^{1/2} \text{sgn}(n-k)\),可对角化为纯虚特征值;非正规部分为秩 1 修正 \(-\frac{1}{2} p p^*\),其中 \((p)_n = (n + 1/2)^{1/2}\)。

**Woodbury 恒等式**:设 \(M = \Lambda - P Q^*\) 为 DPLR 矩阵,则

\[
M^{-1} = \Lambda^{-1} + \Lambda^{-1} P (I - Q^* \Lambda^{-1} P)^{-1} Q^* \Lambda^{-1}.
\tag{7.13}
\]

由于 \(\Lambda^{-1}\) 是对角矩阵,右侧所有运算均可以 \(O(N + r^3)\) 时间完成。这使得在对角形上计算 \((zI - \bar{A})^{-1}\) 变得可行。

**关键算法**:设 \(\Lambda\) 的特征值为 \(\{\lambda_j\}_{j=1}^N\),离散化后 \(\bar{\Lambda} = e^{\Delta\lambda}\)。卷积核的第 \(k\) 个元素可写为

\[
\bar{K}_k = \bar{C} \bar{A}^k \bar{B} = \sum_{j=1}^N c_j \bar{\lambda}_j^k,
\]

其中系数 \(\{c_j\}\) 由伍德伯里（Woodbury）恒等式修正低秩项后得到。将全部 \(L\) 个核元素的生成函数求值:

\[
\hat{K}(z) = \sum_{k=0}^{L-1} \bar{K}_k z^k \approx C(zI - \bar{A})^{-1}\bar{B} = \sum_{j=1}^N \frac{c_j}{z - \bar{\lambda}_j}.
\tag{7.14}
\]

在 \(L\) 个单位根 \(\omega^j\)(其中 \(\omega = e^{2\pi i / L}\))处求值:

\[
\hat{K}(\omega^l) = \sum_{j=1}^N \frac{c_j}{\omega^l - \bar{\lambda}_j}, \quad l = 0, 1, \ldots, L-1.
\]

这等价于 **Cauchy 矩阵向量积** \(\text{diag}(\hat{K})= M_\text{Cauchy} \mathbf{c}\),其中 \((M_\text{Cauchy})_{lj} = 1/(\omega^l - \bar{\lambda}_j)\)。利用快速多极方法或 Vandermonde 矩阵的 FFT 类算法,此计算可在 \(\tilde{O}(N + L)\) 时间内完成。最终通过逆 FFT 恢复时域卷积核 \(\bar{\mathbf{K}}\)。

**整体复杂度**:从 \(O(N^2 L)\) 降至 \(\tilde{O}(N + L)\),提升约 \(N \cdot L / \log(N+L)\) 倍。

**DPLR 参数化的必要性**:若 \(A\) 是一般稠密矩阵,则其矩阵指数的计算为 \(O(N^3)\),后续矩阵幂亦为 \(O(N^2)\) per step,无法实现 Cauchy 核约化。DPLR 结构保证:(i) 矩阵逆由 Woodbury 恒等式精确解析;(ii) 离散化后 \(\bar{A}\) 仍保持 DPLR 结构;(iii) 所有操作在数值稳定的复对角形上进行。S4D(Diagonal S4)进一步将 DPLR 中 \(P, Q\) 简化为零,直接用纯对角矩阵 \(A = \Lambda\),实现更简洁的实现,并在实践中与 S4 性能相当。

---

## §7.4 Mamba 的选择性扫描

### 7.4.1 选择性机制:从 LTI 到输入依赖

S4 的根本局限在于参数 \(A, B, C\) 与输入无关(LTI),因此模型在不同位置对相同幅度的输入给出相同响应,无法进行基于内容的选择性记忆。Mamba(Gu & Dao, 2023)通过使参数依赖于当前输入来打破这一限制。

**选择性参数化**:对输入序列 \(\{x_t\}_{t=1}^L\),Mamba 令

\[
\Delta_t = \text{softplus}(W_\Delta x_t + b_\Delta) \in \mathbb{R}^{D}, \qquad
B_t = W_B x_t \in \mathbb{R}^{N}, \qquad
C_t = W_C x_t \in \mathbb{R}^{N},
\tag{7.15}
\]

其中 \(W_\Delta \in \mathbb{R}^{D \times D}\),\(W_B \in \mathbb{R}^{N \times D}\),\(W_C \in \mathbb{R}^{N \times D}\) 是可学习参数,\(D\) 是输入维度。步长 \(\Delta_t\) 决定了当前时刻离散化的"分辨率",由 \(\text{softplus}(x) = \ln(1 + e^x) > 0\) 保证正性。

**选择性离散化**:对每个时刻 \(t\),独立执行 ZOH 离散化:

\[
\bar{A}_t = e^{\Delta_t A}, \qquad \bar{B}_t = (\Delta_t A)^{-1}(e^{\Delta_t A} - I)\Delta_t B_t,
\tag{7.16}
\]

其中 \(A\) 保持固定(仅结构参数化)。选择性 SSM 的递推为:

\[
h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \qquad y_t = C_t h_t.
\tag{7.17}
\]

**直觉分析**:当 \(\Delta_t \to 0^+\) 时,\(\bar{A}_t = e^{\Delta_t A} \to I\) 而 \(\bar{B}_t \to 0\),模型忽略当前输入、保持历史状态;当 \(\Delta_t \to +\infty\) 时,若 \(A\) 有负特征值,则 \(\bar{A}_t \to 0\) 且 \(\bar{B}_t \to -A^{-1}B_t\),模型聚焦于当前输入、遗忘所有历史。\(\Delta_t\) 由输入内容决定,实现了 "token-wise 的记忆/遗忘控制"。

### 7.4.2 选择性 SSM 的计算挑战

由于 \(\bar{A}_t\) 依赖于 \(t\),序列不再是 LTI 系统,**全局卷积核不再有效**。朴素的顺序递推需要 \(O(LN)\) 时间但无法在 GPU 上并行化,成为训练的瓶颈。解决方案是**并行扫描算法**(Parallel Scan)。

### 7.4.3 并行扫描与关联算子

**引理 7.4 (选择性 SSM 的关联算子)**:定义元素 \((a, b) \in \mathbb{R}^{N \times N} \times \mathbb{R}^N\),其中 \(a\) 表示状态转移矩阵,\(b\) 表示驱动项。定义二元运算

\[
(a_1, b_1) \otimes (a_2, b_2) = (a_2 a_1,\; a_2 b_1 + b_2).
\tag{7.18}
\]

则此运算满足结合律,且递推 \eqref{7.17} 可以写成

\[
(h_t, 1) \stackrel{\text{对应}}{\longleftrightarrow} (\bar{A}_t, \bar{B}_t x_t) \otimes (\bar{A}_{t-1}, \bar{B}_{t-1} x_{t-1}) \otimes \cdots \otimes (\bar{A}_1, \bar{B}_1 x_1).
\tag{7.19}
\]

**证明(结合律)**:

对三个元素 \((a_1, b_1), (a_2, b_2), (a_3, b_3)\),验证左右结合相等。

**左结合**:先计算 \((a_1, b_1) \otimes (a_2, b_2) = (a_2 a_1, a_2 b_1 + b_2)\),再与 \((a_3, b_3)\) 结合:

\[
(a_2 a_1, a_2 b_1 + b_2) \otimes (a_3, b_3) = (a_3(a_2 a_1),\; a_3(a_2 b_1 + b_2) + b_3)
= (a_3 a_2 a_1,\; a_3 a_2 b_1 + a_3 b_2 + b_3).
\]

**右结合**:先计算 \((a_2, b_2) \otimes (a_3, b_3) = (a_3 a_2, a_3 b_2 + b_3)\),再与 \((a_1, b_1)\) 结合:

\[
(a_1, b_1) \otimes (a_3 a_2, a_3 b_2 + b_3) = ((a_3 a_2) a_1,\; (a_3 a_2) b_1 + (a_3 b_2 + b_3))
= (a_3 a_2 a_1,\; a_3 a_2 b_1 + a_3 b_2 + b_3).
\]

两者相等,故结合律成立。\(\square\)

**验证式 \eqref{7.19}**:归纳地,若 \(h_0 = 0\),令 \(a_i = \bar{A}_i\),\(b_i = \bar{B}_i x_i\),则

\[
h_t = a_t a_{t-1} \cdots a_1 \cdot 0 + a_t \cdots a_2 b_1 + a_t \cdots a_3 b_2 + \cdots + b_t,
\]

恰好由运算 \eqref{7.18} 的前缀积给出。此式也可解释为:隐状态是带指数衰减权重的输入历史加权和,权重由路径乘积 \(\prod_{s=j+1}^t \bar{A}_s\) 决定。

### 7.4.4 Blelloch 并行前缀扫描的复杂度

给定长度为 \(L\) 的序列 \(\{(a_t, b_t)\}_{t=1}^L\),计算所有前缀积 \(\{(a_t, b_t) \otimes \cdots \otimes (a_1, b_1)\}_{t=1}^L\):

- **朴素串行**:\(O(L)\) 步骤,无法并行化。
- **Blelloch 并行扫描(Blelloch, 1990)**:利用结合律,将前缀积分为**上扫(up-sweep)**和**下扫(down-sweep)**两个阶段,在 \(O(\log L)\) 深度的并行计算中完成,总工作量(work)为 \(O(L)\)。

具体地,每个 GPU 线程处理 \(O(1)\) 个元素,利用 \(O(\log L)\) 层二叉树归并:

\[
\text{Work} = O(L), \quad \text{Depth} = O(\log L).
\]

这使得长度 \(10^6\) 的序列可以在 \(\sim 20\) 层并行操作中完成状态扫描。Mamba 的**硬件感知实现**将参数和中间状态直接存储在 SRAM 中,避免反复读写 HBM(High Bandwidth Memory),实现了训练时约 5 倍于 Transformer 的吞吐量提升。

---

## §7.5 线性注意力的核函数视角

### 7.5.1 标准归一化指数（Softmax）注意力的复杂度

标准自注意力机制(Vaswani et al., 2017)定义为:

\[
A_{ij} = \frac{\exp(Q_i \cdot K_j / \sqrt{d})}{\sum_{j'} \exp(Q_i \cdot K_{j'} / \sqrt{d})}, \qquad O_i = \sum_j A_{ij} V_j,
\tag{7.20}
\]

其中 \(Q, K, V \in \mathbb{R}^{L \times d}\) 分别是查询、键、值矩阵。计算 \(L \times L\) 注意力矩阵 \(A\) 需要 \(O(L^2 d)\) 时间和 \(O(L^2)\) 空间,在长序列上形成瓶颈。

### 7.5.2 核函数分解

**线性注意力的核心思想**(Katharopoulos et al., ICML 2020):用正半定核函数 \(k(Q_i, K_j) = \phi(Q_i) \cdot \phi(K_j)^\top\) 替换 \(\exp(Q_i \cdot K_j)\),其中 \(\phi : \mathbb{R}^d \to \mathbb{R}^r\) 是特征映射(feature map)。则

\[
O_i = \frac{\sum_j \phi(Q_i) \cdot \phi(K_j)^\top V_j}{\sum_j \phi(Q_i) \cdot \phi(K_j)^\top}.
\tag{7.21}
\]

**关键代数变换**:利用矩阵乘法的结合律

\[
O_i = \frac{\phi(Q_i) \left(\sum_j \phi(K_j)^\top V_j\right)}{\phi(Q_i) \left(\sum_j \phi(K_j)^\top\right)}.
\tag{7.22}
\]

令 \(\mathbf{S} = \sum_j \phi(K_j)^\top V_j \in \mathbb{R}^{r \times d}\)(累积 KV 状态),\(\mathbf{z} = \sum_j \phi(K_j)^\top \in \mathbb{R}^r\)(归一化项),则

\[
O_i = \frac{\phi(Q_i) \mathbf{S}}{\phi(Q_i) \cdot \mathbf{z}}.
\tag{7.23}
\]

在**因果(causal)**设定下(自回归生成),\(\mathbf{S}_t = \sum_{j \leq t} \phi(K_j)^\top V_j\) 可在线更新:

\[
\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(K_t)^\top V_t, \qquad O_t = \frac{\phi(Q_t) \mathbf{S}_t}{\phi(Q_t) \cdot \mathbf{z}_t}.
\tag{7.24}
\]

这使得推理时每个 token 的计算量为 \(O(rd)\)(仅依赖固定大小的状态 \(\mathbf{S}_t \in \mathbb{R}^{r \times d}\)),而非 \(O(td)\)。**推理复杂度从 \(O(L)\) per token 降至 \(O(1)\) per token**。

**复杂度分析**:

- **训练(全序列)**:先计算全局 \(\mathbf{S} = \sum_j \phi(K_j)^\top V_j\)(复杂度 \(O(Lrd)\)),再对所有 \(i\) 计算 \(O_i = \phi(Q_i) \mathbf{S} / (\phi(Q_i) \cdot \mathbf{z})\)(复杂度 \(O(Lrd)\)),总体为 \(O(Lrd)\)。相比 \(O(L^2 d)\),当 \(r \ll L\) 时有显著优势。
- **推理(自回归)**:每步 \(O(rd)\),状态 \(\mathbf{S}_t\) 占 \(O(rd)\) 内存,与序列长度无关。

### 7.5.3 特征映射的选择

不同 \(\phi\) 的选择决定了近似质量与效率的权衡:

**1. ELU+1 (Katharopoulos, 2020)**:

\[
\phi(x)_i = \text{elu}(x_i) + 1 = \begin{cases} x_i + 1 & x_i \geq 0, \\ e^{x_i} & x_i < 0. \end{cases}
\tag{7.25}
\]

此映射保证 \(\phi(x)_i > 0\),使分母 \(\phi(Q_i) \cdot \mathbf{z}\) 保持正定,避免数值不稳定。但此 \(\phi\) 并不对应 softmax 核的精确近似,故可视为一种启发式替代。

**2. FAVOR+ 随机特征 (Performer, Choromanski et al., ICLR 2021)**:

利用 softmax 核的正值随机特征分解。注意到

\[
\exp(Q \cdot K) = \exp\!\left(\frac{\|Q\|^2 + \|K\|^2}{2}\right) \cdot \exp\!\left(-\frac{\|Q - K\|^2}{2}\right),
\]

后者是高斯核。利用 Bochner 定理,高斯核有无偏随机特征展开。FAVOR+ 使用正交正值特征:

\[
\phi^+_\omega(x) = \frac{e^{-\|x\|^2/2}}{\sqrt{r}} \bigl[\exp(\omega_1^\top x), \ldots, \exp(\omega_r^\top x)\bigr], \quad \omega_k \text{ 来自正交矩阵的列}.
\tag{7.26}
\]

可以证明 \(\mathbb{E}_\omega[\phi^+(x) \cdot \phi^+(y)^\top] = \exp(x \cdot y)\),即无偏估计 softmax 核。使用 \(r\) 个随机特征时,方差为 \(O(1/r)\)。

**3. RetNet 的固定衰减 (Sun et al., 2023)**:

RetNet 使用固定的指数衰减,定义保留(Retention)机制:

\[
A_{ij} = \gamma^{i-j} (Q_i K_j^\top / \sqrt{d}) \cdot \mathbf{1}_{i \geq j}, \quad \gamma \in (0, 1),
\tag{7.27}
\]

等价于 \(\phi(Q)_i = Q_i\),\(\phi(K)_j = \gamma^{-j} K_j\),加上按位置的衰减因子 \(\gamma^i\)。多头设定下每个头有不同的 \(\gamma\)("多尺度保留"),提供不同长度的记忆窗口。

### 7.5.4 线性注意力与 Fast Weight Memory 的等价性

**Schlag et al. (2021)** 指出,式 \eqref{7.24} 的更新

\[
\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(K_t)^\top V_t
\]

可以解释为**快速权重存储器(fast weight memory)**:状态矩阵 \(\mathbf{S}_t\) 以外积 \(\phi(K_t)^\top V_t\) 形式存储键值关联,查询时用 \(\phi(Q_t)\) 检索。这与 Hopfield 网络和联想记忆(associative memory)具有深刻联系,揭示了线性注意力的记忆机制本质。在此视角下,外积存储的容量为 \(O(rd)\) 比特,对应约 \(r\) 个精确键值对;当待检索的键值对数目超过 \(r\) 时,干扰效应导致检索精度下降,这从机制上解释了 §7.8 中讨论的检索任务劣势。

---

## §7.6 Mamba、线性注意力（Attention）与 RNN 的统一视角

### 7.6.1 通用递推形式

Yang et al. (GLA, ICML 2024) 和 Dao & Gu (Mamba-2/SSD, ICML 2024) 分别从不同方向指出:Mamba、线性 Attention 及经典门控 RNN 可以统一到同一框架下。

**定义 7.5 (广义线性递推)**:定义隐状态 \(\mathbf{S}_t \in \mathbb{R}^{N \times d}\) 的递推

\[
\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t, \qquad \mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t,
\tag{7.28}
\]

其中 \(\mathbf{G}_t \in \mathbb{R}^{N \times d}\) 是门控矩阵,\(\odot\) 表示 Hadamard 积,\(\mathbf{k}_t, \mathbf{q}_t \in \mathbb{R}^N\),\(\mathbf{v}_t \in \mathbb{R}^d\)。

三类模型在此框架中的实例化:

| 模型 | 门控 \(\mathbf{G}_t\) 的参数化 |
|------|------|
| 线性注意力 (Katharopoulos) | \(\mathbf{G}_t = \mathbf{1}\)(恒等,无遗忘) |
| RetNet (Sun et al., 2023) | \(\mathbf{G}_t = \gamma \cdot \mathbf{1}\)(固定标量衰减) |
| GLA (Yang et al., 2024) | \(\mathbf{G}_t = \sigma(W_g x_t)\)(数据依赖门) |
| Mamba (Gu & Dao, 2023) | \(\mathbf{G}_t = e^{\Delta_t A}\)(输入依赖离散化,结构化对角) |

### 7.6.2 Mamba-2 的结构化状态空间对偶性

**Dao & Gu (SSD, ICML 2024)** 在更严格的数学框架下建立了 SSM 与结构化注意力之间的精确对应:

**定理 7.6 (结构化状态空间对偶)**:考虑标量结构的 SSM(\(A\) 为标量乘以单位矩阵),对每个特征维度:

\[
h_t = \alpha_t h_{t-1} + \beta_t x_t, \qquad y_t = c_t h_t,
\tag{7.29}
\]

其中 \(\alpha_t, \beta_t, c_t \in \mathbb{R}\) 为标量。则此模型的输入-输出映射与如下**半分离(semiseparable)矩阵**乘向量等价:

\[
Y = M X, \quad M_{ij} = c_i \left(\prod_{t=j+1}^{i} \alpha_t\right) \beta_j \cdot \mathbf{1}_{i \geq j},
\tag{7.30}
\]

矩阵 \(M\) 是 1 阶半分离矩阵。在多头设定下,矩阵 \(M\) 升维为具有更丰富块结构的半分离矩阵族,统一了 SSM 的递推视角和注意力的矩阵视角。

此对偶性揭示:SSM 可以被视为结构化、低秩化的注意力矩阵运算。Mamba-2 利用此对偶性设计了新的 SSD(Structured State Space Duality)层,允许更大的状态维度(\(dN\) 而非 Mamba 的 \(dN\)),相较 Mamba 实现了 2-8 倍的速度提升。

### 7.6.3 GLA 的数据依赖门控

**GLA(Gated Linear Attention, Yang et al., ICML 2024)** 在线性注意力框架中引入数据依赖的门控:

\[
\mathbf{S}_t = (\mathbf{G}_t \odot \mathbf{S}_{t-1}) + \mathbf{k}_t^\top \mathbf{v}_t, \quad \mathbf{G}_t = \sigma(W_\alpha x_t) \in (0,1)^{N \times d},
\tag{7.31}
\]

其中 \(\sigma\) 是 sigmoid 函数。GLA 中对 \(\mathbf{G}_t\) 施加了"低秩"结构约束:\(\mathbf{G}_t = \alpha_t \beta_t^\top\)(\(\alpha_t \in \mathbb{R}^N, \beta_t \in \mathbb{R}^d\)),减少参数量同时保持表达能力。GLA 可以证明等价于选择性 SSM:门控 \(\mathbf{G}_t\) 对应 Mamba 中的 \(\bar{A}_t\),而 \(\mathbf{k}_t^\top \mathbf{v}_t\) 对应 \(\bar{B}_t x_t \cdot C_t^\top\)。这一等价性表明,从线性注意力和 SSM 两条独立路径,研究者们殊途同归地发现了同一类数据依赖线性递推。

---

## §7.7 扩散语言模型的数学

### 7.7.1 连续扩散:DDPM 回顾

**去噪扩散概率模型(DDPM, Ho et al., NeurIPS 2020)** 通过马尔可夫链定义数据的渐进加噪过程。给定数据 \(x_0 \sim q_\text{data}\),前向过程为:

\[
q(x_t | x_{t-1}) = \mathcal{N}\!\left(\sqrt{1-\beta_t}\, x_{t-1},\; \beta_t I\right), \quad t = 1, \ldots, T,
\tag{7.32}
\]

其中 \(\{\beta_t\}_{t=1}^T\) 是方差调度(variance schedule)。令 \(\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)\),则边际分布有闭式形式:

\[
q(x_t | x_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t) I\right).
\tag{7.33}
\]

**反向去噪**:学习参数化的反向过程 \(p_\theta(x_{t-1} | x_t)\)。通过变分下界(ELBO)与**去噪分数匹配**的联系,最优化目标简化为

\[
\mathcal{L}_\text{DDPM} = \mathbb{E}_{t, x_0, \epsilon}\left[\left\|\epsilon - \epsilon_\theta\!\left(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\, t\right)\right\|^2\right],
\tag{7.34}
\]

其中 \(\epsilon \sim \mathcal{N}(0, I)\)。这等价于在每个噪声水平 \(t\) 上的**分数匹配**(score matching):

\[
\nabla_{x_t} \log q_t(x_t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}.
\tag{7.35}
\]

**分数函数**的直觉:在样本 \(x_t\) 处,分数 \(\nabla_{x_t} \log q_t(x_t)\) 指向高概率密度方向。去噪网络 \(\epsilon_\theta\) 估计添加的噪声,等价于估计分数函数的负值,从而引导采样轨迹朝数据流形靠拢。

### 7.7.2 离散扩散:吸收态过渡矩阵

连续扩散对连续向量加高斯噪声,但语言模型处理的是离散 token。**离散扩散(Discrete Diffusion)** 通过马尔可夫链在 token 空间 \(\{1, \ldots, V\}^L\) 上定义噪声过程。

**吸收态(Absorbing State)扩散**:引入特殊 [MASK] token(索引 \(M\)),定义前向转移矩阵:

\[
Q_t = (1 - \beta_t) I + \beta_t \mathbf{1} e_M^\top,
\tag{7.36}
\]

其中 \(e_M\) 是 [MASK] 的 one-hot 向量。即以概率 \(\beta_t\) 将当前 token 替换为 [MASK](吸收),以概率 \(1 - \beta_t\) 保持不变。边际分布为

\[
q(x_t | x_0) = \bar{\alpha}_t \delta_{x_t, x_0} + (1 - \bar{\alpha}_t) \delta_{x_t, M}.
\tag{7.37}
\]

其中 \(\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)\)。在时刻 \(t\),token 要么仍是原始 token \(x_0\)(概率 \(\bar{\alpha}_t\)),要么已被 MASK(概率 \(1 - \bar{\alpha}_t\))。

后向条件分布由贝叶斯公式给出:

\[
q(x_{t-1} | x_t, x_0) =
\begin{cases}
\delta_{x_{t-1}, x_0} & \text{若 } x_t \neq M, \\
\frac{(1-\bar{\alpha}_{t-1}) \delta_{x_{t-1}, M} + (\bar{\alpha}_{t-1} - \bar{\alpha}_t) \delta_{x_{t-1}, x_0}}{1 - \bar{\alpha}_t} & \text{若 } x_t = M.
\end{cases}
\tag{7.38}
\]

此闭式形式使得训练中的 ELBO 计算非常高效。

### 7.7.3 MDLM 的简化目标

**MDLM(Masked Diffusion Language Model, Sahoo et al., NeurIPS 2024)** 推导了吸收态离散扩散的变分下界的**Rao-Blackwellized 简化形式**。

由于吸收态的特殊结构,ELBO 化简为：

\[
\mathcal{L}_\text{MDLM} = -\mathbb{E}_{t, x_0}\!\left[\frac{|\alpha_t'|}{\alpha_t} \sum_{i : x_t^{(i)} = M} \log p_\theta(x_0^{(i)} | x_t)\right],
\tag{7.39}
\]

其中 \(\alpha_t' = d\alpha_t / dt\),求和仅对被 MASK 的位置 \(i\) 进行。此目标**本质上是加权的 MLM 损失的混合**,权重系数 \(|\alpha_t'|/\alpha_t\) 对不同噪声水平重要性加权。

**LLaDA(2025)** 将此框架扩展至 8B 参数规模,证明了离散 Mask 扩散与自回归语言模型在同等工程实践下具有可比性能。

### 7.7.4 与自回归模型的本质差异

自回归(AR)语言模型基于因果分解 \(p(x) = \prod_i p(x_i | x_{<i})\),生成时严格左到右顺序进行,时间复杂度为 \(O(L)\) 步。

离散扩散模型则有以下数学差异:

1. **并行解码**:在任意一步去噪 \(p_\theta(x_0 | x_t)\) 中,所有被 MASK 的位置**同时**预测,不存在左到右的顺序依赖。理论上,仅需 1 步去噪即可并行生成整个序列,但质量较差;实际使用 \(T = 64\sim 256\) 步迭代精化。

2. **迭代精化**:每步将部分 MASK token 替换为预测值,逐步精化全序列。理论上,给定足够的步数 \(T\),采样质量随 \(T\) 单调提升,等价于更精确地模拟真实反向马尔可夫链。

3. **消除逆向诅咒(Reversal Curse)**:AR 模型因训练时只看 \(x_{<i}\) 而对逆向查询存在系统性弱点。扩散模型以全序列为条件进行去噪,对所有位置对称建模,在逆向任务上有理论优势,这在 LLaDA 实验中已得到验证。

---

## §7.8 世界模型与预测编码的数学


世界模型的工程目标是让智能体在**表示空间**中对未来状态进行预测,并以此驱动规划与控制。本节从数学角度拆解支撑这一目标的四类形式化:层级预测编码、变分自由能、JEPA 能量函数与防 collapse 正则化、以及 Kalman 滤波作为线性高斯 SSM 的精确推断。它们共同构成当代世界模型的数学骨架。

---

### 7.8.1 预测编码:层级生成模型与误差传递

[Rao & Ballard (1999)](https://pubmed.ncbi.nlm.nih.gov/10195184/) 提出的层级预测编码框架将视觉皮层建模为一个层级生成系统。设共有 \(L+1\) 层,第 \(l\) 层持有表示 \(x_l\),第 \(l\) 层对第 \(l-1\) 层的**自顶向下预测**为

\[
\hat{x}_{l-1} = f_l(x_l),
\]

其中 \(f_l\) 是可学习的生成映射。预测误差(prediction error)定义为

\[
\varepsilon_{l-1} = x_{l-1} - \hat{x}_{l-1}.
\]

联合生成模型写作

\[
p(x_0, x_1, \ldots, x_L) = p(x_L) \prod_{l=0}^{L-1} p(x_l \mid x_{l+1}),
\]

其中每个条件因子在高斯假设下取 \(p(x_l \mid x_{l+1}) = \mathcal{N}(x_l;\, f_{l+1}(x_{l+1}),\, \Pi_{l}^{-1})\),精度矩阵 \(\Pi_l\) 刻画该层预测的置信度。

在此高斯假设下,最大化对数似然等价于最小化**精度加权误差平方和**。对表示 \(x_l\) 求负对数似然的梯度,得到表示更新规则:

\[
\Delta x_l \;\propto\; \Pi_l\,\varepsilon_l \;-\; \frac{\partial f_l(x_l)}{\partial x_l}^\top \Pi_{l-1}\,\varepsilon_{l-1}.
\]

右侧第一项是**来自上方的误差信号**(当前层的预测与实际值的差异),第二项是**来自下方的误差信号**(当前层对下一层的预测残差通过雅可比（Jacobian）反传回来)。这一局部更新规则完全依赖相邻层之间的信号,无需全局梯度通路。

[Whittington & Bogacz (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5467749/) 严格证明:当网络满足"误差单元与表示单元分离"以及"固定预测权重"等条件时,预测编码的权重更新与标准误差反向传播算法在数值上收敛到相同结果。这一等价性表明预测编码并非生物学的特殊曲解,而是反向传播在局部学习规则约束下的一个有效近似。

---

### 7.8.2 自由能原理与变分下界

[Friston (2010)](https://doi.org/10.1038/nrn2787) 将感知与行动统一为一个极小化目标:**变分自由能**(variational free energy)。设观测为 \(x\),隐变量为 \(z\),变分后验为 \(q(z)\),变分自由能定义为

\[
F[q, x] \;=\; \mathbb{E}_{q(z)}\!\left[\log q(z) - \log p(x, z)\right].
\]

将联合概率分解为 \(\log p(x, z) = \log p(x \mid z) + \log p(z)\) 并重组,得到与证据下界(ELBO)的关系:

\[
F[q, x] \;=\; D_{\mathrm{KL}}\!\left(q(z)\,\|\,p(z\mid x)\right) - \log p(x) \;=\; -\,\mathrm{ELBO}(q, x).
\]

由于 \(\log p(x)\) 对 \(q\) 不可优化,最小化 \(F\) 同时实现两个目标:其一,令 \(q(z)\) 靠近真实后验 \(p(z \mid x)\)(感知推断);其二,间接最大化边际似然 \(\log p(x)\)(避免惊奇,即 surprise minimization)。与变分自编码器(VAE)的训练目标完全等价——VAE 的重构损失加 KL 正则项正是 \(F\) 的蒙特卡洛近似。

**主动推断(active inference)**在此框架中把行动 \(a\) 纳入自由能极小化:感知更新通过 \(\partial F / \partial z = 0\) 调整隐变量表示,行动更新通过

\[
\frac{\partial F}{\partial a} = \sum_l \frac{\partial F}{\partial \varepsilon_l} \cdot \frac{\partial \varepsilon_l}{\partial a}
\]

调整运动输出,使感觉预测误差在本体感觉通道中得到抑制。感知与行动因此成为同一极小化问题的两个分支,分别对应变分 E 步与 M 步的连续时间版本。

---

### 7.8.3 JEPA 的能量函数与防 collapse 机制

[LeCun (2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf) 提出**联合嵌入预测架构**(Joint-Embedding Predictive Architecture,JEPA),其核心思想是在**表示空间**而非像素空间中执行预测,从而规避生成模型在像素级重建上的高方差问题。设 \(x\) 为上下文观测,\(y\) 为目标观测,\(s_\phi\) 为编码器,\(P_\psi\) 为预测器,\(z\) 为描述 \(x\) 与 \(y\) 差异的隐变量,能量函数定义为

\[
E(x, y) \;=\; \bigl\|s_\phi(y) \;-\; P_\psi\!\left(s_\phi(x),\, z\right)\bigr\|^2.
\]

最小化 \(E\) 要求预测器能够从上下文表示与差异变量 \(z\) 中复原目标表示。[V-JEPA (Bardes et al., 2024)](https://arxiv.org/abs/2404.08471) 将此框架应用于视频时序预测,在屏蔽的时空块上执行表示层预测。

**Collapse 问题**:若编码器退化为常数映射 \(s_\phi(\cdot) \equiv \mathbf{c}\),则 \(E \equiv 0\) 对任意 \(x, y\) 成立,训练目标被平凡解满足。[VICReg (Bardes et al., ICLR 2022)](https://arxiv.org/abs/2105.04906) 通过三项显式正则化阻止 collapse:

- **不变性项**(Invariance):\(\displaystyle I(Z, Z') = \frac{1}{n}\sum_{i=1}^n \|Z_i - Z'_i\|^2\),最小化两视图嵌入之差;
- **方差项**(Variance):\(\displaystyle V(Z) = \frac{1}{d}\sum_{j=1}^d \max\!\left(0,\; \gamma - \sqrt{\mathrm{Var}(Z_{:,j}) + \epsilon}\right)\),惩罚任意维度标准差低于阈值 \(\gamma\);
- **协方差项**(Covariance):\(\displaystyle C(Z) = \frac{1}{d}\sum_{i \neq j}\bigl[\mathrm{Cov}(Z)\bigr]_{ij}^2\),惩罚不同维度之间的线性相关性。

VICReg 目标函数为

\[
L_{\mathrm{VICReg}} \;=\; \lambda\, I(Z, Z') \;+\; \mu\, V(Z) \;+\; \nu\, C(Z),
\]

其中 \(\lambda, \mu, \nu\) 为超参数。方差项直接对抗维度坍缩,协方差项鼓励嵌入维度间信息不冗余,二者共同维持表示空间的"充盈性"(informativeness)。从信息论角度看,\(C(Z)\) 的极小化是对表示进行近似白化,与去冗余原理(redundancy reduction)一脉相承。

---

### 7.8.4 Kalman 滤波作为线性高斯 SSM 的精确推断,以及神经过程视角

§7.1 已将状态空间模型(SSM)写为 \(x_t = Ax_{t-1} + w_t\)、\(y_t = Cx_t + v_t\),其中 \(w_t \sim \mathcal{N}(0, Q)\),\(v_t \sim \mathcal{N}(0, R)\)。当 \(A, C, Q, R\) 全部已知且线性高斯时,后验 \(p(x_t \mid y_{1:t})\) 保持高斯,可用 [Kalman (1960)](https://doi.org/10.1115/1.3662552) 滤波器在 \(\mathcal{O}(n^3)\) 时间内精确递推。

**预测步**(predict):

\[
\hat{x}_{t\mid t-1} = A\,\hat{x}_{t-1\mid t-1}, \qquad
P_{t\mid t-1} = A\,P_{t-1\mid t-1}\,A^\top + Q.
\]

**更新步**(update):

\[
K_t = P_{t\mid t-1}\,C^\top\!\left(C\,P_{t\mid t-1}\,C^\top + R\right)^{-1},
\]

\[
\hat{x}_{t\mid t} = \hat{x}_{t\mid t-1} + K_t\!\left(y_t - C\,\hat{x}_{t\mid t-1}\right), \qquad
P_{t\mid t} = (I - K_t C)\,P_{t\mid t-1}.
\]

Kalman 增益 \(K_t\) 是协方差矩阵的函数:当预测不确定性 \(P_{t\mid t-1}\) 大时,新观测权重高;当观测噪声 \(R\) 大时,权重低。更新步在结构上与预测编码的误差修正规则 \(\Delta x \propto K(y - C\hat{x})\) 完全对应——\(y_t - C\hat{x}_{t\mid t-1}\) 正是观测层的预测误差。

Mamba(§7.2)可视为 Kalman 滤波的**数据依赖非线性推广**:用输入门控的选择矩阵 \(\bar{A}(x_t), \bar{B}(x_t)\) 替代固定的 \(A, B\),使得噪声协方差的角色由网络动态承担,但预测-更新的递推结构保持不变。

**神经过程(Neural Processes)**[Garnelo et al., ICML 2018](https://arxiv.org/abs/1807.01622) 将贝叶斯滤波的后验近似交给神经网络,以**摊销推断**(amortized inference)替代逐步递推:一个编码器将上下文集合 \(\{(x_i, y_i)\}_{i=1}^n\) 映射为全局潜在表示 \(z\),解码器再条件于 \(z\) 生成目标预测。这一机制在计算上等价于把 Kalman 增益的迭代计算"压缩"进一次前向传播。

**DreamerV3** [Hafner et al. (2023)](https://arxiv.org/abs/2301.04104) 的循环状态空间模型(RSSM)将上述路线推至强化学习世界模型:隐状态分为**确定性分量** \(h_t\)(由 GRU 递推维护,对应 \(\hat{x}_{t\mid t-1}\))和**随机分量** \(z_t \sim q_\phi(z_t \mid h_t, o_t)\)(对应后验更新)。变分目标最小化

\[
\mathcal{L}_{\mathrm{RSSM}} = \mathbb{E}_q\!\left[\sum_t -\log p_\theta(o_t \mid h_t, z_t) + \beta\, D_{\mathrm{KL}}\!\left(q_\phi(z_t \mid h_t, o_t)\,\big\|\,p_\theta(z_t \mid h_t)\right)\right],
\]

其中先验 \(p_\theta(z_t \mid h_t)\) 由确定性递推给出,与 §7.8.2 的自由能框架直接对应。智能体在学到的隐空间中展开纯想象(imagination)轨迹以进行策略优化,无需与真实环境交互。

---

以上四类形式化共同指向一个判断:世界模型的数学骨架并非全新发明,而是把变分推断、Kalman 滤波、对比学习这些已有工具按新的工程目标重新组装。下一节给出每种形式化的边界与失效场景。


---

## §7.9 反类比与边界

### 7.9.1 HiPPO"最优性"的测度依赖

§7.2 的最优性结论具有根本的相对性:

**命题 7.7**:HiPPO-LegT 和 HiPPO-LegS 给出的 \(A\) 矩阵各自在其特定测度下是最优的——LegT 在长度为 \(\theta\) 的**滑动窗口**均匀测度下最优,LegS 在整个历史 \([0, t]\) 的**均匀测度**下最优。在另一种测度下使用对方的矩阵将不是最优近似。

例如,若使用**指数衰减**测度 \(\mu^{(t)}(ds) = e^{-(t-s)} ds\),则最优矩阵 \(A_{\text{LagT}}\) 是 Laguerre 多项式对应的矩阵,与 LegT、LegS 均不同。实践中,S4 使用 LegS 初始化(因为它对全局历史最优),但在学习过程中 \(A\) 矩阵会偏离 LegS 结构,表明任务-特定最优性与理论最优性存在差距。

**推论**:HiPPO 框架的成功部分来自"良好的初始化",而非全局最优。后续工作(How to Train your HIPPO, 2023)表明,S4D(对角 S4)等变体虽然放弃了严格的 HiPPO 最优性,但通过更灵活的参数化在实践中表现更好。此外,不同测度对应不同"时间分辨率":LegT 在近期历史上有更高精度,LegS 对所有时刻均等对待。没有一种测度在所有任务上都最优。

### 7.9.2 Mamba 并行扫描的数值稳定性

选择性 SSM 的并行扫描在数值上面临以下挑战:

**问题 1:矩阵乘积的数值范围**。在式 \eqref{7.19} 的长前缀积中,若状态转移矩阵 \(\bar{A}_k\) 的谱范数大于 1,则乘积指数增长(梯度爆炸);若远小于 1,则乘积指数衰减(梯度消失)。Mamba 通过限制 \(A\) 为具有负实部特征值的对角矩阵(从而 \(\|\bar{A}_t\| = \|e^{\Delta_t A}\| < 1\))来缓解此问题。

**问题 2:与 Softmax Attention 的对比**。Softmax 注意力的归一化操作保证注意力权重之和为 1,这是一种内建的数值归一化机制。对数-求和-指数(log-sum-exp)技巧(FlashAttention)进一步在数值稳定的对数域中计算,避免上溢。状态扫描没有类似的归一化,状态 \(h_t\) 的数值范围可能随序列长度增长而漂移。Mamba 的对数域扫描技巧是部分解决方案,但未被所有实现采用。

**问题 3:长序列下的梯度传播**。在极长序列(\(L > 10^5\))上,即使使用并行扫描,反向传播时 \(O(\log L)\) 层二叉树的激活值存储仍需 \(O(L)\) 显存,可能成为瓶颈。FlashAttention 通过重计算前向激活避免存储 \(O(L^2)\) 注意力矩阵,SSM 的类似技巧(FlashSSM)也在被探索。

### 7.9.3 线性注意力在检索任务上的系统性劣势

**实验规律**:在需要精确键值检索的任务(如 Passkey Retrieval、Copy、归纳头（Induction Heads）)上,线性注意力一致地输给 softmax 注意力,差距在长序列下尤为显著。

**理论解释**:

设查询集中有某个精确匹配的查询 \(Q^*\),对应键 \(K^* = Q^*\)。在 softmax 注意力中:

\[
A_{i^*, j^*} = \frac{e^{\|Q^*\|^2/\sqrt{d}}}{\sum_{j} e^{Q^* \cdot K_j/\sqrt{d}}} \to 1 \quad \text{当 } \|Q^*\|^2/\sqrt{d} \to \infty,
\]

通过增大 \(\|Q^*\|\)(由学习控制)可以任意接近 1,实现"赢者通吃"的精确检索。

而在线性注意力中,分配给 \(j^*\) 的权重为:

\[
w_{j^*} = \frac{\phi(Q^*) \cdot \phi(K^*)}{\sum_j \phi(Q^*) \cdot \phi(K_j)} = \frac{\phi(Q^*) \cdot \phi(K^*)}{\phi(Q^*) \cdot \mathbf{z}}.
\]

**命题 7.8 (线性注意力的检索上界)**:设特征维度为 \(r\),序列中有 \(L\) 个键均匀随机分布。则在最坏情况下,线性注意力分配给正确键的权重期望不超过 \(O(r/L)\)。

**证明思路**:由柯西-施瓦茨不等式,\(|\phi(Q^*) \cdot \phi(K)| \leq \|\phi(Q^*)\| \|\phi(K)\|\)。当 \(L\) 个键均匀分布时,\(\|\mathbf{z}\| = \|\sum_j \phi(K_j)\| \geq \sqrt{L} \|\phi(K^*)\|\)(大数定律),故 \(w_{j^*} \leq \|\phi(K^*)\| / (\sqrt{L} \|\phi(K^*)\|) = 1/\sqrt{L}\)。更精细的分析给出 \(O(r/L)\) 的界。\(\square\)

这从数学上解释了线性注意力在检索任务上的系统性不足:有限秩的特征映射无法模拟 softmax 的无界尖锐化能力。实践中,这一限制导致 GLA、Mamba 等模型在 RULER 等需要长距离精确检索的基准上显著落后于 Transformer。

---

## §7.10 参考文献

1. **Gu, A., Dao, T., Ermon, S., Rudra, A., & Ré, C.** (2020). HiPPO: Recurrent Memory with Optimal Polynomial Projections. *Advances in Neural Information Processing Systems (NeurIPS)*, 33. https://proceedings.neurips.cc/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html

2. **Gu, A., Goel, K., & Ré, C.** (2022). Efficiently Modeling Long Sequences with Structured State Spaces (S4). *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2111.00396

3. **Gu, A., & Dao, T.** (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *arXiv preprint arXiv:2312.00752*. https://arxiv.org/abs/2312.00752

4. **Dao, T., & Gu, A.** (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2/SSD). *International Conference on Machine Learning (ICML)*. https://arxiv.org/abs/2405.21060

5. **Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F.** (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. *International Conference on Machine Learning (ICML)*, PMLR 119. https://proceedings.mlr.press/v119/katharopoulos20a.html

6. **Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlos, T., ... & Weller, A.** (2021). Rethinking Attention with Performers (FAVOR+). *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2009.14794

7. **Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., ... & Wei, F.** (2023). Retentive Network: A Successor to Transformer for Large Language Models. *arXiv preprint arXiv:2307.08621*. https://www.microsoft.com/en-us/research/publication/retentive-network-a-successor-to-transformer-for-large-language-models/

8. **Yang, S., Wang, B., Shen, Y., Panda, R., & Kim, Y.** (2024). Gated Linear Attention Transformers with Hardware-Efficient Training (GLA). *International Conference on Machine Learning (ICML)*. https://arxiv.org/abs/2312.06635

9. **Sahoo, S. S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J. T., Rush, A., & Kuleshov, V.** (2024). Simple and Effective Masked Diffusion Language Models (MDLM). *Advances in Neural Information Processing Systems (NeurIPS)*, 38. https://neurips.cc/virtual/2024/poster/95622

10. **Nie, S., Zhu, F., Du, C., Zhang, T., Ou, Z., Muennighoff, N., ... & Li, C.** (2025). Large Language Diffusion Models (LLaDA). *arXiv preprint arXiv:2502.09992*. https://arxiv.org/abs/2502.09992

11. **Ho, J., Jain, A., & Abbeel, P.** (2020). Denoising Diffusion Probabilistic Models (DDPM). *Advances in Neural Information Processing Systems (NeurIPS)*, 33. https://arxiv.org/abs/2006.11239

12. **Blelloch, G. E.** (1990). Prefix Sums and Their Applications. *Technical Report CMU-CS-90-190*, Carnegie Mellon University. https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf

13. **Schlag, I., Irie, K., & Schmidhuber, J.** (2021). Linear Transformers Are Secretly Fast Weight Programmers. *International Conference on Machine Learning (ICML)*, PMLR 139. https://arxiv.org/abs/2102.11174

14. **Gu, A., Johnson, I., Timalsina, A., Rudra, A., & Ré, C.** (2023). How to Train Your HiPPO: State Space Models with Generalized Orthogonal Basis Projections. *International Conference on Learning Representations (ICLR)*. https://openreview.net/forum?id=klK17OQ3KB

15. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I.** (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30. https://arxiv.org/abs/1706.03762

16. **Gu, A., Goel, K., Gupta, A., & Ré, C.** (2022). On the Parameterization and Initialization of Diagonal State Space Models (S4D). *Advances in Neural Information Processing Systems (NeurIPS)*, 35. https://arxiv.org/abs/2206.11893

17. **Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R.** (2021). Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM). *Advances in Neural Information Processing Systems (NeurIPS)*, 34. https://arxiv.org/abs/2107.03006

18. **Voelker, A., Kajić, I., & Eliasmith, C.** (2019). Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 32. https://proceedings.neurips.cc/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html

---

*本章完。数学卷至此与主本第 1–7 章逐章镜像,公式与推导补完。*



## 后记

本附册的写作目标是让读者「在主本里看到几何，在附册里看到公式」。如果你读完主本之后愿意翻开这里，并在某一节停下来推一遍——那这本书就达成了它最大的野心。

数学之所以重要，不是因为它能预测一切，而是因为它能让我们看清「哪里是确定的、哪里是猜测的、哪里是隐喻」。第 5 章的渗流相变是隐喻；第 1 章的交叉熵是确定的；第 4 章的 μP 是数学定理但前提需要满足；第 6 章的 I/O 下界是经过 1981 年 Hong-Kung 证明的渐近铁律；第 7 章的 HiPPO 是在「平移 Legendre 测度」这一特定选择下最优——换个测度，矩阵就换。把这些边界讲清楚，是这本附册的本职工作。

如果你在阅读中发现任何公式错误、推导疏漏、或者认为某节应该更深入，欢迎寄信至 [fqsx@mail.ustc.edu.cn](mailto:fqsx@mail.ustc.edu.cn) 指正。

——大队长，2026 年 6 月
