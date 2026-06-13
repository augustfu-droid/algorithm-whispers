# 数学卷·第 3 章：注意力机制的矩阵微分

---

## §3.1 从 RNN 到变换器（Transformer）：序列建模的范式革命

```{=latex}
\begin{sectionintuition}
\textbf{为什么 RNN 不能并行}？RNN 把序列理解为一条因果链，每步计算依赖前一步的隐状态。这种设计在直觉上符合「逐字阅读」的体验，却注定无法并行——时序依赖锁死了硬件。\par\medskip
\textbf{第一步：量化梯度消失}。RNN 从时刻 $T$ 向 $t$ 反传梯度，需连乘 $T-t$ 个雅可比矩阵。若其谱半径 $\rho < 1$，梯度以 $\rho^{T-t}$ 指数衰减，模型无法学习长距离依赖；若 $\rho > 1$，梯度爆炸。LSTM 能缓解但不能根治。\par\medskip
\textbf{第二步：Transformer 的切断}。Transformer 彻底去掉时序递推：输入矩阵 $X \in \mathbb{R}^{n \times d}$ 整体存在，每个位置的输出通过矩阵乘法直接从全局上下文「读取」，$n$ 个位置完全并行。\par\medskip
\textbf{第三步：代价是什么}。并行化的代价是失去天然的顺序归纳偏置，必须显式注入位置信息（见 §3.4 RoPE），且计算量是 $O(n^2 d)$，序列越长越昂贵（见 §3.5 FlashAttention）。\par\medskip
\textbf{本节要拿走的一件事}：Transformer 的崛起不是「注意力更聪明」，而是「矩阵乘法比时序递推更适合 GPU」——这是一次范式替换，不是渐进改良。
\end{sectionintuition}
```

**直觉**

循环神经网络（Recurrent Neural Network，RNN）把序列理解为一条因果链：每一步的计算依赖前一步的结果。这种设计在直觉上符合「逐字阅读」的体验，但也带来了一个数学上的死结——序列中任意两步之间的信息传递必须沿着这条链一步一步走，无法跳过，无法并发。Transformer 的根本突破在于：彻底切断时间依赖，让每个位置的表示直接从全局上下文中「读取」，把序列问题还原为一组独立的矩阵运算。

本节出现四个核心概念：**RNN 的时序依赖与串行瓶颈、梯度消失/爆炸的数学根源、Transformer 的并行化设计、自注意力替代时序递推的代价**。它们看似散落，实际上沿着一条严密的逻辑链一个推出一个。理解这条链，是理解为什么深度学习从 RNN 时代跨越到 Transformer 时代的真正原因。下面把这条链一环环铺开。

**第一环：RNN 的时序依赖——为什么递推结构必然串行**

RNN 的隐状态递推公式是 $h_t = \sigma(W h_{t-1} + U x_t + b)$，意思是：第 $t$ 步的隐状态 $h_t$ 是由上一步的隐状态 $h_{t-1}$ 和当前输入 $x_t$ 共同决定的。这在计算图上意味着：$h_t$ 依赖 $h_{t-1}$，$h_{t-1}$ 依赖 $h_{t-2}$，以此类推——这是一条严格的数据流因果链。硬件层面的后果是决定性的：GPU 有数千个并行计算单元，但 RNN 在时间轴上的每一步必须等待前一步完成，这些并行单元在时间维度上大部分时间是空转等待的。这不是实现方式的问题，而是模型定义本身要求的依赖关系。序列长度 $n$ 越大，GPU 的并行潜力被浪费得越多——RNN 天花板由时序链长度，而不是硬件宽度决定。

**第二环：梯度消失/爆炸——长链反传的数学死亡**

并行问题可以忍受，但梯度问题是致命的。在反向传播中，从时刻 $T$ 的损失向时刻 $t$ 传梯度，需要通过链式法则展开一段连乘：

$$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{k=t}^{T-1} \frac{\partial h_{k+1}}{\partial h_k}$$

每一步的雅可比矩阵是 $\frac{\partial h_{k+1}}{\partial h_k} = \mathrm{diag}(\sigma'(\cdot)) W$，它的谱范数大约是 $\rho = \|W\|_2 \cdot \max|\sigma'|$。连乘 $T-t$ 个这样的矩阵后，整个梯度的量级大约是 $\rho^{T-t}$：若 $\rho < 1$，梯度以指数速率衰减至零（梯度消失），模型无法更新远处的参数；若 $\rho > 1$，梯度指数爆炸，训练发散。这不是参数调得不好导致的，而是连乘结构在数学上的必然。稍微多留意一下这个公式就能看出：**这个问题与序列长度 $T-t$ 直接相关**——序列越长，梯度越容易消失。这正是 RNN 在长文本上、在长依赖任务上表现欠佳的根本原因。

**第三环：LSTM 的门控——工程防护栏，而非数学根治**

长短期记忆网络（Long Short-Term Memory，LSTM）是 RNN 时代最成功的改良。它引入了遗忘门、输入门、输出门，以及独立于隐状态的记忆单元（cell state）。门控的作用是让记忆单元的梯度路径更接近恒等映射——如果遗忘门全开（全 1），记忆单元的梯度可以无衰减地回传到早期时刻，有效谱半径接近 1。这在工程上大大缓解了梯度消失问题，使 LSTM 能学到数十步内的依赖关系。

但这只是「防护栏」，不是「根治」。LSTM 仍然是递推结构，仍然无法并行，仍然在极长序列上（数百步以上）遭遇训练困难。更深的问题是：门控参数本身也需要从数据中学习，而它们的正确取值依赖于上下文——在训练初期，门控往往会打开错误的时间段，导致真正重要的远距离信息依然无法有效传播。LSTM 把 RNN 的天花板从大约 50 步提高到大约 200 步，但它没有从根本上改变「距离越长，信号越弱」的衰减规律。

**第四环：Transformer 的并行化切断——把序列问题变成矩阵问题**

Transformer 的设计思路是：彻底放弃时序递推。给定输入序列的嵌入矩阵 $X \in \mathbb{R}^{n \times d}$（所有 $n$ 个 token 同时存在于矩阵的行中），每个位置的输出通过一次矩阵运算直接从整个上下文读取：

$$O = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V, \quad Q = XW^Q,\ K = XW^K,\ V = XW^V$$

这里的关键是：$Q$、$K$、$V$ 全部从同一个矩阵 $X$ 一次性线性投影得到，计算过程没有任何时间轴上的依赖关系。第 $i$ 个 token 的输出需要用到其他所有 token 的信息，但这个「用到」是通过矩阵乘法的一次性全局聚合完成的，而不是通过逐步递推传递的。结果是：$n$ 个位置的输出可以完全并行计算，不需要等待任何「前序状态」。GPU 的所有并行计算单元可以同时满载工作。

更重要的是，任意两个位置之间的梯度传播路径长度从 $O(n)$ 降到了 $O(1)$——在自注意力层内，位置 $i$ 和位置 $j$ 之间只有一层矩阵运算相隔，不管它们在序列中相距多远。这彻底消除了连乘项，梯度消失/爆炸问题在模型的「时序维度」上不再存在（当然，层数增加时仍然存在，但这通过残差连接等机制解决）。

**第五环：并行化的代价——位置信息的缺失与二次复杂度**

任何设计都有代价。Transformer 去掉了时序递推，同时也去掉了时序结构自带的归纳偏置——模型不再天然知道「哪个词在前、哪个词在后」。用矩阵 $X$ 直接描述的 token 集合是无序的，如果不做任何处理，打乱输入序列的顺序，注意力的输出不会改变（除了行排列）。这意味着必须显式地向模型注入位置信息，否则「The dog bites the man」和「The man bites the dog」会得到完全相同的表示。这就是位置编码（positional encoding）出现的原因，§3.4 将专门讨论 RoPE 如何用旋转矩阵优雅地解决这个问题。

第二个代价是计算复杂度。注意力矩阵 $S = QK^T \in \mathbb{R}^{n \times n}$ 的大小随序列长度的平方增长；整个注意力计算的 FLOPs 为 $O(n^2 d)$。当 $n = 1024$ 时这不成问题，但当 $n = 100000$（长文档、代码库）时，$n^2$ 项变得极为昂贵，且 $n \times n$ 的矩阵本身在内存上也是 $O(n^2)$ 的开销。这是 FlashAttention（§3.5）和各种稀疏注意力（sparse attention）方法需要解决的核心瓶颈。

**整条链的回头看**

```
RNN 递推结构
  ↓ 时序依赖 → 串行、无法并行
梯度连乘 ρ^(T-t)
  ↓ ρ<1 消失，ρ>1 爆炸 → 长依赖无法学习
LSTM 门控
  ↓ 缓解但不根治 → 上限仍然存在
Transformer：去掉递推，输入矩阵一次性存在
  ↓ 所有位置并行 → GPU 满载
自注意力 O = softmax(QKᵀ/√dₖ) · V
  ↓ 梯度路径缩短到 O(1) → 长依赖无障碍
代价：无位置归纳偏置 + O(n²d) 复杂度
  ↓ 引出 RoPE（§3.4）和 FlashAttention（§3.5）
```

这条链揭示了一个深层事实：**Transformer 的崛起不是「注意力机制更聪明」，而是一次由硬件约束驱动的架构重设计**。梯度消失是 RNN 的数学死亡，并行化是 GPU 时代的必然要求，自注意力恰好同时解决了两者——代价是引入了新的挑战（位置编码、二次复杂度），而这些挑战又催生了后续的一系列创新。

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

Transformer 完全去掉了时序递推：给定输入矩阵 $X \in \mathbb{R}^{n \times d}$（$n$ 个 token 同时存在于矩阵中），每个位置的输出仅通过矩阵乘法和 softmax 计算，不依赖任何「前序状态」。具体而言，注意力层的输入输出关系为

$$
O = \mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中 $Q, K, V$ 均由 $X$ 一次性线性投影得到，整个计算路径无时间轴上的串行依赖，因此 $n$ 个位置的输出可完全并行计算。

**与原书呼应**：原书 §3.1 提到 RNN「不能并行」和「长距离依赖丢失」两个缺陷，对应本节雅可比连乘的谱半径分析与并行化条件。

---

## §3.2 注意力公式的拆解：QKV 的真实含义

```{=latex}
\begin{sectionintuition}
\textbf{为什么注意力要用三个矩阵 $Q, K, V$}？不用三个，用一个或两个行不行？这不是历史偶然，而是「软检索」语义的最小充分参数化。\par\medskip
\textbf{第一步：把注意力理解为软索引}。给定查询 $Q$，在键集合 $K$ 中按相似度打分，再用归一化的分数加权混合值 $V$。与哈希表不同，这里每个键都有权重，结果是所有值的连续混合。\par\medskip
\textbf{第二步：为什么要除以 $\sqrt{d_k}$}。若 $q$ 和 $k$ 各分量方差为 1，它们的内积方差为 $d_k$，标准差为 $\sqrt{d_k}$。不缩放时 softmax 的输入量级随维度增大，导致概率质量集中于极少数项，梯度趋零。除以 $\sqrt{d_k}$ 将方差还原为 1。\par\medskip
\textbf{第三步：多头的意义}。将 $d$ 维空间切成 $h$ 个子空间，每头在低维子空间中独立做软检索，捕捉不同粒度的语义关系（语法依赖、指代关系、话题关联……），总参数量不变。\par\medskip
\textbf{本节要拿走的一件事}：$\sqrt{d_k}$ 缩放不是调参技巧，是防止高维内积让 softmax 退化为 one-hot 的数学保险。
\end{sectionintuition}
```

**直觉**

注意力公式 $\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^T/\sqrt{d_k})V$ 看上去只有几个符号，但每个设计决策背后都有明确的数学动机。为什么要有三个矩阵而不是一个？为什么要除以 $\sqrt{d_k}$？为什么要用多头？这些问题有确定的答案，而不是「试出来好用就用」。

本节出现五个核心概念：**QKV 的三角色分离、Q≠K 的方向性、scaling 因子的统计动机、softmax 的归一化语义、多头注意力的维度分解**。它们沿着「为什么要这样设计」的逻辑链依次推出。下面把这条链铺开。

**第一环：把注意力理解为软索引——为什么需要 Q、K、V 三者分离**

最直觉的理解：注意力是一种「软哈希表检索」。在硬哈希表里，你用一个键（key）精确匹配到一个值（value）；在软检索里，你的查询（query）和所有键计算相似度，然后按相似度加权汇总所有值。这个框架要求三个独立的角色：

- $Q$（Query）：当前位置想「问什么」，即检索请求；
- $K$（Key）：每个位置「愿意回答什么问题」，即可被检索的标签；
- $V$（Value）：每个位置实际携带的内容，即被检索到时贡献的信息。

为什么这三个必须是独立的矩阵，而不能合并？关键在于 $Q$ 和 $K$ 承担不同职责：$Q$ 描述的是「我需要什么类型的信息」，$K$ 描述的是「我能提供什么类型的信息」。一个 token 在表达「我提供语法主语信息」（K 的角色）和「我在检索动词信息」（Q 的角色）时，使用的特征投影方向可以完全不同。如果强制 $Q = K$（即只用一个投影矩阵），注意力打分矩阵 $QK^T$ 就退化为对称矩阵——但语言中的「关注关系」往往是非对称的：主语需要关注谓语，但谓语不一定需要关注主语。对称矩阵无法表达方向性，会损失大量表达能力。

$V$ 与 $K$ 的分离则是另一层解耦。即使某个 token 在当前上下文中被「高度关注」（高注意力权重），它实际贡献给输出的内容（Value 向量）可以是完全不同维度的特征。举个例子：位置 $j$ 可能在语法上被位置 $i$ 高度关注（$k_j$ 与 $q_i$ 高度相似），但真正传递给 $i$ 的内容（$v_j$）是语义级别的上下文特征，而非语法标签本身。这种「谁关注谁」与「传递什么」的解耦，是注意力比单纯点积相似度强大得多的根本原因。

**第二环：为什么 Q≠K 是必要的——方向性与非对称性**

更精确地说明为什么 $Q$ 和 $K$ 必须是独立的投影。设 $W^Q = W^K$，则 $Q = XW^Q$，$K = XW^K = XW^Q = Q$，注意力打分矩阵变为：

$$A = \frac{QQ^T}{\sqrt{d_k}}$$

$QQ^T$ 是一个对称半正定矩阵（Symmetric Positive Semi-Definite，SPSD）。它的语义是：每个 token 与自身的相似度最高，与其他 token 的相似度由特征空间距离决定，且「$i$ 关注 $j$」和「$j$ 关注 $i$」的分数相同。但语言中大量的依存关系是方向性的：英文中「形容词修饰名词」意味着形容词应该关注名词（从名词的 Key 中读取语法类别），但名词不需要以同等程度关注形容词。强制对称会把这种非对称依存关系的表达能力减半。独立的 $W^Q \neq W^K$ 让注意力矩阵可以是任意非对称方阵，表达能力大幅提升。

**第三环：scaling 因子 $\sqrt{d_k}$——防止高维内积让 softmax 退化**

假设 $q_i$ 和 $k_j$ 的每个分量都独立同分布，均值为 0，方差为 1（这在 Xavier 或 He 初始化后是合理的假设）。则它们的内积 $q_i \cdot k_j = \sum_{l=1}^{d_k} q_{il} k_{jl}$ 是 $d_k$ 个独立零均值随机变量之和，其方差为：

$$\mathrm{Var}(q_i \cdot k_j) = \sum_{l=1}^{d_k} \mathrm{Var}(q_{il})\mathrm{Var}(k_{jl}) = d_k$$

标准差为 $\sqrt{d_k}$。当 $d_k = 64$（典型值），标准差为 8；当 $d_k = 128$，标准差为 11.3。进入 softmax 时，如果输入向量的各分量相差 8 到 11 个单位，$\exp$ 函数会把这个差异放大为约 $e^{11} \approx 60000$ 倍——概率质量几乎全部集中到最大值对应的那个位置，softmax 退化为近似 one-hot，梯度接近零。除以 $\sqrt{d_k}$ 将标准差还原为 1，保持 softmax 输入在合理量级，让各位置都有有效的梯度信号。

这不是「试出来好用的超参」，而是统计上的必然：**内积方差随维度线性增长，缩放因子是对这一增长的精确补偿**。

**第四环：softmax 的归一化语义——为什么是逐行而不是全局**

softmax 沿行（即针对每个 Query）独立归一化：

$$P_{ij} = \frac{\exp(A_{ij})}{\sum_{l=1}^n \exp(A_{il})}$$

这使得第 $i$ 行的权重之和为 1，构成一个概率分布。这里有一个重要的设计选择：是逐行归一化（每个 Query 独立），还是对整个矩阵 $A$ 做全局归一化？答案是逐行，原因是：注意力的语义是「第 $i$ 个 token 从所有其他 token 中按概率分配注意力资源」，每个 token 的关注权重之和为 1 确保了输出的量级不随序列长度变化（否则序列越长，注意力加权和越大，输出越不稳定）。全局归一化会让某些 token 「抢走」其他 token 的注意力预算，破坏每个 Query 独立决策的语义。

**第五环：多头注意力——维度分解带来的容量倍增**

如果只做一次注意力，所有 $d$ 维信息混在一起，模型只能在一个语义层次上「问问题」。多头注意力把 $d$ 维空间分成 $h$ 个 $d/h$ 维子空间，每个头在各自的子空间中独立做注意力：

$$\mathrm{MultiHead}(X) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W^O$$

每个头有独立的 $W^Q_i, W^K_i, W^V_i \in \mathbb{R}^{d \times d/h}$，参数量与单头相同（$3d^2/h \times h = 3d^2$，加上 $W^O$ 后总量约 $4d^2$）。但不同的头会在训练中自发分工：某些头专注句法依存（主语-谓语）、某些头追踪长距离指代（代词-名词）、某些头捕捉局部 $n$-gram 模式。这种分工是**涌现的**，是梯度下降在参数空间中自然寻找到的最优解，不是人为规定的。多头的本质是：用维度分解换取「并行检索不同粒度语义关系」的能力，在不增加参数量的情况下扩展表达能力。

**整条链的回头看**

```
软检索语义
  ↓ 需要「谁问、谁答、答什么」三个独立角色
Q、K、V 三矩阵分离
  ↓ Q≠K 保证非对称方向性
注意力打分 A = QK^T
  ↓ d_k 维内积方差 = d_k，softmax 会退化
除以 sqrt(d_k)，方差还原为 1
  ↓ softmax 逐行归一化，维持每个 Query 的概率语义
P = softmax(A/sqrt(d_k))，输出 O = PV
  ↓ 单头只能在一个层次检索
多头：h 个子空间并行，捕捉不同粒度的语义关系
```

这条链揭示了一个深层事实：**QKV 的每一个设计细节都对应一个数学必然性**。$\sqrt{d_k}$ 是统计补偿，三矩阵分离是表达能力的必要条件，多头是维度分解的容量策略。理解这条链，就理解了为什么注意力公式是这样而不是那样。

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

其中 $\mathrm{head}_i = \mathrm{Attention}(X W^Q_i,\, X W^K_i,\, X W^V_i)$。多头设计的参数总量与单头相同（$d^2$ 量级），但在 $h$ 个低维子空间中并行运行不同的「检索模式」。

**与原书呼应**：原书 §3.2 逐步拆解了 QKV 投影与缩放技巧，对应本节方差推导与 softmax 归一化维度分析。

---

## §3.3 矩阵乘法的大规模并发：为什么 AI 是 GPU 的盛宴

```{=latex}
\begin{sectionintuition}
\textbf{为什么矩阵乘法和 GPU 如此般配}？GPU 的设计哲学是「以宽度换速度」——数千个简单核心并发执行同一条指令，作用于不同数据。矩阵乘法天然吻合：每个输出元素是独立内积，可完全并行。\par\medskip
\textbf{第一步：算术强度决定瓶颈在哪里}。算术强度 $I = \text{FLOPs} / \text{内存字节数}$。大矩阵乘法的 $I \approx 2N/3$ 随矩阵尺寸线性增长，轻松超过 GPU 的「屋脊点」，属于计算受限——GPU 算力能被充分榨取。\par\medskip
\textbf{第二步：softmax 是短板}。逐元素操作（如 $\exp$）算术强度接近 1，远低于屋脊点，属于内存受限。标准 attention 中 softmax 和矩阵读写的来回搬运，才是真正的性能瓶颈（见 §3.5）。\par\medskip
\textbf{第三步：attention 的算力复杂度是 $O(n^2 d)$}。序列长度 $n$ 一旦增大，$n^2$ 项主导，FLOPs 急剧增长——这是长上下文昂贵的根本原因。\par\medskip
\textbf{本节要拿走的一件事}：「AI 是 GPU 的盛宴」不是比喻，是算术强度分析的结论——矩阵乘法恰好落在 GPU 架构最擅长的计算受限区间。
\end{sectionintuition}
```

**直觉**

「AI 是 GPU 的盛宴」这句话是字面成立的，不是修辞。要理解为什么，必须理解 GPU 擅长什么、深度学习计算的性质是什么，以及二者为何在算术强度这个指标上精确匹配。这节的核心任务是把「GPU 适合 AI」从直觉转化为一个可以量化的、可以预测瓶颈的数学分析框架——Roofline 模型。

本节出现五个核心概念：**GPU 的 SIMD 并行架构、算术强度的定义、Roofline 模型与屋脊点、矩阵乘法的算术强度分析、softmax 的内存受限本质**。下面把这条链铺开。

**第一环：GPU 的 SIMD 架构——为什么是「以宽度换速度」**

GPU 和 CPU 的设计哲学截然不同。CPU 的设计目标是「让单条指令流以最快速度执行」：大量晶体管用于分支预测、乱序执行、超标量流水线、大缓存——这些机制减少了单线程的延迟，但每个核心复杂且昂贵，一块 CPU 只有几个到几十个核心。GPU 的设计目标是「让同一条指令同时作用于数千个数据」：SIMD（Single Instruction Multiple Data）架构，每个流处理器（Streaming Multiprocessor，SM）包含数十到数百个 CUDA core，一整块 GPU（如 A100）有超过 6000 个。每个核心很简单，但数量极多。

GPU 擅长的计算必须满足一个条件：**大量独立的、结构相同的操作**。如果操作之间有依赖（结果 A 需要等待结果 B），数千个核心中大多数就会空转等待。矩阵乘法 $C = AB$（$A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$）的每个输出元素 $C_{ij} = \sum_{l} A_{il} B_{lj}$ 是独立内积，不同的 $(i,j)$ 对之间完全没有数据依赖——完美适合 SIMD。这是矩阵乘法与 GPU 架构的第一层匹配：**操作的独立性**。

**第二环：算术强度——量化「计算量 vs 数据量」的比值**

但独立性只是一个必要条件，不是充分条件。GPU 还有第二个瓶颈：内存带宽。计算核心即使再多，如果数据喂不进来，核心也只能空等。这引出了算术强度（Arithmetic Intensity）的概念：

$$I = \frac{\text{浮点运算量（FLOPs）}}{\text{内存访问字节数（Bytes）}}$$

算术强度的单位是 FLOP/Byte，衡量「每读取一个字节的数据，能做多少浮点运算」。这个比值高，说明计算「划算」——读进来的数据被充分利用，内存带宽不是瓶颈；比值低，说明每读一字节只做几个操作，计算很快就完成但要等下一批数据，内存带宽是瓶颈。

现代 GPU 有两种类型的存储：片上 SRAM（Shared Memory）速度极快（带宽 TB/s 量级）但容量小（MB 量级），以及片外 HBM（High Bandwidth Memory）容量大（几十 GB）但相比 SRAM 慢一个数量级（几百 GB/s）。「算术强度」这里说的「内存访问」主要指对 HBM 的访问——片外数据搬运的速度是真正的瓶颈。

**第三环：Roofline 模型——把瓶颈可视化的工具**

Roofline 模型是分析硬件利用率的标准工具。它用一张图描述 GPU 核函数的实际性能上界：

$$P_{\text{actual}} = \min\!\bigl(\text{峰值算力（FLOP/s）},\; I \times \text{峰值内存带宽（Byte/s）}\bigr)$$

当算术强度 $I$ 低于「屋脊点」（ridge point，即 峰值算力/峰值内存带宽）时，核函数受内存带宽限制（memory-bound）：内存带宽决定了数据流入速度，多余的计算单元无法被填满；当 $I$ 高于屋脊点时，核函数受峰值算力限制（compute-bound）：数据足够快，所有计算单元满载。

对 NVIDIA A100 GPU：峰值算力约 312 TFLOP/s（BF16），峰值 HBM 带宽约 2 TB/s，屋脊点约为 $312 / 2 = 156$ FLOP/Byte。这意味着：只要一个操作的算术强度超过 156，它就是计算受限；低于 156，它就是内存受限。

**第四环：矩阵乘法的算术强度——为什么落在计算受限区**

对矩阵乘法 $C = AB$（$A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$，$C \in \mathbb{R}^{m \times n}$）：
- FLOPs：$2mnk$（每个输出元素做 $k$ 次乘加）
- 内存访问字节数：读 $A$（$mk$）、读 $B$（$kn$）、写 $C$（$mn$），以 BF16（2 Byte/元素）计，总计 $2(mk + kn + mn)$ 字节

算术强度为：

$$I = \frac{2mnk}{2(mk + kn + mn)} = \frac{mnk}{mk + kn + mn}$$

当 $m = n = k = N$ 时，$I \approx \frac{N^3}{3N^2} = \frac{N}{3}$，随矩阵尺寸线性增长。对 $N = 4096$（LLM 中典型的隐层维度），$I \approx 1365$ FLOP/Byte，远超 A100 的屋脊点 156 FLOP/Byte。**大矩阵乘法深度落在计算受限区**，GPU 的算力峰值能被充分利用，内存带宽不是瓶颈。Tensor Core 正是为此设计的：以 $16 \times 16$ 或更大的矩阵块为基本操作单元，在片上完成高密度计算，完全隐藏内存访问延迟。

**第五环：softmax 的算术强度——内存受限的短板**

注意力计算不只是矩阵乘法，还有 softmax。softmax 是逐行的操作：对每一行，先找最大值，再计算 $\exp$，再求和，再除以总和。每个元素只需要几次浮点操作（约 3-5 FLOP），但每个元素都要从 HBM 读进来、再写出去。算术强度约为 3-5 FLOP/Byte，比屋脊点低 30 倍以上——典型的内存受限操作。更严重的是，标准 attention 实现要对 $n \times n$ 的矩阵做 softmax，需要把整个矩阵写入 HBM、再读出来处理，HBM 访问量为 $\Theta(n^2)$。这是 attention 计算性能的真正瓶颈——不是矩阵乘法（计算受限，利用率高），而是 softmax 和中间矩阵的读写搬运（内存受限，利用率低）。这直接导致了 FlashAttention 的诞生（§3.5），其核心思路正是消除这一 $\Theta(n^2)$ 的 HBM 访问。

**整条链的回头看**

```
GPU：SIMD 架构，数千核心并行同一操作
  ↓ 需要：大量独立、结构相同的操作
矩阵乘法：每个输出元素 C_ij 是独立内积 ✓
  ↓ 但独立性不够，还需要：每字节数据充分利用
算术强度 I = FLOPs / Bytes
  ↓ 矩阵乘法：I ≈ N/3，随尺寸线性增长
Roofline 模型：I > 屋脊点 → 计算受限
  ↓ 大矩阵乘法落在计算受限区 ✓
attention 的软肋：softmax，I ≈ 3-5，内存受限 ✗
  ↓ 引出 FlashAttention（§3.5）
attention FLOPs = O(n²d)，长上下文时 n² 主导
```

这条链揭示了一个深层事实：**「AI 是 GPU 的盛宴」有精确的数学含义**——矩阵乘法的算术强度随矩阵尺寸线性增长，恰好落在 GPU 算力受限区，让 TFLOP/s 级别的算力得以充分发挥。但注意力中的 softmax 是内存受限的短板，正是它驱动了 FlashAttention 这一系统级创新。

**严格**

**矩阵乘法的算术强度**

对两个矩阵相乘 $C = AB$，其中 $A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$，$C \in \mathbb{R}^{m \times n}$：

- 浮点运算量（FLOPs）：$2mnk$（每个输出元素做 $k$ 次乘加）
- 内存访问字节数（读 $A$、$B$，写 $C$）：$(mk + kn + mn) \times \text{sizeof}(\text{dtype})$

算术强度定义为

$$
I = \frac{2mnk}{mk + kn + mn} \quad (\text{FLOP/byte})
$$

当 $m, n, k$ 均很大（如 $m = n = k = N \gg 1$）时，$I \approx \frac{2N^3}{3N^2} = \frac{2N}{3}$，随矩阵尺寸线性增长。典型大矩阵乘法的算术强度远超 GPU 的「屋脊点」（ridge point，即峰值算力 / 峰值内存带宽），因此属于**计算受限**（compute-bound）操作，GPU 的算力可以充分利用。

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

```{=latex}
\begin{sectionintuition}
\textbf{为什么正弦位置编码不够用}？加法叠加把绝对位置混入 token 表示，让模型无法直接感知「两个 token 相距多远」，且在训练长度之外几乎立即失效。\par\medskip
\textbf{第一步：换一种思路——把位置编码进旋转角}。RoPE 将每对相邻维度 $(q_{2i}, q_{2i+1})$ 看成复平面上的一个点，位置 $m$ 对应旋转 $e^{\mathrm{i}m\theta_i}$，不加法，只旋转。\par\medskip
\textbf{第二步：旋转的魔法——内积只剩位置差}。由于旋转矩阵正交，$\langle \tilde{q}_m, \tilde{k}_n \rangle = q_m^T \mathcal{R}_{n-m} k_n$，绝对位置 $m$、$n$ 从内积中消去，只留下相对位置 $n-m$。这正是语言学上真正有用的信号。\par\medskip
\textbf{第三步：多频率覆盖多尺度}。低维度对应高频旋转（短程依赖），高维度对应低频旋转（长程依赖），类似傅里叶分析的多尺度分解。调整底数可延伸训练外推长度。\par\medskip
\textbf{本节要拿走的一件事}：RoPE 把「相对位置」这一语言学核心信号，用旋转矩阵的正交性以数学方式硬编码进注意力内积——不依赖模型去「学」位置关系，而是直接设计进结构。
\end{sectionintuition}
```

**直觉**

位置编码是 Transformer 的「补丁」：去掉了时序递推后，模型对 token 的顺序一无所知，必须通过某种方式把位置信息注入表示。原始 Transformer 用正弦函数生成一个固定向量加到嵌入上，这个加法方案能用，但有两个根本性缺陷：一是它编码的是绝对位置（token 在序列中的绝对位置编号），而语言学上真正重要的往往是相对位置（两个 token 相距多远）；二是它在训练最大长度之外几乎立即失效（模型从未见过那些位置的向量，无法外推）。RoPE（Rotary Position Embedding）用一个精巧的代数构造解决了这两个问题。

本节出现五个核心概念：**绝对位置编码的局限、旋转的群代数、RoPE 的复数化构造、内积只剩位置差的证明、多频率傅里叶视角**。下面把这条链铺开。

**第一环：绝对位置编码的局限——为什么加法无法编码相对位置**

正弦位置编码的方案是：对位置 $m$，生成一个固定向量 $\mathrm{PE}(m) \in \mathbb{R}^d$，加到 token 嵌入上：$x'_m = x_m + \mathrm{PE}(m)$。然后注意力打分变为：

$$(x'_m W^Q)(x'_n W^K)^T = (x_m + \mathrm{PE}(m))W^Q ((x_n + \mathrm{PE}(n))W^K)^T$$

展开后会有四项：$x_m W^Q (W^K)^T x_n^T$（纯语义）、$\mathrm{PE}(m)W^Q (W^K)^T x_n^T$（查询方的位置×键的语义）、$x_m W^Q (W^K)^T \mathrm{PE}(n)^T$（查询方语义×键的位置）、$\mathrm{PE}(m)W^Q (W^K)^T \mathrm{PE}(n)^T$（两个绝对位置的交叉项）。这四项里，没有任何一项是只关于 $m-n$（位置差）的函数——绝对位置 $m$ 和 $n$ 都混在里面，无法分离。模型如果想利用相对位置信号，必须自己从四项的混合里学会「消去」绝对位置，这是额外的学习负担，且通常学不干净。

更严重的是训练外推问题。如果训练时最大序列长度是 2048，模型从未见过位置 2049 及以后的正弦向量。在推理时遇到更长序列，这些位置的编码向量在训练分布之外，模型的表现会急剧退化。

**第二环：旋转的代数思路——用群结构天然编码「差」**

解决方案的关键洞察是：**如果位置信息以乘法而非加法形式注入，内积就可以天然消去绝对位置**。

群论给出了一个精确的框架。旋转群 $SO(2)$（二维旋转）有一个核心性质：对旋转角度 $\theta_1$ 和 $\theta_2$ 对应的旋转矩阵 $R_{\theta_1}$ 和 $R_{\theta_2}$，有 $R_{\theta_1}^T R_{\theta_2} = R_{\theta_2 - \theta_1}$（两个旋转相乘等于角度相减的旋转）。这意味着：如果对位置 $m$ 的 query 向量施加旋转 $R_{m\theta}$，对位置 $n$ 的 key 向量施加旋转 $R_{n\theta}$，则它们的内积：

$$(R_{m\theta} q)^T (R_{n\theta} k) = q^T R_{m\theta}^T R_{n\theta} k = q^T R_{(n-m)\theta} k$$

绝对位置 $m$ 和 $n$ 精确消去，只剩下相对位置 $n-m$！这不是近似，是旋转矩阵正交性的精确代数恒等式。加法做不到这一点——加法没有这种「两项之差」的代数结构。

**第三环：RoPE 的复数化构造——把高维向量拆成二维旋转对**

如何把这个思路推广到高维向量（$d$ 维）？RoPE 的方案是：把 $d$ 维向量 $q \in \mathbb{R}^d$ 看成 $d/2$ 个复数，每对相邻维度 $(q_{2i}, q_{2i+1})$ 对应一个复数 $z^q_i = q_{2i} + \mathrm{i}\, q_{2i+1}$。对位于位置 $m$ 的 token，每对分量乘以单位复数 $e^{\mathrm{i}m\theta_i}$：

$$\tilde{z}^q_i = z^q_i \cdot e^{\mathrm{i}m\theta_i} = (q_{2i} + \mathrm{i}\, q_{2i+1})(\cos(m\theta_i) + \mathrm{i}\sin(m\theta_i))$$

实数化展开，等价于对每对分量施加一个二维旋转矩阵：

$$\begin{pmatrix} \tilde{q}_{2i} \\ \tilde{q}_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

对整个 $d$ 维向量，RoPE 变换是一个块对角旋转矩阵 $\mathcal{R}_m = \mathrm{blockdiag}(R_m^{(0)}, R_m^{(1)}, \ldots, R_m^{(d/2-1)})$，每个块是一个独立频率的二维旋转。由于每个块都满足旋转矩阵的正交性，第二环的代数性质直接推广到高维：

$$\langle \mathcal{R}_m q_m, \mathcal{R}_n k_n \rangle = q_m^T \mathcal{R}_m^T \mathcal{R}_n k_n = q_m^T \mathcal{R}_{n-m} k_n$$

内积只依赖位置差 $n-m$，精确成立。

**第四环：多频率傅里叶视角——为什么不同维度用不同频率**

如果所有 $d/2$ 对维度都使用同一个频率 $\theta$，每隔 $2\pi/\theta$ 个位置旋转一圈，模型会无法区分相差整数圈的两个位置（比如位置 5 和位置 5 + $2\pi/\theta$）——产生「位置混叠」。RoPE 用不同的频率参数：

$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1$$

低下标 $i$（如 $i=0$）对应 $\theta_0 = 1$，旋转快（高频）——每一步旋转 1 弧度，约 6 步转一圈，对短程依赖敏感；高下标 $i$（如 $i = d/2 - 1$）对应 $\theta_{d/2-1} = 10000^{-1} = 0.0001$，旋转极慢（低频）——要 $2\pi / 0.0001 \approx 62832$ 步才转一圈，对长程依赖编码有效。

这与傅里叶分析完全类似：不同频率的正弦波组合可以表示任意尺度的信号。高频分量感知局部差异（相邻 token），低频分量感知全局结构（远距离 token）。$d/2$ 个不同频率共同构成了一个「位置尺度的频谱」，让模型在 $d/2$ 个独立的尺度上同时感知位置关系。

**第五环：外推能力——频率调整如何延伸上下文长度**

正弦位置编码在训练长度外失效，是因为那些绝对位置的向量没有被模型见过。RoPE 没有这个问题：旋转是一个连续的函数，位置 $m = 10000$ 只是角度 $m\theta_i$ 稍微大了一些，不存在「未见过的向量」的问题。但实际上，当序列长度远超训练时，某些频率的旋转角度进入训练中从未出现过的区间（比如相对位置 $n-m$ 超过训练最大长度），高频分量会出现混叠。

解决方案是调整频率底数（把 10000 换成更大的值，如 $500000$），让低频分量在更大的相对位置范围内仍然处于「未完成一圈旋转」的状态——模型对这段范围的相对位置就有更好的区分能力。YaRN、LongRoPE 等方法的数学本质正是对不同频率段做差异化的频率插值，让模型能够在不重新训练的情况下处理数倍于训练长度的序列。

**整条链的回头看**

```
绝对位置编码（加法）
  ↓ 内积展开有四项，绝对位置无法从内积中消去
需要：能让内积只剩位置差的编码方式
  ↓ 旋转群 SO(2) 的性质：R_a^T R_b = R_{b-a}
旋转能精确消去绝对位置 → 设计 RoPE
  ↓ d 维向量 → 拆成 d/2 对，每对独立旋转
块对角旋转矩阵 R_m，位置 m → 角度 m*theta_i
  ↓ 内积 = q^T R_{n-m} k，只剩位置差 ✓
多频率：theta_i = 10000^{-2i/d}
  ↓ 高频感知短程，低频感知长程（傅里叶视角）
调整底数 → 训练外长度外推
```

这条链揭示了一个深层事实：**RoPE 不是「用旋转来凑一个好用的位置编码」，而是旋转群代数结构的直接应用**。旋转的群性质（角度可加性、正交性）天然给出了「内积只剩差」的性质，多频率设计则是傅里叶分析的自然推广。理解这条链，就理解了为什么 RoPE 能外推而正弦编码不能——不是经验调优，是代数结构的必然结果。

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

```{=latex}
\begin{sectionintuition}
\textbf{为什么标准 attention 慢}？不是因为 FLOPs 太多，而是因为 $n \times n$ 的注意力矩阵要在慢速 HBM 和快速 SRAM 之间反复搬运，内存带宽成了瓶颈。\par\medskip
\textbf{第一步：问题出在写-读-写-读的搬运链上}。标准实现先把 $S = QK^T$ 写到 HBM，再读出来做 softmax，再写回，再读出来乘 $V$——每步都过 HBM，HBM 访问量 $\Theta(n^2)$。\par\medskip
\textbf{第二步：FlashAttention 的思路是「不出芯片」}。把 $Q, K, V$ 切成能放入 SRAM 的小块，在片上完成「乘 + softmax + 再乘」全流程，只将最终输出写回 HBM 一次。\par\medskip
\textbf{第三步：难点在 softmax 需要全局统计量}。FlashAttention 用「在线 softmax」解决：用滚动最大值 $m_j$ 和指数和 $\ell_j$ 递推，无需提前知道全局最大值，可以边读块边更新输出。\par\medskip
\textbf{本节要拿走的一件事}：FlashAttention 的 FLOPs 没有减少，减少的是 HBM 读写量（从 $\Theta(n^2)$ 降至接近线性）——这是一次纯粹的内存工程优化，不改变数学结果，只改变计算路径。
\end{sectionintuition}
```

**直觉**

§3.3 揭示了注意力计算的软肋：softmax 是内存受限操作，标准 attention 实现需要把 $n \times n$ 的矩阵多次搬入搬出 HBM，内存带宽成为真正的瓶颈，而不是 FLOPs 本身。FlashAttention 是针对这个瓶颈的精准外科手术：在不改变数学结果的前提下，重新安排计算顺序，把所有中间量都留在 SRAM 里，只有最终结果才写回 HBM。理解这一方法需要搞清楚：为什么分块计算是可行的，以及 softmax 的全局归一化需求如何被「在线 softmax」精确规避。

本节出现五个核心概念：**HBM 与 SRAM 的速度差异与写读链条问题、分块（tiling）的思路、softmax 的全局依赖障碍、在线 softmax 递推、内存与速度的共赢**。下面把这条链铺开。

**第一环：写-读链条——标准 attention 的内存搬运代价**

标准 attention 实现的执行序列如下：

1. 计算 $S = QK^T / \sqrt{d_k} \in \mathbb{R}^{n \times n}$，结果写入 HBM（SRAM 放不下这个矩阵）；
2. 从 HBM 读出 $S$，逐行做 softmax，得到 $P \in \mathbb{R}^{n \times n}$，写回 HBM；
3. 从 HBM 读出 $P$ 和 $V$，计算 $O = PV \in \mathbb{R}^{n \times d_v}$，写回 HBM。

每次写入/读出 $n \times n$ 矩阵，访问量为 $\Theta(n^2)$ 个元素。对 $n = 8192$，这意味着每次注意力计算需要搬运约 $8192^2 = 6.7 \times 10^7$ 个 FP16 元素，约 134 MB。这看似不多，但注意力计算本身的 FLOPs 并不足以支撑这么多内存访问——算术强度太低，GPU 的计算单元大量空等。

更根本的问题是：SRAM（片上缓存）的容量约为 20 MB（A100），远小于 $n \times n$ 矩阵（134 MB）。所以中间矩阵必须存放在 HBM，而 HBM 的带宽只有约 2 TB/s——相比 SRAM 的约 19 TB/s 慢 10 倍。每次从 HBM 读写 $n \times n$ 矩阵，就是在用最慢的存储层做最频繁的数据交换。

**第二环：分块（Tiling）思路——让计算在 SRAM 内完成**

FlashAttention 的核心思路是：把 $Q, K, V$ 切成能放入 SRAM 的小块，对每个 $(Q\text{块}, K\text{块}, V\text{块})$ 组合在片上完成「矩阵乘 + softmax + 再乘」的完整流程，不写中间矩阵到 HBM。

具体地，把 $K$ 和 $V$ 按行切成 $T$ 个块，每块大小 $B \times d$（$B$ 足够小，使 $B \times d$ 能放入 SRAM）。外层循环遍历 $Q$ 的行块，内层循环遍历 $K, V$ 的列块。对每个 $(i, j)$ 对（$Q$ 的第 $i$ 行块，$K, V$ 的第 $j$ 列块），在 SRAM 内完成：
- 计算局部注意力分数 $S_{ij} = Q_i K_j^T / \sqrt{d_k}$
- 更新在线 softmax 统计量
- 用局部归一化权重更新输出 $O_i$

整个过程中，$n \times n$ 的注意力矩阵 $S$ 从未完整出现——它被分块计算并立即消费，中间结果从不落地到 HBM。

**第三环：softmax 的全局依赖障碍——为什么分块不是显然的**

分块听起来简单，但有一个关键障碍：softmax 需要全局统计量。对第 $i$ 行的 softmax：

$$P_{ij} = \frac{\exp(S_{ij})}{\sum_l \exp(S_{il})}$$

分母 $\sum_l \exp(S_{il})$ 需要这一行所有 $n$ 个元素的信息。如果分块处理，处理第 $j$ 个 $K$ 块时还不知道其他块的值，无法归一化。这是标准 softmax 的「两遍扫描」需求：第一遍找全局最大值（数值稳定性），第二遍做 $\exp$ 和归一化。

在数值稳定版本中（减去最大值防止 $\exp$ 溢出），softmax 公式为：

$$P_{ij} = \frac{\exp(S_{ij} - \max_l S_{il})}{\sum_l \exp(S_{il} - \max_l S_{il})}$$

分母和分子都需要全局最大值 $\max_l S_{il}$，必须先完成全局扫描才能计算。这阻碍了分块计算。

**第四环：在线 softmax——把两遍扫描变成一遍递推**

FlashAttention 用在线 softmax 递推解决这个问题。核心思想是：维护两个滚动统计量，在逐块处理时实时更新，处理完所有块后统计量恰好等于全局精确值，**数学上与两遍扫描完全等价**。

设已处理前 $j-1$ 个 $K$ 块，维护：
- $m_{j-1} \in \mathbb{R}$：前 $j-1$ 块中见过的最大分数（数值稳定的移位基准）
- $\ell_{j-1} \in \mathbb{R}$：移位后的指数和（归一化分母的运行值）
- $O_{j-1} \in \mathbb{R}^{d_v}$：当前输出的未完全归一化版本

处理第 $j$ 块时，令该块的局部最大值 $\tilde{m}_j = \max(S_j)$，局部指数和 $\tilde{\ell}_j = \sum_l \exp(S_{jl} - \tilde{m}_j)$。更新：

$$m_j = \max(m_{j-1}, \tilde{m}_j)$$

$$\ell_j = e^{m_{j-1} - m_j} \ell_{j-1} + e^{\tilde{m}_j - m_j} \tilde{\ell}_j$$

$$O_j = \mathrm{diag}(\ell_j)^{-1}\!\left(\mathrm{diag}(\ell_{j-1}) e^{m_{j-1} - m_j} O_{j-1} + e^{\tilde{m}_j - m_j} \tilde{P}_j V_j\right)$$

这个递推的精妙在于：每次遇到更大的局部最大值时，用比例因子 $e^{m_{\text{old}} - m_{\text{new}}}$ 重新缩放历史统计量——这相当于「追溯修正」之前看到的所有元素，把它们的移位基准更新到新的全局最大值。处理完所有块后，$m_T$ 就是全局最大值，$\ell_T$ 就是全局指数和，$O_T$ 就是精确的注意力输出。全程不需要存储 $n \times n$ 矩阵，只需要两个标量 $m_j, \ell_j$ 和一个 $d_v$ 维向量 $O_j$。

**第五环：内存与速度共赢——代价分析**

这个方案的复杂度如何变化？

- **FLOPs 不变**：FlashAttention 计算的数学结果与标准 attention 完全相同，只是改变了计算路径。对每个 $(Q\text{块}, K\text{块})$ 对，矩阵乘的工作量与标准实现一样，总 FLOPs 仍为 $\Theta(n^2 d)$。
- **HBM 访问量大幅下降**：不再需要写读 $n \times n$ 的中间矩阵。$Q, K, V$ 各读一次（$\Theta(nd)$），$O$ 写一次（$\Theta(nd)$），中间结果在 SRAM 内流转，不落地 HBM。HBM 访问量从 $\Theta(n^2)$ 降至 $\Theta(nd \cdot n/M)$（$M$ 为 SRAM 大小），实践中减少 4-16 倍。
- **显存占用从 $O(n^2)$ 降至 $O(n)$**：不再需要存储整个注意力矩阵，只需要 $m_j, \ell_j$ 两个标量和当前输出向量。这使得更长的上下文在相同显存下成为可能。

反向传播时需要重新计算注意力权重（因为不再存储 $n \times n$ 矩阵），这会增加约 1 倍的 FLOPs，但 HBM 节省的带宽收益远大于这个代价，端到端速度仍然大幅提升（实测 2-4 倍）。

**整条链的回头看**

```
标准 attention：写 S → 读 S → softmax → 写 P → 读 P → 乘 V
  ↓ HBM 访问 Θ(n²)，内存受限，计算单元空等
瓶颈：n×n 矩阵放不入 SRAM，被迫用 HBM
  ↓ 解法：分块计算，让中间结果留在 SRAM
分块障碍：softmax 需要全局最大值（两遍扫描）
  ↓ 解法：在线 softmax 递推（滚动最大值 + 指数和）
递推等价性：m_j, ℓ_j 更新 ≡ 全局精确 softmax ✓
  ↓ 结果
FLOPs 不变 + HBM 访问从 Θ(n²) 降至接近线性
  ↓ attention 从内存受限重新变为计算受限
GPU 利用率大幅提升，端到端 2-4 倍加速
```

这条链揭示了一个深层事实：**FlashAttention 的创新不在数学，而在于识别了「softmax 的全局依赖是分块障碍」这一核心问题，并用在线递推精确消除了这一障碍**。它证明了在不改变任何数学结果的前提下，通过重新安排计算顺序，可以把一个内存受限问题变回计算受限——这是系统级优化与数学保证完美结合的范例。

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

## §3.6 注意力的真实身份：关联记忆，而非「注意力」

```{=latex}
\begin{sectionintuition}
\textbf{为什么「注意力」这个名字有误导性}？认知心理学的注意力是有限资源的主动分配；而 softmax attention 在数学上更接近「内容寻址的关联记忆检索」——给定查询模式，在一组存储模式中检索最接近的内容。\par\medskip
\textbf{第一步：经典 Hopfield 网络的能量函数}。经典版本（1982）用二次能量函数存储模式，存储容量约 $0.14d$，线性于维度。\par\medskip
\textbf{第二步：现代 Hopfield 网络的升级}。将能量函数改为 log-sum-exp 形式，求不动点条件得到更新规则 $\xi^{\text{new}} = X^T \,\mathrm{softmax}(\beta X \xi)$——这与 softmax attention 公式在数学上完全等价（令 $\xi = Q_i$，$X = K$，$\beta = 1/\sqrt{d_k}$）。\par\medskip
\textbf{第三步：存储容量从线性变指数}。现代 Hopfield 网络的容量约为 $O(\exp(\alpha^2 d/2))$，指数级于维度。这解释了为何 Transformer 能以有限参数存储和检索海量知识。\par\medskip
\textbf{本节要拿走的一件事}：softmax attention 是现代 Hopfield 网络一步不动点迭代的精确数学等价——「注意力」的本质是指数容量关联记忆的检索，而非心理学意义上的资源竞争。
\end{sectionintuition}
```

**直觉**

「注意力」这个名字来自认知科学：人类的注意力是有限资源，面对复杂场景时，大脑选择性地将处理能力集中到某些刺激上，忽略其他。这是一种**竞争分配机制**。然而，softmax attention 在数学上并不对应这种语义——它更像是一种**内容寻址的关联记忆检索**：给定一个查询模式，在一个存储了大量模式的系统中，找到最接近的内容并加权汇总。这两种数学结构有本质区别，而 Hopfield 网络理论提供了精确的数学桥梁，揭示了注意力的「真实身份」。

本节出现五个核心概念：**关联记忆与内容寻址、经典 Hopfield 网络的二次能量函数与线性容量、能量函数的 log-sum-exp 改造、现代 Hopfield 网络的不动点等价于 softmax attention、指数容量的跃升**。下面把这条链铺开。

**第一环：关联记忆与内容寻址——比「注意力」更准确的比喻**

关联记忆（Associative Memory）是一种存储和检索模式的系统：存储阶段将若干「记忆模式」$\xi^1, \xi^2, \ldots, \xi^N$ 编码进系统权重；检索阶段给定一个「查询模式」$\xi$（可能不完整或有噪声），系统通过迭代更新收敛到最近的存储模式。这是一种**内容寻址**（content addressing）：不用地址索引，而用内容相似度找到匹配项。

这个框架与注意力机制的结构高度吻合：$K$ 的行向量就是存储模式，$Q_i$ 就是查询模式，注意力计算就是在存储模式集合中按相似度加权检索。关键问题是：这种关联记忆系统的存储容量是多少？——即同时存储多少个不冲突的模式，使得查询时能精确检索到目标？

**第二环：经典 Hopfield 网络——线性容量的限制**

经典 Hopfield 网络（1982，John Hopfield）用一个二次能量函数存储 $N$ 个 $d$ 维二值模式 $\xi^\mu \in \{-1, +1\}^d$：

$$E_{\text{classic}} = -\frac{1}{2} \xi^T W \xi, \quad W = X^T X - NI$$

其中 $X \in \mathbb{R}^{N \times d}$ 是存储模式矩阵，$I$ 是单位矩阵。检索时从查询 $\xi$ 出发，按能量梯度下降更新：$\xi_i^{\text{new}} = \mathrm{sgn}((W\xi)_i)$，迭代直到收敛到能量局部极小（即某个存储模式）。

经典 Hopfield 网络的存储容量约为 $0.14d$（随机模式情形下），即最多约能可靠存储 $0.14d$ 个 $d$ 维模式。更多模式时，不同模式的权重矩阵会产生「串扰」（crosstalk），检索会陷入错误的吸引子（混淆了多个模式的叠加态）。这个线性容量是二次能量函数的根本限制：能量极小值的数量由矩阵 $W$ 的谱结构决定，而对称矩阵的「局部极小」数量在高维空间中是有界的。

**第三环：能量函数的 log-sum-exp 改造——从二次到指数**

2020 年，Ramsauer 等（ICLR 2021）提出现代 Hopfield 网络，把能量函数改为 log-sum-exp 形式：

$$E = -\,\mathrm{lse}(\beta,\, X\xi) + \frac{1}{2}\xi^T\xi + \frac{1}{2\beta}\log N + C$$

其中 $\mathrm{lse}(\beta, z) = \beta^{-1} \log \sum_i \exp(\beta z_i)$ 是 log-sum-exp 算子，$\beta > 0$ 是逆温度参数（类比物理中的温度倒数），$C$ 是与 $\xi$ 无关的常数。

这个改造的直觉是什么？经典的二次能量 $-\xi^T W \xi = -\xi^T X^T X \xi = -\|X\xi\|^2$ 在极小值处对应向量 $X\xi$ 的 $\ell^2$ 范数最大，等价于找最近似的存储模式（线性近似）。log-sum-exp 能量则是 $\ell^2$ 范数的「指数化版本」——它在与某个存储模式几乎完全对齐时，$\exp(\beta z_i)$ 对应项会以指数速率主导，形成一个极其尖锐的吸引子。尖锐的吸引子意味着不同存储模式之间的「分隔」更清晰，更多模式可以共存而不串扰——这是容量从线性跃升至指数的数学根源。

**第四环：不动点条件推导——softmax attention 的精确等价**

对现代 Hopfield 网络的能量函数，求不动点条件（$\partial E / \partial \xi = 0$）：

$$\frac{\partial E}{\partial \xi} = -X^T \mathrm{softmax}(\beta X\xi) + \xi = 0$$

（这里用到了 $\frac{\partial}{\partial \xi}\mathrm{lse}(\beta, X\xi) = X^T \mathrm{softmax}(\beta X\xi)$，以及 $\frac{\partial}{\partial \xi}\frac{1}{2}\xi^T\xi = \xi$。）

不动点条件 $\xi^* = X^T \mathrm{softmax}(\beta X \xi^*)$ 给出了一个不动点迭代更新规则：

$$\xi^{\text{new}} = X^T \,\mathrm{softmax}(\beta X \xi)$$

现在令 $\xi \leftarrow Q_i$（某个查询向量），$X \leftarrow K$（键矩阵的行向量集合），$\beta = 1/\sqrt{d_k}$，则：

$$\xi^{\text{new}} = K^T \,\mathrm{softmax}\!\left(\frac{K Q_i}{\sqrt{d_k}}\right) = K^T \,\mathrm{softmax}\!\left(\frac{Q_i \cdot k_j}{\sqrt{d_k}}\right)_j$$

这精确等价于对查询 $Q_i$ 的单头注意力输出（在 Key = Value 的特殊情形下）！softmax attention 是现代 Hopfield 网络的**一步不动点迭代**——每次前向传播相当于从查询向量出发，做一步朝不动点的跳跃，目标是收敛到「键向量空间中最像查询的那个吸引子」。

**第五环：指数容量——高维空间的记忆奇迹**

现代 Hopfield 网络的存储容量是多少？Ramsauer 等证明，对充分分离的存储模式（模式间 $\ell^2$ 距离至少为 $\Delta$），现代版本能可靠存储的模式数约为：

$$C_{\text{modern}} = O\!\left(\exp\!\left(\frac{\alpha^2 d}{2}\right)\right)$$

其中 $\alpha$ 与模式分离度 $\Delta$ 和维度 $d$ 的比值相关。这是关于 $d$ 的指数函数，而经典版本的容量只有 $O(d)$。当 $d = 128$（典型的 key 维度），假设 $\alpha = 1/\sqrt{d}$（即模式分离度为 1），容量约为 $e^{64} \approx 6 \times 10^{27}$。当然，这是理论上界，实际有效容量取决于模式的相关性和训练动态，但即使打个极大的折扣，仍然远超线性计数的直觉。

这个指数容量有一个深刻的工程含义：**Transformer 模型的「知识存储」能力随维度指数增长**。参数矩阵 $W^K$ 的每一行是一个关联记忆的存储模式，推理时每次注意力计算是一次高效的模式检索。这解释了为什么更大的模型（更大的 $d$）不仅是「更多参数」——它们在关联记忆的意义上具有指数倍更强的知识存储能力，而不只是线性倍。

**整条链的回头看**

```
认知科学的「注意力」：有限资源竞争分配
  ↓ 数学上不准确；更准确的是：
关联记忆：查询模式 → 在存储模式中内容寻址检索
  ↓ 经典 Hopfield 网络（1982）
二次能量函数 E = -½ξ^T Wξ
  ↓ 容量 ≈ 0.14d，线性于维度
改造：E = -lse(β, Xξ) + ½||ξ||²
  ↓ 不动点条件 ∂E/∂ξ = 0
ξ_new = X^T softmax(βXξ)
  ↓ 令 ξ=Q_i, X=K, β=1/√d_k
精确等价于单头 softmax attention ✓
  ↓ 能量函数指数化 → 吸引子更尖锐
容量从 O(d) 跃升至 O(exp(α²d/2))
```

这条链揭示了一个深层事实：**softmax attention 不是一个新颖的神经网络构件，而是一种有 40 年历史的关联记忆系统的现代重新发现**。经典 Hopfield 网络的能量函数改造——从二次到 log-sum-exp——恰好等价于把 $\mathrm{sgn}$ 激活换成 softmax，把离散二值记忆换成连续实数记忆，而这个改造让容量从线性跃升至指数。理解这条链，就理解了为什么 Transformer 能以有限参数「记住」如此海量的世界知识。

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

（$\alpha$ 为模式间最小分离距离的某个函数），即指数级于维度。这解释了为何大型 Transformer 能以有限参数量存储和检索海量知识模式：权重矩阵 $W^K$ 的行向量构成一个指数容量的关联记忆索引，推理时的「理解」本质是高维模式检索的级联。

**与原书呼应**：原书 §3.6 引用了 Ramsauer 等的等价性证明，对应本节能量函数推导与不动点更新规则的显式对应。

---

## §3.7 一个被忽略的争论：注意力权重 ≠ 解释

```{=latex}
\begin{sectionintuition}
\textbf{为什么不能用注意力权重解释模型}？因为注意力权重 $\alpha_{ij}$ 只是计算图中的一个中间量——它衡量的是 Value 向量的线性组合系数，而非输入 token 对最终预测的「因果贡献」。\par\medskip
\textbf{第一步：权重衡量的是什么}。$\alpha_j$ 告诉你「第 $j$ 个 Value 向量被赋予了多少权重」，但 Value 向量本身已经经过线性投影，与原始 token $x_j$ 不是同一件事。\par\medskip
\textbf{第二步：梯度归因走的是不同路径}。$x_j$ 对输出 $y$ 的梯度，不仅经过 Value 分支，还通过 $x_j$ 对 $q$、$k_j$、甚至其他位置 $k_l$ 的影响传播，路径复杂得多。\par\medskip
\textbf{第三步：Jain \& Wallace 的反例}。保持输出 $y$ 近似不变的前提下，可以构造与原始分布 $\ell^1$ 距离很大的替代注意力分布 $\alpha'$——当 Value 矩阵行向量相关性高时，不同权重混合可得到相似输出。\par\medskip
\textbf{本节要拿走的一件事}：$\alpha_j$ 大不等于「第 $j$ 个 token 对预测起决定性作用」；真正的因果归因必须通过梯度方法或激活路径分析（circuit analysis）进行。
\end{sectionintuition}
```

**直觉**

注意力权重可视化是深度学习可解释性研究中最流行的工具之一：展示一张彩色热力图，颜色越深代表模型「越关注」那个词。这种直觉直接来自名字——「注意力」——和视觉上的说服力。但它在数学上是错误的。正确地理解「注意力权重到底衡量了什么、它与解释之间差了什么」，需要追踪计算图中信息流的完整路径。

本节出现四个核心概念：**注意力权重是 Value 路径的系数、梯度归因的多路径性、Jain & Wallace 的反例（Value 矩阵相关性导致注意力可被替换）、解释性研究的边界**。下面把这条链铺开。

**第一环：注意力权重衡量的是什么——Value 路径的系数，而非因果贡献**

注意力的输出是：

$$y = \sum_{j=1}^n \alpha_j v_j = \sum_{j=1}^n \alpha_j (x_j W^V)$$

$\alpha_j$ 是 $v_j$（Value 向量）在线性组合中的权重。但 $v_j = x_j W^V$ 已经是原始输入 $x_j$ 经过线性变换后的结果，它与 $x_j$ 的关系取决于矩阵 $W^V$ 的方向——$W^V$ 可能把 $x_j$ 的某些语义维度完全压缩，突出另一些。因此，$\alpha_j$ 衡量的是：「从位置 $j$ 的 Value 特征空间中，抽取了多少比例的内容」，而不是「位置 $j$ 的原始输入对最终预测有多重要」。

用一个比喻：假设你在读一篇文章，注意力权重是「你目光停留在每行文字上的时间比例」，Value 投影是「每行文字被翻译成另一种语言后的内容」。目光停留时间长不等于那行原文对你理解文章的贡献大——因为翻译可能已经改变了含义，而且你的理解是由翻译后的内容决定的。

**第二环：梯度归因的多路径性——$x_j$ 影响输出的三条路**

真正衡量「$x_j$ 对输出 $y$ 有多重要」的量是梯度 $\partial y / \partial x_j$（或 $\partial \mathcal{L} / \partial x_j$）。计算这个梯度需要追踪所有 $x_j$ 影响 $y$ 的路径：

**路径一（Value 路径）**：$x_j \to v_j = x_j W^V \to y = \sum_l \alpha_l v_l$，梯度为 $\alpha_j W^V$。这是注意力权重 $\alpha_j$ 反映的路径——$\alpha_j$ 大意味着这条路径贡献多。

**路径二（Key 路径）**：$x_j \to k_j = x_j W^K \to \alpha_{ij} = \mathrm{softmax}_j(q_i \cdot k_j / \sqrt{d_k}) \to y$。$x_j$ 通过改变 $k_j$ 来影响位置 $i$ 对位置 $j$ 的注意力权重，进而影响 $y$。这条路径的梯度与 $\alpha_j$ 无关（$\alpha_j$ 只反映最终权重，不反映 $k_j$ 的影响强度）。

**路径三（Query 路径的交叉影响）**：如果 $x_j$ 也参与了某个 Query（在自注意力中，$q_l = x_l W^Q$，$l = j$ 时 $x_j$ 直接贡献到 $q_j$），则 $x_j$ 通过 $q_j$ 影响位置 $j$ 对其他所有位置的注意力权重，这条路径更为复杂。

三条路径的贡献相加才是 $\partial y / \partial x_j$ 的完整计算。注意力权重 $\alpha_j$ 只反映了路径一的强度，完全忽略了路径二和路径三。在实践中，路径二（Key 路径）往往与路径一同等重要甚至更重要——因为 $x_j$ 不仅要「贡献内容」，还要「控制谁关注它」，后者对整个输出矩阵的影响是全局性的。

**第三环：Jain & Wallace 的反例——注意力分布对输出不具唯一决定性**

Jain 和 Wallace（NAACL 2019）提出了一个更强的否定结论：不仅梯度归因与注意力权重不一致，甚至注意力分布本身对输出都不具有唯一决定性。

他们的核心观察是：当 Value 矩阵 $V \in \mathbb{R}^{n \times d_v}$ 的行向量之间高度相关时（实践中这很常见，因为 $V = XW^V$ 中的 $X$ 本身是相关的 token 嵌入），不同的权重向量 $\alpha$ 可以产生几乎相同的输出向量 $y = \alpha^T V$。形式化地，如果 $V$ 的行向量几乎在同一个低维子空间内（即 $V$ 是低秩的），则对任意 $\alpha$，$\alpha^T V$ 都落在这个低维子空间内；在这个子空间内，不同的 $\alpha$ 之间的差异被投影到一个小的范围。这意味着：可以构造一个与原始 $\alpha$ 在 $\ell^1$ 距离上很大的替代权重 $\alpha'$，使得 $\|\alpha'^T V - \alpha^T V\|$ 很小。

从模型可解释性的角度，这意味着：即使把注意力权重换成完全不同的分布，输出（以及下游的预测）可以几乎保持不变。「高度关注词 $j$」不能推断「词 $j$ 对输出至关重要」——因为模型可以通过调整注意力权重（在 Value 矩阵的零空间方向上），在不改变输出的情况下把注意力随意重新分配。

**第四环：解释性的边界——注意力告诉你什么、告诉不了你什么**

理解了上面三环，可以精确地划定注意力权重在可解释性研究中的适用边界：

**可以告诉你的**：哪些 token 的 Value 特征对当前位置的输出有直接的线性贡献（路径一的权重）。这对「模型关注了哪些语义维度的内容」是有信息量的，但这个信息量是局部的、路径特定的。

**告诉不了你的**：（1）哪个输入 token 对最终预测最重要（需要完整的梯度归因）；（2）修改哪个输入会对预测产生最大影响（因果干预，需要 causal tracing 等方法）；（3）模型「为什么」做出某个预测（这需要追踪完整的计算电路，而不只是一层的注意力权重）。

正确的工具选择：若目的是「理解哪个输入特征影响了预测」，应使用 Integrated Gradients、LIME、SHAP 等梯度或扰动方法；若目的是「理解模型内部的计算结构」，应使用激活修补（activation patching）和电路分析（circuit analysis）方法，如 Anthropic 的机械可解释性（mechanistic interpretability）工作中识别出的 induction head 电路。注意力权重可以作为辅助参考，但不能单独用于解释。

**整条链的回头看**

```
注意力权重 α_j
  ↓ 衡量的是：v_j = x_j W^V 在输出线性组合中的比例
α_j ≠ x_j 的重要性（Value 路径只是三条路径之一）
  ↓ 完整梯度 ∂y/∂x_j 需要累加
路径一（Value）+ 路径二（Key）+ 路径三（Query 交叉）
  ↓ α_j 只反映路径一
Jain & Wallace 反例：V 低秩 → 不同 α 可得相同输出
  ↓ 注意力分布对输出不具唯一决定性
「α_j 大」⇏「x_j 对预测重要」
  ↓ 真正的因果归因需要
梯度方法 / 激活修补 / 电路分析
```

这条链揭示了一个深层事实：**注意力权重是计算图中的一个中间结果，它描述的是信息流在一条特定路径上的分配比例，而不是输入的重要性**。把注意力权重当作解释，相当于用「货物在某一条路上的运输量」来衡量「起始城市对目的地的经济影响」——忽略了其他所有路径和网络效应。可解释性研究的真正困难，正是在于 Transformer 的信息流是多路径的、非线性的、层次叠加的，没有任何单一的简单量可以直接等价于「重要性」。

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

对足够小的 $\delta$ 和足够大的 $\epsilon$，这说明注意力分布本身对输出不具有唯一决定性——当 Value 矩阵 $V$ 的行向量之间相关性高时，不同的权重混合可以得到接近相同的输出向量。因此，$\alpha_j$ 大不能直接推断「第 $j$ 个 token 对预测起决定性作用」；真正的因果归因必须通过梯度方法（gradient attribution）或激活路径分析（如 circuit analysis）来进行。

**与原书呼应**：原书 §3.7 引用了 Jain & Wallace 的核心发现及 induction head 电路分析，对应本节注意力权重与梯度归因不等价的形式化说明。

---

## §3.8 跨界映射：宏观工业供应链

```{=latex}
\begin{sectionintuition}
\textbf{为什么用供应链类比 attention}？注意力机制的「匹配-归一化-加权汇总」流程，在逻辑结构上与全球供应链的「需求发布-能力匹配-资源调度」完全对应，能把抽象矩阵运算还原为可感知的经济直觉。\par\medskip
\textbf{第一步：Query 是需求方，Key 是供应商标签，Value 是实际物料}。每个生产环节（token）发出需求向量 $Q$，每个供应商（token）公布能力标签 $K$，softmax 将匹配分数归一化为调度权重，最终按权重汇总实际物料 $V$。\par\medskip
\textbf{第二步：多头 = 多条并行供应链}。每个注意力头按不同逻辑（语法、语义、话题）独立调度，就像多条并行运作的专业供应链，最后在「总装车间」（输出投影 $W^O$）合并。\par\medskip
\textbf{第三步：这个类比的价值是量级直觉}。当序列长度 $n = 8192$，$h = 32$ 个头，每步注意力相当于 32 条供应链各自对 8192 个供应商做一轮全局询价——这是人类供应链系统完全无法实现的规模。\par\medskip
\textbf{本节要拿走的一件事}：供应链类比把 attention 从「心理学隐喻」还原为「可量化的资源匹配」，但它有根本性的局限——见 §3.9。
\end{sectionintuition}
```

**直觉**

类比是理解抽象数学的强大工具，但也是最容易固化错误直觉的工具。这一节先把供应链类比尽量用好，把它能解释的东西讲清楚；下一节（§3.9）再逐一拆解它的失效边界。

注意力机制的「匹配-归一化-加权汇总」逻辑，与全球供应链的「需求发布-能力匹配-资源调度」在结构上高度吻合。下面分析这个类比成立的三个数学根源，以及它的量级价值在哪里。

**供应链类比成立的根源一：三角色的分离**

在现实供应链中，一个供应商同时扮演三个角色：它**发布需求**（向上游采购），它**公布能力**（告诉下游客户能提供什么），它**提供实物**（真正交付的货物）。这三个角色对应注意力中的 Query（发布需求）、Key（公布能力标签）和 Value（交付的实际内容）。

现实供应链中，「能力标签」（Key）和「实际物料」（Value）是分开的：供应商公布的规格书（Key）描述了型号、参数、价格，而真正交货的零部件（Value）是物理实体，两者虽然对应，但完全不同的东西。注意力也是如此：Key 是用于匹配打分的向量，Value 是真正混合到输出里的内容，两者由不同的线性投影矩阵（$W^K$ 和 $W^V$）生成，解耦的设计原因（见 §3.2）在这里体现为「规格书与实物分开管理」的供应链逻辑。

**供应链类比成立的根源二：softmax 归一化等价于调度权重的比例分配**

在供应链调度中，当多个供应商都能满足需求时，采购部门通常不会把所有订单都给评分最高的那一个（风险集中），而是按能力匹配度**比例分配订单**：最高分供应商拿 60%，次高分拿 30%，其余分散 10%。这正是 softmax 做的事：把原始打分向量映射到概率单纯形（所有权重非负且和为 1），使得匹配度高的供应商（token）获得更大但不是唯一的份额，仍然保留了「软检索」而非「硬选择」的特性。

这个类比帮助解释为什么不用 argmax（硬性选择得分最高的一个 key 对应的 value）：硬选择不可微，无法反向传播梯度；而且在语言中，「最相关的词」不止一个，输出应该是多个相关词的加权混合。softmax 的连续性和可微性，正是「比例分配」语义的数学实现。

**供应链类比成立的根源三：多头等价于多条专业供应链的分工**

现代供应链管理中，大型制造商往往维护多条平行的供应链体系：一条专注于标准零件（低成本、大批量），一条专注于精密部件（高质量、小批量），一条专注于紧急备料（快速响应）。每条供应链有自己的评估标准（Key）和物料库（Value），最终产品是多条供应链交付结果的集成。

多头注意力完全对应这个结构：$h$ 个头各自维护独立的 $W^Q_i, W^K_i, W^V_i$，在不同的语义子空间（不同的「采购标准」）里独立做注意力，输出通过 $W^O$（「总装」）合并。不同的头会自发专门化——实验研究发现，有的头专注句法依存（语法供应链）、有的头追踪长距离指代（语义供应链）、有的头捕捉局部搭配（词汇供应链）。这种分工不是预先设计的，而是训练压力下的涌现，就像市场竞争下自然形成的专业化分工。

**供应链类比的量级价值**

这个类比最重要的作用是帮助建立规模直觉。一个真实的汽车制造商可能管理数千个供应商，但每次采购决策通常只涉及少数几家候选。注意力的不同之处：每个 token 每次前向传播都对序列中所有 $n$ 个 token 做完整评分。当 $n = 8192$（GPT-4 的上下文长度），$h = 32$ 个头，每次前向传播相当于 32 条供应链，各自对 8192 个供应商做一轮完整的全球询价。一个 Transformer 层的一次前向传播，完成的「全球询价」次数是 $n \times h = 8192 \times 32 = 262144$ 次——对于人类供应链系统，这是完全不可想象的规模；对于 GPU，这只是一次矩阵乘法。这种量级感的震撼，正是供应链类比的核心价值：把抽象的矩阵运算转换成可以用现实世界的操作规模感受的东西。

---

## §3.9 反类比：当供应链隐喻失效

```{=latex}
\begin{sectionintuition}
\textbf{为什么供应链类比有根本性局限}？供应链有两个注意力机制没有的核心特征：硬性产能约束和明确的优化目标。这两处差异揭示了 attention 作为数学对象的真实本质。\par\medskip
\textbf{第一步：Value 向量可被无限次「调用」，没有产能耗尽}。同一个 token 可以同时以最高权重响应序列中所有其他位置，不存在资源争抢。attention 调度的实质是相似度计算，而非真正的权衡分配。\par\medskip
\textbf{第二步：注意力头没有明确的语义目标}。供应链的优化目标（成本、时效）是可测量的单一指标；而单个注意力头的「功能」是反向传播在高维参数空间中塑造的中间张量，是涌现的、上下文相关的，通常无法用自然语言直接描述。\par\medskip
\textbf{第三步：attention 数学透明但语义不透明}。注意力权重可以精确计算，但它们不等于解释（见 §3.7）；attention 机制可以精确描述，但单个头的「功能」不能被先验地规定。\par\medskip
\textbf{本节要拿走的一件事}：好的类比帮助建立直觉，坏的类比固化误解——供应链隐喻的价值在于规模感知，其局限在于它用「有约束的竞争分配」来描述一个「无约束的相似度加权」，会系统性地误导对 attention 瓶颈和可解释性的判断。
\end{sectionintuition}
```

**直觉**

§3.8 展示了供应链类比在建立「规模直觉」上的价值。这一节要做相反的工作：逐一拆解类比失效的数学根源，说明把供应链思维带入注意力机制时会产生哪些系统性误判。分析类比在哪里失效，往往比分析它在哪里成立更能揭示对象的本质。

供应链类比在以下四个数学根源上失效。下面逐一深入分析。

**失效根源一：Value 向量无产能约束——softmax 归一化不是资源守恒**

供应链中，物料是守恒的：如果某个供应商把 60% 的产能给了客户 A，它只剩 40% 给客户 B。资源分配是一个零和博弈，一方多则另一方少。

注意力中，Value 向量完全没有这种约束。Position $j$ 的 Value 向量 $v_j$ 可以同时以权重 1 贡献给序列中所有 $n$ 个其他位置的输出——它不会因为「被太多人关注」而「产能耗尽」。Position $i$ 的 softmax 权重 $\alpha_{ij}$ 之和等于 1（行归一化），但这只是位置 $i$ 自己的注意力权重的约束，不是 $v_j$ 被「消耗」的约束。$v_j$ 可以同时满载地参与 $n$ 个位置的计算。

这个差异的数学含义是：softmax 的归一化确保的是「每个 Query 的关注权重之和为 1」，而不是「每个 Key/Value 的被关注量有上限」。这是一个**统计归一化**（保证权重是概率分布），不是一个**物理守恒律**（保证资源总量不变）。用供应链思维理解注意力，会系统性地高估「激烈竞争」的程度：实际上没有竞争，每个 Value 向量对所有想要它的 Query 都「来者不拒」。

更进一步：正因为没有产能约束，注意力的计算可以完全并行——不同 Query 的输出之间没有任何资源争抢关系，可以同时计算，完全独立。如果真的是供应链（有产能约束），不同客户的分配决策就会互相依赖，就无法并行。注意力的可并行性，恰恰来自它不是真正意义上的资源分配。

**失效根源二：注意力头没有先验语义目标——功能是涌现的，不是设计的**

供应链中，每条链的优化目标是外生给定的（成本、时效、质量评分），可以被明确陈述、事前规定、事后审计。「这条链负责精密部件采购」是一个可以写进合同的规定。

注意力头没有这种先验语义目标。第 $k$ 个头的投影矩阵 $W^Q_k, W^K_k, W^V_k$ 是通过反向传播在整个训练过程中共同优化的，它的「功能」完全由数据分布和损失函数塑造，是一个涌现属性，不是设计属性。

确实，机械可解释性（Mechanistic Interpretability）研究发现了一些可命名的头：「induction head」（复制前面出现过的模式）、「name mover head」（把命名实体从主语位置传递到谓语位置）等。但这些是事后发现的少数特例，不是设计目标。大多数注意力头的功能在不同上下文中会变化（上下文相关的多义性），无法用一句话描述，在不同的输入分布下甚至会改变激活模式。用供应链类比「这个头负责语法关系，那个头负责长距离指代」，会误导你认为头的职责是固定的、可预先规定的，而实际上它是流动的、上下文依赖的。

**失效根源三：attention 是线性加权，而非优化决策**

供应链调度是一个优化问题：给定约束（产能、成本），求最优解（最大化利润、最小化延误）。这是一个带目标函数的决策过程，决策者在权衡不同选项。

注意力是一个确定性的线性函数：给定 $Q, K, V$，输出 $\mathrm{softmax}(QK^T/\sqrt{d_k})V$ 没有任何优化过程，没有约束，没有「权衡」——只是一个矩阵乘法加非线性变换，结果是唯一确定的。softmax 引入的「竞争感」只是视觉上的：概率质量从低分项转移到高分项，但这只是 $\exp$ 函数的放大效应，而不是某种零和竞争博弈的均衡解。

这个失效意味着：不能用供应链的「瓶颈分析」（哪个供应商产能不足导致整体延误）来理解注意力的计算瓶颈。注意力的真正瓶颈是算术强度（§3.3）和内存带宽（§3.5），这些是纯粹的数值计算资源问题，与「谁关注谁」的分配结构完全无关。

**失效根源四：注意力权重不等于解释——供应链审计的不可移植性**

供应链的可审计性来自其设计：每个决策对应一个可追溯的人类可读的原因（这个供应商评分高是因为交期短、价格低）。这种审计是直接的、一对一的。

注意力权重看起来可以「审计」，但 §3.7 已经证明，高注意力权重 $\alpha_j$ 不等于「token $j$ 对预测起决定作用」。把供应链的审计思维移植到注意力，会导致一个系统性错误：把「高注意力权重的词」当作「模型关注的理由」，而实际上真正的因果路径需要通过梯度归因和激活修补来确定。这种误植正是 §3.7 批评的注意力权重可解释性误用的来源。

**类比失效的总结：四处失效共同揭示了 attention 的真实数学本质**

```
供应链特征                      Attention 的真实情况
──────────────────────────────────────────────────────
有产能约束（零和分配）           无产能约束（线性叠加，无守恒）
优化目标外生给定               功能由训练涌现，无先验语义
竞争调度（博弈均衡）            确定性线性变换（矩阵乘法 + softmax）
可审计（决策有因果链）          权重 ≠ 解释（因果需梯度方法）
```

供应链类比的价值在于：量级直觉（$n \times h$ 次全局询价的规模感）、三角色分离的直觉（QKV 解耦）、多条专业化子链的直觉（多头分工）。它的局限在于：一旦涉及「约束」「竞争」「目标」「可解释」这些词，类比就开始系统性地误导。好的工程师知道何时抓住类比、何时放下类比。

---
