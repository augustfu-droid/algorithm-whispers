# 算法的低语 · The Whisper of Algorithms

> 从概率论到复杂系统 —— 大模型背后的数学引擎
>
> **Version 2.2 · 教学级配图版**
> 完稿于 2026 年 6 月 · 约 4.5 万字 · 覆盖至 2026 Q2 前沿

一本面向广义读者讲故事、一本给愿意推公式的人留住硬核——《算法的低语》进入**教学级配图双册版**:主本 7 张 3B1B 风格图、数学卷 8 张推导插图,全部嵌入正文,直接陪你看懂。

---

## 📘 主本 [`article.pdf`](./article.pdf) — 69 页 (v2.2)

**导言 + 7 章 + 尾声 + 变更日志**,新增 7 张教学线稿图。

| 章节 | 标题 | 关键词 |
|---|---|---|
| 导言 | 为什么要理解大模型 | 主流研究界的共识与分歧 |
| 第一章 | 从 N-Gram 到神经语言模型 | 概率链、共现、词向量 |
| 第二章 | Transformer 与注意力机制 | QKV、自注意力、位置编码 |
| 第三章 | Scaling Law 与涌现 | Chinchilla、Epoch AI、相变 |
| 第四章 | RLHF 与对齐 | 奖励模型、PPO、宪法 AI |
| 第五章 | 推理时代与思维链 | o1、链式推理、自洽性 |
| **第六章** | **数学为何变成万亿美元——从 Attention 到 AI 资本周期** | HBM、CoWoS、光模块、Hyperscaler capex |
| **第七章** | **后 Transformer 时代——SSM、线性注意力与下一代架构** | Mamba、SSD、Jamba、扩散语言模型、世界模型 |
| 尾声 | 硅基演化的下一幕 | 7 条洞见 |

### 主本配图(v2.2 新增 · 3B1B 教学线稿风格)

1. 高维几何与距离的诅咒
2. Word2Vec 词向量加减法
3. Attention QKV 三件套
4. Scaling Law 幂律曲线
5. Grokking 顿悟相变
6. NVIDIA 三层护城河
7. 后变换器时代候选谱系

### 第六章亮点(约 1.22 万字 · 39 引用 · 4 表)
从算术强度 `I = 2mnk / (mk+kn+mn)` 推到 HBM 内存带宽墙 → CoWoS 封装瓶颈 → 中际旭创 1.6T 光模块 50%+ 市占 → Hyperscaler Q2 FY2026 单季 capex $37.5B → GB200 NVL72 整柜 120kW 液冷拐点 → token 价格 3 年降 1000×。

### 第七章亮点(约 8000 字)
从 Duman Keles 的 SETH 下界证明出发,梳理 HiPPO → S4 → Mamba → Mamba-2/SSD → RetNet/RWKV-7/GLA → Jamba 1.5/Zamba2/Samba 混合架构 → 长上下文工程派 (GQA/YaRN/Ring Attention) → MDLM/LLaDA/Mercury 扩散语言模型 → Genie 3/V-JEPA 2 世界模型。

---

## 📐 数学卷 [`mathvol.pdf`](./mathvol.pdf) — 76 页 (v1.4 · 7 章镜像主本 · 8 张推导插图)

独立封面 + 7 章严格推导(与主本逐章镜像):

1. 信息论与语言的概率结构
2. 嵌入空间与高维几何
3. 注意力机制的矩阵微分
4. 非凸优化与超参数标度律(含 Adam Sharpness、μP 推导)
5. 标度律、相变与涌现的数理基础
6. **硬件感知算法的数学**——Roofline、I/O 复杂度与 FlashAttention 推导(Hong-Kung 1981 下界、在线 softmax 递推、RingAttention 通信复杂度)
7. **后 Transformer 时代的数学**——状态空间模型、线性注意力、扩散语言模型与世界模型(HiPPO LegS 矩阵推导、S4 ZOH 离散化、Mamba 并行扫描关联律、GLA/SSD 统一视角、DDPM/MDLM;§7.8 Rao-Ballard 预测编码 / Friston 变分自由能 / JEPA 能量函数与 VICReg 防坍缩 / Kalman 滤波 · DreamerV3 RSSM)

### 数学卷配图(v1.4 新增)
余弦相似度几何直观 · softmax 温度对照 · 因果掩码 · Chinchilla 配比 · loss 幂律 · KV cache 增长 · FlashAttention I/O · 状态空间扫描。

主本读故事,数学卷查公式,两本互不干扰。

---

## 变更日志

按「同一天改动合为一个版本」原则归并。

- **v2.2 / 数学卷 v1.4 (2026-06-10)** — 教学级配图批量补入(3B1B 风格)
  - 主本 7 张线稿图嵌入正文(高维几何、Word2Vec、Attention、Scaling Law、Grokking、NVIDIA 护城河、后变换器)
  - 数学卷 8 张推导插图嵌入(余弦相似度、softmax 温度、因果掩码、Chinchilla 配比、loss 幂律、KV cache、FlashAttention I/O、状态空间扫描)
  - LaTeX preamble 加 `\usepackage{float}` 修复浮位
  - 改动记录重整为「同一天一个版本」结构
  - 主本封面 v2.0 → v2.2;数学卷封面 v1.3 → v1.4

- **v2.1.2 (2026-06-09)** — 数学卷术语本地化:「中文(English)」样式统一;数学卷封面 v1.2 → v1.3。

- **v2.1 (2026-06-08)** — 主本重构为七章本·数学卷拆为独立附册·§7.8 世界模型补全
  - 合并原 v2.0 / v2.0.1 / v2.1 / v2.1.1 — 同一日全部变动
  - 数学卷扩为 7 章与主本镜像;新增主本第六章产业链、第七章后 Transformer
  - 尾声重写为 7 条洞见;数学卷独立出册
  - 数学卷新增 §7.8《世界模型与预测编码的数学》
  - 导言与主本第五章末同步为「上下两半场」架构
  - 主本封面 v1.x → v2.0;数学卷封面 v1.0 → v1.2

- **v1.0 (2026-06-07)** — 首发·三轮事实校验后定稿
  - 合并原 v1.0 / v1.1 — 同日首发与校验修订

---

## 作者

**大队长** · USTC alumnus
联系: <fqsx@mail.ustc.edu.cn>

---

## License

文本与图表 © 大队长,保留所有权利。引用请注明出处。
