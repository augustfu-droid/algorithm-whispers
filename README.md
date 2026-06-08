# 算法的低语 · The Whisper of Algorithms

> 从概率论到复杂系统 —— 大模型背后的数学引擎
>
> **Version 2.0 · 上下两半场**
> 完稿于 2026 年 6 月 · 约 4.5 万字 · 覆盖至 2026 Q2 前沿

一本面向广义读者讲故事、一本给愿意推公式的人留住硬核——《算法的低语》正式进入**双册时代**。

---

## 📘 主本 [`article.pdf`](./article.pdf) — 66 页

**导言 + 7 章 + 尾声 + 变更日志**

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

### 第六章亮点（约 1.22 万字 · 39 引用 · 4 表）
从算术强度 `I = 2mnk / (mk+kn+mn)` 推到 HBM 内存带宽墙 → CoWoS 封装瓶颈 → 中际旭创 1.6T 光模块 50%+ 市占 → Hyperscaler Q2 FY2026 单季 capex $37.5B → GB200 NVL72 整柜 120kW 液冷拐点 → token 价格 3 年降 1000×。

### 第七章亮点（约 8000 字）
从 Duman Keles 的 SETH 下界证明出发，梳理 HiPPO → S4 → Mamba → Mamba-2/SSD → RetNet/RWKV-7/GLA → Jamba 1.5/Zamba2/Samba 混合架构 → 长上下文工程派 (GQA/YaRN/Ring Attention) → MDLM/LLaDA/Mercury 扩散语言模型 → Genie 3/V-JEPA 2 世界模型。

---

## 📐 数学卷 [`mathvol.pdf`](./mathvol.pdf) — 42 页

独立封面 + 5 章严格推导：

1. 概率论与信息论基础
2. 线性代数与梯度
3. Transformer 完整推导
4. 最优化算法（含 Adam Sharpness 不等式）
5. Scaling Law 与涌现

主本读故事,数学卷查公式,两本互不干扰。

---

## 变更日志

- **v2.0 (2026-06-08)** — 拆分双册:新增第六章产业链、新增第七章后 Transformer、尾声重写为 7 条洞见、数学卷独立出册
- **v1.2 (2026-06-08)** — 合刊版,数学卷作为附录
- **v1.1 (2026-06-08)** — Round-3 事实核查 + 措辞微调
- **v1.0 (2026-06-08)** — 首版发布

---

## 作者

**大队长 · 付强** · USTC alumnus
联系: <fqsx@mail.ustc.edu.cn>

---

## License

文本与图表 © 大队长,保留所有权利。引用请注明出处。
