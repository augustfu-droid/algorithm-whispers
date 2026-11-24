[![YouTube 中文教学频道](https://img.shields.io/badge/YouTube-%E5%A2%A8%E7%82%B9%E6%88%90%E6%98%9F%C2%B7%E7%A7%91%E5%AD%A6%E4%B8%8EAI%20%28%40InkStarSciAICN%29-C8483B?style=for-the-badge&logo=youtube)](https://www.youtube.com/@InkStarSciAICN?sub_confirmation=1)
[![YouTube English Science & AI](https://img.shields.io/badge/YouTube-InkStar%20Science%20%26%20AI%20%28%40InkStarSciAI%29-FF0000?style=for-the-badge&logo=youtube)](https://www.youtube.com/@InkStarSciAI?sub_confirmation=1)
[![ML 108 中文播放列表](https://img.shields.io/badge/Playlist-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%8E%E6%95%B0%E5%AD%A6%E6%8E%A8%E5%AF%BC%20108%20%E8%AE%B2-2B5B84?style=for-the-badge&logo=youtube)](https://www.youtube.com/playlist?list=PLB52YSkb7icg)
[![ML 108 Open Book](https://img.shields.io/badge/Open%20Book-%E5%A2%A8%E7%82%B9%E6%88%90%E6%98%9F%EF%BC%9A%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%20108%20%E8%AE%B2-E3A857?style=for-the-badge&logo=github)](https://github.com/augustfu-droid/ml108)

> [!TIP]
> **🎬 配套 4K 数学推导与 AI 原理官方视频频道（Official Video Series）**
>
> - 🇨🇳 **中文科学/AI 教学专号**：[**墨点成星·科学与AI (`@InkStarSciAICN`)**](https://www.youtube.com/@InkStarSciAICN)（[一键订阅](https://www.youtube.com/@InkStarSciAICN?sub_confirmation=1)）— 大模型数学引擎、纸笔手算推导与前沿 AI 原理新课首发阵地
> - 🌐 **English Science & AI Channel**：[**InkStar Science & AI (`@InkStarSciAI`)**](https://www.youtube.com/@InkStarSciAI)（[Subscribe](https://www.youtube.com/@InkStarSciAI?sub_confirmation=1)）— First-principles mathematics, Transformer mechanics & frontier AI derivations
> - 📺 **已上线配套系列播放列表**：[**▶ 中文《机器学习与量化理工 108 讲》(`PLB52YSkb7icg`)**](https://www.youtube.com/playlist?list=PLB52YSkb7icg) ｜ [**▶ English *108 Lectures on Machine Learning* (`PLQ8UeKfoqQSw`)**](https://www.youtube.com/playlist?list=PLQ8UeKfoqQSw)
> - 📖 **姊妹开源项目**：[《墨点成星：机器学习 108 讲》(`ml108`)](https://github.com/augustfu-droid/ml108) ｜ [《代码驱动的金融数学》(`finance-math-for-geeks`)](https://github.com/augustfu-droid/finance-math-for-geeks) ｜ [《2026 AI 周期研究》(`ai-bubble-cycle-research`)](https://github.com/augustfu-droid/ai-bubble-cycle-research)

---

# 算法的低语 · The Whisper of Algorithms

从概率论到复杂系统，从 Transformer 数学到 AI 基础设施周期。

## 当前版本

**2026-11-17 增订版：数学卷 v1.8 / 习题伴读 v1.3。** 保留基础篇，新增第 8、9 章与 36 道完整解析习题。核对 2026 年 6 月至 11 月的六篇研究，区分论文结果、经典工具与自拟教学模型。

| 新版文件 | 说明 |
|---|---|
| [数学卷 PDF](output/pdf/算法的低语_数学卷_v1.8.pdf) | 245 页，九章完整重排，含更新导读与原始文献链接 |
| [习题伴读 PDF](output/pdf/算法的低语_数学卷习题伴读_v1.3_全册.pdf) | 387 页，基础全册及新增 36 题，逐题完整解答 |
| [数学卷源稿](算法的低语_数学卷_v1.8.md) | 基于 `_math_full_v1.7.md` 整合，保留旧版文件 |
| [习题源稿](算法的低语_数学卷习题伴读_v1.3_全册.md) | 基础篇已含旧 95 题补编，不重复拼接 |
| [更新证据](research/2026-11-17_evidence.md) | 日期、版本、原始链接与适用边界 |
| [构建与校验说明](revisions/2026-11/BUILD.md) | 可重复生成与验证 |

新增内容覆盖推理预算、选择器误差、策略梯度、KL 与组内基线、课程分解、证明搜索、稀疏注意力修补、状态压缩和残差扩散。勘误范围与验证边界见新版导读及 CHANGELOG。

## 历史版本

| 文件 | 说明 |
|---|---|
| `article.md` | 主本 Markdown 源稿，v2.4.1（主要改动：第六章过渡引言重写为「上下半场」版） |
| `article.pdf` | 主本完整重排 PDF，共 70 页（v2.4.1） |
| `mathvol.md` | 数学卷 Markdown 源稿，v1.6（在 v1.5 编号修复基础上追加工程直觉引子与排版统一） |
| `mathvol.pdf` | 数学卷完整重排 PDF，共 78 页（v1.6） |
| `CHANGELOG.md` | 版本修改记录 |
| `REVIEW_NOTES.md` | 审校问题、已处理项与后续阻塞项 |

## 主本

主本 `article.md` / `article.pdf` 保持七章结构：

1. 信息论与语言的概率论本质
2. 嵌入空间与高维几何
3. Transformer 与注意力机制
4. 非凸优化与深度学习训练
5. Scaling Law、涌现与相变隐喻
6. 从 Attention 到 AI 资本周期
7. 后 Transformer 时代的架构候选

### v2.4.1 在 v2.4 基础上的调整

- **第六章过渡引言重写**：以「从双对数直线跌进重工业齿轮」取代 v2.4 的「当指数遇见指数」，强化「上下半场」叙事张力、Attention \(O(n^2)\)复杂度墙与 HBM/CoWoS/光模块/GW 集群的物质映射。

### v2.4 在 v2.3 基础上的追加
- **九段真实历史引文题记**：导言（Clarke 1962）、Ch1（Shannon 1948）、Ch2（Diaconis & Freedman）、Ch3（Vaswani 2017）、Ch4（Dauphin 2014）、Ch5（P. W. Anderson 1972）、Ch6（Jensen Huang GTC 2024）、Ch7（Albert Gu S4 2022）、尾声（McLuhan 1964）。

### v2.3 已完成的表达校订（保留）

- 将“涌现”“理解”“原罪”等过强表述改为带边界的论断。
- 补充第六章动态数据口径：Capex、市占率、供应链与价格曲线等数字需继续绑定财报、官方规格或具名研究报告。

## 数学卷

数学卷 `mathvol.md` / `mathvol.pdf` 是主本的数学推导附册，覆盖信息论、嵌入几何、注意力矩阵、非凸优化、标度律、硬件感知算法与后 Transformer 架构数学。

### v1.6 在 v1.5 基础上的追加

- **§7.1 新增「工程直觉引子」**：从 EWMA / 滑动窗口 + 缓存状态逐步过渡到连续 ODE，为后续 HiPPO / S4 / Mamba 数学搭建直觉地基。
- **全卷推开「粗体关键转折 + 摘要框」**：§6.2 Hong-Kung 证明、§6.3 FlashAttention 在线 softmax、§4.7 μP 缩放律、§7.2 HiPPO 推导链、§7.3 S4 卷积消元等关键推导节点。
- **目录修复**：§6.1.1 / §6.2 标题误装正文已拆出独立行；§7.8 子子节移除 § 前缀。

### v1.5 已完成的工作（保留）

- 修复 §7.9 下小节误写为 7.8.x 的编号问题。
- 确认 Markdown 源稿中未残留 `(??)` 交叉引用占位。

## 变更记录

- **v2.4.1 (2026-06-12 晚)**：主本第六章过渡引言重写。
- **v2.4 / 数学卷 v1.6 (2026-06-12)**：叙事打磨 + 真实题记 + 全卷排版统一。基于 v2.3 / v1.5 的表达校订线，追加 6 月 12 日批次的叙事与排版改动；完整重排 PDF。
- **v2.3 / 数学卷 v1.5 (2026-06-11)**：源码级表达校订，追加 PDF 修订附录，记录缺失构建资源。
- **v2.2 / 数学卷 v1.4 (2026-06-10)**：教学级配图批量补入，主本与数学卷 PDF 发布，并上传 Markdown 源稿。
- **v2.1.2 (2026-06-09)**：数学卷术语本地化，封面版本修订。
- **v2.1 (2026-06-08)**：主本重构为七章本，数学卷拆为独立附册，新增产业链与后 Transformer 章节。
- **v1.0 (2026-06-07)**：首发。

## 作者

大队长（墨点成星 / InkStar Science & AI）  
交流与勘误：欢迎通过本仓库 [GitHub Issues](https://github.com/augustfu-droid/algorithm-whispers/issues) 提交讨论与反馈。

## License

文本与图表 © 大队长。引用请注明出处。

---

## 技术科普与合规免责声明

> **【技术科普与免责声明】**：本仓库书稿、数学推导及配套视频资料系作者利用业余非工作时间整理的算法原理、应用数学与开源算力工程个人技术学习笔记。全部工程参数与学术资料均 100% 源自公网公开技术文献与开源行业报告，不涉及任何未公开信息或职务信息。本仓库旨在客观探讨人工智能与计算机科学底层数学机理，纯属个人业余理工科技术交流，与任何金融机构、企事业单位、雇主或组织无关，亦不代表任何机构立场；不涉及任何中国境内（A股）上市公司、港股标的或境内基金产品，在任何情况下均不构成任何投资、商业采购、工程施工或证券/基金交易决策建议。读者据此进行的任何判断或操作，风险完全自担。
