# 算法的低语 · The Whisper of Algorithms

从概率论到复杂系统，从 Transformer 数学到 AI 基础设施周期。

## 当前版本

| 文件 | 说明 |
|---|---|
| `article.md` | 主本 Markdown 源稿，v2.3 校订版 |
| `article.pdf` | 主本原出版 PDF + 1 页 v2.3 修订附录，共 70 页 |
| `mathvol.md` | 数学卷 Markdown 源稿，v1.5 校订版 |
| `mathvol.pdf` | 数学卷原出版 PDF + 1 页 v1.5 修订附录，共 77 页 |
| `CHANGELOG.md` | 版本修改记录 |
| `REVIEW_NOTES.md` | 审校问题、已处理项与后续阻塞项 |

## 主本

主本 `article.md` / `article.pdf` 仍保持七章结构：

1. 信息论与语言的概率论本质
2. 嵌入空间与高维几何
3. Transformer 与注意力机制
4. 非凸优化与深度学习训练
5. Scaling Law、涌现与相变隐喻
6. 从 Attention 到 AI 资本周期
7. 后 Transformer 时代的架构候选

v2.3 主要校订：

- 将“涌现”“理解”“原罪”等过强表述改为带边界的论断。
- 补充第六章动态数据口径：Capex、市占率、供应链与价格曲线等数字需继续绑定财报、官方规格或具名研究报告。
- PDF 保留原出版排版，并追加一页修订附录；正文修订以 `article.md` 为准。

## 数学卷

数学卷 `mathvol.md` / `mathvol.pdf` 是主本的数学推导附册，覆盖信息论、嵌入几何、注意力矩阵、非凸优化、标度律、硬件感知算法与后 Transformer 架构数学。

v1.5 主要校订：

- 修复 `§7.9` 下小节误写为 `7.8.x` 的编号问题。
- 确认 Markdown 源稿中未残留 `(??)` 交叉引用占位。
- PDF 保留原出版排版，并追加一页修订附录；完整重排仍需补齐图片和 LaTeX 构建链。

## 可重编性说明

当前仓库已经有 Markdown 源稿，但尚未包含完整重编 PDF 所需的图片资源与构建模板。缺失资源包括：

- 主本章节图：`ch1_information.png`、`ch2_geometry.png`、`ch3_attention.png`、`ch4_optimization.png`、`ch5_emergence.png`
- 主本插图：`assets/figs/fig_main_02_01_word2vec_geom.png`、`fig_main_01_01_geometry.png`、`fig_main_03_01_attention_headhunter.png`、`fig_main_04_01_scaling_law.png`、`fig_main_05_01_grokking.png`、`fig_main_06_01_compute_stack.png`、`fig_main_07_01_arch_tree.png`
- 数学卷插图：`assets/figs/fig_math_03_01_skipgram.png`、`fig_math_05_01_transformer_block.png`、`fig_math_02_01_softmax.png`、`fig_math_03_02_attention_matmul.png`、`fig_math_05_02_rope.png`、`fig_math_06_01_kv_cache.png`、`fig_math_04_01_gradient_descent.png`、`fig_math_07_01_moe.png`

补齐上述资源和 LaTeX 模板后，应重新生成出版级 PDF，并复查目录、书签、公式交叉引用和图像引用。

## 变更记录

- **v2.3 / 数学卷 v1.5 (2026-06-11)**：源码级校订，追加 PDF 修订附录，记录缺失构建资源。
- **v2.2 / 数学卷 v1.4 (2026-06-10)**：教学级配图批量补入，主本与数学卷 PDF 发布，并上传 Markdown 源稿。
- **v2.1.2 (2026-06-09)**：数学卷术语本地化，封面版本修订。
- **v2.1 (2026-06-08)**：主本重构为七章本，数学卷拆为独立附册，新增产业链与后 Transformer 章节。
- **v1.0 (2026-06-07)**：首发。

## 作者

大队长 · USTC alumnus  
联系：<fqsx@mail.ustc.edu.cn>

## License

文本与图表 © 大队长。引用请注明出处。
