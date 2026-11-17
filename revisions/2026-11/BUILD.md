# 2026 年 11 月版构建与验证

原始基线提交为 `c70daaf9d0e7b5e0f732c3e82e3cb788e5ecbbea`。该提交日期为 2026-10-23，更新 README；数学卷内容的基线来自 2026-06-15 的 v1.7.5。

构建输入为仓库内 `_math_full_v1.7.md`、`算法的低语数学卷习题伴读_v12_全册.md` 和本目录分章源稿。旧习题全册已经覆盖补编全部 95 个题号，补编不再重复并入。

依赖：Python、Pandoc、XeLaTeX（含 ctex、Fandol、tcolorbox、xurl 等包），Python 的 PyMuPDF 与 matplotlib。构建使用 matplotlib 随附的 DejaVu 字体，不依赖新增的系统字体安装。

在仓库根目录运行：

```powershell
python -X utf8 build_2026_11.py all
python -X utf8 verify_2026_11.py --pdf
```

分别构建可使用 `math`、`workbook`。完整 Markdown 输出位于根目录，PDF 位于 `output/pdf/`。中间 LaTeX、日志与页面预览位于被 Git 忽略的 `tmp/pdfs/`。

验证覆盖新增 36 题的题号与解答映射、4 个代码块执行、独立算例、公式号唯一性、公式引用标签、PDF 缺字与未解析引用诊断。PDF 页面另外经过渲染检查；机器检查不能替代数学证明或视觉检查。

数学复核对新增内容逐项进行，并处理了本轮发现的历史勘误；不代表全部旧章已重新证明。原文、预印本结果和自拟教学推导的边界见数学卷导读。研究证据见 `research/2026-11-17_evidence.md`。

参考文件 `validation.json` 记录本次实际执行结果。源稿中既有的公式短编号仍保留，新增公式使用 8.x.y、9.x.y，不改变旧读者的引用位置。
