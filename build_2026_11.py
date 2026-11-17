"""Reproducible November 2026 edition. Run with Python -X utf8.

Requires pandoc, XeLaTeX and PyMuPDF. Old edition files are read-only inputs.
"""
from pathlib import Path
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REV = ROOT / 'revisions' / '2026-11'
TMP = ROOT / 'tmp' / 'pdfs'
OUT = ROOT / 'output' / 'pdf'

PREAMBLE = r'''
\usepackage{fontspec}
\usepackage[fontset=fandol]{ctex}
\setmainfont{DejaVuSans.ttf}[Path=@@FONTDIR@@,BoldFont=DejaVuSans-Bold.ttf,ItalicFont=DejaVuSans-Oblique.ttf,BoldItalicFont=DejaVuSans-BoldOblique.ttf]
\setsansfont{DejaVuSans.ttf}[Path=@@FONTDIR@@,BoldFont=DejaVuSans-Bold.ttf]
\setmonofont{DejaVuSans.ttf}[Path=@@FONTDIR@@,Scale=0.82,BoldFont=DejaVuSans-Bold.ttf,ItalicFont=DejaVuSans-Oblique.ttf]
\usepackage{geometry}
\geometry{a4paper,top=24mm,bottom=24mm,left=25mm,right=25mm,headheight=15pt,footskip=12mm}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{graphicx,float,xcolor,booktabs,array,calc}
\usepackage{hyperref}
\usepackage{xurl}
\usepackage[open,openlevel=1]{bookmark}
\usepackage{fancyhdr,titlesec,enumitem}
\usepackage[most]{tcolorbox}
\definecolor{teal1}{RGB}{0,77,89}
\definecolor{teal2}{RGB}{15,118,135}
\definecolor{teal3}{RGB}{38,162,180}
\AtBeginDocument{\setcounter{secnumdepth}{3}}
\AtBeginDocument{\setcounter{tocdepth}{2}}
\renewcommand{\thesection}{}
\renewcommand{\thesubsection}{}
\renewcommand{\thesubsubsection}{}
\titleformat{\section}{\Large\bfseries\color{teal1}}{}{0pt}{}
\titleformat{\subsection}{\large\bfseries\color{teal2}}{}{0pt}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{teal2}}{}{0pt}{}
\titlespacing*{\section}{0pt}{1.8em}{0.8em}
\titlespacing*{\subsection}{0pt}{1.3em}{0.5em}
\newtcolorbox{sectionintuition}{enhanced,breakable,colback=teal3!5,colframe=teal2,boxrule=0.5pt,arc=2pt,left=9pt,right=9pt,top=7pt,bottom=7pt,title=这一节在讲什么,fonttitle=\bfseries}
\newtcolorbox{keytermsbox}{enhanced,breakable,colback=teal3!4,colframe=teal3!25,boxrule=0.4pt,arc=2pt,left=7pt,right=7pt,top=4pt,bottom=4pt,fontupper=\small}
\newcommand{\keyterms}[1]{\begin{keytermsbox}\textbf{\color{teal1}核心概念}\quad #1\end{keytermsbox}}
\newcommand{\faIconStub}{}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\linespread{1.28}
\setlist{itemsep=0.2em,parsep=0.15em,topsep=0.3em}
\setlength{\emergencystretch}{3em}
\sloppy
\allowdisplaybreaks[2]
\newsavebox{\displaymeasure}
\newcommand{\fitdisplay}[1]{\sbox{\displaymeasure}{$\displaystyle #1$}\ifdim\wd\displaymeasure>\dimexpr\linewidth-55pt\relax\resizebox{\dimexpr\linewidth-55pt\relax}{!}{\usebox{\displaymeasure}}\else\usebox{\displaymeasure}\fi}
\newsavebox{\inlinemeasure}
\DeclareRobustCommand{\fitinline}[1]{\sbox{\inlinemeasure}{$#1$}\ifdim\wd\inlinemeasure>0.90\linewidth\relax\par\noindent\resizebox{0.90\linewidth}{!}{\usebox{\inlinemeasure}}\par\else$#1$\fi}
\AtBeginEnvironment{longtable}{\small\setlength{\tabcolsep}{4pt}}
\makeatletter
\renewcommand\@pnumwidth{2.8em}
\renewcommand\@tocrmarg{3.8em}
\makeatother
\hypersetup{colorlinks=true,linkcolor=teal2,urlcolor=teal2,citecolor=teal2,pdfauthor={大队长},pdftitle={@@TITLE@@}}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\color{gray}算法的低语 · @@SHORT@@}
\fancyhead[R]{\small\color{gray}2026 年 11 月增订版}
\fancyfoot[C]{\small\color{gray}\thepage}
\fancyfoot[R]{\small\color{gray}大队长出品}
\renewcommand{\headrulewidth}{0.3pt}
\newcommand{\makeeditioncover}{
  \begingroup\thispagestyle{empty}\pagecolor{teal1}\color{white}
  \vspace*{25mm}
  {\small 2026 · NOVEMBER\par}
  \vspace{12mm}{\fontsize{42}{52}\selectfont\bfseries 算法的低语\par}
  \vspace{5mm}{\fontsize{28}{36}\selectfont\bfseries @@SHORT@@\par}
  \vspace{8mm}{\Large @@SUBTITLE@@\par}
  \vspace{12mm}{\color{teal3}\rule{0.85\linewidth}{1pt}\par}
  \vspace{8mm}\begin{minipage}{0.88\linewidth}\large
  @@BLURB@@
  \end{minipage}
  \vfill{\Large 大队长\par}
  \vspace{4mm}{\normalsize @@VERSION@@ · 2026 年 11 月 17 日\par}
  \vspace{3mm}{\small\color{white!75}fqsx@mail.ustc.edu.cn\par}
  \clearpage\nopagecolor\endgroup
}
\newcommand{\makemathcoverpage}{\makeeditioncover}
'''

def read(name):
    return (ROOT / name).read_text(encoding='utf-8-sig')

def repair_source(content):
    content = re.sub(r'\\tag\{([^}]+)\}(?!\\label)',r'\\tag{\1}\\label{\1}',content)
    return content.replace('乾釥函数','生成函数').replace('ℋ', 'H')

def correct_historical_math(content):
    new_scan = r'''\begin{aligned}
(P_t,q_t)&=(\bar A_1,\bar B_1x_1)\otimes\cdots\otimes(\bar A_t,\bar B_tx_t),\\
h_t&=P_t h_0+q_t.
\end{aligned}'''
    for tag in ('7.4.2','7.19'):
        hits = []
        def replace_equation(m):
            if r'\tag{'+tag+'}' not in m.group(1):
                return m.group(0)
            hits.append(tag)
            return '$$\n'+new_scan+r'\tag{'+tag+'}\n$$'
        content = re.sub(r'\$\$(.*?)\$\$',replace_equation,content,flags=re.S)
        assert len(hits)==1, (tag,hits)
    content = content.replace(r'\{(a_t, b_t) \otimes \cdots \otimes (a_1, b_1)\}_{t=1}^L',
                              r'\{(a_1, b_1) \otimes \cdots \otimes (a_t, b_t)\}_{t=1}^L')
    content = content.replace('**引理 7.4 (选择性 SSM 的关联算子)**:',
       '**v1.8 方向校正**：本节算子按时间正序复合。$P_t$ 是矩阵、$q_t$ 是向量；$h_0=0$ 时 $h_t=q_t$。矩阵乘法的书写顺序与该算子的参数顺序不同，不能互换。\n\n**引理 7.4 (选择性 SSM 的关联算子)**:')
    old = '其精度为 $O(\\Delta^2)$ 而 ZOH 同阶。两者实践差异较小,但 Mamba 重新引入 ZOH 以支持更直观的选择机制'
    new = '它对齐次状态转移的单步误差为 $O(\\Delta^3)$，在通常稳定性与正则条件下有二阶全局精度；ZOH 在每步内矩阵固定且输入保持常数时给出精确采样递推，数值计算误差另计。Mamba 使用 ZOH 形式支持直观的选择机制'
    assert old in content
    content = content.replace(old,new)
    note = r'''**v1.8 条件补充**：以下含 $A^{-1}$ 的写法要求 $A$ 可逆；一般矩阵应使用
$\bar B=(\int_0^\Delta e^{\tau A}\,d\tau)B=\Delta\varphi_1(\Delta A)B$，
其中 $\varphi_1(Z)=\sum_{j=0}^{\infty}Z^j/(j+1)!$。
这一定义也适用于奇异矩阵。本条件同时适用于式 (7.3.1)、(7.10)、(7.3.7)、(7.3.8) 以及下一节逐步变化的 ZOH 表达式。
'''
    content = content.replace('### 7.3.1 零阶保持离散化','### 7.3.1 零阶保持离散化\n\n'+note,1)
    return content

def fit_tex(text):
    def display(match):
        body = match.group(1)
        tags = re.findall(r'\\tag\{([^}]+)\}', body)
        labels = re.findall(r'\\label\{([^}]+)\}', body)
        body = re.sub(r'\\(?:tag|label)\{[^}]+\}', '', body)
        suffix = ''.join(r'\tag{'+v+'}' for v in tags) + ''.join(r'\label{'+v+'}' for v in labels)
        return '\\begin{equation*}\n\\fitdisplay{'+body.strip()+'}'+suffix+'\n\\end{equation*}'
    text = re.sub(r'\\\((.*?)\\\)',lambda m:(r'\fitinline{'+m.group(1)+'}') if len(m.group(1))>140 else m.group(0), text, flags=re.S)
    text = re.sub(r'\\\[(.*?)\\\]', display, text, flags=re.S)
    def table_widths(match):
        block = match.group(0)
        nums = re.findall(r'\\real\{([\d.]+)\}', block)
        if not nums:
            return block
        weights = [float(x) for x in nums]
        floor = min(0.14,0.8/len(weights))
        deficits = sum(max(0,floor-w) for w in weights)
        slack = sum(max(0,w-floor) for w in weights)
        if deficits and slack:
            weights = [floor if w<floor else w-deficits*(w-floor)/slack for w in weights]
        iterator = iter(weights)
        return re.sub(r'\\real\{[\d.]+\}',lambda _:r'\real{'+f'{next(iterator):.6f}'+'}',block)
    return re.sub(r'\\begin\{longtable\}.*?\\toprule',table_widths,text,flags=re.S)

def assemble():
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    math = correct_historical_math(read('_math_full_v1.7.md'))
    math = math.replace('## 前言：为什么需要这本附册', '## 基础篇前言：为什么需要这本附册', 1)
    math = math.replace('## 改动记录', '## 基础篇改动记录', 1)
    math = math.replace('## 后记', '## 2026 年 6 月版后记', 1)
    math = math.replace('交叉熵不是「随便挑的损失」，它是「让模型分布逼近数据分布」的唯一信息论意义上正确的目标。',
                        '交叉熵来自对数评分与最大似然；在真实分布固定时，最小化交叉熵等价于最小化正向 KL。它不是分布学习中唯一合理的损失。')
    math = math.replace('已知信息只能减少不确定性，不可能增加。',
                        '该不等式对条件熵的平均值成立；固定某个观测后，条件分布的熵仍可能高于原来的边缘熵。')
    # The historical preface keeps its date. Put the new reader guide before it.
    cover_end = math.index('```', math.index('```') + 3) + 3
    math = math[:cover_end] + '\n\n' + read('revisions/2026-11/math_intro.md') + '\n\n' + math[cover_end:]
    math += '\n\n\\clearpage\n\n' + read('revisions/2026-11/math_ch8.md')
    math += '\n\n\\clearpage\n\n' + read('revisions/2026-11/math_ch9.md')
    math += '\n\n\\clearpage\n\n' + read('revisions/2026-11/references.md')
    workbook = read('算法的低语数学卷习题伴读_v12_全册.md')
    workbook = workbook.replace('Adam/AdamW','Adam / AdamW')
    workbook = re.sub(r'\A---\n.*?\n---\n', '', workbook, count=1, flags=re.S)
    workbook = workbook.replace('# 前言', '# 基础篇前言', 1)
    workbook = workbook.replace('而固定参数的 Kalman 只能处理平稳线性系统。',
        '经典 Kalman 滤波允许已知的时变线性系统，也不要求状态过程必须平稳；与 Mamba 的关键差别是建模目的、参数生成方式及概率假设。')
    workbook = workbook.replace(r'两者 $O(\Delta)$ 精度相同。',
        r'两者的一阶泰勒展开相同，这不意味着离散化误差阶相同。ZOH 在保持输入假设下精确；双线性方法对状态转移的单步误差为 $O(\Delta^3)$。')
    workbook = workbook.replace('ZOH 和双线性离散化在小步长下精度相同，但 Mamba 为何选 ZOH？关键是哪个极限行为？',
        'ZOH 在保持输入假设下精确，双线性方法在小步长下近似它；对于负实标量状态系数，两者在大步长极限下的遗忘行为有何不同？')
    workbook = workbook.replace(r'从 $(1, 0)$（初始态 $h_0 = 0$）出发：',
        r'令 $Z_0=(I_N,0_N)$ 为单位仿射变换，按时间顺序计算 $Z_t=Z_{t-1}\otimes(a_t,b_t)=(P_t,q_t)$。由于 $h_0=0$，隐状态为 $h_t=q_t$：')
    workbook = workbook.replace('## §7.3 S4/Mamba 离散化与选择性 SSM',
        '## §7.3 S4/Mamba 离散化与选择性 SSM\n\n'+r'**v1.3 条件补充**：本节公式卡片和题目中的逆矩阵表达式假设 $A$ 可逆；一般情形应采用 $\bar B=(\int_0^\Delta e^{\tau A}\,d\tau)B$。T-7.3.2 的具体对角矩阵可逆。',1)
    workbook = '```{=latex}\n\\makeeditioncover\n\\tableofcontents\n\\clearpage\n```\n\n' + read('revisions/2026-11/workbook_intro.md') + '\n\n' + workbook
    exercises8 = REV / 'exercises_ch8.md'
    if not exercises8.exists() and 'math' not in sys.argv:
        raise FileNotFoundError(exercises8)
    workbook += '\n\n\\clearpage\n\n' + (read('revisions/2026-11/exercises_ch8.md') if exercises8.exists() else '')
    workbook += '\n\n\\clearpage\n\n' + read('revisions/2026-11/exercises_ch9.md')
    specs = [
      ('math', '算法的低语_数学卷_v1.8', math, '数学卷', '从公式到推导，从生成到验证', 'v1.8',
       '保留七章基础推导，新增推理时计算、可验证奖励与证明搜索，以及注意力修补、状态表示和残差扩散。\\par\\medskip 定义、假设、推导与边界，沿着同一条逻辑链展开。'),
      ('workbook', '算法的低语_数学卷习题伴读_v1.3_全册', workbook, '习题伴读', '把新方法放到纸笔和代码上', 'v1.3',
       '保留基础篇全册与既有补题，新增第 8、9 章共 36 道分层习题与完整解答。\\par\\medskip 从手算与反例，到证明与可运行实验，逐节对齐数学卷。'),
    ]
    result = []
    for key, stem, content, short, subtitle, version, blurb in specs:
        if key == 'workbook' and not exercises8.exists():
            continue
        md = ROOT / (stem + '.md')
        md.write_text(repair_source(content), encoding='utf-8')
        pre = PREAMBLE
        import matplotlib
        fontdir = (Path(matplotlib.get_data_path())/'fonts'/'ttf').as_posix()+'/'
        pre = pre.replace('@@FONTDIR@@',fontdir)
        for field, value in {'TITLE':'算法的低语 · '+short+' '+version,'SHORT':short,'SUBTITLE':subtitle,'VERSION':version,'BLURB':blurb}.items():
            pre = pre.replace('@@'+field+'@@', value)
        header = TMP / (key + '_preamble.tex')
        header.write_text(pre, encoding='utf-8')
        tex = TMP / (key + '.tex')
        subprocess.run(['pandoc', str(md), '-f', 'markdown+tex_math_dollars+raw_tex+autolink_bare_uris', '-s', '-t', 'latex',
                        '--top-level-division=section', '--toc-depth=2', '-V', 'fontsize=11pt',
                        '--include-in-header='+str(header), '-o', str(tex)], check=True, cwd=ROOT)
        tex.write_text(fit_tex(tex.read_text(encoding='utf-8')),encoding='utf-8')
        result.append((key, stem, tex))
    return result

def compile_one(key, stem, tex):
    for run in range(1, 4):
        log = TMP / f'{key}_run{run}.txt'
        with log.open('w', encoding='utf-8') as out:
            proc = subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','-no-shell-escape',
                '-output-directory='+str(TMP),str(tex)], cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
        if proc.returncode:
            print(log.read_text(encoding='utf-8')[-6000:])
            raise RuntimeError(f'XeLaTeX failed; see {log}')
    target = OUT / (stem + '.pdf')
    target.write_bytes((TMP / (key + '.pdf')).read_bytes())
    import fitz
    doc = fitz.open(target)
    print(json.dumps({'file':str(target),'pages':len(doc),'bytes':target.stat().st_size},ensure_ascii=False))
    return target

if __name__ == '__main__':
    specs = assemble()
    if '--sources-only' not in sys.argv:
        wanted = sys.argv[1] if len(sys.argv)>1 else 'all'
        for key, stem, tex in specs:
            if wanted in ('all',key):
                compile_one(key,stem,tex)
