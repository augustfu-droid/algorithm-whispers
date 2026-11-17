"""Render new material and representative full-book pages for visual QA."""
from pathlib import Path
import re
import subprocess
import sys
import fitz
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'tmp'/'pdfs'/'preview'
OUT.mkdir(parents=True,exist_ok=True)

for key,stem in [('math','算法的低语_数学卷_v1.8'),('workbook','算法的低语_数学卷习题伴读_v1.3_全册')]:
    if len(sys.argv)>1 and key!=sys.argv[1]:
        continue
    doc=fitz.open(ROOT/'output'/'pdf'/(stem+'.pdf'))
    toc=doc.get_toc()
    starts=[p-1 for level,title,p in toc if level==1 and re.search(r'第\s*8\s*章',title)]
    # PDF bookmarks are the authoritative page numbers, not a text match in TOC.
    if not starts:
        raise RuntimeError(f'Missing chapter 8 bookmark in {key}')
    start=min(starts)
    samples=sorted(set([0,1,2,5,10,len(doc)//3,2*len(doc)//3]+list(range(start,len(doc)))))
    for batch_start in range(0,len(samples),12):
        group=samples[batch_start:batch_start+12]
        sheet=Image.new('RGB',(4*300,3*445),'#dce5e8')
        draw=ImageDraw.Draw(sheet)
        for j,idx in enumerate(group):
            pix=doc[idx].get_pixmap(matrix=fitz.Matrix(0.55,0.55),alpha=False)
            img=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
            img.thumbnail((288,414))
            x=(j%4)*300+(300-img.width)//2
            y=(j//4)*445+22
            sheet.paste(img,(x,y))
            draw.text(((j%4)*300+10,(j//4)*445+5),f'{key} p{idx+1}',fill='black')
        target=OUT/f'{key}_contact_{batch_start//12+1}.png'
        sheet.save(target)
        print(target)
    for idx in [0,start,min(start+1,len(doc)-1),len(doc)-1]:
        target=OUT/f'{key}_page_{idx+1}.png'
        subprocess.run(['pdftoppm','-f',str(idx+1),'-l',str(idx+1),'-r','120',
            '-png','-singlefile',str(ROOT/'output'/'pdf'/(stem+'.pdf')),str(target.with_suffix(''))],
            check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        print(target)
