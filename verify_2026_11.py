"""Check new exercises, independent numerical examples, and final PDF structure."""
from pathlib import Path
from collections import Counter
import contextlib
import io
import json
import math
import re
import sys

ROOT = Path(__file__).resolve().parent
REV = ROOT / 'revisions' / '2026-11'

def main():
    result = {'exercise_counts':{},'code_blocks':[],'formula_counts':{}}
    for chapter, expected in [(8,24),(9,12)]:
        text = (REV/f'exercises_ch{chapter}.md').read_text(encoding='utf-8')
        ids = re.findall(r'^### (T-\d+\.\d+\.\d+)【',text,re.M)
        answers = re.findall(r'^(?:\*\*|### )(T-\d+\.\d+\.\d+) 解答',text,re.M)
        assert len(ids)==expected and len(set(ids))==expected, (chapter,ids)
        assert Counter(ids)==Counter(answers),(chapter,ids,answers)
        assert text.count('$$')%2==0
        result['exercise_counts'][chapter]=expected
        fence=chr(96)*3
        blocks = re.findall(r'(?:'+fence+r'|~~~)python\n(.*?)(?:'+fence+r'|~~~)',text,re.S)
        for i,code in enumerate(blocks,1):
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                exec(compile(code,f'chapter{chapter}_example{i}','exec'),{})
            result['code_blocks'].append({'chapter':chapter,'example':i,'output':capture.getvalue().strip()})
        body=(REV/f'math_ch{chapter}.md').read_text(encoding='utf-8')
        tags=re.findall(r'\\tag\{([^}]+)\}',body)
        assert len(tags)==len(set(tags))
        assert body.count('$$')%2==0
        result['formula_counts'][chapter]=len(tags)
    assert abs((2*1+6*5)/(2+6)-4)<1e-12
    assert abs((0+1+2)/3-(1+2)/2)==0.5
    assert abs((1/4+3)**-1*6-24/13)<1e-12
    assert math.ceil(1000/16)==63
    assert (128+64)/(4+8)==16
    entropy=lambda p:-sum(x*math.log(x) for x in p if x)
    assert entropy([0.01,0.99])<entropy([0.6,0.4])
    assert -math.log(0.01)>-math.log(0.6)
    a1,b1,a2,b2=2,1,3,4
    assert a2*b1+b2==7 and a1*b2+b1==9
    result['independent_numerical_checks']='passed'
    math_source=(ROOT/'算法的低语_数学卷_v1.8.md').read_text(encoding='utf-8')
    tags=re.findall(r'\\tag\{([^}]+)\}',math_source)
    labels=set(re.findall(r'\\label\{([^}]+)\}',math_source))
    refs=set(re.findall(r'\\eqref\{([^}]+)\}',math_source))
    assert len(tags)==len(set(tags))
    assert set(tags)<=labels and refs<=labels
    result['all_numbered_formulas']=len(tags)
    result['equation_references']='all resolve to source labels'
    if '--pdf' in sys.argv:
        import fitz
        result['pdfs']=[]
        for key,stem in [('math','算法的低语_数学卷_v1.8'),('workbook','算法的低语_数学卷习题伴读_v1.3_全册')]:
            path=ROOT/'output'/'pdf'/(stem+'.pdf')
            doc=fitz.open(path)
            texts=[p.get_text() for p in doc]
            bad=[i+1 for i,t in enumerate(texts) if '\ufffd' in t or '(??)' in t]
            assert not bad,(path,bad)
            assert len(doc.get_toc())>20
            assert '2026' in texts[0] and '11' in texts[0]
            outside=[]
            for i,page in enumerate(doc):
                for block in page.get_text('dict')['blocks']:
                    for line in block.get('lines',[]):
                        for span in line.get('spans',[]):
                            if not span['text'].strip():
                                continue
                            x0,y0,x1,y1=span['bbox']
                            if x0<35 or x1>page.rect.width-35 or y0<15 or y1>page.rect.height-15:
                                outside.append(i+1)
            log=(ROOT/'tmp'/'pdfs'/f'{key}_run3.txt').read_text(encoding='utf-8')
            missing=re.findall(r'Missing character: .+',log)
            unresolved=[l for l in log.splitlines() if 'undefined' in l.lower()]
            assert not missing,(key,missing)
            assert not unresolved,(key,unresolved)
            result['pdfs'].append({'file':str(path),'pages':len(doc),'bookmarks':len(doc.get_toc()),'margin_review_pages':sorted(set(outside)),'missing_glyphs':len(missing),'unresolved_references':len(unresolved)})
    print(json.dumps(result,ensure_ascii=False,indent=2))
    (REV/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
