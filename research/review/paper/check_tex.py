"""Structural checks for the paper when no LaTeX install is available. Does not replace compiling.

    python research/review/paper/check_tex.py
"""
import glob
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
R = os.path.dirname(os.path.abspath(__file__))
files = [os.path.join(R, "main.tex")] + sorted(glob.glob(os.path.join(R, "sections", "*.tex")))
raw = {f: open(f, encoding="utf-8").read() for f in files}
txt = {f: re.sub(r"(?<!\\)%.*", "", t) for f, t in raw.items()}
bib = open(os.path.join(R, "references.bib"), encoding="utf-8").read()
keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))

cited, labels, refs = set(), set(), set()
for f, t in txt.items():
    for m in re.findall(r"\\cite\{([^}]*)\}", t):
        cited |= {x.strip() for x in m.split(",")}
    labels |= set(re.findall(r"\\label\{([^}]*)\}", t))
    refs |= set(re.findall(r"\\ref\{([^}]*)\}", t))
    opens, closes = len(re.findall(r"(?<!\\)\{", t)), len(re.findall(r"(?<!\\)\}", t))
    if opens != closes:
        print("BRACES", os.path.basename(f), opens, closes)
    begins, ends = re.findall(r"\\begin\{([^}]*)\}", t), re.findall(r"\\end\{([^}]*)\}", t)
    if sorted(begins) != sorted(ends):
        print("ENVIRONMENTS", os.path.basename(f), set(begins) ^ set(ends))
    for m in re.finditer(r"(?<!\\)[&_#]", re.sub(r"\\begin\{tabularx\}.*?\\end\{tabularx\}|\\(cite|label|ref|includegraphics)(\[[^\]]*\])?\{[^}]*\}|\\url\{[^}]*\}", "", t, flags=re.S)):
        print("UNESCAPED", os.path.basename(f), repr(m.group()))
    for m in re.findall(r"\d%", raw[f]):
        print("BARE PERCENT", os.path.basename(f), m)
    for m in re.finditer(r"\\begin\{tabularx\}\{[^\n]*?\}\{@\{\}(.*?)@\{\}\}\n(.*?)\\end\{tabularx\}", t, flags=re.S):
        cols = len(re.findall(r"L\{|Y", m.group(1)))
        for row in m.group(2).split("\\\\"):
            row = re.sub(r"\\(top|mid|bottom)rule", "", row).strip()
            if not row or row.startswith("\\multicolumn"):
                continue
            n = len(re.findall(r"(?<!\\)&", row)) + 1
            if n != cols:
                print("COLUMNS", os.path.basename(f), f"expected {cols}, got {n}:", row[:70])

print(f"citations used: {len(cited)}, bibliography entries: {len(keys)}")
print("cited but missing from references.bib:", sorted(cited - keys))
print("in references.bib but not cited:", sorted(keys - cited))
print("undefined \\ref:", sorted(refs - labels), "| labels never referenced:", sorted(labels - refs))
for f, t in txt.items():
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", t):
        print("figure", g, "exists" if os.path.exists(os.path.join(R, "figures", g + ".pdf")) else "MISSING")
print("approx. words:", sum(len(re.sub(r"\\[a-zA-Z]+|[{}$&]", " ", t).split()) for t in txt.values()))
print("red TODO markers:", sum(t.count("\\todo{") for t in txt.values()) - 1)
