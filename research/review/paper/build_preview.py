"""Renders the LaTeX draft to one self-contained HTML page for reading. It does not replace compiling on Overleaf:
the layout is single-column, not IEEE two-column. It only handles the LaTeX this paper uses.

    python research/review/paper/build_preview.py
"""
import base64
import html
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["00_abstract", "01_introduction", "02_background", "03_taxonomy", "04_modalities", "05_datasets", "06_evaluation", "07_challenges", "08_conclusion"]
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


def strip_comments(t):
    return re.sub(r"(?<!\\)%.*", "", t)


src = "\n\n".join(strip_comments(open(os.path.join(HERE, "sections", n + ".tex"), encoding="utf-8").read()) for n in ORDER)
main_tex = strip_comments(open(os.path.join(HERE, "main.tex"), encoding="utf-8").read())

# --- Numbering: sections in Roman, subsections as VI-E, tables in Roman, figures in Arabic, citations by first use.
labels, sec, sub, tab, fig = {}, 0, 0, 0, 0
current = ""
for m in re.finditer(r"\\section\{|\\subsection\{|\\begin\{table\*?\}|\\begin\{figure\*?\}|\\label\{([^}]*)\}", src):
    tok = m.group(0)
    if tok.startswith("\\section"):
        sec, sub = sec + 1, 0
        current = ROMAN[sec]
    elif tok.startswith("\\subsection"):
        sub += 1
        current = f"{ROMAN[sec]}-{chr(64 + sub)}"
    elif tok.startswith("\\begin{table"):
        tab += 1
        current = ROMAN[tab]
    elif tok.startswith("\\begin{figure"):
        fig += 1
        current = str(fig)
    else:
        labels[m.group(1)] = current

cite_no = {}
for m in re.finditer(r"\\cite\{([^}]*)\}", src):
    for k in m.group(1).split(","):
        cite_no.setdefault(k.strip(), len(cite_no) + 1)


def cite_html(m):
    nums = sorted(cite_no[k.strip()] for k in m.group(1).split(","))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"{nums[i]}" if j == i else (f"{nums[i]}, {nums[j]}" if j == i + 1 else f"{nums[i]}&ndash;{nums[j]}"))
        i = j + 1
    links = ", ".join(parts)
    return f'<a class="cite" href="#ref{nums[0]}">[{links}]</a>'


def inline(t):
    t = t.strip()
    t = re.sub(r"\\IEEEPARstart\{(.)\}\{([^}]*)\}", r"\1\2", t)
    t = html.escape(t, quote=False)
    t = re.sub(r"\\cite\{([^}]*)\}", cite_html, t)
    t = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels[m.group(1)], t)
    t = re.sub(r"\\url\{([^}]*)\}", r'<a href="\1">\1</a>', t)
    t = re.sub(r"\\todo\{([^}]*)\}", r'<mark class="todo">[\1]</mark>', t)
    for _ in range(3):
        t = re.sub(r"\\textit\{([^{}]*)\}", r"<em>\1</em>", t)
        t = re.sub(r"\\textbf\{([^{}]*)\}", r"<strong>\1</strong>", t)
    math = {r"\times": "×", r"\le": "≤", r"\ge": "≥", r"\pm": "±", "^{\\dagger}": "<sup>†</sup>", "R^2": "R<sup>2</sup>"}
    for a, b in math.items():
        t = t.replace(a, b)
    t = t.replace("$", "")
    t = t.replace("\\%", "%").replace("\\&amp;", "&amp;").replace("\\_", "_").replace("{,}", ",").replace("\\ ", " ").replace("~", "&nbsp;")
    t = t.replace("``", "“").replace("''", "”").replace("---", "—").replace("--", "–")
    return t


def table_html(block, number):
    caption = re.search(r"\\caption\{(.*)\}\s*\n\\label", block, re.S).group(1)
    label = re.search(r"\\label\{([^}]*)\}", block).group(1)
    body = re.search(r"\\begin\{tabularx\}\{[^\n]*?\n(.*?)\\end\{tabularx\}", block, re.S).group(1)
    wide = "table*" in block.split("\n", 1)[0]
    rows, head_done = [], False
    for raw in re.split(r"\\\\", body):
        rule = "\\midrule" in raw
        raw = re.sub(r"\\(top|mid|bottom)rule", "", raw).strip()
        if not raw:
            continue
        mc = re.match(r"\\multicolumn\{(\d+)\}\{[^\n]*?\}\{(.*)\}\s*$", raw, re.S)
        if mc:
            # IEEE-style table note or group label spanning the whole table
            rows.append(f'<tr class="span{" rule" if rule else ""}"><td colspan="{mc.group(1)}">{inline(mc.group(2))}</td></tr>')
            continue
        cells = re.split(r"(?<!\\)&", raw)
        tag = "th" if not head_done else "td"
        cls = ' class="rule"' if rule and head_done else ""
        rows.append(f"<tr{cls}>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells) + "</tr>")
        head_done = True
    return (f'<figure class="tbl{" wide" if wide else ""}" id="{label}"><figcaption><strong>Table {number}.</strong> {inline(caption)}</figcaption>'
            f'<div class="scroll"><table>{"".join(rows)}</table></div></figure>')


def figure_html(block, number):
    name = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", block).group(1)
    caption = re.search(r"\\caption\{(.*)\}\s*\n\\label", block, re.S).group(1)
    label = re.search(r"\\label\{([^}]*)\}", block).group(1)
    data = base64.b64encode(open(os.path.join(HERE, "figures", name + ".png"), "rb").read()).decode()
    return (f'<figure class="fig" id="{label}"><img alt="Figure {number}" src="data:image/png;base64,{data}"/>'
            f"<figcaption><strong>Fig. {number}.</strong> {inline(caption)}</figcaption></figure>")


# --- Blocks
out, buf, sec, sub, tab, fig = [], [], 0, 0, 0, 0


def flush():
    global buf
    text = " ".join(buf).strip()
    if text:
        out.append(f"<p>{inline(text)}</p>")
    buf = []


pattern = re.compile(r"\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}|\\begin\{abstract\}.*?\\end\{abstract\}|\\begin\{IEEEkeywords\}.*?\\end\{IEEEkeywords\}|"
                     r"\\begin\{(itemize|enumerate)\}.*?\\end\{\2\}", re.S)
pos = 0
pieces = []
for m in pattern.finditer(src):
    pieces.append(("text", src[pos:m.start()]))
    pieces.append(("block", m.group(0)))
    pos = m.end()
pieces.append(("text", src[pos:]))

for kind, chunk in pieces:
    if kind == "block":
        flush()
        if chunk.startswith("\\begin{table"):
            tab += 1
            out.append(table_html(chunk, ROMAN[tab]))
        elif chunk.startswith("\\begin{figure"):
            fig += 1
            out.append(figure_html(chunk, fig))
        elif chunk.startswith("\\begin{abstract"):
            body = re.search(r"\\begin\{abstract\}(.*)\\end\{abstract\}", chunk, re.S).group(1)
            out.append(f'<p class="abstract"><strong><em>Abstract</em>&mdash;</strong>{inline(body)}</p>')
        elif chunk.startswith("\\begin{IEEEkeywords"):
            body = re.search(r"\\begin\{IEEEkeywords\}(.*)\\end\{IEEEkeywords\}", chunk, re.S).group(1)
            out.append(f'<p class="abstract"><strong><em>Index Terms</em>&mdash;</strong>{inline(body)}</p>')
        else:
            tag = "ol" if "enumerate" in chunk.split("\n", 1)[0] else "ul"
            items = re.split(r"\\item\s+", re.sub(r"\\(begin|end)\{(itemize|enumerate)\}", "", chunk))
            out.append(f"<{tag}>" + "".join(f"<li>{inline(i)}</li>" for i in items if i.strip()) + f"</{tag}>")
        continue
    for line in chunk.split("\n"):
        s = line.strip()
        m = re.match(r"\\section\{(.*)\}$", s)
        if m:
            flush()
            sec, sub = sec + 1, 0
            out.append(f"<h2>{ROMAN[sec]}. {inline(m.group(1))}</h2>")
            continue
        m = re.match(r"\\subsection\{(.*)\}$", s)
        if m:
            flush()
            sub += 1
            out.append(f"<h3>{chr(64 + sub)}. {inline(m.group(1))}</h3>")
            continue
        if s.startswith("\\label{"):
            continue
        if not s:
            flush()
        else:
            buf.append(s)
flush()

# --- References, in order of first citation
ACCENT = {r'{\"o}': "ö", r'{\"u}': "ü", r"{\~a}": "ã", r"{\c{c}}": "ç", r"{\'e}": "é", r"{\i}": "ı", r"\&": "&", r"\_": "_", r"\%": "%", "``": "“", "''": "”", "--": "–"}


def clean(v):
    for a, b in ACCENT.items():
        v = v.replace(a, b)
    return v.replace("{", "").replace("}", "").strip()


bib = open(os.path.join(HERE, "references.bib"), encoding="utf-8").read()
entries = {}
for m in re.finditer(r"@(\w+)\{([^,\s]+),\n(.*?)\n\}", bib, re.S):
    fields = dict(re.findall(r"^\s*(\w+) = \{(.*)\},?\s*$", m.group(3), re.M))
    entries[m.group(2)] = fields


def author_list(a):
    names = []
    for n in clean(a).split(" and "):
        if "," in n:
            fam, giv = [x.strip() for x in n.split(",", 1)]
            initials = " ".join(p[0] + "." for p in re.split(r"[\s-]+", giv) if p)
            names.append(f"{initials} {fam}".strip())
        else:
            names.append(n)
    if len(names) > 6:
        return ", ".join(names[:6]) + " <em>et al.</em>"
    return ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]


refs = []
for key, no in sorted(cite_no.items(), key=lambda kv: kv[1]):
    e = entries[key]
    parts = [author_list(e["author"]) + ",", f"“{html.escape(clean(e['title']), quote=False)},”"]
    venue = e.get("journal") or e.get("booktitle")
    if venue:
        parts.append(f"<em>{html.escape(clean(venue), quote=False)}</em>,")
    if e.get("volume"):
        parts.append(f"vol. {clean(e['volume'])},")
    if e.get("number"):
        parts.append(f"no. {clean(e['number'])},")
    if e.get("pages"):
        parts.append(f"pp. {clean(e['pages'])},")
    parts.append(f"{clean(e['year'])}.")
    if e.get("note"):
        parts.append(html.escape(clean(e["note"]), quote=False) + ".")
    if e.get("doi"):
        parts.append(f'<a href="https://doi.org/{e["doi"]}">doi:{e["doi"]}</a>')
    refs.append(f'<li id="ref{no}" value="{no}">' + " ".join(parts) + "</li>")

title = inline(re.search(r"\\title\{(.*)\}", main_tex).group(1))
authors = inline(re.search(r"\\author\{(.*?)\n\\thanks", main_tex, re.S).group(1))
thanks = inline(re.search(r"\\thanks\{(.*)\}\}", main_tex).group(1))
ack = inline(re.search(r"\\section\*\{Acknowledgment\}\n(.*?)\n\n\\bibliographystyle", main_tex, re.S).group(1))

CSS = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:#f4f3f0;color:#1a1a19;font:15.5px/1.55 Georgia,'Times New Roman',serif}
main{max-width:860px;margin:0 auto;background:#fff;padding:44px 56px 64px;box-shadow:0 1px 6px rgba(0,0,0,.08)}
.banner{font:12.5px/1.45 system-ui,sans-serif;background:#f9eeda;border-radius:6px;padding:9px 13px;margin-bottom:26px;color:#52514e}
h1{font-size:27px;line-height:1.22;font-weight:400;text-align:center;margin:0 0 12px}
.authors{text-align:center;margin:0 0 3px}.thanks{text-align:center;font-size:12.5px;color:#52514e;margin:0 0 24px}
h2{font-size:16px;text-align:center;text-transform:uppercase;letter-spacing:.04em;font-weight:600;margin:34px 0 12px}
h3{font-size:15.5px;font-style:italic;font-weight:600;margin:24px 0 8px}
p{margin:0 0 11px;text-align:justify;hyphens:auto}
.abstract{font-size:14px;font-weight:600;line-height:1.45}.abstract strong{font-weight:700}
ul,ol{margin:0 0 12px;padding-left:24px}li{margin-bottom:4px}
a{color:#1f5fae;text-decoration:none}a:hover{text-decoration:underline}a.cite{white-space:nowrap}
mark.todo{background:#fbebeb;color:#b3261e;padding:0 3px;border-radius:3px;font:13px system-ui,sans-serif}
figure{margin:22px 0}figure.fig img{display:block;max-width:100%;margin:0 auto}
figcaption{font-size:12.5px;line-height:1.4;margin-top:7px}figure.tbl figcaption{margin:0 0 7px;text-align:center}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font:11.6px/1.34 system-ui,sans-serif;border-top:1.4px solid #1a1a19;border-bottom:1.4px solid #1a1a19}
th{text-align:left;vertical-align:bottom;border-bottom:.8px solid #1a1a19;padding:5px 6px}
td{vertical-align:top;padding:4px 6px;border-bottom:.5px solid #e4e3df}
tr.rule td{border-top:.8px solid #1a1a19}tr.span td{font-size:11px;color:#52514e;border-bottom:none}
.refs{font-size:12.8px;line-height:1.4;padding-left:34px}.refs li{margin-bottom:5px;overflow-wrap:anywhere}
@media(max-width:700px){main{padding:24px 16px}p{text-align:left}}
"""
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>XAI Drowsiness Review Draft</title><style>{CSS}</style></head><body><main>
<div class="banner">Draft preview for reading. Layout is single-column; the IEEE two-column PDF comes from compiling on Overleaf. Red items are still to be done.</div>
<h1>{title}</h1><p class="authors">{authors}</p><p class="thanks">{thanks}</p>
{"".join(out)}
<h2>Acknowledgment</h2><p>{ack}</p>
<h2>References</h2><ol class="refs">{"".join(refs)}</ol>
</main></body></html>"""
path = os.path.join(HERE, "paper_preview.html")
open(path, "w", encoding="utf-8").write(page)

visible = re.sub(r'src="data:[^"]*"', "", page)
leftover = sorted(set(re.findall(r"\\[a-zA-Z]+", visible.split("</style>")[1])))
print(f"written {os.path.getsize(path)//1024} KB; sections {sec}, tables {tab}, figures {fig}, references {len(refs)}")
print("leftover LaTeX commands:", leftover)
print("stray braces in text:", len(re.findall(r"[{}]", re.sub(r"<[^>]+>", "", visible.split("</style>")[1]))))
print("todo markers:", visible.count('class="todo"'))
