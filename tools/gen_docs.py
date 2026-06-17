#!/usr/bin/env python3
"""Generate detailed per-widget documentation (MyST markdown) + catalog from
the widget source files by static parsing (no Orange import required)."""
import os, re, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WIDGETS = os.path.join(ROOT, "orangebib", "widgets")
DOCS = os.path.join(ROOT, "docs")
WDOCS = os.path.join(DOCS, "widgets")
os.makedirs(WDOCS, exist_ok=True)

SECTIONS = [
    (0, 99, "Data input & preparation"),
    (100, 199, "Overview, counts & statistics"),
    (200, 299, "Time, trends & dynamics"),
    (300, 349, "Citation analysis"),
    (350, 399, "Text, topics & concepts"),
    (400, 499, "Networks"),
    (500, 599, "Geography, SDG & disciplines"),
    (600, 759, "Groups, comparison & inference"),
    (760, 9999, "Niche, external & output"),
]

def section_for(p):
    for lo, hi, t in SECTIONS:
        if lo <= p <= hi:
            return t
    return "Other"

def first_docstring(src):
    m = re.search(r'^("""|\'\'\')(.*?)(\1)', src, re.S | re.M)
    if not m:
        return ""
    body = m.group(2).strip()
    lines = [ln for ln in body.splitlines() if not re.match(r'^[=\-]{3,}$', ln.strip())]
    return "\n".join(lines).strip()

def grab(pattern, src, default=""):
    m = re.search(pattern, src)
    return m.group(1).strip() if m else default

def block_after(src, header_re):
    m = re.search(header_re, src)
    if not m:
        return ""
    start = m.end()
    # capture until a line that dedents to 4-space class-level construct
    rest = src[start:]
    out = []
    for ln in rest.splitlines():
        if ln.strip() and not ln.startswith(("        ", "    \t", "\t")) and \
           re.match(r'^    [a-zA-Z_]', ln) and "= Input(" not in ln and \
           "= Output(" not in ln and "Msg(" not in ln:
            break
        out.append(ln)
    return "\n".join(out)

def signals(src, cls):
    blk = block_after(src, r'\n    class %s:' % cls)
    found = re.findall(
        r'(?:Input|Output)\(\s*"([^"]+)"\s*,\s*([A-Za-z_][\w\.]*)'
        r'(?:[^)]*?doc\s*=\s*"([^"]*)")?', blk, re.S)
    return [(n, t, (d or "")) for n, t, d in found]

def messages(src):
    out = {}
    for kind in ("Error", "Warning", "Information"):
        blk = block_after(src, r'\n    class %s\(OWWidget\.%s\):' % (kind, kind))
        msgs = re.findall(r'=\s*Msg\(\s*(?:"([^"]*)"|\'([^\']*)\')', blk)
        texts = [a or b for a, b in msgs]
        if texts:
            out[kind] = texts
    return out

def controls(src):
    boxes = re.findall(r'gui\.widgetBox\([^,]+,\s*"([^"]+)"', src)
    labels = re.findall(r'label\s*=\s*"([^"]+)"', src)
    # checkBox(box, self, "attr", "Label text")
    cbs = re.findall(r'gui\.checkBox\([^,]+,\s*self,\s*"[^"]+",\s*"([^"]+)"', src)
    qbtn = re.findall(r'QPushButton\("([^"]+)"\)', src)
    gbtn = re.findall(r'gui\.button\([^,]+,\s*self,\s*"([^"]+)"', src)
    qlab = re.findall(r'QLabel\("([^":]+):"\)', src)
    ctrl = []
    for x in labels + cbs + qlab:
        x = x.rstrip(":").strip()
        if x and x not in ctrl:
            ctrl.append(x)
    btns = []
    for x in qbtn + gbtn:
        if x not in btns:
            btns.append(x)
    return boxes, ctrl, btns

widgets = []
for f in sorted(glob.glob(os.path.join(WIDGETS, "ow*.py"))):
    src = open(f, encoding="utf-8").read()
    name = grab(r'\n    name = "([^"]+)"', src)
    if not name:
        continue
    desc = grab(r'\n    description = (?:\(\s*)?["\']([^"\']+)["\']', src)
    pr = int(grab(r'\n    priority = (\d+)', src, "9999"))
    boxes, ctrl, btns = controls(src)
    widgets.append(dict(
        file=os.path.basename(f), name=name, desc=desc, priority=pr,
        inputs=signals(src, "Inputs"), outputs=signals(src, "Outputs"),
        msgs=messages(src), boxes=boxes, controls=ctrl, buttons=btns,
        doc=first_docstring(src)))

widgets.sort(key=lambda w: w["priority"])
slug = lambda fn: fn[:-3]

for w in widgets:
    L = [f"# {w['name']}", ""]
    if w["desc"]:
        L += [f"> {w['desc']}", ""]
    img_rel = f"_static/img/{slug(w['file'])}.png"
    if os.path.exists(os.path.join(DOCS, img_rel)):
        L += [f"```{{figure}} ../{img_rel}", ":alt: " + w["name"],
              ":class: widget-screenshot", "", w["name"] + " widget.", "```", ""]
    else:
        L += ["```{admonition} Screenshot",
              ":class: tip",
              f"Add `docs/{img_rel}` (run `python tools/capture_screenshots.py` "
              "with Orange's Python, then re-run `tools/gen_docs.py`).",
              "```", ""]
    # Overview
    body = w["doc"]
    if body:
        b2 = "\n".join(body.splitlines()[1:]).strip() or body
        L += ["## Overview", "", b2, ""]
    # Inputs / Outputs
    L += ["## Inputs", ""]
    if w["inputs"]:
        for n, t, d in w["inputs"]:
            L.append(f"- **{n}** (`{t}`)" + (f" — {d}" if d else ""))
    else:
        L.append("- *(none)*")
    L += ["", "## Outputs", ""]
    if w["outputs"]:
        for n, t, d in w["outputs"]:
            L.append(f"- **{n}** (`{t}`)" + (f" — {d}" if d else ""))
    else:
        L.append("- *(none)*")
    L += [""]
    # Controls
    if w["boxes"] or w["controls"] or w["buttons"]:
        L += ["## Controls", ""]
        if w["boxes"]:
            L += ["**Sections:** " + ", ".join(f"*{b}*" for b in w["boxes"]), ""]
        for c in w["controls"]:
            L.append(f"- **{c}** — _describe what this does_")
        if w["buttons"]:
            L += ["", "**Actions:** " + ", ".join(f"`{b}`" for b in w["buttons"]), ""]
        L += [""]
    # Messages
    if w["msgs"]:
        L += ["## Messages", ""]
        for kind, texts in w["msgs"].items():
            L.append(f"*{kind}:*")
            for t in texts:
                L.append(f"- {t}")
            L.append("")
    page = os.path.join(WDOCS, slug(w["file"]) + ".md")
    if os.path.exists(page) and os.environ.get("OVERWRITE") != "1":
        pass  # keep hand-written content
    else:
        open(page, "w", encoding="utf-8").write("\n".join(L))

# index
toc = ["```{toctree}", ":hidden:", ":maxdepth: 1", ""]
toc += [slug(w["file"]) for w in widgets] + ["```", ""]
idx = ["# Widget reference", "",
       f"Biblium contributes {len(widgets)} widgets, grouped by analysis stage. "
       "Each page documents the inputs, outputs, controls and messages.", ""]
cur = None
for w in widgets:
    sec = section_for(w["priority"])
    if sec != cur:
        cur = sec; idx += ["", f"## {sec}", ""]
    idx.append(f"- [{w['name']}]({slug(w['file'])}.md) — {w['desc']}")
open(os.path.join(WDOCS, "index.md"), "w", encoding="utf-8").write(
    "\n".join(toc + idx))

cat = ["| # | Widget | Stage | Description |", "|---|--------|-------|-------------|"]
for i, w in enumerate(widgets, 1):
    cat.append(f"| {i} | **{w['name']}** | {section_for(w['priority'])} | {w['desc']} |")
open(os.path.join(HERE, "_catalog.md"), "w", encoding="utf-8").write("\n".join(cat))
print("generated", len(widgets), "detailed widget pages")
