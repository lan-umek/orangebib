#!/usr/bin/env python3
"""Capture one PNG per Biblium widget into docs/_static/img/<module>.png.

Each widget renders in its OWN subprocess (a hard crash in one cannot abort the
run). System fonts are loaded so control labels are readable, and an optional
sample dataset is fed to widgets that accept a "Data" input so views populate.

Run with ORANGE's Python, ideally with a sample bibliographic export::

    "%LOCALAPPDATA%\\Programs\\Orange\\python.exe" tools/capture_screenshots.py "C:\\path\\to\\sample.xlsx"
"""
import os
import sys
import glob
import subprocess
import importlib.util

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Make Qt find fonts (Orange's bundled Qt often ships none) so labels render.
for _fdir in (r"C:\Windows\Fonts", "/usr/share/fonts",
              "/System/Library/Fonts", os.path.expanduser("~/.fonts")):
    if os.path.isdir(_fdir):
        os.environ.setdefault("QT_QPA_FONTDIR", _fdir)
        break

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WIDGETS = os.path.join(ROOT, "orangebib", "widgets")
OUT = os.path.join(ROOT, "docs", "_static", "img")
os.makedirs(OUT, exist_ok=True)
REPORT = os.path.join(HERE, "capture_report.txt")


def _ensure_fonts(app):
    from AnyQt.QtGui import QFont, QFontDatabase
    loaded = False
    fdir = os.environ.get("QT_QPA_FONTDIR", "")
    for ttf in ("arial.ttf", "segoeui.ttf", "tahoma.ttf", "calibri.ttf",
                "DejaVuSans.ttf"):
        p = os.path.join(fdir, ttf)
        if os.path.exists(p):
            QFontDatabase.addApplicationFont(p)
            loaded = True
    fams = QFontDatabase().families() if hasattr(QFontDatabase, "families") \
        else QFontDatabase.families()
    fam = "Arial" if "Arial" in fams else (fams[0] if fams else "")
    if fam:
        app.setFont(QFont(fam, 9))
    return loaded


def _capture_one(path, data_path):
    from AnyQt.QtWidgets import QApplication
    from AnyQt.QtCore import Qt
    try:
        QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    except Exception:
        pass
    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        _ensure_fonts(app)
    except Exception:
        pass

    name = "owdoc_" + os.path.basename(path)[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from Orange.widgets.widget import OWWidget
    cls = None
    for obj in vars(mod).values():
        if isinstance(obj, type) and issubclass(obj, OWWidget) \
                and obj.__module__ == mod.__name__ and getattr(obj, "name", None):
            cls = obj
            break
    if cls is None:
        return 2

    w = cls()
    if data_path and os.path.exists(data_path) and hasattr(w, "set_data"):
        try:
            from Orange.data import Table
            data = Table(data_path)
            w.set_data(data)
            if hasattr(w, "handleNewSignals"):
                w.handleNewSignals()
        except Exception:
            pass
    # show off-screen so the layout is fully realised (subprocess-isolated, so a
    # paint crash in one widget cannot abort the whole run), then size the widget
    # to its content before grabbing so nothing is clipped.
    w.show()
    app.processEvents()
    try:
        w.adjustSize()
    except Exception:
        pass
    sh = w.sizeHint()
    w.resize(max(1100, sh.width()), max(780, sh.height()))
    for _ in range(4):
        app.processEvents()
    out_png = os.path.join(OUT, os.path.basename(path)[:-3] + ".png")
    pix = w.grab()
    pix.save(out_png)
    try:
        w.close()
    except Exception:
        pass
    return 0 if os.path.exists(out_png) else 3


def _run_parent(data_path):
    files = sorted(glob.glob(os.path.join(WIDGETS, "ow*.py")))
    results = []
    for f in files:
        base = os.path.basename(f)[:-3]
        env = dict(os.environ, BIBLIUM_ONE=f)
        if data_path:
            env["BIBLIUM_DATA"] = data_path
        r = None
        try:
            r = subprocess.run([sys.executable, os.path.abspath(__file__)],
                               env=env, capture_output=True, text=True,
                               timeout=180)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = -9
        png = os.path.join(OUT, base + ".png")
        if code == 0 and os.path.exists(png):
            results.append(("ok", base, ""))
            print("captured", base)
        else:
            err = "exit %s" % code
            try:
                lines = [ln for ln in r.stderr.splitlines() if ln.strip()]
                if lines:
                    err = lines[-1][:200]
            except Exception:
                pass
            results.append(("FAILED", base, "exit=%s %s" % (code, err)))
            print("FAILED", base, "-> exit", code)

    ok = sum(1 for s, _, _ in results if s == "ok")
    fail = len(results) - ok
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("captured=%d failed=%d  (data=%s)\n\n" % (ok, fail, data_path or "none"))
        for status, base, err in results:
            if status != "ok":
                fh.write("%s\t%s\t%s\n" % (status, base, err))
    print("\nDone: %d captured, %d failed -> %s" % (ok, fail, OUT))
    print("report ->", REPORT)
    failed = [b for s, b, _ in results if s != "ok"]
    if failed:
        print("FAILED widgets:", ", ".join(failed))


def main():
    one = os.environ.get("BIBLIUM_ONE")
    data_path = os.environ.get("BIBLIUM_DATA")
    if not data_path and len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        data_path = sys.argv[1]
    if one:
        sys.exit(_capture_one(one, data_path))
    _run_parent(data_path)


if __name__ == "__main__":
    main()
