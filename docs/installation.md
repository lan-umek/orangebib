# Installation

## Requirements

- Python 3.9–3.12
- Orange3 ≥ 3.36
- `biblium` ≥ 2.16 (the analysis engine; installed automatically from PyPI)

## From the Orange desktop app

Open **Options ▸ Add-ons ▸ Add more…**, search for *Orange3-Biblium* and
install it, then restart Orange. The **Biblium** category appears in the widget
toolbox.

## With pip

```bash
pip install Orange3-Biblium
```

When installing into an existing Orange desktop installation, use **Orange's**
bundled Python interpreter, for example on Windows:

```bat
"%LOCALAPPDATA%\Programs\Orange\python.exe" -m pip install Orange3-Biblium
```

## From source (development)

```bash
git clone https://github.com/lan-umek/orange3-biblium
cd orange3-biblium
pip install -e .
```

Restart Orange after installation so the widget registry is rebuilt.
