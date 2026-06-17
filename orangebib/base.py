# -*- coding: utf-8 -*-
"""
Shared base module for Orange3-Biblium widgets.
================================================

Provides:
- BaseBibliumWidget: common widget class with shared lifecycle hooks
- table_to_df / df_to_table: bidirectional Orange Table <-> pandas DataFrame
- find_column / detect_separator: column resolution helpers
- get_biblium / has_biblium: lazy biblium import (so widgets don't pay
  the import cost at module load time)
- COLUMN_PATTERNS: canonical column name lookups shared across widgets

Compatible with Biblium 2.16+ API (utilsbib, bibstats, bibgroup, ...).
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from Orange.data import (
    ContinuousVariable,
    Domain,
    StringVariable,
    Table,
)
from Orange.widgets.widget import Msg, OWWidget

logger = logging.getLogger(__name__)


# =============================================================================
# Lazy biblium accessor — avoids paying the heavy import cost at widget load
# =============================================================================

_BIBLIUM = None
_BIBLIUM_TRIED = False
_HAS_BIBLIUM = False


def get_biblium():
    """Return the imported `biblium` module or None.

    Imports lazily on first call. Subsequent calls are O(1).
    Compatible with biblium 2.16's lazy-loaded submodule structure.
    """
    global _BIBLIUM, _BIBLIUM_TRIED, _HAS_BIBLIUM
    if not _BIBLIUM_TRIED:
        _BIBLIUM_TRIED = True
        try:
            import biblium  # noqa: PLC0415
            _BIBLIUM = biblium
            _HAS_BIBLIUM = True
            logger.debug("Loaded biblium %s", getattr(biblium, "__version__", "?"))
        except ImportError as exc:
            logger.info("biblium not available: %s", exc)
            _BIBLIUM = None
            _HAS_BIBLIUM = False
    return _BIBLIUM


def has_biblium() -> bool:
    """Return True if biblium can be imported."""
    if not _BIBLIUM_TRIED:
        get_biblium()
    return _HAS_BIBLIUM


def get_biblium_submodule(name: str):
    """Return `biblium.<name>` (e.g. "utilsbib") or None."""
    bib = get_biblium()
    if bib is None:
        return None
    try:
        return getattr(bib, name, None)
    except Exception:
        return None


# =============================================================================
# Column-name patterns — shared across most widgets
# =============================================================================

COLUMN_PATTERNS: dict[str, list[str]] = {
    "year":         ["Year", "Publication Year", "publication_year", "PY"],
    "citations":    ["Cited by", "Times Cited", "Citation Count",
                     "cited_by_count", "TC"],
    "authors":      ["Authors", "Author", "AU", "Author full names",
                     "Author(s) ID"],
    "author_ids":   ["Author(s) ID", "Author IDs", "AID"],
    "source":       ["Source title", "Source", "Journal", "SO"],
    "doctype":      ["Document Type", "Document type", "type", "DT"],
    "language":     ["Language of Original Document", "Language", "LA"],
    "open_access":  ["Open Access", "open_access", "OA"],
    "country":      ["Countries of Authors", "Countries", "Country",
                     "authorships.countries"],
    "affiliation":  ["Affiliations", "Affiliation", "C1"],
    "keywords":     ["Author Keywords", "Keywords", "DE"],
    "index_kw":     ["Index Keywords", "Index keywords", "ID"],
    "references":   ["References", "Cited References", "CR"],
    "title":        ["Title", "Document Title", "TI"],
    "abstract":     ["Abstract", "AB"],
    "doi":          ["DOI", "doi"],
}


def find_column(df: pd.DataFrame | None,
                key_or_candidates: str | Sequence[str]) -> str | None:
    """Return the first matching column name in *df* or None.

    *key_or_candidates* can be:
    - a key in COLUMN_PATTERNS (e.g. "year")
    - an explicit list/tuple of candidates
    """
    if df is None or df.empty:
        return None
    if isinstance(key_or_candidates, str):
        candidates = COLUMN_PATTERNS.get(key_or_candidates, [key_or_candidates])
    else:
        candidates = list(key_or_candidates)
    cols = set(df.columns)
    for cand in candidates:
        if cand in cols:
            return cand
    return None


def detect_separator(series: pd.Series, default: str = "; ") -> str:
    """Detect delimiter used inside string-list cells (";", "|", ", ")."""
    sample = series.dropna() if series is not None else pd.Series(dtype=object)
    if len(sample) == 0:
        return default
    sample_str = str(sample.iloc[0])
    if "|" in sample_str:
        return "|"
    if ";" in sample_str:
        return "; "
    if "," in sample_str and " " in sample_str:
        return ", "
    return default


def split_list_cell(value: Any, sep: str) -> list[str]:
    """Split a multi-valued cell into a list of trimmed non-empty items."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [s.strip() for s in str(value).split(sep) if s and s.strip()]


# =============================================================================
# Orange Table <-> pandas DataFrame
# =============================================================================

def table_to_df(table: Table | None) -> pd.DataFrame:
    """Convert an Orange Table to a pandas DataFrame.

    Uses Table.get_column for each variable across attributes, class_vars,
    and metas. Returns an empty DataFrame when *table* is None or empty.
    """
    if table is None or len(table) == 0:
        return pd.DataFrame()

    data: dict[str, Any] = {}
    domain = table.domain
    for group in (domain.attributes, domain.class_vars, domain.metas):
        for var in group:
            try:
                data[var.name] = table.get_column(var)
            except Exception:  # noqa: BLE001
                # Fall back to indexed access
                try:
                    data[var.name] = [row[var] for row in table]
                except Exception:
                    continue
    return pd.DataFrame(data)


def df_to_table(df: pd.DataFrame | None,
                prefer_strings: bool = False) -> Table | None:
    """Convert a pandas DataFrame to an Orange Table.

    Numeric columns become ContinuousVariable attributes; non-numeric
    columns become StringVariable metas. Returns None for empty input.
    Set *prefer_strings* to keep all columns as string metas (useful for
    summary tables that mix text and numbers).
    """
    if df is None or df.empty:
        return None

    if prefer_strings:
        metas = [StringVariable(str(c)) for c in df.columns]
        domain = Domain([], metas=metas)
        meta_data = df.astype(str).values
        return Table.from_numpy(domain, np.empty((len(df), 0)), metas=meta_data)

    attrs = []
    metas = []
    attr_cols: list[str] = []
    meta_cols: list[str] = []

    for col in df.columns:
        col_data = df[col]
        if pd.api.types.is_numeric_dtype(col_data.dtype):
            attrs.append(ContinuousVariable(str(col)))
            attr_cols.append(col)
        else:
            metas.append(StringVariable(str(col)))
            meta_cols.append(col)

    domain = Domain(attrs, metas=metas)
    n_rows = len(df)

    if attrs:
        X = np.empty((n_rows, len(attrs)), dtype=float)
        for i, col in enumerate(attr_cols):
            X[:, i] = pd.to_numeric(df[col], errors="coerce").values
    else:
        X = np.empty((n_rows, 0), dtype=float)

    if metas:
        M = np.empty((n_rows, len(metas)), dtype=object)
        for i, col in enumerate(meta_cols):
            M[:, i] = df[col].astype(object).where(df[col].notna(), "").values
    else:
        M = np.empty((n_rows, 0), dtype=object)

    return Table.from_numpy(domain, X, metas=M)


# =============================================================================
# Tiny formatting helpers used by many widgets
# =============================================================================

def fmt_value(v: Any) -> str:
    """Format a scalar for display in a QTableWidget."""
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return str(v)


def safe_numeric(series: pd.Series) -> pd.Series:
    """to_numeric(errors='coerce') with NaN-fill compatible across pandas."""
    return pd.to_numeric(series, errors="coerce")


def unique_items_in_list_column(series: pd.Series, sep: str) -> set[str]:
    """Return the set of unique items across a multi-valued column."""
    out: set[str] = set()
    for val in series.dropna():
        out.update(split_list_cell(val, sep))
    return out


def count_items_in_list_column(series: pd.Series, sep: str) -> int:
    """Total occurrences across a multi-valued column."""
    total = 0
    for val in series.dropna():
        total += sum(1 for _ in split_list_cell(val, sep))
    return total


# =============================================================================
# BaseBibliumWidget
# =============================================================================

class BaseBibliumWidget(OWWidget):
    """Common base class for Biblium-powered Orange widgets.

    Subclasses get:

    - ``self.has_biblium``: bool, set in __init__
    - ``self.biblium``: imported biblium module or None
    - Helpers: ``_table_to_df``, ``_df_to_table``, ``_find_column``,
      ``_detect_separator``
    - Standard ``Error.no_data`` / ``Warning.no_biblium`` /
      ``Information.computed`` messages — subclasses can extend or override

    Subclasses must still set ``name``, ``description``, ``icon``,
    ``priority``, ``category`` class attributes per Orange's contract.
    """

    category = "Biblium"

    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        compute_error = Msg("Computation error: {}")

    class Warning(OWWidget.Warning):
        no_biblium = Msg(
            "Biblium not installed — using fallback implementation. "
            "Install biblium>=2.16 for full functionality."
        )
        missing_columns = Msg("Required column(s) not found: {}")

    class Information(OWWidget.Information):
        computed = Msg("Processed {:,} documents")

    def __init__(self):
        super().__init__()
        self.biblium = get_biblium()
        self.has_biblium = self.biblium is not None
        if not self.has_biblium:
            self.Warning.no_biblium()

    # ------------------------------------------------------------------
    # Helper methods (instance-level wrappers around module functions)
    # ------------------------------------------------------------------

    @staticmethod
    def _table_to_df(table: Table | None) -> pd.DataFrame:
        return table_to_df(table)

    @staticmethod
    def _df_to_table(df: pd.DataFrame | None,
                     prefer_strings: bool = False) -> Table | None:
        return df_to_table(df, prefer_strings=prefer_strings)

    @staticmethod
    def _find_column(df: pd.DataFrame | None,
                     candidates: str | Sequence[str]) -> str | None:
        return find_column(df, candidates)

    @staticmethod
    def _detect_separator(series: pd.Series, default: str = "; ") -> str:
        return detect_separator(series, default=default)

    @staticmethod
    def _split_list(value: Any, sep: str) -> list[str]:
        return split_list_cell(value, sep)

    # ------------------------------------------------------------------
    # Lifecycle hooks subclasses commonly need
    # ------------------------------------------------------------------

    def clear_messages(self):
        """Clear Error/Warning/Information messages, preserving no_biblium."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        if not self.has_biblium:
            self.Warning.no_biblium()


# =============================================================================
# Consistent group colours (stable per group *name*, shared across widgets)
# =============================================================================

GROUP_PALETTE = [
    "#4a90d9", "#e8743b", "#5aa454", "#c0504d", "#8064a2", "#1aa8a8",
    "#d9a441", "#9b59b6", "#16a085", "#e74c3c", "#2c3e50", "#f39c12",
]


# User-assigned overrides (set in Setup Groups, read everywhere). Lives at
# module level so all widgets in the same Orange session share it.
_USER_GROUP_COLORS: dict = {}


def set_group_color(name, color) -> None:
    """Assign a colour to a group name (overrides the deterministic one)."""
    if color:
        _USER_GROUP_COLORS[str(name)] = str(color)


def clear_group_colors() -> None:
    _USER_GROUP_COLORS.clear()


def group_color(name) -> str:
    """Return a hex colour for a group name.

    Uses the user-assigned override if set (see set_group_color); otherwise a
    deterministic (crc32-based) colour so the same group gets the same colour
    in every widget that displays groups, regardless of group ordering.
    """
    key = str(name)
    if key in _USER_GROUP_COLORS:
        return _USER_GROUP_COLORS[key]
    idx = zlib.crc32(key.encode("utf-8")) % len(GROUP_PALETTE)
    return GROUP_PALETTE[idx]



__all__ = [
    "BaseBibliumWidget",
    "GROUP_PALETTE",
    "group_color",
    "set_group_color",
    "clear_group_colors",
    "COLUMN_PATTERNS",
    "count_items_in_list_column",
    "detect_separator",
    "df_to_table",
    "find_column",
    "fmt_value",
    "get_biblium",
    "get_biblium_submodule",
    "has_biblium",
    "safe_numeric",
    "split_list_cell",
    "table_to_df",
    "unique_items_in_list_column",
]
