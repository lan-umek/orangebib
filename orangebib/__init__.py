# Orange3-Biblium package
"""Orange3-Biblium — bibliometric analysis widgets for Orange3.

Powered by Biblium 2.16+. Widgets degrade gracefully when biblium
isn't installed (a fallback implementation is used where possible).
"""

__version__ = "0.2.0"

from orangebib.base import (
    BaseBibliumWidget,
    COLUMN_PATTERNS,
    count_items_in_list_column,
    detect_separator,
    df_to_table,
    find_column,
    fmt_value,
    get_biblium,
    group_color,
    set_group_color,
    get_biblium_submodule,
    has_biblium,
    safe_numeric,
    split_list_cell,
    table_to_df,
    unique_items_in_list_column,
)

__all__ = [
    "__version__",
    "BaseBibliumWidget",
    "COLUMN_PATTERNS",
    "count_items_in_list_column",
    "detect_separator",
    "df_to_table",
    "find_column",
    "fmt_value",
    "get_biblium",
    "group_color",
    "set_group_color",
    "get_biblium_submodule",
    "has_biblium",
    "safe_numeric",
    "split_list_cell",
    "table_to_df",
    "unique_items_in_list_column",
]
