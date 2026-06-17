# -*- coding: utf-8 -*-
"""
Citation Network Widget
=======================
Orange widget using Biblium's citation network implementation.
- OpenAlex: Exact ID matching (no threshold needed)
- Scopus/WoS: Fuzzy title matching
"""

import logging
import re
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (QLabel, QPushButton, QComboBox, QCheckBox,
                             QHBoxLayout, QFileDialog, QApplication)
from AnyQt.QtCore import Qt
try:
    import pyqtgraph as pg
    HAS_PG = True
except Exception:  # noqa: BLE001
    pg = None
    HAS_PG = False

from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

try:
    from orangecontrib.network import Network
    HAS_NETWORK = True
except ImportError:
    HAS_NETWORK = False
    Network = None

try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    sp = None

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

# Fuzzy matching for Scopus/WoS
try:
    from thefuzz import fuzz
    HAS_FUZZ = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz
        HAS_FUZZ = True
    except ImportError:
        try:
            from rapidfuzz import fuzz
            HAS_FUZZ = True
        except ImportError:
            HAS_FUZZ = False


MAIN_PATH_METHODS = [
    ("SPC: Search Path Count", "SPC"),
    ("SPLC: Normalized by path length", "SPLC"),
    ("SPNP: Normalized by node pairs", "SPNP"),
]


# =============================================================================
# OpenAlex Citation Network (exact ID matching)
# =============================================================================

def build_openalex_citation_network(
    df: pd.DataFrame,
    id_col: str = "unique-id",
    refs_col: str = "referenced_works", 
    title_col: str = "title",
    year_col: str = "publication_year",
    citations_col: str = "cited_by_count",
    sep: str = "|",
    keep_largest_component: bool = True,
    verbose: bool = False,
) -> Tuple[nx.DiGraph, Dict]:
    """
    Build citation network from OpenAlex data using exact ID matching.
    No fuzzy matching needed - OpenAlex provides exact work IDs.
    """
    tail_pat = re.compile(r"(W\d+)$")
    
    def to_short(s: str) -> str:
        """Normalize OpenAlex ID to short form (W123456789)."""
        m = tail_pat.search(str(s))
        return m.group(1) if m else str(s)
    
    # Find columns with fallbacks
    def find_col(options):
        for opt in options:
            if opt in df.columns:
                return opt
        return None
    
    # references first — their ID space dictates which id column to use
    actual_refs_col = find_col([refs_col, "oa_referenced_works", "referenced_works",
                                "References", "references", "Cited References", "CR"])
    oa_refs = actual_refs_col in ("oa_referenced_works", "referenced_works")
    if oa_refs:
        # references are OpenAlex work IDs -> the document id MUST also be its
        # OpenAlex work id (matching DOIs against OpenAlex IDs yields no edges,
        # which is why Scopus-enriched-with-OpenAlex looked like isolated nodes)
        actual_id_col = find_col([id_col, "oa_openalex_id", "openalex_id",
                                  "ids.openalex", "OpenAlex ID", "oa_id",
                                  "work_id", "id", "unique-id"])
    else:
        actual_id_col = find_col([id_col, "id", "unique-id", "DOI", "doi",
                                  "EID", "UT", "PubMed ID", "pmid"])
    # auto-detect the reference separator
    if actual_refs_col is not None:
        _samp = " ".join(df[actual_refs_col].dropna().astype(str).head(20))
        sep = next((c for c in ["||", "|", "; ", ";", ", ", " "] if c in _samp), sep)
    actual_title_col = find_col([title_col, "title", "Title", "display_name"])
    actual_year_col = find_col([year_col, "publication_year", "Year", "year", "PY"])
    actual_cite_col = find_col([citations_col, "cited_by_count", "Cited by", "Times Cited", "TC"])
    auth_col = find_col(["Authors", "Author", "Author full names", "AU", "authorships.author.display_name"])
    src_col = find_col(["Source title", "Source", "Journal", "SO", "host_venue.display_name"])
    doi_col = find_col(["DOI", "doi"])
    eid_col = find_col(["EID", "UT"])
    wid_col = find_col(["oa_openalex_id", "openalex_id", "OpenAlex ID", "ids.openalex"])

    def _first_author(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        s0 = str(v)
        for sp in (";", "|", ","):
            if sp in s0:
                return s0.split(sp)[0].strip()
        return s0.strip()
    
    if actual_id_col is None:
        if oa_refs:
            raise ValueError(
                "References are OpenAlex IDs but no OpenAlex work-id column "
                "(e.g. 'oa_openalex_id') was found. Re-run the OpenAlex "
                "Enrichment so each document keeps its OpenAlex ID.")
        raise ValueError(f"No ID column found. Available: {list(df.columns)[:10]}")
    if actual_refs_col is None:
        raise ValueError(f"No references column found. Available: {list(df.columns)[:10]}")
    
    if verbose:
        print("OpenAlex Citation Network")
        print(f"  ID column: {actual_id_col}")
        print(f"  References column: {actual_refs_col}")
        print(f"  Documents: {len(df)}")
    
    # Normalize IDs and build set
    ids = df[actual_id_col].dropna().astype(str).str.strip().map(to_short)
    id_set = set(ids)
    
    # Build ID to row data mapping
    id_to_data = {}
    for idx, row in df.iterrows():
        raw_id = row.get(actual_id_col)
        if pd.isna(raw_id):
            continue
        node_id = to_short(str(raw_id).strip())
        
        title = row.get(actual_title_col, "") if actual_title_col else ""
        year = row.get(actual_year_col, 2000) if actual_year_col else 2000
        citations = row.get(actual_cite_col, 0) if actual_cite_col else 0
        
        id_to_data[node_id] = {
            "title": str(title)[:100] if pd.notna(title) else node_id,
            "year": int(year) if pd.notna(year) and year else 2000,
            "citations": int(citations) if pd.notna(citations) else 0,
            "author": _first_author(row.get(auth_col)) if auth_col else "",
            "source": (str(row.get(src_col))[:60] if src_col and pd.notna(row.get(src_col)) else ""),
            "doi": (str(row.get(doi_col)) if doi_col and pd.notna(row.get(doi_col)) else ""),
            "eid": (str(row.get(eid_col)) if eid_col and pd.notna(row.get(eid_col)) else ""),
            "workid": (str(row.get(wid_col)) if wid_col and pd.notna(row.get(wid_col)) else node_id),
        }
    
    # Process references - explode pipe-separated IDs
    edges_list = []
    total_refs = 0
    
    for idx, row in df.iterrows():
        citing_id = row.get(actual_id_col)
        refs = row.get(actual_refs_col)
        
        if pd.isna(citing_id) or pd.isna(refs):
            continue
        
        citing_id = to_short(str(citing_id).strip())
        
        # Split references by separator (pipe for OpenAlex)
        ref_list = [r.strip() for r in str(refs).split(sep) if r.strip()]
        total_refs += len(ref_list)
        
        for ref in ref_list:
            cited_id = to_short(ref)
            # Only keep edges where cited document is in our dataset
            if cited_id in id_set and cited_id != citing_id:
                edges_list.append((citing_id, cited_id))
    
    # Build graph
    G = nx.DiGraph()
    
    for node_id in id_set:
        data = id_to_data.get(node_id, {"title": node_id, "year": 2000, "citations": 0})
        G.add_node(node_id, **data)
    
    G.add_edges_from(edges_list)
    
    # Remove duplicates
    G = nx.DiGraph(G)  # This removes duplicate edges
    
    stats = {
        "total_documents": len(df),
        "total_references": total_refs,
        "internal_refs": len(edges_list),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
    }
    
    if verbose:
        print(f"  Total references: {total_refs}")
        print(f"  Internal (within corpus): {len(edges_list)}")
        print(f"  Edges after dedup: {G.number_of_edges()}")
    
    # Keep largest component
    if keep_largest_component and G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        largest = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    
    stats["nodes_final"] = G.number_of_nodes()
    stats["edges_final"] = G.number_of_edges()
    stats["match_rate"] = len(edges_list) / total_refs if total_refs > 0 else 0
    
    if verbose:
        print(f"  Final: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G, stats


# =============================================================================
# Scopus/WoS Citation Network (fuzzy title matching)
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text: lowercase, remove punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[\W_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_title_from_reference(ref: str) -> Optional[str]:
    """Extract title from reference string (Author, Title, Journal, ...)."""
    parts = [p.strip() for p in ref.split(",")]
    if len(parts) >= 2:
        for i, part in enumerate(parts[1:], start=1):
            if re.match(r"^\d+$", part):
                continue
            if re.match(r"^pp?\.\s*\d+", part):
                continue
            if re.match(r"^\(\d{4}\)$", part):
                continue
            if len(part) < 10:
                continue
            return part
    return None


def build_fuzzy_citation_network(
    df: pd.DataFrame,
    title_col: str,
    ref_col: str,
    id_col: str,
    threshold: int = 80,
    verbose: bool = False,
) -> Tuple[nx.DiGraph, Dict]:
    """Build citation network using fuzzy title matching (for Scopus/WoS)."""
    if not HAS_FUZZ:
        raise ImportError("thefuzz/fuzzywuzzy required for fuzzy matching")
    
    titles = df[title_col].tolist()
    doc_ids = df[id_col].tolist()
    norm_titles = [normalize_text(str(t)) if pd.notna(t) else "" for t in titles]

    title_to_idx = {}
    for idx, nt in enumerate(norm_titles):
        if nt and nt not in title_to_idx:
            title_to_idx[nt] = idx

    def _col(*names):
        for n in names:
            if n in df.columns:
                return n
        low = {str(c).lower(): c for c in df.columns}
        for n in names:
            if n.lower() in low:
                return low[n.lower()]
        return None

    a_col = _col("Authors", "Author", "Author full names", "AU")
    s_col = _col("Source title", "Source", "Journal", "SO")
    y_col = _col("Year", "Publication Year", "PY")
    c_col = _col("Cited by", "Times Cited", "TC", "cited_by_count")
    d_col = _col("DOI", "doi"); e_col = _col("EID", "UT")

    def _fa(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        s0 = str(v)
        for sp in (";", "|", ","):
            if sp in s0:
                return s0.split(sp)[0].strip()
        return s0.strip()

    def _get(idx, col):
        if col is None:
            return ""
        try:
            v = df[col].iloc[idx]
            return "" if pd.isna(v) else v
        except Exception:  # noqa: BLE001
            return ""

    G = nx.DiGraph()
    for idx, doc_id in enumerate(doc_ids):
        yr = _get(idx, y_col)
        cc = _get(idx, c_col)
        G.add_node(doc_id, title=titles[idx], index=idx,
                   year=int(float(yr)) if str(yr).strip() not in ("", "nan") else 2000,
                   citations=int(float(cc)) if str(cc).strip() not in ("", "nan") else 0,
                   author=_fa(_get(idx, a_col)),
                   source=str(_get(idx, s_col))[:60],
                   doi=str(_get(idx, d_col)), eid=str(_get(idx, e_col)),
                   workid=str(doc_id))
    
    total_refs = 0
    matched_refs = 0
    
    if verbose:
        print(f"Processing {len(df)} documents with fuzzy matching...")
    
    for idx, row in df.iterrows():
        refs = row[ref_col]
        source_id = row[id_col]
        
        if not isinstance(refs, str) or pd.isna(refs):
            continue
        
        ref_list = [r.strip() for r in refs.split(";") if r.strip()]
        
        for ref in ref_list:
            total_refs += 1
            matched = False
            
            extracted_title = extract_title_from_reference(ref)
            search_texts = []
            if extracted_title:
                search_texts.append(normalize_text(extracted_title))
            search_texts.append(normalize_text(ref))
            
            for search_text in search_texts:
                if matched:
                    break
                
                if search_text in title_to_idx:
                    tgt_idx = title_to_idx[search_text]
                    tgt_id = doc_ids[tgt_idx]
                    if tgt_id != source_id:
                        G.add_edge(source_id, tgt_id)
                        matched = True
                        matched_refs += 1
                        break
                
                best_score, best_idx = 0, None
                for j, nt in enumerate(norm_titles):
                    if not nt or j == idx:
                        continue
                    score = fuzz.token_set_ratio(search_text, nt)
                    if score > best_score:
                        best_score, best_idx = score, j
                
                if best_score >= threshold and best_idx is not None:
                    tgt_id = doc_ids[best_idx]
                    G.add_edge(source_id, tgt_id)
                    matched = True
                    matched_refs += 1
                    break
    
    G.remove_edges_from(nx.selfloop_edges(G))
    
    stats = {
        "total_documents": len(df),
        "total_references": total_refs,
        "matched_references": matched_refs,
        "match_rate": matched_refs / total_refs if total_refs > 0 else 0,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
    }
    
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    
    stats["nodes_final"] = G.number_of_nodes()
    stats["edges_final"] = G.number_of_edges()
    
    return G, stats


# =============================================================================
# Main Path Analysis
# =============================================================================

def compute_main_path(G: nx.DiGraph, method: str = "SPC") -> Tuple[List, Dict]:
    """Compute main path using SPC/SPLC/SPNP weights."""
    if not HAS_NETWORKX or G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return [], {}
    
    if not nx.is_directed_acyclic_graph(G):
        G = nx.condensation(G)
    
    sources = [n for n in G.nodes() if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    if not sources or not sinks:
        return [], {}
    
    paths_from_source = {n: 0 for n in G.nodes()}
    for source in sources:
        paths_from_source[source] = 1
    for node in nx.topological_sort(G):
        for pred in G.predecessors(node):
            paths_from_source[node] += paths_from_source[pred]
    
    paths_to_sink = {n: 0 for n in G.nodes()}
    for sink in sinks:
        paths_to_sink[sink] = 1
    for node in reversed(list(nx.topological_sort(G))):
        for succ in G.successors(node):
            paths_to_sink[node] += paths_to_sink[succ]
    
    total_paths = sum(paths_from_source[sink] for sink in sinks)
    n_pairs = len(sources) * len(sinks)
    
    edge_weights = {}
    for u, v in G.edges():
        spc = paths_from_source[u] * paths_to_sink[v]
        if method == "SPC":
            edge_weights[(u, v)] = spc
        elif method == "SPLC":
            edge_weights[(u, v)] = spc / total_paths if total_paths > 0 else 0
        else:
            edge_weights[(u, v)] = spc / n_pairs if n_pairs > 0 else 0
    
    max_weight = {n: float("-inf") for n in G.nodes()}
    predecessor = {n: None for n in G.nodes()}
    for source in sources:
        max_weight[source] = 0
    
    for node in nx.topological_sort(G):
        for succ in G.successors(node):
            edge_w = edge_weights.get((node, succ), 0)
            new_weight = max_weight[node] + edge_w
            if new_weight > max_weight[succ]:
                max_weight[succ] = new_weight
                predecessor[succ] = node
    
    best_sink = max(sinks, key=lambda s: max_weight[s])
    if max_weight[best_sink] == float("-inf"):
        return [], edge_weights
    
    path = []
    node = best_sink
    while node is not None:
        path.append(node)
        node = predecessor[node]
    
    return list(reversed(path)), edge_weights


# =============================================================================
# Widget
# =============================================================================

class OWCitationNetwork(OWWidget):
    """Build document citation networks and main paths."""
    
    name = "Citation Network"
    description = "Document citation network with main path analysis"
    icon = "icons/citation_network.svg"
    priority = 420
    keywords = ["citation", "network", "main path"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table)
    
    class Outputs:
        network = Output("Network", Network) if HAS_NETWORK else Output("Network", object, auto_summary=False)
        main_path = Output("Main Path", Network) if HAS_NETWORK else Output("Main Path", object, auto_summary=False)
        node_data = Output("Node Data", Table)
        edge_data = Output("Edge Data", Table)
        main_path_data = Output("Main Path Data", Table)
    
    min_citations = settings.Setting(0)
    top_n_docs = settings.Setting(50)
    match_threshold = settings.Setting(80)
    main_path_method = settings.Setting(0)
    layout_index = settings.Setting(1)  # default: hierarchical by year
    node_size_by = settings.Setting(0)   # 0 citations, 1 in-degree, 2 out-degree, 3 uniform
    curved_edges = settings.Setting(True)
    show_labels = settings.Setting(True)
    label_field = settings.Setting(0)
    highlight_main_path = settings.Setting(True)

    want_main_area = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        build_failed = Msg("{}")
        no_network_addon = Msg("Orange Network add-on not installed")
    
    class Warning(OWWidget.Warning):
        no_edges = Msg("No citation links found")
        no_main_path = Msg("Could not compute main path")
    
    class Information(OWWidget.Information):
        network_built = Msg("{} nodes, {} edges, main path: {} nodes")
        using_openalex = Msg("OpenAlex detected - using exact ID matching")
        using_fuzzy = Msg("Using fuzzy title matching (threshold {}%)")
    
    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._columns = []
        self._G = None
        self._main_path = []
        self._setup_gui()
        if HAS_PG:
            self.graph = pg.PlotWidget(background="w")
            self.graph.hideAxis("bottom"); self.graph.hideAxis("left")
            self.graph.setAspectLocked(False)
            self.mainArea.layout().addWidget(self.graph)
        else:
            self.mainArea.layout().addWidget(
                QLabel("pyqtgraph not available — plotting disabled"))
    
    def _setup_gui(self):
        box = gui.widgetBox(self.controlArea, "Parameters")
        gui.spin(box, self, "min_citations", 0, 1000, label="Min Citations:",
                 callback=self._on_change)
        gui.spin(box, self, "top_n_docs", 5, 500, label="Top N Documents:",
                 callback=self._on_change)
        gui.spin(box, self, "match_threshold", 50, 100, 
                 label="Match Threshold (%):",
                 callback=self._on_change,
                 tooltip="For Scopus/WoS only. OpenAlex uses exact matching.")
        
        box = gui.widgetBox(self.controlArea, "Main Path Analysis")
        gui.comboBox(box, self, "main_path_method",
                     items=[m[0] for m in MAIN_PATH_METHODS],
                     callback=self._on_change)
        
        info = QLabel("<small><i>SPC: Search Path Count<br>"
                      "SPLC: Normalized by path length<br>"
                      "SPNP: Normalized by node pairs</i></small>")
        info.setStyleSheet("color: #666;")
        box.layout().addWidget(info)
        
        abox = gui.widgetBox(self.controlArea, "Plot aesthetics")
        gui.comboBox(abox, self, "layout_index", label="Layout:",
                     orientation="horizontal",
                     items=["Spring (force)", "Hierarchical (by year)",
                            "Circular", "Kamada-Kawai"],
                     callback=self._redraw, sendSelectedValue=False)
        gui.comboBox(abox, self, "node_size_by", label="Node size:",
                     orientation="horizontal",
                     items=["Citations", "In-degree", "Out-degree", "Uniform"],
                     callback=self._redraw, sendSelectedValue=False)
        gui.checkBox(abox, self, "curved_edges", "Curved edges", callback=self._redraw)
        gui.checkBox(abox, self, "show_labels", "Show labels", callback=self._redraw)
        gui.comboBox(abox, self, "label_field", label="Label format:",
                     orientation="horizontal",
                     items=["Author, Year", "Author, Year, Source",
                            "Author, Year, Title, Source", "Identifier (DOI/EID/WorkID)",
                            "Title"],
                     callback=self._redraw, sendSelectedValue=False)
        gui.checkBox(abox, self, "highlight_main_path", "Highlight main path",
                     callback=self._redraw)

        gui.button(self.controlArea, self, "Build Network", callback=self._build_network)

        ebox = gui.widgetBox(self.controlArea, "Export (Pajek)")
        row = QHBoxLayout()
        b1 = QPushButton(".net"); b1.clicked.connect(lambda: self._export_pajek("net"))
        b2 = QPushButton(".clu"); b2.clicked.connect(lambda: self._export_pajek("clu"))
        b3 = QPushButton(".vec"); b3.clicked.connect(lambda: self._export_pajek("vec"))
        for b in (b1, b2, b3):
            row.addWidget(b)
        ebox.layout().addLayout(row)
        ball = QPushButton("Save all (.net + .clu + .vec)")
        ball.clicked.connect(lambda: self._export_pajek("all"))
        ebox.layout().addWidget(ball)
        self.controlArea.layout().addStretch()
    
    def _on_change(self):
        pass
    
    def _find_column(self, *names) -> Optional[str]:
        for name in names:
            if name in self._columns:
                return name
            for col in self._columns:
                if name.lower() == col.lower():
                    return col
        for name in names:
            for col in self._columns:
                if name.lower() in col.lower():
                    return col
        return None
    
    def _is_openalex(self) -> bool:
        """Check if data is from OpenAlex."""
        # Check for OpenAlex-specific columns
        openalex_indicators = [
            "referenced_works", "cited_by_api_url", "ids.openalex",
            "OpenAlex ID", "cited_by_count"
        ]
        for col in self._columns:
            if col in openalex_indicators:
                return True
            if "openalex" in col.lower():
                return True
        
        # Check if References column contains OpenAlex URLs
        refs_col = self._find_column("References", "referenced_works")
        if refs_col:
            sample = self._df[refs_col].dropna()
            if len(sample) > 0:
                first_ref = str(sample.iloc[0])
                if "openalex.org" in first_ref or first_ref.startswith("https://openalex"):
                    return True
                # Check for pipe-separated W* IDs
                if "|" in first_ref and re.search(r"W\d+", first_ref):
                    return True
        
        return False
    
    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        self._data = data
        self._df = None
        self._columns = []
        
        if data is None:
            self.Error.no_data()
            self._clear_outputs()
            return
        
        d = {}
        for var in data.domain.attributes:
            d[var.name] = data.get_column(var)
        for var in data.domain.metas:
            d[var.name] = data.get_column(var)
        for var in data.domain.class_vars:
            d[var.name] = data.get_column(var)
        self._df = pd.DataFrame(d)
        self._columns = list(self._df.columns)
    
    def _clear_outputs(self):
        self.Outputs.network.send(None)
        self.Outputs.main_path.send(None)
        self.Outputs.node_data.send(None)
        self.Outputs.edge_data.send(None)
        self.Outputs.main_path_data.send(None)
    
    def _build_network(self):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None:
            self._clear_outputs()
            return
        
        try:
            df = self._df.copy()
            
            # Filter by citations
            cite_col = self._find_column("Cited by", "cited_by_count", "Times Cited", "TC")
            if self.min_citations > 0 and cite_col:
                df = df[df[cite_col].fillna(0).astype(float) >= self.min_citations]
            
            # Top N
            if len(df) > self.top_n_docs:
                if cite_col:
                    df = df.nlargest(self.top_n_docs, cite_col)
                else:
                    df = df.head(self.top_n_docs)
            
            # Detect data source and build network. Prefer OpenAlex exact
            # matching, but fall back to fuzzy title matching if the OpenAlex
            # path cannot find an ID/references column.
            G = stats = None
            if self._is_openalex():
                try:
                    self.Information.using_openalex()
                    G, stats = build_openalex_citation_network(
                        df, keep_largest_component=True, verbose=True)
                except Exception as oa_exc:  # noqa: BLE001
                    logger.warning("OpenAlex citation path failed: %s", oa_exc)
                    G = None
            if G is None:
                self.Information.using_fuzzy(self.match_threshold)
                title_col = self._find_column("Title", "TI", "title", "display_name")
                ref_col = self._find_column(
                    "References", "Cited References", "CR", "oa_referenced_works",
                    "referenced_works")
                id_col = self._find_column("EID", "DOI", "UT", "id", "unique-id",
                                           "oa_id", "PubMed ID")
                if not title_col:
                    self.Error.build_failed("No Title column found")
                    self._clear_outputs()
                    return
                if not ref_col:
                    self.Error.build_failed(
                        "No references column found. The citation network needs a "
                        "References column or OpenAlex 'referenced_works' "
                        "(enrich the data with OpenAlex, including referenced works).")
                    self._clear_outputs()
                    return
                if not id_col:
                    df["_doc_id"] = [f"DOC_{i}" for i in range(len(df))]
                    id_col = "_doc_id"
                G, stats = build_fuzzy_citation_network(
                    df, title_col, ref_col, id_col,
                    threshold=self.match_threshold, verbose=True)
            
            if G.number_of_nodes() == 0:
                self.Error.build_failed("No connected documents found")
                self._clear_outputs()
                return
            
            if G.number_of_edges() == 0:
                self.Warning.no_edges()
            
            # Main path
            method = MAIN_PATH_METHODS[self.main_path_method][1]
            main_path_nodes, edge_weights = compute_main_path(G, method)
            
            if G.number_of_edges() > 0 and not main_path_nodes:
                self.Warning.no_main_path()
            
            self.Information.network_built(
                G.number_of_nodes(), G.number_of_edges(), len(main_path_nodes)
            )
            
            self._G = G
            self._main_path = main_path_nodes or []
            self._send_outputs(G, main_path_nodes, edge_weights)
            self._redraw()

        except Exception as e:
            logger.exception(f"Build failed: {e}")
            self.Error.build_failed(str(e))
            self._clear_outputs()
    
    # ----------------------------------------------------------- plotting
    def _layout(self, G, nodes):
        try:
            if self.layout_index == 1:   # hierarchical / chronological by year
                years = {n: float(G.nodes[n].get("year", 0) or 0) for n in nodes}
                if all(years[n] == 0 for n in nodes):
                    return nx.spring_layout(G, seed=42)
                # group by year; within a year, order nodes by the average year
                # of their neighbours so citation arrows mostly flow forward and
                # cross less.
                cols = {}
                for n in nodes:
                    cols.setdefault(years[n], []).append(n)
                def _key(n):
                    nb = list(G.predecessors(n)) + list(G.successors(n))
                    nb_years = [years[m] for m in nb if years.get(m)]
                    return (np.mean(nb_years) if nb_years else years[n],
                            -(G.in_degree(n)))
                pos = {}
                yrs = sorted(cols)
                vgap = 1.6  # vertical gap between stacked nodes (room for labels)
                for yr in yrs:
                    members = sorted(cols[yr], key=_key)
                    k = len(members)
                    for i, n in enumerate(members):
                        pos[n] = (float(yr), (i - (k - 1) / 2.0) * vgap)
                return pos
            if self.layout_index == 2:
                return nx.circular_layout(G)
            if self.layout_index == 3:
                return nx.kamada_kawai_layout(G)
            return nx.spring_layout(G, seed=42, k=1.5 / (len(nodes) ** 0.5 or 1))
        except Exception:  # noqa: BLE001
            try:
                return nx.circular_layout(G)
            except Exception:  # noqa: BLE001
                return {n: (0.0, 0.0) for n in nodes}

    def _node_sizes(self, G, nodes):
        if self.node_size_by == 3:
            return [12.0] * len(nodes)
        if self.node_size_by == 1:
            vals = [G.in_degree(n) for n in nodes]
        elif self.node_size_by == 2:
            vals = [G.out_degree(n) for n in nodes]
        else:
            vals = [float(G.nodes[n].get("citations", 0) or 0) for n in nodes]
        vmax = max(vals) if vals else 1
        return [8 + 26 * (v / vmax if vmax else 0) for v in vals]

    def _node_label(self, n):
        d = self._G.nodes[n]
        au = str(d.get("author", "") or "").split(",")[0].split(";")[0].strip()
        yr = d.get("year", "") or ""
        yr = "" if yr in (0, "0") else str(int(yr)) if str(yr).isdigit() else str(yr)
        src = str(d.get("source", "") or "")
        ttl = str(d.get("title", n) or "")
        mode = self.label_field
        if mode == 0:
            base = ", ".join(x for x in (au, yr) if x)
        elif mode == 1:
            base = ", ".join(x for x in (au, yr, src[:24]) if x)
        elif mode == 2:
            base = ", ".join(x for x in (au, yr, ttl[:30], src[:20]) if x)
        elif mode == 3:
            base = (str(d.get("doi", "")) or str(d.get("eid", ""))
                    or str(d.get("workid", "")) or str(n))
        else:
            base = ttl[:30]
        return base or str(n)[:18]

    def _redraw(self):
        if not HAS_PG or not hasattr(self, "graph"):
            return
        self.graph.clear()
        # show a year axis only for the chronological layout
        if self.layout_index == 1:
            self.graph.showAxis("bottom")
            self.graph.setLabel("bottom", "Publication year")
        else:
            self.graph.hideAxis("bottom")
        G = self._G
        if G is None or G.number_of_nodes() == 0:
            return
        nodes = list(G.nodes())
        pos = self._layout(G, nodes)
        main_set = (set(zip(self._main_path[:-1], self._main_path[1:]))
                    if (self.highlight_main_path and self._main_path) else set())
        xs, ys = [], []
        mxs, mys = [], []
        for u, v in G.edges():
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]; x1, y1 = pos[v]
            if self.curved_edges:
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                dx, dy = x1 - x0, y1 - y0
                norm = (dx * dx + dy * dy) ** 0.5 or 1
                cx, cy = mx - dy / norm * 0.08 * norm, my + dx / norm * 0.08 * norm
                t = np.linspace(0, 1, 12)
                bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
                by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
                seg_x, seg_y = list(bx) + [np.nan], list(by) + [np.nan]
            else:
                seg_x, seg_y = [x0, x1, np.nan], [y0, y1, np.nan]
            if (u, v) in main_set:
                mxs += seg_x; mys += seg_y
            else:
                xs += seg_x; ys += seg_y
        if xs:
            self.graph.addItem(pg.PlotCurveItem(
                x=np.array(xs), y=np.array(ys), connect="finite",
                pen=pg.mkPen((150, 150, 150, 110), width=1)))
        if mxs:
            self.graph.addItem(pg.PlotCurveItem(
                x=np.array(mxs), y=np.array(mys), connect="finite",
                pen=pg.mkPen((217, 83, 79), width=2.5)))
        sizes = self._node_sizes(G, nodes)
        years = [float(G.nodes[n].get("year", 0) or 0) for n in nodes]
        ymin = min([y for y in years if y] or [0]); ymax = max(years or [1])
        spots = []
        for i, n in enumerate(nodes):
            if n not in pos:
                continue
            if n in (self._main_path or []):
                brush = pg.mkBrush(217, 83, 79)
            else:
                t = (years[i] - ymin) / (ymax - ymin) if ymax > ymin else 0.5
                brush = pg.mkBrush(int(60 + 150 * (1 - t)), int(120 + 80 * t),
                                   int(200 - 120 * t))
            spots.append({"pos": pos[n], "size": sizes[i], "brush": brush,
                          "pen": pg.mkPen("w", width=0.5),
                          "data": str(G.nodes[n].get("title", n))[:60]})
        sc = pg.ScatterPlotItem(hoverable=True, tip=None)
        sc.addPoints(spots)
        self.graph.addItem(sc)
        if self.show_labels:
            order = sorted(range(len(nodes)),
                           key=lambda i: -(G.nodes[nodes[i]].get("citations", 0) or 0))
            labelled = [nodes[i] for i in order[:25] if nodes[i] in pos]
            # small alternating vertical offset to reduce label collisions
            labelled.sort(key=lambda n: (pos[n][0], pos[n][1]))
            for j, n in enumerate(labelled):
                dy = 0.35 if (j % 2 == 0) else 0.9   # stagger above the node
                t = pg.TextItem(self._node_label(n)[:40],
                                color=(40, 40, 40), anchor=(0.5, 1.0))
                t.setPos(pos[n][0], pos[n][1] + dy)
                self.graph.addItem(t)
        self.graph.getViewBox().autoRange()

    def _export_pajek(self, kind):
        if self._G is None or self._G.number_of_nodes() == 0:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Pajek", "citation_network",
            "Pajek (*.net *.clu *.vec);;All files (*)")
        if not path:
            return
        base = path
        for ext in (".net", ".clu", ".vec"):
            if base.lower().endswith(ext):
                base = base[:-4]
        G = self._G
        nodes = list(G.nodes())
        idx = {n: i + 1 for i, n in enumerate(nodes)}
        mp = set(self._main_path or [])
        try:
            if kind in ("net", "all"):
                with open(base + ".net", "w", encoding="utf-8") as f:
                    f.write(f"*Vertices {len(nodes)}\n")
                    for n in nodes:
                        lbl = str(G.nodes[n].get("title", n)).replace('"', "'")[:60]
                        f.write(f'{idx[n]} "{lbl}"\n')
                    f.write("*Arcs\n")
                    for u, v in G.edges():
                        f.write(f"{idx[u]} {idx[v]} 1\n")
            if kind in ("clu", "all"):
                with open(base + ".clu", "w", encoding="utf-8") as f:
                    f.write(f"*Vertices {len(nodes)}\n")
                    for n in nodes:
                        f.write(f"{2 if n in mp else 1}\n")
            if kind in ("vec", "all"):
                with open(base + ".vec", "w", encoding="utf-8") as f:
                    f.write(f"*Vertices {len(nodes)}\n")
                    for n in nodes:
                        f.write(f"{float(G.nodes[n].get('citations', 0) or 0):.4f}\n")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pajek export failed")
            self.Error.build_failed(f"Export failed: {exc}")

    def _send_outputs(self, G, main_path_nodes, edge_weights):
        n = G.number_of_nodes()
        nodes = list(G.nodes())
        
        # Node data
        cont_vars = [
            ContinuousVariable("Citations"),
            ContinuousVariable("In_Degree"),
            ContinuousVariable("Out_Degree"),
            ContinuousVariable("Year"),
        ]
        meta_vars = [StringVariable("ID"), StringVariable("Label"),
                     StringVariable("Author"), StringVariable("Title"),
                     StringVariable("Source"), StringVariable("DOI/EID/WorkID")]

        X = np.zeros((n, 4))
        metas = []

        for i, node in enumerate(nodes):
            data = G.nodes[node]
            X[i, 0] = data.get("citations", 0)
            X[i, 1] = G.in_degree(node)
            X[i, 2] = G.out_degree(node)
            X[i, 3] = data.get("year", 0)
            ident = (str(data.get("doi", "")) or str(data.get("eid", ""))
                     or str(data.get("workid", "")) or str(node))
            metas.append([str(node), self._node_label(node),
                          str(data.get("author", "")),
                          str(data.get("title", node))[:120],
                          str(data.get("source", "")), ident])

        domain = Domain(cont_vars, metas=meta_vars)
        metas_arr = np.array(metas, dtype=object)
        node_table = Table.from_numpy(domain, X, metas=metas_arr)
        self.Outputs.node_data.send(node_table)
        
        # Edge data
        main_set = set(zip(main_path_nodes[:-1], main_path_nodes[1:])) if main_path_nodes else set()
        edges = []
        for u, v in G.edges():
            w = edge_weights.get((u, v), 1.0)
            mp = 1.0 if (u, v) in main_set else 0.0
            edges.append([str(u)[:40], str(v)[:40], float(w), mp])
        
        if edges:
            ea = np.array(edges, dtype=object)
            ed = Domain([ContinuousVariable("Weight"), ContinuousVariable("MainPath")],
                        metas=[StringVariable("From"), StringVariable("To")])
            et = Table.from_numpy(ed, ea[:, 2:4].astype(float), metas=ea[:, :2])
            self.Outputs.edge_data.send(et)
        else:
            self.Outputs.edge_data.send(None)
        
        # Main path data
        if main_path_nodes:
            mp = []
            for i, node in enumerate(main_path_nodes):
                title = G.nodes[node].get("title", str(node))[:60]
                year = G.nodes[node].get("year", 0)
                mp.append([str(node), title, year, i + 1])
            
            ma = np.array(mp, dtype=object)
            md = Domain([ContinuousVariable("Year"), ContinuousVariable("Order")],
                        metas=[StringVariable("ID"), StringVariable("Title")])
            mt = Table.from_numpy(md, ma[:, 2:4].astype(float), metas=ma[:, :2])
            self.Outputs.main_path_data.send(mt)
        else:
            self.Outputs.main_path_data.send(None)
        
        # Network outputs
        if HAS_NETWORK and HAS_SCIPY:
            try:
                node_to_idx = {node: i for i, node in enumerate(nodes)}
                matrix = np.zeros((n, n))
                for u, v in G.edges():
                    i, j = node_to_idx[u], node_to_idx[v]
                    matrix[i, j] = edge_weights.get((u, v), 1.0)
                
                sparse = sp.csr_matrix(matrix)
                network = Network(node_table, sparse)
                self.Outputs.network.send(network)
                
                if main_path_nodes:
                    mp_mat = np.zeros((n, n))
                    for k in range(len(main_path_nodes) - 1):
                        u, v = main_path_nodes[k], main_path_nodes[k + 1]
                        if u in node_to_idx and v in node_to_idx:
                            i, j = node_to_idx[u], node_to_idx[v]
                            mp_mat[i, j] = edge_weights.get((u, v), 1.0)
                    mp_net = Network(node_table, sp.csr_matrix(mp_mat))
                    self.Outputs.main_path.send(mp_net)
                else:
                    self.Outputs.main_path.send(None)
            except Exception as e:
                logger.exception(f"Network creation failed: {e}")
                self.Outputs.network.send(None)
                self.Outputs.main_path.send(None)
        else:
            if not HAS_NETWORK:
                self.Error.no_network_addon()
            self.Outputs.network.send(None)
            self.Outputs.main_path.send(None)


if __name__ == "__main__":
    WidgetPreview(OWCitationNetwork).run()
