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
from pathlib import Path

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLabel

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
    url_prefix = "https://openalex.org/"
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
    
    actual_id_col = find_col([id_col, "id", "unique-id", "ids.openalex", "work_id", "OpenAlex ID"])
    actual_refs_col = find_col([refs_col, "referenced_works", "References", "references"])
    actual_title_col = find_col([title_col, "title", "Title", "display_name"])
    actual_year_col = find_col([year_col, "publication_year", "Year", "year", "PY"])
    actual_cite_col = find_col([citations_col, "cited_by_count", "Cited by", "Times Cited", "TC"])
    
    if actual_id_col is None:
        raise ValueError(f"No ID column found. Available: {list(df.columns)[:10]}")
    if actual_refs_col is None:
        raise ValueError(f"No references column found. Available: {list(df.columns)[:10]}")
    
    if verbose:
        print(f"OpenAlex Citation Network")
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
    
    G = nx.DiGraph()
    for idx, doc_id in enumerate(doc_ids):
        G.add_node(doc_id, title=titles[idx], index=idx)
    
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
    priority = 110
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
    
    want_main_area = False
    
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
        self._setup_gui()
    
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
        
        gui.button(self.controlArea, self, "Build Network", callback=self._build_network)
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
            
            # Detect data source and build network
            if self._is_openalex():
                self.Information.using_openalex()
                G, stats = build_openalex_citation_network(
                    df,
                    keep_largest_component=True,
                    verbose=True
                )
            else:
                self.Information.using_fuzzy(self.match_threshold)
                
                title_col = self._find_column("Title", "TI", "title", "display_name")
                ref_col = self._find_column("References", "Cited References", "CR")
                id_col = self._find_column("EID", "DOI", "UT", "id", "unique-id")
                
                if not title_col:
                    self.Error.build_failed(f"No Title column found")
                    self._clear_outputs()
                    return
                if not ref_col:
                    self.Error.build_failed(f"No References column found")
                    self._clear_outputs()
                    return
                
                if not id_col:
                    df["_doc_id"] = [f"DOC_{i}" for i in range(len(df))]
                    id_col = "_doc_id"
                
                G, stats = build_fuzzy_citation_network(
                    df, title_col, ref_col, id_col,
                    threshold=self.match_threshold,
                    verbose=True
                )
            
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
            
            self._send_outputs(G, main_path_nodes, edge_weights)
            
        except Exception as e:
            logger.exception(f"Build failed: {e}")
            self.Error.build_failed(str(e))
            self._clear_outputs()
    
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
        meta_vars = [StringVariable("ID"), StringVariable("Title")]
        
        X = np.zeros((n, 4))
        metas = []
        
        for i, node in enumerate(nodes):
            data = G.nodes[node]
            X[i, 0] = data.get("citations", 0)
            X[i, 1] = G.in_degree(node)
            X[i, 2] = G.out_degree(node)
            X[i, 3] = data.get("year", 0)
            metas.append([str(node), str(data.get("title", node))[:80]])
        
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
