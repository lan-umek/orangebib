# -*- coding: utf-8 -*-
"""
Network Co-occurrence Widget
============================
Orange widget for building bibliometric co-occurrence networks.
Outputs to Orange Network add-on for visualization.
Computes partition and vector-based node properties.
"""

import logging
import re
from typing import Optional, List, Dict, Set
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QWidget

from Orange.data import Table, Domain, StringVariable, ContinuousVariable, DiscreteVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)

# Try to import Orange Network
try:
    from orangecontrib.network import Network
    HAS_NETWORK = True
except ImportError:
    HAS_NETWORK = False
    Network = None

# Try to import scipy for sparse matrices
try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    sp = None

# Try to import networkx for metrics
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None


NETWORK_TYPES = [
    ("Author Keywords Co-occurrence", "author_keywords"),
    ("Index Keywords Co-occurrence", "index_keywords"),
    ("All Keywords Co-occurrence", "all_keywords"),
    ("Co-authorship", "coauthorship"),
    ("Reference Co-citation", "co_citation"),
    ("Source Co-citation", "source_cocitation"),
    ("Country Collaboration", "country_collab"),
    ("Institution Collaboration", "institution_collab"),
    ("Title N-grams Co-occurrence", "title_ngrams"),
    ("Abstract N-grams Co-occurrence", "abstract_ngrams"),
]


class NetworkBuilder:
    """Build co-occurrence networks from bibliographic data."""
    
    def __init__(self, df: pd.DataFrame, columns: List[str]):
        self.df = df
        self.columns = columns
        self._column_map = {col.lower(): col for col in columns}
    
    def _find_column(self, *names: str) -> Optional[str]:
        for name in names:
            if name in self.columns:
                return name
            if name.lower() in self._column_map:
                return self._column_map[name.lower()]
        for name in names:
            for col in self.columns:
                if name.lower() in col.lower():
                    return col
        return None
    
    def _extract_entities(self, column: str) -> Dict[int, List[str]]:
        doc_entities = {}
        for idx in range(len(self.df)):
            val = self.df.iloc[idx][column]
            if pd.isna(val):
                continue
            val_str = str(val).strip()
            if not val_str:
                continue
            if ";" in val_str:
                entities = [e.strip() for e in val_str.split(";") if e.strip()]
            elif "|" in val_str:
                entities = [e.strip() for e in val_str.split("|") if e.strip()]
            else:
                entities = [val_str]
            if entities:
                doc_entities[idx] = entities
        return doc_entities
    
    def _extract_ngrams(self, column: str, n: int = 2) -> Dict[int, List[str]]:
        doc_ngrams = {}
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
                    'that', 'these', 'those', 'it', 'its', 'we', 'our', 'their', 'there'}
        for idx in range(len(self.df)):
            val = self.df.iloc[idx][column]
            if pd.isna(val):
                continue
            text = str(val).lower()
            words = re.findall(r'\b[a-z]{3,}\b', text)
            words = [w for w in words if w not in stopwords]
            ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
            if ngrams:
                doc_ngrams[idx] = ngrams
        return doc_ngrams
    
    def _build_network(self, doc_entities: Dict[int, List[str]], top_n: int = 50,
                       min_occ: int = 2, min_edge: int = 1,
                       include_set: Set[str] = None, exclude_set: Set[str] = None,
                       include_regex: str = None, exclude_regex: str = None,
                       normalize: bool = False):
        
        entity_counts = Counter()
        for entities in doc_entities.values():
            entity_counts.update(entities)
        
        filtered = set()
        for entity, count in entity_counts.items():
            if count < min_occ:
                continue
            if include_set and entity.lower() not in include_set:
                continue
            if exclude_set and entity.lower() in exclude_set:
                continue
            if include_regex:
                try:
                    if not re.search(include_regex, entity, re.IGNORECASE):
                        continue
                except:
                    pass
            if exclude_regex:
                try:
                    if re.search(exclude_regex, entity, re.IGNORECASE):
                        continue
                except:
                    pass
            filtered.add(entity)
        
        top_entities = [e for e, _ in entity_counts.most_common() if e in filtered][:top_n]
        if not top_entities:
            return [], np.array([]), {'error': 'No entities after filtering'}
        
        top_set = set(top_entities)
        n_nodes = len(top_entities)
        entity_idx = {e: i for i, e in enumerate(top_entities)}
        
        matrix = np.zeros((n_nodes, n_nodes), dtype=float)
        for entities in doc_entities.values():
            doc_filtered = [e for e in entities if e in top_set]
            for e1, e2 in combinations(set(doc_filtered), 2):
                i, j = entity_idx[e1], entity_idx[e2]
                matrix[i, j] += 1
                matrix[j, i] += 1
        
        matrix[matrix < min_edge] = 0
        if normalize and matrix.max() > 0:
            matrix = matrix / matrix.max()
        
        # Compute node properties
        node_props = self._compute_node_properties(top_entities, matrix, entity_counts)
        
        return top_entities, matrix, node_props
    
    def _compute_node_properties(self, nodes: List[str], matrix: np.ndarray, 
                                  entity_counts: Counter) -> Dict:
        """Compute partition and vector-based node properties."""
        n = len(nodes)
        
        props = {
            'occurrences': [entity_counts[e] for e in nodes],
            'degree': [0] * n,
            'weighted_degree': [0.0] * n,
            'betweenness': [0.0] * n,
            'closeness': [0.0] * n,
            'eigenvector': [0.0] * n,
            'pagerank': [0.0] * n,
            'clustering': [0.0] * n,
            'community': [0] * n,
        }
        
        if not HAS_NETWORKX or n == 0:
            return props
        
        # Build networkx graph
        G = nx.Graph()
        for i, node in enumerate(nodes):
            G.add_node(i, label=node)
        
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j] > 0:
                    G.add_edge(i, j, weight=matrix[i, j])
        
        if G.number_of_edges() == 0:
            return props
        
        # Degree
        for i in range(n):
            props['degree'][i] = G.degree(i)
            props['weighted_degree'][i] = G.degree(i, weight='weight')
        
        # Betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(G, weight='weight')
            for i in range(n):
                props['betweenness'][i] = betweenness.get(i, 0)
        except:
            pass
        
        # Closeness centrality
        try:
            closeness = nx.closeness_centrality(G)
            for i in range(n):
                props['closeness'][i] = closeness.get(i, 0)
        except:
            pass
        
        # Eigenvector centrality
        try:
            eigenvector = nx.eigenvector_centrality_numpy(G, weight='weight')
            for i in range(n):
                props['eigenvector'][i] = eigenvector.get(i, 0)
        except:
            try:
                eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
                for i in range(n):
                    props['eigenvector'][i] = eigenvector.get(i, 0)
            except:
                pass
        
        # PageRank
        try:
            pagerank = nx.pagerank(G, weight='weight')
            for i in range(n):
                props['pagerank'][i] = pagerank.get(i, 0)
        except:
            pass
        
        # Clustering coefficient
        try:
            clustering = nx.clustering(G, weight='weight')
            for i in range(n):
                props['clustering'][i] = clustering.get(i, 0)
        except:
            pass
        
        # Community detection (Louvain)
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, weight='weight', seed=42)
            for comm_id, comm in enumerate(communities):
                for node_id in comm:
                    props['community'][node_id] = comm_id
        except:
            try:
                # Fallback to label propagation
                from networkx.algorithms.community import label_propagation_communities
                communities = list(label_propagation_communities(G))
                for comm_id, comm in enumerate(communities):
                    for node_id in comm:
                        props['community'][node_id] = comm_id
            except:
                pass
        
        return props
    
    def build(self, network_type: str, top_n: int = 50, min_occ: int = 2, 
              min_edge: int = 1, include_set: Set[str] = None, 
              exclude_set: Set[str] = None, include_regex: str = None,
              exclude_regex: str = None, normalize: bool = False, 
              ngram_n: int = 2):
        """Build network of specified type."""
        
        kwargs = {
            'top_n': top_n,
            'min_occ': min_occ,
            'min_edge': min_edge,
            'include_set': include_set,
            'exclude_set': exclude_set,
            'include_regex': include_regex,
            'exclude_regex': exclude_regex,
            'normalize': normalize,
        }
        
        if network_type == "author_keywords":
            col = self._find_column("Author Keywords", "DE", "Keywords")
            if not col:
                return [], np.array([]), {'error': 'Author Keywords not found'}
            return self._build_network(self._extract_entities(col), **kwargs)
        
        elif network_type == "index_keywords":
            col = self._find_column("Index Keywords", "ID", "Keywords Plus")
            if not col:
                return [], np.array([]), {'error': 'Index Keywords not found'}
            return self._build_network(self._extract_entities(col), **kwargs)
        
        elif network_type == "all_keywords":
            ak = self._find_column("Author Keywords", "DE")
            ik = self._find_column("Index Keywords", "ID")
            doc_entities = {}
            for idx in range(len(self.df)):
                entities = []
                for col in [ak, ik]:
                    if col:
                        val = self.df.iloc[idx][col]
                        if not pd.isna(val):
                            val_str = str(val)
                            if ";" in val_str:
                                entities.extend([e.strip() for e in val_str.split(";") if e.strip()])
                            elif "|" in val_str:
                                entities.extend([e.strip() for e in val_str.split("|") if e.strip()])
                            else:
                                entities.append(val_str.strip())
                if entities:
                    doc_entities[idx] = entities
            if not doc_entities:
                return [], np.array([]), {'error': 'No keywords found'}
            return self._build_network(doc_entities, **kwargs)
        
        elif network_type == "coauthorship":
            col = self._find_column("Authors", "AU", "Author")
            if not col:
                return [], np.array([]), {'error': 'Authors not found'}
            return self._build_network(self._extract_entities(col), **kwargs)
        
        elif network_type == "co_citation":
            col = self._find_column("References", "Cited References", "CR")
            if not col:
                return [], np.array([]), {'error': 'References not found'}
            return self._build_network(self._extract_entities(col), **kwargs)
        
        elif network_type == "source_cocitation":
            col = self._find_column("References", "Cited References", "CR")
            if not col:
                return [], np.array([]), {'error': 'References not found'}
            doc_entities = {}
            for idx in range(len(self.df)):
                val = self.df.iloc[idx][col]
                if pd.isna(val):
                    continue
                sources = []
                for ref in str(val).split(";"):
                    parts = ref.split(",")
                    if len(parts) >= 3:
                        sources.append(parts[2].strip())
                if sources:
                    doc_entities[idx] = sources
            return self._build_network(doc_entities, **kwargs)
        
        elif network_type == "country_collab":
            col = self._find_column("Countries", "Country", "CU")
            if not col:
                return [], np.array([]), {'error': 'Countries not found'}
            return self._build_network(self._extract_entities(col), **kwargs)
        
        elif network_type == "institution_collab":
            col = self._find_column("Affiliations", "Institutions", "C1")
            if not col:
                return [], np.array([]), {'error': 'Affiliations not found'}
            return self._build_network(self._extract_entities(col), **kwargs)
        
        elif network_type == "title_ngrams":
            col = self._find_column("Title", "TI", "Document Title")
            if not col:
                return [], np.array([]), {'error': 'Title not found'}
            return self._build_network(self._extract_ngrams(col, ngram_n), **kwargs)
        
        elif network_type == "abstract_ngrams":
            col = self._find_column("Abstract", "AB", "Description")
            if not col:
                return [], np.array([]), {'error': 'Abstract not found'}
            return self._build_network(self._extract_ngrams(col, ngram_n), **kwargs)
        
        return [], np.array([]), {'error': f'Unknown type: {network_type}'}


class OWNetworkCooccurrence(OWWidget):
    """Build bibliometric co-occurrence networks."""
    
    name = "Network Co-occurrence"
    description = "Build co-occurrence networks from bibliographic data"
    icon = "icons/network.svg"
    priority = 100
    keywords = ["network", "cooccurrence", "collaboration", "keywords"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table)
    
    class Outputs:
        network = Output("Network", Network) if HAS_NETWORK else Output("Network", object, auto_summary=False)
        node_data = Output("Node Data", Table)
        edge_data = Output("Edge Data", Table)
    
    network_type_index = settings.Setting(0)
    top_n_nodes = settings.Setting(50)
    min_occurrences = settings.Setting(2)
    include_filter = settings.Setting("")
    exclude_filter = settings.Setting("")
    use_include_regex = settings.Setting(False)
    use_exclude_regex = settings.Setting(False)
    min_edge_weight = settings.Setting(1)
    normalize_weights = settings.Setting(False)
    ngram_n = settings.Setting(2)
    
    want_main_area = False
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        build_failed = Msg("{}")
        no_network_addon = Msg("Orange Network add-on not installed")
    
    class Warning(OWWidget.Warning):
        few_nodes = Msg("Only {} nodes")
        no_edges = Msg("No edges with current settings")
        no_networkx = Msg("NetworkX not installed - metrics not computed")
    
    class Information(OWWidget.Information):
        network_built = Msg("{} nodes, {} edges, {} communities")
    
    def __init__(self):
        super().__init__()
        self._data = None
        self._df = None
        self._columns = []
        self._setup_gui()
    
    def _setup_gui(self):
        # Type
        box = gui.widgetBox(self.controlArea, "Network Type")
        gui.comboBox(box, self, "network_type_index",
                     items=[t[0] for t in NETWORK_TYPES],
                     callback=self._on_change)
        
        self.ngram_box = gui.widgetBox(box, "")
        gui.spin(self.ngram_box, self, "ngram_n", 1, 5, label="N-gram size:",
                 callback=self._on_change)
        self.ngram_box.setVisible(False)
        
        # Nodes
        box = gui.widgetBox(self.controlArea, "Node Options")
        gui.spin(box, self, "top_n_nodes", 5, 500, label="Top N Nodes:",
                 callback=self._on_change)
        gui.spin(box, self, "min_occurrences", 1, 100, label="Min Occurrences:",
                 callback=self._on_change)
        
        # Filters
        box = gui.widgetBox(self.controlArea, "Entity Filtering")
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Include:"))
        self.include_edit = QLineEdit()
        self.include_edit.setPlaceholderText("entity1; entity2; ...")
        self.include_edit.editingFinished.connect(self._on_filter)
        row.addWidget(self.include_edit)
        w = QWidget()
        w.setLayout(row)
        box.layout().addWidget(w)
        gui.checkBox(box, self, "use_include_regex", "Use regex", callback=self._on_change)
        
        row = QHBoxLayout()
        row.addWidget(QLabel("Exclude:"))
        self.exclude_edit = QLineEdit()
        self.exclude_edit.setPlaceholderText("entity1; entity2; ...")
        self.exclude_edit.editingFinished.connect(self._on_filter)
        row.addWidget(self.exclude_edit)
        w = QWidget()
        w.setLayout(row)
        box.layout().addWidget(w)
        gui.checkBox(box, self, "use_exclude_regex", "Use regex", callback=self._on_change)
        
        # Edges
        box = gui.widgetBox(self.controlArea, "Edge Options")
        gui.spin(box, self, "min_edge_weight", 1, 100, label="Min Edge Weight:",
                 callback=self._on_change)
        gui.checkBox(box, self, "normalize_weights", "Normalize weights",
                     callback=self._on_change)
        
        # Build button
        gui.button(self.controlArea, self, "Build Network", callback=self._do_commit)
        
        self.controlArea.layout().addStretch()
    
    def _do_commit(self):
        self.commit()
    
    def _on_change(self):
        net_type = NETWORK_TYPES[self.network_type_index][1]
        self.ngram_box.setVisible("ngrams" in net_type)
    
    def _on_filter(self):
        self.include_filter = self.include_edit.text()
        self.exclude_filter = self.exclude_edit.text()
    
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
        
        # Convert to DataFrame
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
        self.Outputs.node_data.send(None)
        self.Outputs.edge_data.send(None)
    
    def commit(self):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None:
            self._clear_outputs()
            return
        
        if not HAS_NETWORKX:
            self.Warning.no_networkx()
        
        try:
            builder = NetworkBuilder(self._df, self._columns)
            net_type = NETWORK_TYPES[self.network_type_index][1]
            
            # Parse filters
            inc_set = exc_set = inc_re = exc_re = None
            if self.include_filter.strip():
                if self.use_include_regex:
                    inc_re = self.include_filter.strip()
                else:
                    inc_set = {e.strip().lower() for e in self.include_filter.split(";") if e.strip()}
            if self.exclude_filter.strip():
                if self.use_exclude_regex:
                    exc_re = self.exclude_filter.strip()
                else:
                    exc_set = {e.strip().lower() for e in self.exclude_filter.split(";") if e.strip()}
            
            nodes, matrix, props = builder.build(
                net_type,
                top_n=self.top_n_nodes,
                min_occ=self.min_occurrences,
                min_edge=self.min_edge_weight,
                include_set=inc_set,
                exclude_set=exc_set,
                include_regex=inc_re,
                exclude_regex=exc_re,
                normalize=self.normalize_weights,
                ngram_n=self.ngram_n,
            )
            
            if 'error' in props:
                self.Error.build_failed(props['error'])
                self._clear_outputs()
                return
            
            if len(nodes) == 0:
                self.Error.build_failed("No nodes found")
                self._clear_outputs()
                return
            
            n_edges = int(np.sum(matrix > 0) / 2)
            n_communities = len(set(props.get('community', [0])))
            
            if len(nodes) < 3:
                self.Warning.few_nodes(len(nodes))
            if n_edges == 0:
                self.Warning.no_edges()
            
            self.Information.network_built(len(nodes), n_edges, n_communities)
            self._send_outputs(nodes, matrix, props)
            
        except Exception as e:
            logger.exception(f"Build failed: {e}")
            self.Error.build_failed(str(e))
            self._clear_outputs()
    
    def _send_outputs(self, nodes, matrix, props):
        """Send network and data outputs with computed properties."""
        n = len(nodes)
        
        # Build node table with all properties
        # Continuous variables for metrics
        cont_vars = [
            ContinuousVariable("Occurrences"),
            ContinuousVariable("Degree"),
            ContinuousVariable("Weighted Degree"),
            ContinuousVariable("Betweenness"),
            ContinuousVariable("Closeness"),
            ContinuousVariable("Eigenvector"),
            ContinuousVariable("PageRank"),
            ContinuousVariable("Clustering"),
        ]
        
        # Discrete variable for community
        n_communities = len(set(props.get('community', [0])))
        community_values = [str(i) for i in range(n_communities)]
        community_var = DiscreteVariable("Community", values=community_values)
        
        # Meta for entity name
        meta_vars = [StringVariable("Entity")]
        
        domain = Domain(cont_vars, class_vars=[community_var], metas=meta_vars)
        
        # Build data arrays
        X = np.column_stack([
            props.get('occurrences', [0]*n),
            props.get('degree', [0]*n),
            props.get('weighted_degree', [0.0]*n),
            props.get('betweenness', [0.0]*n),
            props.get('closeness', [0.0]*n),
            props.get('eigenvector', [0.0]*n),
            props.get('pagerank', [0.0]*n),
            props.get('clustering', [0.0]*n),
        ]).astype(float)
        
        Y = np.array(props.get('community', [0]*n)).astype(float)
        metas = np.array(nodes).reshape(-1, 1)
        
        node_table = Table.from_numpy(domain, X, Y, metas)
        self.Outputs.node_data.send(node_table)
        
        # Edge table
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i, j] > 0:
                    edges.append([nodes[i], nodes[j], matrix[i, j]])
        
        if edges:
            edges_arr = np.array(edges)
            edge_domain = Domain(
                [ContinuousVariable("Weight")],
                metas=[StringVariable("Source"), StringVariable("Target")]
            )
            edge_table = Table.from_numpy(
                edge_domain,
                X=edges_arr[:, 2:3].astype(float),
                metas=edges_arr[:, :2]
            )
            self.Outputs.edge_data.send(edge_table)
        else:
            self.Outputs.edge_data.send(None)
        
        # Network output
        if HAS_NETWORK and HAS_SCIPY:
            try:
                sparse = sp.csr_matrix(matrix)
                network = Network(node_table, sparse)
                self.Outputs.network.send(network)
            except Exception as e:
                logger.exception(f"Network creation failed: {e}")
                self.Error.no_network_addon()
                self.Outputs.network.send(None)
        else:
            if not HAS_NETWORK:
                self.Error.no_network_addon()
            self.Outputs.network.send(None)


if __name__ == "__main__":
    WidgetPreview(OWNetworkCooccurrence).run()
