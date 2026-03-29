# -*- coding: utf-8 -*-
"""
Thematic Map Widget
===================
Strategic diagram showing research themes by centrality and density.

Computes co-occurrence network, clusters it, and calculates centrality/density
for each cluster. Outputs data for visualization in Orange's Scatter Plot.

Quadrants:
- Motor themes (high centrality, high density): Well-developed and important
- Niche themes (low centrality, high density): Well-developed but peripheral
- Emerging/Declining (low centrality, low density): New or fading themes
- Basic themes (high centrality, low density): Important but underdeveloped
"""

import logging
from typing import Optional, Dict, List, Tuple, Set
from collections import Counter

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QSpinBox,
    QGroupBox, QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QTabWidget, QFrame,
    QTextEdit, QLineEdit,
)
from AnyQt.QtCore import Qt

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

# Try networkx for graph operations
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# Try community detection
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

# Try biblium
try:
    import biblium
    from biblium import utilsbib
    HAS_BIBLIUM = True
except ImportError:
    HAS_BIBLIUM = False

logger = logging.getLogger(__name__)


# =============================================================================
# FIELD DEFINITIONS
# =============================================================================

FIELD_OPTIONS = [
    ("Keywords", ["Author Keywords", "Keywords", "DE", "author_keywords"]),
    ("Index Keywords", ["Index Keywords", "Keywords Plus", "ID", "indexed_keywords"]),
    ("Authors", ["Authors", "AU", "author_name"]),
    ("Sources", ["Source title", "Source", "SO", "source_title", "Journal"]),
    ("Countries", ["Countries", "Country", "CU", "countries"]),
    ("Institutions", ["Affiliations", "Institutions", "C1", "affiliations"]),
]

BUBBLE_SIZE_OPTIONS = [
    ("Occurrences", "occurrences"),
    ("Documents", "documents"),
    ("Centrality", "centrality"),
    ("Density", "density"),
]


# =============================================================================
# THEMATIC MAP COMPUTATION
# =============================================================================

def build_cooccurrence_network(items_per_doc: List[List[str]], min_occurrences: int = 2) -> nx.Graph:
    """
    Build co-occurrence network from items.
    
    Args:
        items_per_doc: List of item lists, one per document
        min_occurrences: Minimum occurrences for an item to be included
    
    Returns:
        NetworkX graph with items as nodes and co-occurrence counts as edge weights
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required for thematic map")
    
    # Count item occurrences
    item_counts = Counter()
    for items in items_per_doc:
        item_counts.update(items)
    
    # Filter by min occurrences
    valid_items = {item for item, count in item_counts.items() if count >= min_occurrences}
    
    # Build co-occurrence matrix
    cooccur = Counter()
    for items in items_per_doc:
        valid = [i for i in items if i in valid_items]
        for i, item1 in enumerate(valid):
            for item2 in valid[i+1:]:
                pair = tuple(sorted([item1, item2]))
                cooccur[pair] += 1
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with occurrence counts
    for item in valid_items:
        G.add_node(item, occurrences=item_counts[item])
    
    # Add edges with co-occurrence weights
    for (item1, item2), weight in cooccur.items():
        if weight > 0:
            G.add_edge(item1, item2, weight=weight)
    
    return G


def detect_communities(G: nx.Graph) -> Dict[str, int]:
    """
    Detect communities in the graph using Louvain algorithm.
    
    Returns:
        Dictionary mapping node to cluster ID
    """
    if HAS_LOUVAIN:
        return community_louvain.best_partition(G, weight='weight')
    else:
        # Fallback: use connected components or simple clustering
        partition = {}
        for i, component in enumerate(nx.connected_components(G)):
            for node in component:
                partition[node] = i
        return partition


def compute_thematic_map(G: nx.Graph, partition: Dict[str, int]) -> pd.DataFrame:
    """
    Compute thematic map metrics for each cluster.
    
    Args:
        G: Co-occurrence network
        partition: Node to cluster mapping
    
    Returns:
        DataFrame with cluster metrics
    """
    clusters = {}
    for node, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node)
    
    results = []
    
    # Calculate total external links for normalization
    total_external = 0
    cluster_external = {}
    
    for cluster_id, nodes in clusters.items():
        node_set = set(nodes)
        external_weight = 0
        
        for node in nodes:
            for neighbor in G.neighbors(node):
                if neighbor not in node_set:
                    external_weight += G[node][neighbor].get('weight', 1)
        
        cluster_external[cluster_id] = external_weight
        total_external += external_weight
    
    # Calculate metrics for each cluster
    for cluster_id, nodes in clusters.items():
        node_set = set(nodes)
        n_items = len(nodes)
        
        if n_items == 0:
            continue
        
        # Internal links (within cluster)
        internal_weight = 0
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if G.has_edge(node1, node2):
                    internal_weight += G[node1][node2].get('weight', 1)
        
        # External links (to other clusters)
        external_weight = cluster_external[cluster_id]
        
        # Occurrences
        total_occurrences = sum(G.nodes[node].get('occurrences', 1) for node in nodes)
        
        # Density: internal cohesion
        # Formula: sum of internal link weights / number of possible internal links
        max_internal_links = n_items * (n_items - 1) / 2 if n_items > 1 else 1
        density = internal_weight / max_internal_links if max_internal_links > 0 else 0
        
        # Centrality: importance to the field
        # Formula: external links / total external links (normalized)
        centrality = external_weight / total_external if total_external > 0 else 0
        
        # Get top keywords for label
        sorted_nodes = sorted(nodes, key=lambda n: G.nodes[n].get('occurrences', 0), reverse=True)
        top_keywords = sorted_nodes[:5]
        label = "; ".join(top_keywords[:3])
        all_keywords = "; ".join(sorted_nodes)
        
        # Count documents containing cluster items
        n_documents = len(nodes)  # Approximation - actual would need original data
        
        results.append({
            'Cluster': cluster_id,
            'Label': label,
            'Centrality': centrality,
            'Density': density,
            'Occurrences': total_occurrences,
            'Items': n_items,
            'Internal Links': internal_weight,
            'External Links': external_weight,
            'Keywords': all_keywords,
            'Top Keywords': "; ".join(top_keywords),
        })
    
    df = pd.DataFrame(results)
    
    # Add quadrant labels
    if len(df) > 0:
        median_centrality = df['Centrality'].median()
        median_density = df['Density'].median()
        
        def get_quadrant(row):
            high_cent = row['Centrality'] >= median_centrality
            high_dens = row['Density'] >= median_density
            
            if high_cent and high_dens:
                return "Motor Themes"
            elif not high_cent and high_dens:
                return "Niche Themes"
            elif high_cent and not high_dens:
                return "Basic Themes"
            else:
                return "Emerging/Declining"
        
        df['Quadrant'] = df.apply(get_quadrant, axis=1)
    
    return df


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWThematicMap(OWWidget):
    """Strategic diagram showing research themes by centrality and density."""
    
    name = "Thematic Map"
    description = "Strategic diagram showing research themes by centrality and density"
    icon = "icons/thematic_map.svg"
    priority = 60
    keywords = ["thematic", "strategic", "map", "centrality", "density", "clusters", "keywords"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data table")
    
    class Outputs:
        thematic_data = Output("Thematic Data", Table, doc="Cluster data for scatter plot")
        selected_documents = Output("Selected Documents", Table, doc="Documents in selected cluster")
        network = Output("Network", Table, doc="Co-occurrence network edges")
    
    # Settings
    field_index = settings.Setting(0)
    min_occurrences = settings.Setting(5)
    top_n_items = settings.Setting(100)
    auto_apply = settings.Setting(True)
    
    want_main_area = True
    resizing_enabled = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_field = Msg("Selected field not found in data")
        no_networkx = Msg("NetworkX required - install with: pip install networkx")
        compute_error = Msg("Computation error: {}")
    
    class Warning(OWWidget.Warning):
        no_louvain = Msg("python-louvain not installed - using basic clustering")
        few_items = Msg("Only {} items found after filtering")
    
    class Information(OWWidget.Information):
        clusters_found = Msg("Found {} clusters from {} items")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._result_df: Optional[pd.DataFrame] = None
        self._graph: Optional[nx.Graph] = None
        self._partition: Optional[Dict] = None
        self._items_per_doc: Optional[List[List[str]]] = None
        
        if not HAS_NETWORKX:
            self.Error.no_networkx()
        
        if not HAS_LOUVAIN:
            self.Warning.no_louvain()
        
        self._setup_control_area()
        self._setup_main_area()
    
    def _setup_control_area(self):
        """Build control area."""
        # Analysis Settings
        analysis_box = gui.widgetBox(self.controlArea, "Analysis Settings")
        
        # Field selection
        field_layout = QHBoxLayout()
        field_layout.addWidget(QLabel("Field:"))
        self.field_combo = QComboBox()
        for name, _ in FIELD_OPTIONS:
            self.field_combo.addItem(name)
        self.field_combo.setCurrentIndex(self.field_index)
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        field_layout.addWidget(self.field_combo)
        analysis_box.layout().addLayout(field_layout)
        
        # Min occurrences
        gui.spin(analysis_box, self, "min_occurrences", 1, 100,
                 label="Min Occurrences:", callback=self._on_settings_changed)
        
        # Top N items
        gui.spin(analysis_box, self, "top_n_items", 10, 500,
                 label="Top N Items:", callback=self._on_settings_changed)
        
        # Synonyms box (placeholder for future)
        synonyms_box = gui.widgetBox(self.controlArea, "Synonyms (Merge Items)")
        self.synonyms_text = QTextEdit()
        self.synonyms_text.setPlaceholderText("Enter synonyms to merge:\nterm1; term2; term3\nword1; word2")
        self.synonyms_text.setMaximumHeight(80)
        synonyms_box.layout().addWidget(self.synonyms_text)
        
        # Generate button
        self.generate_btn = gui.button(
            self.controlArea, self, "Generate Thematic Map",
            callback=self.commit, autoDefault=False
        )
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setStyleSheet("background-color: #4a90d9; color: white; font-weight: bold;")
        
        gui.checkBox(self.controlArea, self, "auto_apply", "Auto apply")
        
        self.controlArea.layout().addStretch(1)
    
    def _setup_main_area(self):
        """Build main area with results display."""
        main_layout = QVBoxLayout()
        self.mainArea.layout().addLayout(main_layout)
        
        # Header
        header = QLabel("Thematic Map Results")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(header)
        
        self.info_label = QLabel("Load data and click 'Generate Thematic Map' to see results")
        self.info_label.setStyleSheet("color: #6c757d;")
        main_layout.addWidget(self.info_label)
        
        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Clusters table
        self.clusters_widget = QWidget()
        clusters_layout = QVBoxLayout(self.clusters_widget)
        
        self.clusters_table = QTableWidget()
        self.clusters_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.clusters_table.setSelectionMode(QTableWidget.SingleSelection)
        self.clusters_table.itemSelectionChanged.connect(self._on_cluster_selected)
        clusters_layout.addWidget(self.clusters_table)
        
        self.tabs.addTab(self.clusters_widget, "📊 Clusters")
        
        # Quadrant summary
        self.quadrant_widget = QWidget()
        quadrant_layout = QVBoxLayout(self.quadrant_widget)
        self.quadrant_table = QTableWidget()
        self.quadrant_table.setSelectionBehavior(QTableWidget.SelectRows)
        quadrant_layout.addWidget(self.quadrant_table)
        self.tabs.addTab(self.quadrant_widget, "🎯 Quadrants")
        
        # Keywords detail
        self.keywords_widget = QWidget()
        keywords_layout = QVBoxLayout(self.keywords_widget)
        self.keywords_text = QTextEdit()
        self.keywords_text.setReadOnly(True)
        keywords_layout.addWidget(self.keywords_text)
        self.tabs.addTab(self.keywords_widget, "🔑 Keywords")
        
        # Interpretation guide
        self.guide_widget = QWidget()
        guide_layout = QVBoxLayout(self.guide_widget)
        guide_text = QTextEdit()
        guide_text.setReadOnly(True)
        guide_text.setHtml("""
        <h3>Thematic Map Interpretation</h3>
        <p>The thematic map plots research themes based on two dimensions:</p>
        
        <p><b>Centrality (X-axis):</b> Measures the importance of a theme to the field.
        High centrality means the theme is strongly connected to other themes.</p>
        
        <p><b>Density (Y-axis):</b> Measures the internal coherence of a theme.
        High density means the keywords within the theme are strongly interconnected.</p>
        
        <h4>Quadrant Interpretation:</h4>
        <ul>
        <li><b>Motor Themes</b> (upper-right): Well-developed and central to the field.
        These are the main themes driving research.</li>
        
        <li><b>Niche Themes</b> (upper-left): Well-developed but peripheral.
        Specialized topics with strong internal structure but limited connection to mainstream research.</li>
        
        <li><b>Emerging/Declining Themes</b> (lower-left): Weakly developed and peripheral.
        Either new emerging topics or declining ones losing relevance.</li>
        
        <li><b>Basic Themes</b> (lower-right): Central but not well-developed.
        Important foundational topics that span across the field but lack internal cohesion.</li>
        </ul>
        
        <h4>How to Use:</h4>
        <ol>
        <li>Connect the <b>Thematic Data</b> output to a <b>Scatter Plot</b> widget</li>
        <li>Set X-axis to Centrality, Y-axis to Density</li>
        <li>Set Size to Occurrences or Items</li>
        <li>Set Label to Label or Top Keywords</li>
        <li>Color by Quadrant for easy interpretation</li>
        </ol>
        """)
        guide_layout.addWidget(guide_text)
        self.tabs.addTab(self.guide_widget, "📖 Guide")
    
    def _on_field_changed(self, index):
        self.field_index = index
        if self.auto_apply and self._df is not None:
            self.commit()
    
    def _on_settings_changed(self):
        if self.auto_apply and self._df is not None:
            self.commit()
    
    def _on_cluster_selected(self):
        """Handle cluster selection in table."""
        selected = self.clusters_table.selectedItems()
        if not selected or self._result_df is None or self._data is None:
            self.Outputs.selected_documents.send(None)
            return
        
        row = selected[0].row()
        if row >= len(self._result_df):
            return
        
        # Get keywords for this cluster
        keywords = self._result_df.iloc[row]['Keywords'].split("; ")
        
        # Find documents containing any of these keywords
        self._send_documents_for_keywords(keywords)
    
    def _send_documents_for_keywords(self, keywords: List[str]):
        """Send documents containing specified keywords."""
        if self._items_per_doc is None or self._data is None:
            return
        
        keyword_set = set(k.lower() for k in keywords)
        
        indices = []
        for i, items in enumerate(self._items_per_doc):
            if any(item.lower() in keyword_set for item in items):
                indices.append(i)
        
        if indices:
            selected = self._data[indices]
            self.Outputs.selected_documents.send(selected)
        else:
            self.Outputs.selected_documents.send(None)
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        """Receive input data."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._result_df = None
        self._graph = None
        self._partition = None
        self._items_per_doc = None
        
        self._clear_displays()
        
        if data is None:
            self.Error.no_data()
            return
        
        self._df = self._table_to_df(data)
        
        if self.auto_apply:
            self.commit()
    
    def _table_to_df(self, table: Table) -> pd.DataFrame:
        """Convert Orange Table to pandas DataFrame."""
        data = {}
        for var in table.domain.attributes:
            data[var.name] = table.get_column(var)
        for var in table.domain.class_vars:
            data[var.name] = table.get_column(var)
        for var in table.domain.metas:
            data[var.name] = table.get_column(var)
        return pd.DataFrame(data)
    
    def _find_field_column(self) -> Optional[str]:
        """Find the column for selected field."""
        if self._df is None:
            return None
        
        _, candidates = FIELD_OPTIONS[self.field_index]
        
        for col in self._df.columns:
            if col in candidates:
                return col
            # Case-insensitive match
            for candidate in candidates:
                if col.lower() == candidate.lower():
                    return col
        
        return None
    
    def _get_synonyms(self) -> Dict[str, str]:
        """Parse synonyms from text input."""
        synonyms = {}
        text = self.synonyms_text.toPlainText()
        
        for line in text.strip().split('\n'):
            if ';' in line:
                terms = [t.strip().lower() for t in line.split(';') if t.strip()]
                if len(terms) > 1:
                    canonical = terms[0]
                    for term in terms[1:]:
                        synonyms[term] = canonical
        
        return synonyms
    
    def _extract_items(self, column: str) -> List[List[str]]:
        """Extract items from column, one list per document."""
        items_per_doc = []
        synonyms = self._get_synonyms()
        
        for value in self._df[column]:
            if pd.isna(value) or value == '':
                items_per_doc.append([])
                continue
            
            value_str = str(value)
            
            # Detect separator
            if '; ' in value_str:
                items = [i.strip() for i in value_str.split('; ')]
            elif ';' in value_str:
                items = [i.strip() for i in value_str.split(';')]
            elif '|' in value_str:
                items = [i.strip() for i in value_str.split('|')]
            elif ',' in value_str and self.field_index not in [2]:  # Not for authors
                items = [i.strip() for i in value_str.split(',')]
            else:
                items = [value_str.strip()]
            
            # Clean and apply synonyms
            cleaned = []
            for item in items:
                item = item.strip()
                if item:
                    # Apply synonyms
                    item_lower = item.lower()
                    if item_lower in synonyms:
                        item = synonyms[item_lower]
                    cleaned.append(item)
            
            items_per_doc.append(cleaned)
        
        return items_per_doc
    
    def _clear_displays(self):
        """Clear all displays."""
        self.clusters_table.clear()
        self.clusters_table.setRowCount(0)
        self.quadrant_table.clear()
        self.quadrant_table.setRowCount(0)
        self.keywords_text.clear()
        self.info_label.setText("Load data and click 'Generate Thematic Map'")
    
    def commit(self):
        """Generate thematic map."""
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._result_df = None
        self._graph = None
        self._partition = None
        
        if not HAS_NETWORKX:
            self.Error.no_networkx()
            self._send_outputs()
            return
        
        if self._df is None:
            self.Error.no_data()
            self._send_outputs()
            return
        
        # Find field column
        field_col = self._find_field_column()
        if field_col is None:
            self.Error.no_field()
            self._send_outputs()
            return
        
        try:
            # Extract items
            self._items_per_doc = self._extract_items(field_col)
            
            # Build network
            self._graph = build_cooccurrence_network(
                self._items_per_doc,
                min_occurrences=self.min_occurrences
            )
            
            n_nodes = self._graph.number_of_nodes()
            
            if n_nodes < 3:
                self.Warning.few_items(n_nodes)
                self._send_outputs()
                return
            
            # Limit to top N items by occurrence
            if n_nodes > self.top_n_items:
                top_nodes = sorted(
                    self._graph.nodes(),
                    key=lambda n: self._graph.nodes[n].get('occurrences', 0),
                    reverse=True
                )[:self.top_n_items]
                self._graph = self._graph.subgraph(top_nodes).copy()
            
            # Detect communities
            self._partition = detect_communities(self._graph)
            
            # Compute thematic map
            self._result_df = compute_thematic_map(self._graph, self._partition)
            
            n_clusters = len(self._result_df)
            n_items = self._graph.number_of_nodes()
            
            self.Information.clusters_found(n_clusters, n_items)
            self.info_label.setText(f"Found {n_clusters} clusters from {n_items} items")
            
            self._update_displays()
            self._send_outputs()
            
        except Exception as e:
            import traceback
            logger.error(f"Thematic map error: {e}\n{traceback.format_exc()}")
            self.Error.compute_error(str(e))
            self._send_outputs()
    
    def _update_displays(self):
        """Update all display elements."""
        if self._result_df is None or len(self._result_df) == 0:
            return
        
        df = self._result_df
        
        # Clusters table
        columns = ['Cluster', 'Label', 'Centrality', 'Density', 'Occurrences', 'Items', 'Quadrant']
        self.clusters_table.clear()
        self.clusters_table.setRowCount(len(df))
        self.clusters_table.setColumnCount(len(columns))
        self.clusters_table.setHorizontalHeaderLabels(columns)
        
        for i, row in df.iterrows():
            for j, col in enumerate(columns):
                if col in ['Centrality', 'Density']:
                    val = f"{row[col]:.4f}"
                else:
                    val = str(row[col])
                self.clusters_table.setItem(i, j, QTableWidgetItem(val))
        
        self.clusters_table.resizeColumnsToContents()
        
        # Quadrant summary
        quadrant_counts = df['Quadrant'].value_counts()
        quadrant_order = ['Motor Themes', 'Niche Themes', 'Basic Themes', 'Emerging/Declining']
        
        self.quadrant_table.clear()
        self.quadrant_table.setRowCount(len(quadrant_order))
        self.quadrant_table.setColumnCount(4)
        self.quadrant_table.setHorizontalHeaderLabels(['Quadrant', 'Clusters', 'Total Items', 'Description'])
        
        descriptions = {
            'Motor Themes': 'Well-developed and important',
            'Niche Themes': 'Well-developed but peripheral',
            'Basic Themes': 'Important but underdeveloped',
            'Emerging/Declining': 'New or fading themes',
        }
        
        for i, quad in enumerate(quadrant_order):
            count = quadrant_counts.get(quad, 0)
            total_items = df[df['Quadrant'] == quad]['Items'].sum() if count > 0 else 0
            
            self.quadrant_table.setItem(i, 0, QTableWidgetItem(quad))
            self.quadrant_table.setItem(i, 1, QTableWidgetItem(str(count)))
            self.quadrant_table.setItem(i, 2, QTableWidgetItem(str(int(total_items))))
            self.quadrant_table.setItem(i, 3, QTableWidgetItem(descriptions[quad]))
        
        self.quadrant_table.resizeColumnsToContents()
        
        # Keywords detail
        keywords_html = "<h3>Cluster Keywords</h3>"
        for i, row in df.iterrows():
            quad_color = {
                'Motor Themes': '#28a745',
                'Niche Themes': '#17a2b8',
                'Basic Themes': '#fd7e14',
                'Emerging/Declining': '#6c757d',
            }.get(row['Quadrant'], '#000')
            
            keywords_html += f"""
            <p><b style='color:{quad_color}'>Cluster {row['Cluster']}: {row['Label']}</b><br>
            <small>Quadrant: {row['Quadrant']} | Centrality: {row['Centrality']:.4f} | Density: {row['Density']:.4f}</small><br>
            Keywords: {row['Keywords']}</p>
            """
        
        self.keywords_text.setHtml(keywords_html)
    
    def _send_outputs(self):
        """Send all outputs."""
        if self._result_df is None or len(self._result_df) == 0:
            self.Outputs.thematic_data.send(None)
            self.Outputs.selected_documents.send(None)
            self.Outputs.network.send(None)
            return
        
        # Thematic data for scatter plot
        df = self._result_df.copy()
        
        # Create Orange Table with proper variable types
        # Continuous: Centrality, Density, Occurrences, Items
        # Discrete: Quadrant
        # String: Label, Keywords, Top Keywords
        
        cont_vars = [
            ContinuousVariable('Centrality'),
            ContinuousVariable('Density'),
            ContinuousVariable('Occurrences'),
            ContinuousVariable('Items'),
            ContinuousVariable('Internal Links'),
            ContinuousVariable('External Links'),
        ]
        
        quadrants = ['Motor Themes', 'Niche Themes', 'Basic Themes', 'Emerging/Declining']
        class_var = DiscreteVariable('Quadrant', values=quadrants)
        
        meta_vars = [
            StringVariable('Label'),
            StringVariable('Top Keywords'),
            StringVariable('Keywords'),
        ]
        
        domain = Domain(cont_vars, class_var, meta_vars)
        
        # Build arrays
        X = df[['Centrality', 'Density', 'Occurrences', 'Items', 'Internal Links', 'External Links']].values.astype(float)
        
        # Map quadrant to index
        y = np.array([quadrants.index(q) if q in quadrants else 0 for q in df['Quadrant']])
        
        metas = df[['Label', 'Top Keywords', 'Keywords']].values
        
        table = Table.from_numpy(domain, X, y, metas)
        self.Outputs.thematic_data.send(table)
        
        # Network edges output
        if self._graph is not None:
            edges = []
            for u, v, data in self._graph.edges(data=True):
                edges.append({
                    'Source': u,
                    'Target': v,
                    'Weight': data.get('weight', 1)
                })
            
            if edges:
                edges_df = pd.DataFrame(edges)
                edge_metas = [StringVariable('Source'), StringVariable('Target'), StringVariable('Weight')]
                edge_domain = Domain([], metas=edge_metas)
                edge_table = Table.from_numpy(edge_domain, np.empty((len(edges), 0)), metas=edges_df.astype(str).values)
                self.Outputs.network.send(edge_table)
            else:
                self.Outputs.network.send(None)


if __name__ == "__main__":
    WidgetPreview(OWThematicMap).run()
