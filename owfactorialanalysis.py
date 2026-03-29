# -*- coding: utf-8 -*-
"""
Factorial Analysis Widget
=========================
Orange widget for factorial analysis of bibliometric data.

Combines dimensionality reduction (MCA, CA, PCA, SVD) with clustering
(K-means, hierarchical, spectral, DBSCAN) and visualization.
"""

import logging
from typing import Optional, List, Dict, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import (QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QTableWidget, QTableWidgetItem, QToolTip,
                              QHeaderView, QSplitter)

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable, DiscreteVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.visualize.utils.plotutils import AxisItem, PlotWidget
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

FIELD_TYPES = [
    ("Author Keywords", "Author Keywords"),
    ("Index Keywords", "Index Keywords"),
    ("Title", "Title"),
    ("Abstract", "Abstract"),
    ("Authors", "Authors"),
    ("Sources", "Source title"),
    ("Countries", "Countries"),
    ("Affiliations", "Affiliations"),
    ("References", "References"),
]

DR_METHODS = [
    ("MCA", "mca"),
    ("CA", "ca"),
    ("PCA", "pca"),
    ("SVD", "svd"),
]

DTM_METHODS = [
    ("Count", "count"),
    ("Binary", "binary"),
    ("TF-IDF", "tfidf"),
]

CLUSTER_METHODS = [
    ("K-Means", "kmeans"),
    ("Hierarchical", "hierarchical"),
    ("Spectral", "spectral"),
    ("DBSCAN", "dbscan"),
]

PLOT_TYPES = [
    ("Word Map", "wordmap"),
    ("Topic Dendrogram", "dendrogram"),
    ("Cluster Heatmap", "heatmap"),
    ("Terms by Cluster", "terms_cluster"),
]

# Cluster colors
CLUSTER_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b",
    "#2980b9", "#27ae60", "#d35400", "#8e44ad", "#17a2b8",
]


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def create_document_term_matrix(df: pd.DataFrame, column: str, 
                                 n_terms: int = 100, min_freq: int = 2,
                                 method: str = "count") -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Create document-term matrix from text column.
    
    Returns:
        (matrix, term_names, doc_indices)
    """
    # Extract terms from each document
    doc_terms = []
    doc_indices = []
    
    for idx, row in df.iterrows():
        val = row[column]
        if pd.isna(val):
            continue
        
        val_str = str(val)
        
        # Split by separators
        terms = []
        for sep in [";", "|"]:
            if sep in val_str:
                terms = [t.strip().lower() for t in val_str.split(sep) if t.strip()]
                break
        else:
            # For text fields (Title, Abstract), split by words
            if column.lower() in ["title", "abstract"]:
                terms = [w.lower() for w in val_str.split() if len(w) > 2]
            else:
                terms = [val_str.strip().lower()] if val_str.strip() else []
        
        if terms:
            doc_terms.append(terms)
            doc_indices.append(idx)
    
    if not doc_terms:
        return np.array([]), [], []
    
    # Count term frequencies
    term_counter = Counter()
    for terms in doc_terms:
        term_counter.update(set(terms))  # Count document frequency
    
    # Filter by min frequency and take top N
    filtered_terms = [(t, c) for t, c in term_counter.items() if c >= min_freq]
    filtered_terms.sort(key=lambda x: -x[1])
    top_terms = [t[0] for t in filtered_terms[:n_terms]]
    
    if not top_terms:
        return np.array([]), [], []
    
    # Create term index
    term_to_idx = {t: i for i, t in enumerate(top_terms)}
    
    # Build matrix
    n_docs = len(doc_terms)
    n_terms_final = len(top_terms)
    matrix = np.zeros((n_docs, n_terms_final))
    
    for i, terms in enumerate(doc_terms):
        term_counts = Counter(terms)
        for term, count in term_counts.items():
            if term in term_to_idx:
                if method == "binary":
                    matrix[i, term_to_idx[term]] = 1
                else:
                    matrix[i, term_to_idx[term]] = count
    
    # Apply TF-IDF if requested
    if method == "tfidf":
        # TF-IDF transformation
        tf = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-10)
        df_term = (matrix > 0).sum(axis=0)
        idf = np.log(n_docs / (df_term + 1)) + 1
        matrix = tf * idf
    
    return matrix, top_terms, doc_indices


def perform_pca(matrix: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform PCA on matrix."""
    # Center the data
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    
    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    # Get components
    n_comp = min(n_components, len(S))
    
    # Document coordinates (scores)
    doc_coords = U[:, :n_comp] * S[:n_comp]
    
    # Term coordinates (loadings)
    term_coords = Vt[:n_comp, :].T * S[:n_comp]
    
    # Explained variance
    var_explained = (S ** 2) / (S ** 2).sum()
    
    return doc_coords, term_coords, var_explained[:n_comp]


def perform_ca(matrix: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform Correspondence Analysis."""
    # Ensure non-negative
    matrix = np.maximum(matrix, 0)
    
    # Total
    N = matrix.sum()
    if N == 0:
        return np.zeros((matrix.shape[0], n_components)), \
               np.zeros((matrix.shape[1], n_components)), \
               np.zeros(n_components)
    
    # Row and column profiles
    P = matrix / N
    r = P.sum(axis=1, keepdims=True)
    c = P.sum(axis=0, keepdims=True)
    
    # Avoid division by zero
    r = np.maximum(r, 1e-10)
    c = np.maximum(c, 1e-10)
    
    # Standardized residuals
    S = (P - r @ c) / np.sqrt(r @ c)
    
    # SVD
    U, sigma, Vt = np.linalg.svd(S, full_matrices=False)
    
    n_comp = min(n_components, len(sigma))
    
    # Row coordinates
    row_coords = (U[:, :n_comp] * sigma[:n_comp]) / np.sqrt(r)
    
    # Column coordinates  
    col_coords = (Vt[:n_comp, :].T * sigma[:n_comp]) / np.sqrt(c.T)
    
    # Inertia explained
    inertia = sigma ** 2
    inertia_explained = inertia / inertia.sum() if inertia.sum() > 0 else inertia
    
    return row_coords, col_coords, inertia_explained[:n_comp]


def perform_svd(matrix: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform truncated SVD."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    n_comp = min(n_components, len(S))
    
    doc_coords = U[:, :n_comp] * S[:n_comp]
    term_coords = Vt[:n_comp, :].T
    
    var_explained = (S ** 2) / (S ** 2).sum()
    
    return doc_coords, term_coords, var_explained[:n_comp]


def perform_mca(matrix: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Multiple Correspondence Analysis.
    For continuous data, we discretize first.
    """
    # Binarize the matrix (presence/absence)
    binary = (matrix > 0).astype(float)
    
    # Use CA on the indicator matrix
    return perform_ca(binary, n_components)


def perform_clustering(coords: np.ndarray, method: str = "kmeans", 
                       n_clusters: int = 5) -> np.ndarray:
    """Perform clustering on coordinates."""
    from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
    
    if len(coords) < n_clusters:
        return np.zeros(len(coords), dtype=int)
    
    try:
        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(coords)
        
        elif method == "hierarchical":
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(coords)
        
        elif method == "spectral":
            n_clust = min(n_clusters, len(coords) - 1)
            model = SpectralClustering(n_clusters=n_clust, random_state=42, 
                                       affinity='nearest_neighbors', n_neighbors=min(10, len(coords)-1))
            labels = model.fit_predict(coords)
        
        elif method == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=2)
            labels = model.fit_predict(coords)
            # DBSCAN returns -1 for noise, convert to separate cluster
            labels = np.where(labels == -1, labels.max() + 1, labels)
        
        else:
            labels = np.zeros(len(coords), dtype=int)
        
        return labels
    
    except Exception as e:
        logger.warning(f"Clustering failed: {e}")
        return np.zeros(len(coords), dtype=int)


# =============================================================================
# PLOT WIDGETS
# =============================================================================

class WordMapPlot(PlotWidget):
    """Word map scatter plot showing terms in reduced space."""
    
    def __init__(self, master, parent=None):
        super().__init__(
            parent=parent,
            enableMenu=False,
            axisItems={
                "bottom": AxisItem(orientation="bottom"),
                "left": AxisItem(orientation="left"),
            }
        )
        
        self.master = master
        self.scatter_items = []
        self.text_items = []
        self.term_data = []
        self.selected_indices = []
        
        self.getPlotItem().buttonsHidden = True
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)
        self.showGrid(x=False, y=False, alpha=0.3)
        
        self.legend = pg.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.getPlotItem())
        self.legend.hide()
        
        self.setMouseTracking(True)
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)
    
    def clear_plot(self):
        self.clear()
        self.scatter_items = []
        self.text_items = []
        self.term_data = []
        self.selected_indices = []
        self.legend.clear()
        self.legend.hide()
    
    def plot_wordmap(self, coords: np.ndarray, terms: List[str], 
                     labels: np.ndarray, var_explained: np.ndarray,
                     show_labels: bool = True):
        """Plot word map with terms colored by cluster."""
        self.clear_plot()
        
        if len(coords) == 0:
            return
        
        self.term_data = [(terms[i], coords[i], labels[i]) for i in range(len(terms))]
        
        # Get unique clusters
        unique_labels = np.unique(labels)
        
        # Plot each cluster
        for cluster_id in unique_labels:
            mask = labels == cluster_id
            cluster_coords = coords[mask]
            cluster_terms = [terms[i] for i in range(len(terms)) if mask[i]]
            
            color = CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]
            qcolor = QColor(color)
            
            scatter = pg.ScatterPlotItem(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                pen=pg.mkPen(qcolor.darker(120), width=1),
                brush=pg.mkBrush(qcolor),
                size=10,
                name=f"Cluster {cluster_id + 1}"
            )
            self.addItem(scatter)
            self.scatter_items.append(scatter)
            
            # Add labels
            if show_labels and len(cluster_terms) <= 100:
                for i, (term, coord) in enumerate(zip(cluster_terms, cluster_coords)):
                    # Truncate long terms
                    display_term = term[:20] + "..." if len(term) > 20 else term
                    text = pg.TextItem(display_term, color=qcolor.darker(150), anchor=(0, 0.5))
                    text.setPos(coord[0], coord[1])
                    text.setFont(pg.QtGui.QFont("Arial", 8))
                    self.addItem(text)
                    self.text_items.append(text)
        
        # Axis labels with variance explained
        if len(var_explained) >= 2:
            self.setLabel('bottom', f'Dim 1 ({var_explained[0]*100:.1f}%)')
            self.setLabel('left', f'Dim 2 ({var_explained[1]*100:.1f}%)')
        else:
            self.setLabel('bottom', 'Dimension 1')
            self.setLabel('left', 'Dimension 2')
        
        self.setTitle("Word Map - Factorial Analysis")
        
        # Legend
        self.legend.clear()
        for i, cluster_id in enumerate(unique_labels[:10]):  # Max 10 in legend
            color = CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]
            sample = pg.ScatterPlotItem(brush=pg.mkBrush(color), size=10)
            self.legend.addItem(sample, f"Cluster {cluster_id + 1}")
        self.legend.show()
        
        self.autoRange()
    
    def _on_mouse_moved(self, pos):
        if not self.term_data:
            QToolTip.hideText()
            return
        
        mouse_point = self.getPlotItem().vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Find nearest term
        min_dist = float('inf')
        nearest = None
        
        for term, coord, label in self.term_data:
            dist = (x - coord[0])**2 + (y - coord[1])**2
            if dist < min_dist:
                min_dist = dist
                nearest = (term, coord, label)
        
        view_rect = self.getPlotItem().vb.viewRect()
        threshold = (view_rect.width() ** 2 + view_rect.height() ** 2) * 0.001
        
        if min_dist < threshold and nearest:
            tooltip = f"<b>{nearest[0]}</b><br>Cluster: {int(nearest[2]) + 1}"
            global_pos = self.mapToGlobal(self.mapFromScene(pos))
            QToolTip.showText(global_pos, tooltip)
        else:
            QToolTip.hideText()
    
    def _on_mouse_clicked(self, event):
        if not self.term_data:
            return
        
        pos = event.scenePos()
        mouse_point = self.getPlotItem().vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Find clicked term
        min_dist = float('inf')
        clicked_idx = None
        
        for i, (term, coord, label) in enumerate(self.term_data):
            dist = (x - coord[0])**2 + (y - coord[1])**2
            if dist < min_dist:
                min_dist = dist
                clicked_idx = i
        
        view_rect = self.getPlotItem().vb.viewRect()
        threshold = (view_rect.width() ** 2 + view_rect.height() ** 2) * 0.002
        
        if min_dist < threshold and clicked_idx is not None:
            modifiers = event.modifiers()
            if modifiers & Qt.ControlModifier:
                if clicked_idx in self.selected_indices:
                    self.selected_indices.remove(clicked_idx)
                else:
                    self.selected_indices.append(clicked_idx)
            else:
                self.selected_indices = [clicked_idx]
            
            self.master.term_selection_changed()


class DendrogramWidget(QWidget):
    """Widget showing hierarchical clustering dendrogram."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        self.plot = pg.PlotWidget()
        self.plot.setBackground('w')
        self.layout.addWidget(self.plot)
    
    def clear_plot(self):
        self.plot.clear()
    
    def plot_dendrogram(self, coords: np.ndarray, terms: List[str], 
                        n_clusters: int = 5):
        """Plot dendrogram using hierarchical clustering."""
        self.clear_plot()
        
        if len(coords) < 3:
            return
        
        try:
            # Compute linkage
            Z = linkage(coords, method='ward')
            
            # Get dendrogram data
            dend = dendrogram(Z, no_plot=True, labels=terms)
            
            # Plot the dendrogram manually using pyqtgraph
            icoord = np.array(dend['icoord'])
            dcoord = np.array(dend['dcoord'])
            colors = dend['color_list']
            
            for i in range(len(icoord)):
                x = icoord[i]
                y = dcoord[i]
                
                # Draw U-shape
                self.plot.plot(x, y, pen=pg.mkPen('#3498db', width=1.5))
            
            # Add term labels at bottom
            ivl = dend['ivl']
            leaves = dend['leaves']
            
            # X positions for labels
            x_positions = np.arange(5, 5 + len(ivl) * 10, 10)
            
            for i, (term, x_pos) in enumerate(zip(ivl, x_positions)):
                display_term = term[:15] + "..." if len(term) > 15 else term
                text = pg.TextItem(display_term, angle=-90, anchor=(0, 0.5))
                text.setPos(x_pos, -0.5)
                text.setFont(pg.QtGui.QFont("Arial", 7))
                self.plot.addItem(text)
            
            self.plot.setLabel('left', 'Distance')
            self.plot.setTitle('Hierarchical Clustering Dendrogram')
            
        except Exception as e:
            logger.warning(f"Dendrogram plotting failed: {e}")


class ClusterHeatmapWidget(QWidget):
    """Widget showing cluster-term heatmap."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        self.table = QTableWidget()
        self.layout.addWidget(self.table)
    
    def clear_plot(self):
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
    
    def plot_heatmap(self, matrix: np.ndarray, terms: List[str], 
                     labels: np.ndarray):
        """Show heatmap of term frequencies by cluster."""
        self.clear_plot()
        
        if len(matrix) == 0:
            return
        
        # Aggregate by cluster
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        n_terms = min(len(terms), 50)  # Limit terms
        
        # Calculate mean term frequency per cluster
        cluster_term_means = np.zeros((n_clusters, n_terms))
        
        for i, cluster_id in enumerate(unique_labels):
            mask = labels == cluster_id
            if mask.any():
                cluster_term_means[i] = matrix[mask, :n_terms].mean(axis=0)
        
        # Setup table
        self.table.setRowCount(n_clusters)
        self.table.setColumnCount(n_terms)
        
        # Headers
        self.table.setHorizontalHeaderLabels([t[:15] for t in terms[:n_terms]])
        self.table.setVerticalHeaderLabels([f"Cluster {i+1}" for i in range(n_clusters)])
        
        # Fill with colored cells
        max_val = cluster_term_means.max() if cluster_term_means.max() > 0 else 1
        
        for i in range(n_clusters):
            for j in range(n_terms):
                val = cluster_term_means[i, j]
                item = QTableWidgetItem(f"{val:.2f}")
                
                # Color based on intensity
                intensity = int(255 * (1 - val / max_val))
                item.setBackground(QColor(intensity, intensity, 255))
                
                self.table.setItem(i, j, item)
        
        # Resize
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)


class TermsClusterWidget(QWidget):
    """Widget showing top terms per cluster."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        self.table = QTableWidget()
        self.layout.addWidget(self.table)
    
    def clear_plot(self):
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
    
    def show_terms_by_cluster(self, matrix: np.ndarray, terms: List[str],
                               labels: np.ndarray, top_n: int = 10):
        """Show top terms for each cluster."""
        self.clear_plot()
        
        if len(matrix) == 0:
            return
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Setup table
        self.table.setRowCount(top_n)
        self.table.setColumnCount(n_clusters * 2)  # Term and score for each cluster
        
        headers = []
        for i in range(n_clusters):
            headers.extend([f"Cluster {i+1}", "Score"])
        self.table.setHorizontalHeaderLabels(headers)
        
        # For each cluster, find top terms
        for i, cluster_id in enumerate(unique_labels):
            mask = labels == cluster_id
            if not mask.any():
                continue
            
            # Mean term frequency in this cluster
            cluster_means = matrix[mask].mean(axis=0)
            
            # Sort by frequency
            sorted_indices = np.argsort(-cluster_means)[:top_n]
            
            for j, term_idx in enumerate(sorted_indices):
                if term_idx < len(terms):
                    term = terms[term_idx]
                    score = cluster_means[term_idx]
                    
                    self.table.setItem(j, i*2, QTableWidgetItem(term))
                    self.table.setItem(j, i*2+1, QTableWidgetItem(f"{score:.2f}"))
        
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWFactorialAnalysis(OWWidget):
    """Factorial analysis with MCA, CA, PCA, SVD and clustering."""
    
    name = "Factorial Analysis"
    description = "MCA, CA, PCA with clustering and visualization"
    icon = "icons/factorial.svg"
    priority = 90
    keywords = ["factorial", "mca", "pca", "ca", "svd", "clustering", "correspondence"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")
    
    class Outputs:
        embeddings = Output("Embeddings", Table, doc="Term embeddings")
        clusters = Output("Clusters", Table, doc="Cluster assignments")
    
    # Settings
    plot_type_index = settings.Setting(0)
    field_index = settings.Setting(0)
    column_name = settings.Setting("")
    n_terms = settings.Setting(100)
    min_doc_freq = settings.Setting(2)
    dr_method_index = settings.Setting(0)
    n_components = settings.Setting(2)
    dtm_method_index = settings.Setting(0)
    cluster_method_index = settings.Setting(0)
    n_clusters = settings.Setting(5)
    show_labels = settings.Setting(True)
    
    want_main_area = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_column = Msg("Selected column not found")
        no_terms = Msg("No terms found with current settings")
        analysis_failed = Msg("Analysis failed: {}")
    
    class Warning(OWWidget.Warning):
        few_terms = Msg("Only {} terms found")
    
    class Information(OWWidget.Information):
        analyzed = Msg("Analyzed {} terms in {} clusters")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._columns: List[str] = []
        
        # Analysis results
        self._matrix: Optional[np.ndarray] = None
        self._terms: List[str] = []
        self._doc_indices: List[int] = []
        self._term_coords: Optional[np.ndarray] = None
        self._doc_coords: Optional[np.ndarray] = None
        self._var_explained: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        
        self._setup_gui()
    
    def _setup_gui(self):
        # Visualization
        viz_box = gui.widgetBox(self.controlArea, "Visualization")
        
        gui.comboBox(
            viz_box, self, "plot_type_index",
            items=[p[0] for p in PLOT_TYPES],
            label="Plot Type:",
            callback=self._on_plot_type_changed,
            orientation=Qt.Horizontal,
        )
        
        # Field Selection
        field_box = gui.widgetBox(self.controlArea, "Field Selection")
        
        gui.label(field_box, self, "Field:")
        self.col_combo = gui.comboBox(
            field_box, self, "column_name",
            sendSelectedValue=True,
            callback=self._on_column_changed,
        )
        
        gui.spin(field_box, self, "n_terms", minv=10, maxv=500,
                 label="N Terms:", callback=self._on_setting_changed)
        
        gui.spin(field_box, self, "min_doc_freq", minv=1, maxv=50,
                 label="Min Doc Freq:", callback=self._on_setting_changed)
        
        # Analysis Settings
        analysis_box = gui.widgetBox(self.controlArea, "Analysis Settings")
        
        gui.comboBox(
            analysis_box, self, "dr_method_index",
            items=[m[0] for m in DR_METHODS],
            label="DR Method:",
            callback=self._on_setting_changed,
            orientation=Qt.Horizontal,
        )
        
        gui.spin(analysis_box, self, "n_components", minv=2, maxv=10,
                 label="Components:", callback=self._on_setting_changed)
        
        gui.comboBox(
            analysis_box, self, "dtm_method_index",
            items=[m[0] for m in DTM_METHODS],
            label="DTM Method:",
            callback=self._on_setting_changed,
            orientation=Qt.Horizontal,
        )
        
        # Clustering
        cluster_box = gui.widgetBox(self.controlArea, "Clustering")
        
        gui.comboBox(
            cluster_box, self, "cluster_method_index",
            items=[m[0] for m in CLUSTER_METHODS],
            label="Method:",
            callback=self._on_setting_changed,
            orientation=Qt.Horizontal,
        )
        
        gui.spin(cluster_box, self, "n_clusters", minv=2, maxv=20,
                 label="N Clusters:", callback=self._on_setting_changed)
        
        # Display options
        gui.checkBox(self.controlArea, self, "show_labels", "Show term labels",
                     callback=self._update_plot)
        
        # Run button
        self.run_btn = gui.button(
            self.controlArea, self, "Run Analysis",
            callback=self._run_analysis,
        )
        self.run_btn.setMinimumHeight(35)
        
        self.controlArea.layout().addStretch(1)
        
        # Main area with tabs
        self.tabs = QTabWidget()
        self.mainArea.layout().addWidget(self.tabs)
        
        # Plot tab
        self.wordmap_plot = WordMapPlot(master=self)
        self.tabs.addTab(self.wordmap_plot, "📊 Plot")
        
        # Clusters tab
        self.terms_cluster = TermsClusterWidget()
        self.tabs.addTab(self.terms_cluster, "📋 Clusters")
        
        # Embeddings tab
        self.embeddings_table = QTableWidget()
        self.tabs.addTab(self.embeddings_table, "📈 Embeddings")
    
    def _on_plot_type_changed(self):
        self._update_plot()
    
    def _on_column_changed(self):
        pass
    
    def _on_setting_changed(self):
        pass
    
    def _update_plot(self):
        if self._term_coords is None:
            return
        
        plot_type = PLOT_TYPES[self.plot_type_index][1]
        
        if plot_type == "wordmap":
            self.wordmap_plot.plot_wordmap(
                self._term_coords, self._terms, self._labels,
                self._var_explained, self.show_labels
            )
        elif plot_type == "dendrogram":
            # Create dendrogram widget if needed
            pass
        elif plot_type == "heatmap":
            pass
        elif plot_type == "terms_cluster":
            self.terms_cluster.show_terms_by_cluster(
                self._matrix, self._terms, self._labels
            )
    
    def term_selection_changed(self):
        """Handle term selection in plot."""
        pass
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._columns = []
        self._clear_results()
        
        self.col_combo.clear()
        
        if data is None:
            self.Error.no_data()
            return
        
        self._df = self._table_to_df(data)
        self._columns = list(self._df.columns)
        
        self.col_combo.addItems(self._columns)
        self._suggest_column()
    
    def _table_to_df(self, table: Table) -> pd.DataFrame:
        data = {}
        for var in table.domain.attributes:
            data[var.name] = table.get_column(var)
        for var in table.domain.metas:
            data[var.name] = table.get_column(var)
        for var in table.domain.class_vars:
            data[var.name] = table.get_column(var)
        return pd.DataFrame(data)
    
    def _suggest_column(self):
        if not self._columns:
            return
        
        patterns = ["keyword", "author", "title", "abstract"]
        
        for col in self._columns:
            col_lower = col.lower()
            for pattern in patterns:
                if pattern in col_lower:
                    idx = self._columns.index(col)
                    self.col_combo.setCurrentIndex(idx)
                    self.column_name = col
                    return
    
    def _clear_results(self):
        self._matrix = None
        self._terms = []
        self._doc_indices = []
        self._term_coords = None
        self._doc_coords = None
        self._var_explained = None
        self._labels = None
        
        self.wordmap_plot.clear_plot()
        self.terms_cluster.clear_plot()
    
    def _run_analysis(self):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        self._clear_results()
        
        if self._df is None:
            self.Error.no_data()
            return
        
        if not self.column_name or self.column_name not in self._df.columns:
            self.Error.no_column()
            return
        
        try:
            # Create document-term matrix
            dtm_method = DTM_METHODS[self.dtm_method_index][1]
            self._matrix, self._terms, self._doc_indices = create_document_term_matrix(
                self._df, self.column_name,
                n_terms=self.n_terms,
                min_freq=self.min_doc_freq,
                method=dtm_method
            )
            
            if len(self._terms) < 3:
                self.Error.no_terms()
                return
            
            if len(self._terms) < self.n_terms:
                self.Warning.few_terms(len(self._terms))
            
            # Perform dimensionality reduction
            dr_method = DR_METHODS[self.dr_method_index][1]
            
            if dr_method == "pca":
                self._doc_coords, self._term_coords, self._var_explained = \
                    perform_pca(self._matrix.T, self.n_components)  # Transpose for term analysis
            elif dr_method == "ca":
                self._doc_coords, self._term_coords, self._var_explained = \
                    perform_ca(self._matrix, self.n_components)
            elif dr_method == "svd":
                self._doc_coords, self._term_coords, self._var_explained = \
                    perform_svd(self._matrix.T, self.n_components)
            else:  # mca
                self._doc_coords, self._term_coords, self._var_explained = \
                    perform_mca(self._matrix, self.n_components)
            
            # Perform clustering on term coordinates
            cluster_method = CLUSTER_METHODS[self.cluster_method_index][1]
            self._labels = perform_clustering(
                self._term_coords, cluster_method, self.n_clusters
            )
            
            self.Information.analyzed(len(self._terms), len(np.unique(self._labels)))
            
            # Update visualizations
            self._update_plot()
            self._update_embeddings_table()
            self._send_outputs()
            
        except Exception as e:
            logger.exception(f"Analysis failed: {e}")
            self.Error.analysis_failed(str(e))
    
    def _update_embeddings_table(self):
        """Update embeddings table."""
        if self._term_coords is None:
            return
        
        n_terms = len(self._terms)
        n_dims = self._term_coords.shape[1]
        
        self.embeddings_table.setRowCount(n_terms)
        self.embeddings_table.setColumnCount(n_dims + 2)
        
        headers = ["Term", "Cluster"] + [f"Dim {i+1}" for i in range(n_dims)]
        self.embeddings_table.setHorizontalHeaderLabels(headers)
        
        for i in range(n_terms):
            self.embeddings_table.setItem(i, 0, QTableWidgetItem(self._terms[i]))
            self.embeddings_table.setItem(i, 1, QTableWidgetItem(str(int(self._labels[i]) + 1)))
            for j in range(n_dims):
                self.embeddings_table.setItem(i, j + 2, 
                    QTableWidgetItem(f"{self._term_coords[i, j]:.4f}"))
        
        self.embeddings_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    
    def _send_outputs(self):
        """Send output tables."""
        if self._term_coords is None:
            self.Outputs.embeddings.send(None)
            self.Outputs.clusters.send(None)
            return
        
        # Embeddings table
        n_dims = self._term_coords.shape[1]
        
        emb_domain = Domain(
            [ContinuousVariable(f"Dim_{i+1}") for i in range(n_dims)],
            metas=[StringVariable("Term"), DiscreteVariable("Cluster", 
                   values=[str(i+1) for i in range(self.n_clusters)])]
        )
        
        metas = np.column_stack([
            np.array(self._terms, dtype=object),
            self._labels.astype(str)
        ])
        
        emb_table = Table.from_numpy(
            emb_domain,
            X=self._term_coords,
            metas=metas
        )
        
        self.Outputs.embeddings.send(emb_table)
        
        # Clusters table
        cluster_domain = Domain(
            [ContinuousVariable("Cluster")],
            metas=[StringVariable("Term")]
        )
        
        cluster_table = Table.from_numpy(
            cluster_domain,
            X=self._labels.reshape(-1, 1),
            metas=np.array(self._terms, dtype=object).reshape(-1, 1)
        )
        
        self.Outputs.clusters.send(cluster_table)


if __name__ == "__main__":
    WidgetPreview(OWFactorialAnalysis).run()
