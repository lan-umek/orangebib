# -*- coding: utf-8 -*-
"""
K-Fields Plot Widget
====================
Orange widget for visualizing relationships between K bibliometric fields
using an interactive Sankey diagram.

Features:
- Hover tooltips on nodes and flows
- Draggable nodes within fields
- Click selection
- Zoom and pan
"""

import logging
from typing import Optional, List, Dict, Tuple
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QPointF, QRectF, QTimer, pyqtSignal
from AnyQt.QtGui import (QColor, QPainter, QPainterPath, QBrush, QPen, QFont, 
                          QLinearGradient, QCursor, QTransform)
from AnyQt.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QScrollArea, QFrame, QSizePolicy, QToolTip,
                              QApplication)

import pyqtgraph as pg

from Orange.data import Table, Domain, ContinuousVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Input, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

FIELD_TYPES = [
    ("Authors", "Authors"),
    ("Author Keywords", "Author Keywords"),
    ("Index Keywords", "Index Keywords"),
    ("Sources", "Source title"),
    ("Countries", "Countries"),
    ("Affiliations", "Affiliations"),
    ("References", "References"),
    ("Year", "Year"),
]

COLOR_BY_OPTIONS = [
    ("Average Year", "avg_year"),
    ("Document Count", "doc_count"),
    ("Citation Count", "citations"),
    ("Flow Weight", "weight"),
]

# Color palettes for fields
FIELD_COLORS = [
    "#3498db",  # Blue
    "#e74c3c",  # Red
    "#2ecc71",  # Green
    "#f39c12",  # Orange
    "#9b59b6",  # Purple
    "#1abc9c",  # Teal
]


# =============================================================================
# INTERACTIVE SANKEY DIAGRAM WIDGET
# =============================================================================

class InteractiveSankeyWidget(QWidget):
    """Interactive Sankey diagram with hover, drag, and selection."""
    
    # Signals
    nodeClicked = pyqtSignal(int, int, str)  # level, idx, name
    flowClicked = pyqtSignal(dict)  # flow data
    selectionChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(700, 500)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Data
        self.nodes: Dict[int, List[Dict]] = {}
        self.flows: List[Dict] = []
        self.color_values: Dict[str, float] = {}
        self.color_by = "weight"
        self.color_min = 0
        self.color_max = 1
        
        # Layout
        self.margin = 80
        self.node_width = 25
        self.node_spacing = 15
        self.min_node_height = 20
        
        # Computed positions
        self.node_rects: Dict[Tuple[int, int], QRectF] = {}
        self.flow_paths: List[Tuple[Dict, QPainterPath]] = []
        
        # Interaction state
        self.hovered_node: Optional[Tuple[int, int]] = None
        self.hovered_flow: Optional[int] = None
        self.selected_nodes: List[Tuple[int, int]] = []
        self.selected_flows: List[int] = []
        
        # Dragging
        self.dragging_node: Optional[Tuple[int, int]] = None
        self.drag_start_pos: Optional[QPointF] = None
        self.drag_start_y: float = 0
        
        # View transform (for zoom/pan)
        self.scale_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        self.panning = False
        self.pan_start = QPointF(0, 0)
        
        # Tooltip timer
        self.tooltip_timer = QTimer()
        self.tooltip_timer.setSingleShot(True)
        self.tooltip_timer.timeout.connect(self._show_tooltip)
        self.tooltip_pos = QPointF(0, 0)
        
    def clear(self):
        """Clear the diagram."""
        self.nodes = {}
        self.flows = []
        self.color_values = {}
        self.node_rects = {}
        self.flow_paths = []
        self.hovered_node = None
        self.hovered_flow = None
        self.selected_nodes = []
        self.selected_flows = []
        self.dragging_node = None
        self.update()
    
    def set_data(self, nodes: Dict[int, List[Dict]], flows: List[Dict],
                 color_values: Dict[str, float] = None, color_by: str = "weight"):
        """Set diagram data."""
        self.nodes = nodes
        self.flows = flows
        self.color_values = color_values or {}
        self.color_by = color_by
        
        # Initialize node positions (y_offset for custom positioning)
        for level, level_nodes in self.nodes.items():
            for node in level_nodes:
                if 'y_offset' not in node:
                    node['y_offset'] = 0
        
        # Calculate color range from both flows and nodes
        all_color_values = list(self.color_values.values())
        for level_nodes in self.nodes.values():
            for node in level_nodes:
                if 'color_value' in node and node['color_value'] > 0:
                    all_color_values.append(node['color_value'])
        
        if all_color_values:
            self.color_min = min(all_color_values)
            self.color_max = max(all_color_values)
        else:
            self.color_min = 0
            self.color_max = 1
        
        # Compute node colors from connected flows (weighted average)
        self._compute_node_colors()
        
        self._compute_layout()
        self.update()
    
    def _compute_node_colors(self):
        """Calculate node colors as weighted average of connected flow colors."""
        node_color_sums = defaultdict(float)
        node_color_weights = defaultdict(float)
        
        for flow in self.flows:
            src_key = (flow['source_level'], flow['source_idx'])
            tgt_key = (flow['target_level'], flow['target_idx'])
            flow_key = f"{flow['source_level']}_{flow['source_idx']}_{flow['target_level']}_{flow['target_idx']}"
            
            if flow_key in self.color_values:
                color_val = self.color_values[flow_key]
                weight = flow.get('value', 1)
                
                node_color_sums[src_key] += color_val * weight
                node_color_weights[src_key] += weight
                
                node_color_sums[tgt_key] += color_val * weight
                node_color_weights[tgt_key] += weight
        
        # Store in nodes
        for level, level_nodes in self.nodes.items():
            for idx, node in enumerate(level_nodes):
                key = (level, idx)
                if node_color_weights[key] > 0:
                    node['color_value'] = node_color_sums[key] / node_color_weights[key]
                elif 'color_value' not in node:
                    node['color_value'] = (self.color_min + self.color_max) / 2
    
    def _compute_layout(self):
        """Compute node rectangles and flow paths."""
        if not self.nodes:
            return
        
        width = self.width()
        height = self.height()
        
        n_levels = len(self.nodes)
        if n_levels < 2:
            return
        
        # Apply scale
        eff_width = width / self.scale_factor
        eff_height = height / self.scale_factor
        
        # Level positions
        level_width = (eff_width - 2 * self.margin - self.node_width) / (n_levels - 1)
        
        # Compute node positions
        self.node_rects = {}
        
        for level in range(n_levels):
            if level not in self.nodes:
                continue
            
            level_nodes = self.nodes[level]
            total_value = sum(max(n.get('value', 1), 1) for n in level_nodes)
            
            x = self.margin + level * level_width
            available_height = eff_height - 2 * self.margin - (len(level_nodes) - 1) * self.node_spacing
            
            y = self.margin
            for idx, node in enumerate(level_nodes):
                value = max(node.get('value', 1), 1)
                node_height = max(self.min_node_height, (value / total_value) * available_height)
                
                # Apply custom y_offset from dragging
                y_with_offset = y + node.get('y_offset', 0)
                
                # Clamp to valid range
                y_with_offset = max(self.margin, min(eff_height - self.margin - node_height, y_with_offset))
                
                self.node_rects[(level, idx)] = QRectF(x, y_with_offset, self.node_width, node_height)
                y += node_height + self.node_spacing
        
        # Compute flow paths
        self._compute_flow_paths()
    
    def _compute_flow_paths(self):
        """Compute bezier paths for all flows."""
        self.flow_paths = []
        
        # Track flow offsets within each node
        src_offsets = defaultdict(float)
        tgt_offsets = defaultdict(float)
        
        for flow in self.flows:
            src_key = (flow['source_level'], flow['source_idx'])
            tgt_key = (flow['target_level'], flow['target_idx'])
            
            if src_key not in self.node_rects or tgt_key not in self.node_rects:
                continue
            
            src_rect = self.node_rects[src_key]
            tgt_rect = self.node_rects[tgt_key]
            
            # Get node values for proportion calculation
            src_node = self.nodes[flow['source_level']][flow['source_idx']]
            tgt_node = self.nodes[flow['target_level']][flow['target_idx']]
            
            src_total = max(src_node.get('value', 1), 1)
            tgt_total = max(tgt_node.get('value', 1), 1)
            flow_value = flow.get('value', 1)
            
            # Calculate flow heights proportional to value
            flow_height_src = (flow_value / src_total) * src_rect.height()
            flow_height_tgt = (flow_value / tgt_total) * tgt_rect.height()
            
            # Source and target points
            sx = src_rect.right()
            sy1 = src_rect.top() + src_offsets[src_key]
            sy2 = sy1 + flow_height_src
            
            tx = tgt_rect.left()
            ty1 = tgt_rect.top() + tgt_offsets[tgt_key]
            ty2 = ty1 + flow_height_tgt
            
            # Update offsets
            src_offsets[src_key] += flow_height_src
            tgt_offsets[tgt_key] += flow_height_tgt
            
            # Create bezier path
            path = QPainterPath()
            cx = (sx + tx) / 2
            
            path.moveTo(sx, sy1)
            path.cubicTo(cx, sy1, cx, ty1, tx, ty1)
            path.lineTo(tx, ty2)
            path.cubicTo(cx, ty2, cx, sy2, sx, sy2)
            path.closeSubpath()
            
            self.flow_paths.append((flow, path))
    
    def _get_flow_color(self, flow: Dict, highlighted: bool = False, 
                        selected: bool = False) -> QColor:
        """Get color for a flow using viridis colormap."""
        key = f"{flow['source_level']}_{flow['source_idx']}_{flow['target_level']}_{flow['target_idx']}"
        
        # Higher alpha for better color visibility
        alpha = 255 if highlighted else (240 if selected else 200)
        
        if key in self.color_values and self.color_max > self.color_min:
            val = self.color_values[key]
            norm = (val - self.color_min) / (self.color_max - self.color_min)
            norm = max(0.0, min(1.0, norm))
            
            # Viridis colormap - accurate saturated values
            viridis = [
                (0.00, (68, 1, 84)),      # Dark purple
                (0.10, (72, 35, 116)),    # Purple
                (0.20, (64, 67, 135)),    # Blue-purple
                (0.30, (52, 94, 141)),    # Blue
                (0.40, (41, 120, 142)),   # Teal-blue
                (0.50, (32, 144, 140)),   # Teal
                (0.60, (34, 167, 132)),   # Teal-green
                (0.70, (68, 190, 112)),   # Green
                (0.80, (121, 209, 81)),   # Light green
                (0.90, (189, 223, 38)),   # Yellow-green
                (1.00, (253, 231, 37)),   # Yellow
            ]
            
            # Find the two colors to interpolate between
            for i in range(len(viridis) - 1):
                if viridis[i][0] <= norm <= viridis[i + 1][0]:
                    t1, c1 = viridis[i]
                    t2, c2 = viridis[i + 1]
                    t = (norm - t1) / (t2 - t1) if t2 > t1 else 0
                    
                    r = int(c1[0] + t * (c2[0] - c1[0]))
                    g = int(c1[1] + t * (c2[1] - c1[1]))
                    b = int(c1[2] + t * (c2[2] - c1[2]))
                    
                    return QColor(r, g, b, alpha)
            
            # Handle edge case for norm = 1.0
            r, g, b = viridis[-1][1]
            return QColor(r, g, b, alpha)
        
        return QColor(150, 150, 150, alpha)
    
    def _get_node_color_from_value(self, value: float) -> QColor:
        """Get solid color for a node using viridis colormap."""
        if self.color_max <= self.color_min:
            return QColor(100, 100, 100)
        
        norm = (value - self.color_min) / (self.color_max - self.color_min)
        norm = max(0.0, min(1.0, norm))
        
        # Same viridis colormap as flows
        viridis = [
            (0.00, (68, 1, 84)),
            (0.10, (72, 35, 116)),
            (0.20, (64, 67, 135)),
            (0.30, (52, 94, 141)),
            (0.40, (41, 120, 142)),
            (0.50, (32, 144, 140)),
            (0.60, (34, 167, 132)),
            (0.70, (68, 190, 112)),
            (0.80, (121, 209, 81)),
            (0.90, (189, 223, 38)),
            (1.00, (253, 231, 37)),
        ]
        
        for i in range(len(viridis) - 1):
            if viridis[i][0] <= norm <= viridis[i + 1][0]:
                t1, c1 = viridis[i]
                t2, c2 = viridis[i + 1]
                t = (norm - t1) / (t2 - t1) if t2 > t1 else 0
                
                r = int(c1[0] + t * (c2[0] - c1[0]))
                g = int(c1[1] + t * (c2[1] - c1[1]))
                b = int(c1[2] + t * (c2[2] - c1[2]))
                
                return QColor(r, g, b)
        
        # Handle edge case for norm = 1.0
        r, g, b = viridis[-1][1]
        return QColor(r, g, b)
    
    def _get_node_at(self, pos: QPointF) -> Optional[Tuple[int, int]]:
        """Find node at position."""
        # Transform position
        transformed = QPointF(pos.x() / self.scale_factor - self.pan_offset.x(),
                             pos.y() / self.scale_factor - self.pan_offset.y())
        
        for key, rect in self.node_rects.items():
            if rect.contains(transformed):
                return key
        return None
    
    def _get_flow_at(self, pos: QPointF) -> Optional[int]:
        """Find flow at position."""
        transformed = QPointF(pos.x() / self.scale_factor - self.pan_offset.x(),
                             pos.y() / self.scale_factor - self.pan_offset.y())
        
        for i, (flow, path) in enumerate(self.flow_paths):
            if path.contains(transformed):
                return i
        return None
    
    def _show_tooltip(self):
        """Show tooltip for hovered element."""
        if self.hovered_node is not None:
            level, idx = self.hovered_node
            node = self.nodes[level][idx]
            
            text = f"<b>{node['name']}</b><br>"
            text += f"Field: {node.get('field_name', f'Field {level+1}')}<br>"
            text += f"Documents: {node.get('value', 0):,}"
            
            if 'citations' in node:
                text += f"<br>Citations: {node['citations']:,}"
            
            QToolTip.showText(QCursor.pos(), text, self)
            
        elif self.hovered_flow is not None and self.hovered_flow < len(self.flow_paths):
            flow, _ = self.flow_paths[self.hovered_flow]
            
            src_node = self.nodes[flow['source_level']][flow['source_idx']]
            tgt_node = self.nodes[flow['target_level']][flow['target_idx']]
            
            text = f"<b>{src_node['name']}</b> → <b>{tgt_node['name']}</b><br>"
            text += f"Co-occurrences: {flow.get('value', 0):,}"
            
            if 'color_value' in flow:
                color_label = COLOR_BY_OPTIONS[[c[1] for c in COLOR_BY_OPTIONS].index(self.color_by)][0] if self.color_by in [c[1] for c in COLOR_BY_OPTIONS] else "Value"
                text += f"<br>{color_label}: {flow['color_value']:.1f}"
            
            QToolTip.showText(QCursor.pos(), text, self)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for hover and drag."""
        pos = event.position() if hasattr(event, 'position') else event.localPos()
        
        # Handle panning
        if self.panning:
            delta = pos - self.pan_start
            self.pan_offset += delta / self.scale_factor
            self.pan_start = pos
            self._compute_layout()
            self.update()
            return
        
        # Handle node dragging
        if self.dragging_node is not None:
            level, idx = self.dragging_node
            node = self.nodes[level][idx]
            
            # Calculate new y_offset
            delta_y = (pos.y() - self.drag_start_pos.y()) / self.scale_factor
            node['y_offset'] = self.drag_start_y + delta_y
            
            self._compute_layout()
            self.update()
            return
        
        # Check for hover
        old_node = self.hovered_node
        old_flow = self.hovered_flow
        
        self.hovered_node = self._get_node_at(pos)
        
        if self.hovered_node is None:
            self.hovered_flow = self._get_flow_at(pos)
        else:
            self.hovered_flow = None
        
        # Update cursor
        if self.hovered_node is not None:
            self.setCursor(Qt.OpenHandCursor)
        elif self.hovered_flow is not None:
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        
        # Schedule tooltip
        if self.hovered_node != old_node or self.hovered_flow != old_flow:
            QToolTip.hideText()
            self.tooltip_timer.stop()
            if self.hovered_node is not None or self.hovered_flow is not None:
                self.tooltip_timer.start(500)
            self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse press for selection and drag start."""
        pos = event.position() if hasattr(event, 'position') else event.localPos()
        
        if event.button() == Qt.MiddleButton:
            # Start panning
            self.panning = True
            self.pan_start = pos
            self.setCursor(Qt.ClosedHandCursor)
            return
        
        if event.button() == Qt.LeftButton:
            # Check for node
            node_key = self._get_node_at(pos)
            
            if node_key is not None:
                # Start dragging
                self.dragging_node = node_key
                self.drag_start_pos = pos
                self.drag_start_y = self.nodes[node_key[0]][node_key[1]].get('y_offset', 0)
                self.setCursor(Qt.ClosedHandCursor)
                
                # Toggle selection
                if event.modifiers() & Qt.ControlModifier:
                    if node_key in self.selected_nodes:
                        self.selected_nodes.remove(node_key)
                    else:
                        self.selected_nodes.append(node_key)
                else:
                    self.selected_nodes = [node_key]
                
                level, idx = node_key
                self.nodeClicked.emit(level, idx, self.nodes[level][idx]['name'])
                self.selectionChanged.emit()
                self.update()
                return
            
            # Check for flow
            flow_idx = self._get_flow_at(pos)
            if flow_idx is not None:
                if event.modifiers() & Qt.ControlModifier:
                    if flow_idx in self.selected_flows:
                        self.selected_flows.remove(flow_idx)
                    else:
                        self.selected_flows.append(flow_idx)
                else:
                    self.selected_flows = [flow_idx]
                
                self.flowClicked.emit(self.flow_paths[flow_idx][0])
                self.selectionChanged.emit()
                self.update()
                return
            
            # Click on empty space - clear selection
            if not (event.modifiers() & Qt.ControlModifier):
                self.selected_nodes = []
                self.selected_flows = []
                self.selectionChanged.emit()
                self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MiddleButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
        
        if event.button() == Qt.LeftButton:
            if self.dragging_node is not None:
                self.dragging_node = None
                self.setCursor(Qt.OpenHandCursor if self.hovered_node else Qt.ArrowCursor)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y()
        
        if delta > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor /= 1.1
        
        self.scale_factor = max(0.5, min(3.0, self.scale_factor))
        
        self._compute_layout()
        self.update()
    
    def mouseDoubleClickEvent(self, event):
        """Reset view on double click."""
        self.scale_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        
        # Reset node positions
        for level_nodes in self.nodes.values():
            for node in level_nodes:
                node['y_offset'] = 0
        
        self._compute_layout()
        self.update()
    
    def leaveEvent(self, event):
        """Handle mouse leave."""
        self.hovered_node = None
        self.hovered_flow = None
        self.tooltip_timer.stop()
        QToolTip.hideText()
        self.setCursor(Qt.ArrowCursor)
        self.update()
    
    def resizeEvent(self, event):
        """Handle resize."""
        super().resizeEvent(event)
        self._compute_layout()
    
    def paintEvent(self, event):
        """Paint the Sankey diagram."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        
        if not self.nodes:
            painter.setPen(QPen(QColor(150, 150, 150)))
            painter.drawText(self.rect(), Qt.AlignCenter, 
                           "Click 'Generate Plot' to create diagram\n\n"
                           "Mouse controls:\n"
                           "• Drag nodes to reposition\n"
                           "• Scroll to zoom\n"
                           "• Middle-click drag to pan\n"
                           "• Double-click to reset")
            return
        
        # Apply transform
        painter.scale(self.scale_factor, self.scale_factor)
        painter.translate(self.pan_offset)
        
        n_levels = len(self.nodes)
        
        # Draw flows first (behind nodes)
        for i, (flow, path) in enumerate(self.flow_paths):
            is_highlighted = (i == self.hovered_flow)
            is_selected = (i in self.selected_flows)
            
            # Also highlight flows connected to selected/hovered nodes
            src_key = (flow['source_level'], flow['source_idx'])
            tgt_key = (flow['target_level'], flow['target_idx'])
            
            if src_key in self.selected_nodes or tgt_key in self.selected_nodes:
                is_selected = True
            if src_key == self.hovered_node or tgt_key == self.hovered_node:
                is_highlighted = True
            
            color = self._get_flow_color(flow, is_highlighted, is_selected)
            
            if is_highlighted:
                # Draw glow effect
                glow_color = QColor(color)
                glow_color.setAlpha(50)
                painter.setPen(QPen(glow_color, 8))
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawPath(path)
            
            if is_selected:
                # Draw selection outline
                painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)
        
        # Draw nodes
        for level in range(n_levels):
            if level not in self.nodes:
                continue
            
            for idx, node in enumerate(self.nodes[level]):
                key = (level, idx)
                if key not in self.node_rects:
                    continue
                
                rect = self.node_rects[key]
                
                is_hovered = (key == self.hovered_node)
                is_selected = (key in self.selected_nodes)
                
                # Get node color from viridis based on color_value
                base_color = self._get_node_color_from_value(node.get('color_value', 0))
                
                # Determine color with hover/selection modifiers
                if is_hovered:
                    color = base_color.lighter(120)
                elif is_selected:
                    color = base_color.lighter(110)
                else:
                    color = base_color
                
                # Draw shadow for hovered/selected
                if is_hovered or is_selected:
                    shadow_rect = rect.adjusted(2, 2, 2, 2)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(QColor(0, 0, 0, 50)))
                    painter.drawRoundedRect(shadow_rect, 3, 3)
                
                # Draw node
                pen_width = 3 if is_selected else (2 if is_hovered else 1)
                pen_color = QColor(255, 255, 255) if is_selected else color.darker(130)
                
                painter.setPen(QPen(pen_color, pen_width))
                painter.setBrush(QBrush(color))
                painter.drawRoundedRect(rect, 3, 3)
                
                # Draw label
                painter.setPen(QPen(QColor(30, 30, 30)))
                font = QFont("Arial", 9)
                if is_hovered or is_selected:
                    font.setBold(True)
                painter.setFont(font)
                
                label = node['name']
                if len(label) > 25:
                    label = label[:22] + "..."
                
                # Position label
                if level == 0:
                    text_rect = QRectF(0, rect.top(), rect.left() - 10, rect.height())
                    painter.drawText(text_rect, Qt.AlignRight | Qt.AlignVCenter, label)
                elif level == n_levels - 1:
                    text_rect = QRectF(rect.right() + 10, rect.top(), 200, rect.height())
                    painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, label)
                else:
                    # Middle labels - show on hover only or if selected
                    if is_hovered or is_selected:
                        text_rect = QRectF(rect.left() - 80, rect.top() - 18, 
                                          160 + self.node_width, 16)
                        painter.drawText(text_rect, Qt.AlignCenter, label)
        
        # Draw field headers
        painter.setPen(QPen(QColor(50, 50, 50)))
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)
        
        width = self.width() / self.scale_factor
        level_width = (width - 2 * self.margin - self.node_width) / (n_levels - 1) if n_levels > 1 else 0
        
        for level in range(n_levels):
            if level not in self.nodes or not self.nodes[level]:
                continue
            
            x = self.margin + level * level_width
            field_name = self.nodes[level][0].get('field_name', f'Field {level + 1}')
            
            # Draw background for header
            header_rect = QRectF(x - 60, 8, 120 + self.node_width, 25)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(FIELD_COLORS[level % len(FIELD_COLORS)]).lighter(150)))
            painter.drawRoundedRect(header_rect, 5, 5)
            
            painter.setPen(QPen(QColor(50, 50, 50)))
            painter.drawText(header_rect, Qt.AlignCenter, field_name)
        
        # Draw color legend
        self._draw_legend(painter)
    
    def _draw_legend(self, painter: QPainter):
        """Draw color legend."""
        if not self.color_values:
            return
        
        width = self.width() / self.scale_factor
        height = self.height() / self.scale_factor
        
        legend_width = 150
        legend_height = 20
        legend_x = width - legend_width - 20
        legend_y = height - 50
        
        # Background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
        painter.drawRoundedRect(QRectF(legend_x - 10, legend_y - 25, legend_width + 20, 55), 5, 5)
        
        # Title
        painter.setPen(QPen(QColor(50, 50, 50)))
        font = QFont("Arial", 9)
        painter.setFont(font)
        
        color_label = "Value"
        for name, key in COLOR_BY_OPTIONS:
            if key == self.color_by:
                color_label = name
                break
        
        painter.drawText(QRectF(legend_x, legend_y - 20, legend_width, 15), 
                        Qt.AlignCenter, color_label)
        
        # Gradient bar
        gradient = QLinearGradient(legend_x, legend_y, legend_x + legend_width, legend_y)
        
        # Viridis colors - matching the flow colors
        gradient.setColorAt(0.0, QColor(68, 1, 84))
        gradient.setColorAt(0.1, QColor(72, 35, 116))
        gradient.setColorAt(0.2, QColor(64, 67, 135))
        gradient.setColorAt(0.3, QColor(52, 94, 141))
        gradient.setColorAt(0.4, QColor(41, 120, 142))
        gradient.setColorAt(0.5, QColor(32, 144, 140))
        gradient.setColorAt(0.6, QColor(34, 167, 132))
        gradient.setColorAt(0.7, QColor(68, 190, 112))
        gradient.setColorAt(0.8, QColor(121, 209, 81))
        gradient.setColorAt(0.9, QColor(189, 223, 38))
        gradient.setColorAt(1.0, QColor(253, 231, 37))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.drawRect(QRectF(legend_x, legend_y, legend_width, legend_height))
        
        # Labels
        font = QFont("Arial", 8)
        painter.setFont(font)
        painter.setPen(QPen(QColor(50, 50, 50)))
        
        min_str = f"{self.color_min:.0f}" if self.color_min > 100 else f"{self.color_min:.1f}"
        max_str = f"{self.color_max:.0f}" if self.color_max > 100 else f"{self.color_max:.1f}"
        
        painter.drawText(QRectF(legend_x, legend_y + legend_height + 2, 50, 15),
                        Qt.AlignLeft, min_str)
        painter.drawText(QRectF(legend_x + legend_width - 50, legend_y + legend_height + 2, 50, 15),
                        Qt.AlignRight, max_str)


# =============================================================================
# MAIN WIDGET
# =============================================================================

class OWKFieldsPlot(OWWidget):
    """Visualize relationships between K bibliometric fields using Sankey diagram."""
    
    name = "K-Fields Plot"
    description = "Visualize relationships between K bibliometric fields using Sankey diagram"
    icon = "icons/kfields.svg"
    priority = 96
    keywords = ["sankey", "fields", "relationships", "flow", "connections"]
    category = "Biblium"
    
    class Inputs:
        data = Input("Data", Table, doc="Bibliographic data")
    
    class Outputs:
        flow_data = Output("Flow Data", Table, doc="Flow relationships between fields")
        selected_data = Output("Selected Documents", Table, doc="Data for selected nodes/flows")
    
    # Settings
    n_fields = settings.Setting(3)
    
    field1 = settings.Setting("Authors")
    field2 = settings.Setting("Author Keywords")
    field3 = settings.Setting("Sources")
    field4 = settings.Setting("Countries")
    field5 = settings.Setting("Affiliations")
    field6 = settings.Setting("References")
    
    top1 = settings.Setting(10)
    top2 = settings.Setting(10)
    top3 = settings.Setting(10)
    top4 = settings.Setting(10)
    top5 = settings.Setting(10)
    top6 = settings.Setting(10)
    
    color_by_index = settings.Setting(0)
    
    want_main_area = True
    
    class Error(OWWidget.Error):
        no_data = Msg("No input data")
        no_flows = Msg("No connections found between selected fields")
        analysis_failed = Msg("Analysis failed: {}")
    
    class Warning(OWWidget.Warning):
        few_connections = Msg("Only {} connections found")
        column_not_found = Msg("Column '{}' not found, using fallback")
    
    class Information(OWWidget.Information):
        generated = Msg("Generated diagram with {} nodes and {} flows")
    
    def __init__(self):
        super().__init__()
        
        self._data: Optional[Table] = None
        self._df: Optional[pd.DataFrame] = None
        self._columns: List[str] = []
        self._entity_doc_map: Dict[str, List[int]] = {}
        
        self._setup_gui()
    
    def _setup_gui(self):
        # Number of Fields
        k_box = gui.widgetBox(self.controlArea, "Number of Fields (K)")
        
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("K:"))
        self.k_spin = gui.spin(
            k_box, self, "n_fields", minv=2, maxv=6,
            callback=self._on_k_changed,
            controlWidth=60,
        )
        k_layout.addWidget(QLabel("(2-6 fields)"))
        k_layout.addStretch()
        
        # Field Selection
        self.field_box = gui.widgetBox(self.controlArea, "Field Selection")
        
        self.field_combos = []
        self.top_spins = []
        
        for i in range(6):
            row = QHBoxLayout()
            
            label = QLabel(f"Field {i+1}:")
            label.setFixedWidth(50)
            row.addWidget(label)
            
            combo = gui.comboBox(
                None, self, f"field{i+1}",
                items=[f[0] for f in FIELD_TYPES],
                sendSelectedValue=True,
            )
            combo.setFixedWidth(130)
            row.addWidget(combo)
            self.field_combos.append(combo)
            
            row.addWidget(QLabel("Top:"))
            
            spin = gui.spin(
                None, self, f"top{i+1}",
                minv=5, maxv=50,
                controlWidth=50,
            )
            row.addWidget(spin)
            self.top_spins.append(spin)
            
            row.addStretch()
            
            container = QWidget()
            container.setLayout(row)
            self.field_box.layout().addWidget(container)
            
            setattr(self, f"field_row_{i}", container)
        
        self._update_field_visibility()
        
        # Coloring
        color_box = gui.widgetBox(self.controlArea, "Coloring")
        
        gui.comboBox(
            color_box, self, "color_by_index",
            items=[c[0] for c in COLOR_BY_OPTIONS],
            label="Color By:",
            orientation=Qt.Horizontal,
        )
        
        # Generate button
        self.gen_btn = gui.button(
            self.controlArea, self, "Generate Plot",
            callback=self._generate_plot,
        )
        self.gen_btn.setMinimumHeight(35)
        self.gen_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        
        # Help text
        help_label = QLabel(
            "<small>• Drag nodes to reposition<br>"
            "• Scroll to zoom<br>"
            "• Middle-click to pan<br>"
            "• Double-click to reset<br>"
            "• Ctrl+click for multi-select</small>"
        )
        help_label.setStyleSheet("color: #7f8c8d;")
        self.controlArea.layout().addWidget(help_label)
        
        self.controlArea.layout().addStretch(1)
        
        # Main area - Interactive Sankey
        self.sankey = InteractiveSankeyWidget()
        self.sankey.selectionChanged.connect(self._on_selection_changed)
        
        self.mainArea.layout().addWidget(self.sankey)
    
    def _on_k_changed(self):
        self._update_field_visibility()
    
    def _update_field_visibility(self):
        for i in range(6):
            row = getattr(self, f"field_row_{i}")
            row.setVisible(i < self.n_fields)
    
    def _on_selection_changed(self):
        """Handle selection changes in Sankey diagram."""
        self._send_selected_data()
    
    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        self._data = data
        self._df = None
        self._columns = []
        self._entity_doc_map = {}
        
        self.sankey.clear()
        
        if data is None:
            self.Error.no_data()
            return
        
        self._df = self._table_to_df(data)
        self._columns = list(self._df.columns)
    
    def _table_to_df(self, table: Table) -> pd.DataFrame:
        data = {}
        for var in table.domain.attributes:
            data[var.name] = table.get_column(var)
        for var in table.domain.metas:
            data[var.name] = table.get_column(var)
        for var in table.domain.class_vars:
            data[var.name] = table.get_column(var)
        return pd.DataFrame(data)
    
    def _get_column_for_field(self, field_name: str) -> Optional[str]:
        for display, col in FIELD_TYPES:
            if display == field_name:
                if col in self._columns:
                    return col
                break
        
        field_lower = field_name.lower()
        for col in self._columns:
            if field_lower in col.lower():
                return col
        
        return None
    
    def _extract_entities(self, column: str, top_n: int) -> List[str]:
        counter = Counter()
        
        for idx, val in enumerate(self._df[column]):
            if pd.isna(val):
                continue
            
            val_str = str(val)
            
            for sep in [";", "|"]:
                if sep in val_str:
                    entities = [e.strip() for e in val_str.split(sep) if e.strip()]
                    break
            else:
                entities = [val_str.strip()] if val_str.strip() else []
            
            for entity in entities:
                counter[entity] += 1
                # Track entity -> document mapping
                if entity not in self._entity_doc_map:
                    self._entity_doc_map[entity] = []
                self._entity_doc_map[entity].append(idx)
        
        return [e[0] for e in counter.most_common(top_n)]
    
    def _compute_flows(self, fields: List[Tuple[str, str, int]]) -> Tuple[Dict, List]:
        field_entities = []
        for field_name, col_name, top_n in fields:
            entities = self._extract_entities(col_name, top_n)
            field_entities.append(entities)
        
        nodes = {}
        for level, (entities, (field_name, col_name, top_n)) in enumerate(zip(field_entities, fields)):
            nodes[level] = [
                {"name": e, "value": 0, "field_name": field_name, "color_value": 0, "color_count": 0}
                for e in entities
            ]
        
        flows = []
        
        entity_to_idx = {}
        for level, entities in enumerate(field_entities):
            for idx, entity in enumerate(entities):
                entity_to_idx[(level, entity)] = idx
        
        # Get year and citation columns
        year_col = None
        cit_col = None
        for col in self._columns:
            col_lower = col.lower()
            if 'year' in col_lower and year_col is None:
                year_col = col
            if ('cited' in col_lower or 'citation' in col_lower) and cit_col is None:
                cit_col = col
        
        # Track node color values
        node_years = defaultdict(list)
        node_citations = defaultdict(int)
        node_docs = defaultdict(int)
        
        for level in range(len(fields) - 1):
            col1 = fields[level][1]
            col2 = fields[level + 1][1]
            entities1 = set(field_entities[level])
            entities2 = set(field_entities[level + 1])
            
            flow_counts = defaultdict(int)
            flow_years = defaultdict(list)
            flow_citations = defaultdict(int)
            
            for idx, row in self._df.iterrows():
                val1 = row[col1]
                val2 = row[col2]
                
                if pd.isna(val1) or pd.isna(val2):
                    continue
                
                e1_list = []
                e2_list = []
                
                val1_str = str(val1)
                val2_str = str(val2)
                
                for sep in [";", "|"]:
                    if sep in val1_str:
                        e1_list = [e.strip() for e in val1_str.split(sep) if e.strip()]
                        break
                else:
                    e1_list = [val1_str.strip()] if val1_str.strip() else []
                
                for sep in [";", "|"]:
                    if sep in val2_str:
                        e2_list = [e.strip() for e in val2_str.split(sep) if e.strip()]
                        break
                else:
                    e2_list = [val2_str.strip()] if val2_str.strip() else []
                
                year = None
                citations = 0
                if year_col:
                    try:
                        year = float(row[year_col])
                    except:
                        pass
                if cit_col:
                    try:
                        citations = float(row[cit_col])
                    except:
                        pass
                
                # Track node color values for all entities in this document
                for e1 in e1_list:
                    if e1 in entities1:
                        node_key = (level, e1)
                        node_docs[node_key] += 1
                        if year:
                            node_years[node_key].append(year)
                        node_citations[node_key] += citations
                
                for e2 in e2_list:
                    if e2 in entities2:
                        node_key = (level + 1, e2)
                        node_docs[node_key] += 1
                        if year:
                            node_years[node_key].append(year)
                        node_citations[node_key] += citations
                
                for e1 in e1_list:
                    if e1 not in entities1:
                        continue
                    for e2 in e2_list:
                        if e2 not in entities2:
                            continue
                        
                        key = (e1, e2)
                        flow_counts[key] += 1
                        if year:
                            flow_years[key].append(year)
                        flow_citations[key] += citations
            
            for (e1, e2), count in flow_counts.items():
                if count == 0:
                    continue
                
                src_idx = entity_to_idx.get((level, e1))
                tgt_idx = entity_to_idx.get((level + 1, e2))
                
                if src_idx is None or tgt_idx is None:
                    continue
                
                nodes[level][src_idx]['value'] += count
                nodes[level + 1][tgt_idx]['value'] += count
                
                color_by = COLOR_BY_OPTIONS[self.color_by_index][1]
                if color_by == "avg_year" and flow_years[(e1, e2)]:
                    color_val = np.mean(flow_years[(e1, e2)])
                elif color_by == "citations":
                    color_val = flow_citations[(e1, e2)]
                elif color_by == "doc_count":
                    color_val = count
                else:
                    color_val = count
                
                flows.append({
                    'source_level': level,
                    'source_idx': src_idx,
                    'target_level': level + 1,
                    'target_idx': tgt_idx,
                    'value': count,
                    'color_value': color_val,
                    'source_name': e1,
                    'target_name': e2,
                })
        
        # Calculate node color values
        color_by = COLOR_BY_OPTIONS[self.color_by_index][1]
        for level in range(len(fields)):
            for idx, node in enumerate(nodes[level]):
                entity = node['name']
                node_key = (level, entity)
                
                if color_by == "avg_year" and node_years[node_key]:
                    node['color_value'] = np.mean(node_years[node_key])
                elif color_by == "citations":
                    node['color_value'] = node_citations[node_key]
                elif color_by == "doc_count":
                    node['color_value'] = node_docs[node_key]
                else:
                    node['color_value'] = node_docs[node_key]
        
        return nodes, flows
    
    def _generate_plot(self):
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        
        if self._df is None:
            self.Error.no_data()
            return
        
        try:
            self._entity_doc_map = {}
            
            fields = []
            for i in range(self.n_fields):
                field_name = getattr(self, f"field{i+1}")
                top_n = getattr(self, f"top{i+1}")
                
                col_name = self._get_column_for_field(field_name)
                if col_name is None:
                    self.Warning.column_not_found(field_name)
                    continue
                
                fields.append((field_name, col_name, top_n))
            
            if len(fields) < 2:
                self.Error.no_flows()
                return
            
            nodes, flows = self._compute_flows(fields)
            
            if not flows:
                self.Error.no_flows()
                return
            
            color_values = {}
            for flow in flows:
                key = f"{flow['source_level']}_{flow['source_idx']}_{flow['target_level']}_{flow['target_idx']}"
                color_values[key] = flow.get('color_value', flow['value'])
            
            self.sankey.set_data(nodes, flows, color_values, 
                               COLOR_BY_OPTIONS[self.color_by_index][1])
            
            n_nodes = sum(len(n) for n in nodes.values())
            n_flows = len(flows)
            
            self.Information.generated(n_nodes, n_flows)
            
            if n_flows < 10:
                self.Warning.few_connections(n_flows)
            
            self._send_output(flows, fields)
            
        except Exception as e:
            logger.exception(f"Plot generation failed: {e}")
            self.Error.analysis_failed(str(e))
    
    def _send_output(self, flows: List[Dict], fields: List[Tuple]):
        if not flows:
            self.Outputs.flow_data.send(None)
            return
        
        data = []
        for flow in flows:
            src_level = flow['source_level']
            tgt_level = flow['target_level']
            
            data.append({
                'Source_Field': fields[src_level][0],
                'Source': flow['source_name'],
                'Target_Field': fields[tgt_level][0],
                'Target': flow['target_name'],
                'Weight': flow['value'],
                'Color_Value': flow.get('color_value', flow['value']),
            })
        
        df = pd.DataFrame(data)
        
        domain = Domain(
            [ContinuousVariable("Weight"), ContinuousVariable("Color_Value")],
            metas=[StringVariable("Source_Field"), StringVariable("Source"),
                   StringVariable("Target_Field"), StringVariable("Target")]
        )
        
        table = Table.from_numpy(
            domain,
            X=df[["Weight", "Color_Value"]].values,
            metas=df[["Source_Field", "Source", "Target_Field", "Target"]].values.astype(object)
        )
        
        self.Outputs.flow_data.send(table)
    
    def _send_selected_data(self):
        """Send data for selected nodes/flows."""
        if self._data is None:
            self.Outputs.selected_data.send(None)
            return
        
        selected_doc_indices = set()
        
        # Get documents for selected nodes
        for level, idx in self.sankey.selected_nodes:
            if level in self.sankey.nodes and idx < len(self.sankey.nodes[level]):
                entity = self.sankey.nodes[level][idx]['name']
                if entity in self._entity_doc_map:
                    selected_doc_indices.update(self._entity_doc_map[entity])
        
        # Get documents for selected flows
        for flow_idx in self.sankey.selected_flows:
            if flow_idx < len(self.sankey.flow_paths):
                flow, _ = self.sankey.flow_paths[flow_idx]
                src_entity = flow.get('source_name', '')
                tgt_entity = flow.get('target_name', '')
                
                src_docs = set(self._entity_doc_map.get(src_entity, []))
                tgt_docs = set(self._entity_doc_map.get(tgt_entity, []))
                
                # Intersection - documents with both entities
                selected_doc_indices.update(src_docs & tgt_docs)
        
        if selected_doc_indices:
            selected_indices = sorted(selected_doc_indices)
            selected_table = self._data[selected_indices]
            self.Outputs.selected_data.send(selected_table)
        else:
            self.Outputs.selected_data.send(None)


if __name__ == "__main__":
    WidgetPreview(OWKFieldsPlot).run()
