# -*- coding: utf-8 -*-
"""
Load Bibliographic Data Widget - Orange widget for bibliometric data.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

from AnyQt.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox,
    QGroupBox, QRadioButton, QButtonGroup, QFileDialog,
    QProgressBar, QTabWidget, QCheckBox, QPlainTextEdit,
    QScrollArea, QFrame, QApplication,
)
from AnyQt.QtCore import Qt, QThread, pyqtSignal, QTimer

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
from Orange.widgets import gui, settings
from Orange.widgets.widget import OWWidget, Output, Msg
from Orange.widgets.utils.widgetpreview import WidgetPreview

# Try to import biblium
HAS_BIBLIUM = False
BIBLIUM_DIR = None
BIBLIUM_DATA_DIR = None
BIBLIUM_ADDITIONAL_DIR = None
COUNTRY_CODE_TO_NAME = {}

try:
    import biblium
    from biblium import readbib
    from biblium.biblium_main import BiblioAnalysis
    BIBLIUM_DIR = os.path.dirname(biblium.__file__)
    BIBLIUM_DATA_DIR = os.path.join(BIBLIUM_DIR, "data")
    BIBLIUM_ADDITIONAL_DIR = os.path.join(BIBLIUM_DIR, "additional files")
    if os.path.exists(BIBLIUM_DATA_DIR):
        HAS_BIBLIUM = True
    
    # Load country code to name mapping from Biblium
    countries_file = os.path.join(BIBLIUM_ADDITIONAL_DIR, "countries.xlsx")
    if os.path.exists(countries_file):
        try:
            df_countries = pd.read_excel(countries_file)
            COUNTRY_CODE_TO_NAME = df_countries.set_index("Code").to_dict()["Name"]
        except Exception:
            pass
except Exception:
    pass

# Fallback country codes if Biblium not available
if not COUNTRY_CODE_TO_NAME:
    COUNTRY_CODE_TO_NAME = {
        "US": "United States", "GB": "United Kingdom", "DE": "Germany",
        "FR": "France", "IT": "Italy", "ES": "Spain", "NL": "Netherlands",
        "BE": "Belgium", "AT": "Austria", "CH": "Switzerland", "SE": "Sweden",
        "NO": "Norway", "DK": "Denmark", "FI": "Finland", "PL": "Poland",
        "CZ": "Czech Republic", "HU": "Hungary", "PT": "Portugal", "GR": "Greece",
        "IE": "Ireland", "RO": "Romania", "BG": "Bulgaria", "HR": "Croatia",
        "SI": "Slovenia", "SK": "Slovakia", "LT": "Lithuania", "LV": "Latvia",
        "EE": "Estonia", "CY": "Cyprus", "LU": "Luxembourg", "MT": "Malta",
        "CN": "China", "JP": "Japan", "KR": "South Korea", "IN": "India",
        "AU": "Australia", "NZ": "New Zealand", "CA": "Canada", "MX": "Mexico",
        "BR": "Brazil", "AR": "Argentina", "CL": "Chile", "CO": "Colombia",
        "RU": "Russia", "UA": "Ukraine", "TR": "Turkey", "IL": "Israel",
        "SA": "Saudi Arabia", "AE": "United Arab Emirates", "EG": "Egypt",
        "ZA": "South Africa", "NG": "Nigeria", "KE": "Kenya", "MA": "Morocco",
        "TW": "Taiwan", "HK": "Hong Kong", "SG": "Singapore", "MY": "Malaysia",
        "TH": "Thailand", "ID": "Indonesia", "PH": "Philippines", "VN": "Vietnam",
        "PK": "Pakistan", "BD": "Bangladesh", "IR": "Iran", "IQ": "Iraq",
    }

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


DATABASE_OPTIONS = {
    "Auto-detect": "",
    "OpenAlex": "oa",
    "Scopus": "scopus",
    "Web of Science": "wos",
    "PubMed": "pubmed",
    "Dimensions": "dimensions",
    "Lens.org": "lens",
    "Generic CSV": "generic",
}

PREPROCESS_LEVELS = {
    0: "0 - None (raw data)",
    1: "1 - Basic (countries, labels)",
    2: "2 - Standard (keywords, text)",
    3: "3 - Extended (science mappings)",
    4: "4 - Full (interdisciplinarity)",
}

SAMPLE_DATASETS = {
    "none": {"name": "None", "filename": None, "db": "", "records": ""},
    "scopus": {"name": "Scopus Sample", "filename": "scopus dataset.csv", "db": "scopus", "records": "~200"},
    "wos": {"name": "WoS Sample", "filename": "wos dataset.txt", "db": "wos", "records": "~500"},
    "openalex": {"name": "OpenAlex Sample", "filename": "open alex dataset.csv", "db": "oa", "records": "~600"},
}

SEARCH_BY_OPTIONS = ["Keywords", "Author", "Institution", "ISSN/Journal", "DOI List"]
DOC_TYPE_OPTIONS = ["All", "article", "review", "book-chapter", "book", "proceedings-article", "preprint", "dissertation"]


class FetchWorker(QThread):
    """Background worker for API fetching with progress signals."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, str)  # result, error
    
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    def run(self):
        try:
            import requests
            
            query = self.params["query"]
            search_type = self.params["search_type"]
            year_from = self.params["year_from"]
            year_to = self.params["year_to"]
            doc_type = self.params.get("doc_type")
            max_results = self.params["max_results"]
            open_access = self.params.get("open_access", False)
            email = self.params.get("email", "")
            
            self.progress.emit(5, "Building query...")
            
            # Build filters
            filter_parts = [f"publication_year:{year_from}-{year_to}"]
            
            if doc_type:
                filter_parts.append(f"type:{doc_type}")
            if open_access:
                filter_parts.append("is_oa:true")
            
            # Handle search types - two-step process for author/institution
            search_param = None
            if search_type == "Keywords":
                search_param = query
            elif search_type == "Author":
                if "0000-" in query:
                    # ORCID - use directly
                    orcid = query.strip()
                    if not orcid.startswith("https://"):
                        orcid = f"https://orcid.org/{orcid}"
                    filter_parts.append(f"author.orcid:{orcid}")
                else:
                    # Two-step: search authors first, then filter works by author ID
                    self.progress.emit(7, "Finding author...")
                    try:
                        author_resp = requests.get(
                            "https://api.openalex.org/authors",
                            params={"search": query, "per_page": 5},
                            timeout=15
                        )
                        if author_resp.status_code == 200:
                            author_data = author_resp.json()
                            authors = author_data.get("results", [])
                            if authors:
                                author_ids = [a.get("id") for a in authors[:3] if a.get("id")]
                                if author_ids:
                                    filter_parts.append(f"authorships.author.id:{('|').join(author_ids)}")
                                else:
                                    self.finished.emit(None, f"No authors found matching '{query}'")
                                    return
                            else:
                                self.finished.emit(None, f"No authors found matching '{query}'")
                                return
                        else:
                            self.finished.emit(None, f"Author search failed: {author_resp.status_code}")
                            return
                    except Exception as e:
                        self.finished.emit(None, f"Author search error: {e}")
                        return
            elif search_type == "Institution":
                # Two-step: search institutions first
                self.progress.emit(7, "Finding institution...")
                try:
                    inst_resp = requests.get(
                        "https://api.openalex.org/institutions",
                        params={"search": query, "per_page": 3},
                        timeout=15
                    )
                    if inst_resp.status_code == 200:
                        inst_data = inst_resp.json()
                        institutions = inst_data.get("results", [])
                        if institutions:
                            inst_ids = [i.get("id") for i in institutions[:2] if i.get("id")]
                            if inst_ids:
                                filter_parts.append(f"institutions.id:{('|').join(inst_ids)}")
                            else:
                                self.finished.emit(None, f"No institutions found matching '{query}'")
                                return
                        else:
                            self.finished.emit(None, f"No institutions found matching '{query}'")
                            return
                    else:
                        self.finished.emit(None, f"Institution search failed: {inst_resp.status_code}")
                        return
                except Exception as e:
                    self.finished.emit(None, f"Institution search error: {e}")
                    return
            elif search_type == "ISSN/Journal":
                if query.replace("-", "").isdigit():
                    filter_parts.append(f"primary_location.source.issn:{query}")
                else:
                    filter_parts.append(f"primary_location.source.display_name.search:{query}")
            elif search_type == "DOI List":
                dois = [d.strip() for d in query.split(",") if d.strip()]
                if dois:
                    filter_parts.append(f"doi:{('|').join(dois)}")
            
            base_url = "https://api.openalex.org/works"
            per_page = min(200, max_results)
            
            params = {
                "per_page": per_page,
                "filter": ",".join(filter_parts),
                "cursor": "*",
            }
            if search_param:
                params["search"] = search_param
            if email:
                params["mailto"] = email
            
            self.progress.emit(10, "Connecting to OpenAlex...")
            
            records = []
            page = 0
            max_pages = (max_results // per_page) + 2
            
            while len(records) < max_results and page < max_pages:
                if self._cancelled:
                    self.finished.emit(None, "Cancelled")
                    return
                
                page += 1
                pct = min(85, 10 + int(75 * len(records) / max_results))
                self.progress.emit(pct, f"Page {page}: {len(records)} records...")
                
                try:
                    resp = requests.get(base_url, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                except requests.exceptions.Timeout:
                    self.finished.emit(None, "Request timed out (30s). Try reducing max results.")
                    return
                except requests.exceptions.ConnectionError as e:
                    self.finished.emit(None, f"Connection error: {e}")
                    return
                except requests.exceptions.RequestException as e:
                    self.finished.emit(None, f"Request error: {e}")
                    return
                
                results = data.get("results", [])
                if not results:
                    break
                
                for work in results:
                    rec = self._parse_work(work)
                    records.append(rec)
                    if len(records) >= max_results:
                        break
                
                cursor = data.get("meta", {}).get("next_cursor")
                if not cursor:
                    break
                params["cursor"] = cursor
            
            self.progress.emit(90, f"Creating DataFrame ({len(records)} records)...")
            
            df = pd.DataFrame(records) if records else pd.DataFrame()
            
            self.progress.emit(100, "Complete")
            self.finished.emit(df, "")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(None, f"{type(e).__name__}: {e}")
    
    def _parse_work(self, work):
        """Parse OpenAlex work to dict with Bibliometric Counts-compatible column names."""
        authors = []
        author_ids = []
        institutions = []
        country_codes = set()
        
        for auth in work.get("authorships", []):
            ai = auth.get("author", {})
            if ai.get("display_name"):
                authors.append(ai["display_name"])
            if ai.get("id"):
                author_ids.append(ai["id"])
            for inst in auth.get("institutions", []):
                if inst.get("display_name"):
                    institutions.append(inst["display_name"])
                if inst.get("country_code"):
                    country_codes.add(inst["country_code"])
            for c in auth.get("countries", []):
                country_codes.add(c)
        
        # Convert country codes to names
        country_names = []
        for code in country_codes:
            if code:
                name = COUNTRY_CODE_TO_NAME.get(code, code)
                country_names.append(name)
        
        source_name = None
        source_issn = None
        loc = work.get("primary_location", {})
        if loc and loc.get("source"):
            src = loc["source"]
            source_name = src.get("display_name")
            source_issn = src.get("issn_l")
        
        # Extract keywords from multiple OpenAlex fields
        concepts = [c.get("display_name", "") for c in work.get("concepts", [])[:10] if c.get("display_name")]
        topics = [t.get("display_name", "") for t in work.get("topics", [])[:5] if t.get("display_name")]
        keywords_raw = [k.get("display_name", "") or k.get("keyword", "") for k in work.get("keywords", [])]
        keywords_raw = [k for k in keywords_raw if k]
        sdgs = [s.get("display_name", "") for s in work.get("sustainable_development_goals", []) if s.get("display_name")]
        
        # Combine all keyword-like fields
        all_keywords = list(set(keywords_raw + concepts + topics))
        
        oa = work.get("open_access", {})
        refs = work.get("referenced_works", [])
        
        # Reconstruct abstract from inverted index
        abstract = ""
        abs_inv = work.get("abstract_inverted_index")
        if abs_inv:
            try:
                positions = []
                for word, pos_list in abs_inv.items():
                    for pos in pos_list:
                        positions.append((pos, word))
                positions.sort(key=lambda x: x[0])
                abstract = " ".join(word for _, word in positions)
            except:
                pass
        
        # Return dict with column names matching Bibliometric Counts expectations
        return {
            # Identifiers
            "openalex_id": work.get("id", ""),
            "DOI": work.get("doi"),
            
            # Core fields with standard names
            "Title": work.get("title") or work.get("display_name"),
            "Year": work.get("publication_year"),
            "Publication Year": work.get("publication_year"),
            "publication_date": work.get("publication_date"),
            
            # Document type
            "Document Type": work.get("type"),
            "type": work.get("type"),
            
            # Citation metrics
            "Cited by": work.get("cited_by_count", 0),
            "cited_by_count": work.get("cited_by_count", 0),
            
            # Open access
            "is_oa": oa.get("is_oa", False),
            "oa_status": oa.get("oa_status"),
            
            # Authors
            "Authors": "|".join(authors),
            "author_ids": "|".join(author_ids),
            "Authors Count": len(authors),
            
            # Affiliations/Institutions
            "Affiliations": "|".join(list(set(institutions))),
            
            # Countries - converted from codes to full names
            "Countries": "|".join(country_names),
            
            # Keywords - multiple columns for compatibility
            "Author Keywords": "|".join(all_keywords),
            "Keywords": "|".join(all_keywords),
            "Index Keywords": "|".join(concepts),
            
            # Topics/Subject Areas - both names for compatibility
            "Topics": "|".join(topics),
            "Subject Area": "|".join(topics),
            "SDGs": "|".join(sdgs),
            
            # Source/Journal
            "Source title": source_name,
            "Source": source_name,
            "ISSN": source_issn,
            
            # Abstract
            "Abstract": abstract,
            
            # References - as pipe-separated list for counting
            "References": "|".join(refs) if refs else "",
            "References Count": len(refs),
        }


class OWBibliographicData(OWWidget):
    """Load Bibliographic Data with preprocessing options."""
    
    name = "Bibliographic Data"
    description = "Load bibliometric data from files or OpenAlex API"
    icon = "icons/bibliographic_data.svg"
    priority = 10
    keywords = ["bibliometric", "scopus", "web of science", "openalex"]
    category = "Biblium"
    
    class Outputs:
        data = Output("Data", Table)
    
    # Settings
    selected_sample = settings.Setting("none")
    recent_files = settings.Setting([])
    selected_database = settings.Setting("Auto-detect")
    preprocess_level = settings.Setting(2)
    use_stopwords = settings.Setting(True)
    stopwords_file = settings.Setting("")
    selected_categories = settings.Setting([])
    extra_stopwords = settings.Setting("")
    user_stopwords_file = settings.Setting("")
    synonyms_text = settings.Setting("")
    
    # API settings
    api_email = settings.Setting("")
    api_query = settings.Setting("")
    api_search_by = settings.Setting(0)
    api_year_from = settings.Setting(2016)
    api_year_to = settings.Setting(2026)
    api_doc_type = settings.Setting(0)
    api_max_results = settings.Setting(200)
    api_open_access = settings.Setting(False)
    
    want_main_area = False
    
    class Warning(OWWidget.Warning):
        no_biblium = Msg("Biblium not available")
    
    class Error(OWWidget.Error):
        load_error = Msg("Load error: {}")
        empty_data = Msg("No data loaded")
    
    class Information(OWWidget.Information):
        loaded = Msg("Loaded {:,} records with {} columns")
    
    def __init__(self):
        super().__init__()
        self._data: Optional[pd.DataFrame] = None
        self._worker: Optional[FetchWorker] = None
        self._current_file: Optional[str] = None
        
        # Stopwords
        self._general_stopwords: List[str] = []
        self._specific_stopwords: Dict[str, List[str]] = {}
        self._user_stopwords: List[str] = []
        
        # Track widgets for syncing
        self._level_combos = []
        self._stopword_checks = []
        self._category_checkboxes: Dict[str, List[QCheckBox]] = {}
        self._extra_stopwords_edits = []
        self._user_sw_edits = []
        self._synonyms_edits = []
        
        if not HAS_BIBLIUM:
            self.Warning.no_biblium()
        
        self._load_stopwords()
        self._setup_gui()
    
    def _load_stopwords(self):
        """Load stopwords from Biblium."""
        if not HAS_BIBLIUM or not BIBLIUM_ADDITIONAL_DIR:
            return
        
        sw_path = os.path.join(BIBLIUM_ADDITIONAL_DIR, "stopwords.xlsx")
        if not os.path.exists(sw_path):
            return
        
        try:
            general_df = pd.read_excel(sw_path, sheet_name="general")
            col = general_df.columns[0] if len(general_df.columns) > 0 else None
            if col:
                self._general_stopwords = general_df[col].dropna().str.lower().tolist()
            
            specific_df = pd.read_excel(sw_path, sheet_name="specific")
            if "Category" in specific_df.columns:
                for cat in specific_df["Category"].unique():
                    if pd.notna(cat):
                        cat_df = specific_df[specific_df["Category"] == cat]
                        word_col = None
                        for c in ["Stopword", "Word", "Term"]:
                            if c in cat_df.columns:
                                word_col = c
                                break
                        if word_col is None and len(cat_df.columns) > 1:
                            word_col = cat_df.columns[1]
                        if word_col:
                            self._specific_stopwords[cat] = cat_df[word_col].dropna().str.lower().tolist()
            
            self.stopwords_file = sw_path
        except Exception as e:
            logger.error(f"Failed to load stopwords: {e}")
    
    def _setup_gui(self):
        """Build GUI."""
        self.tabs = QTabWidget()
        self.controlArea.layout().addWidget(self.tabs)
        
        self._create_load_tab()
        self._create_api_tab()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.controlArea.layout().addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.controlArea.layout().addWidget(self.status_label)
    
    def _create_preprocessing_group(self) -> QGroupBox:
        """Create preprocessing options."""
        group = QGroupBox("Preprocessing Options")
        layout = QVBoxLayout(group)
        
        level_row = QHBoxLayout()
        level_row.addWidget(QLabel("Level:"))
        combo = QComboBox()
        for lvl, desc in PREPROCESS_LEVELS.items():
            combo.addItem(desc, lvl)
        combo.setCurrentIndex(self.preprocess_level)
        combo.currentIndexChanged.connect(self._on_level_changed)
        level_row.addWidget(combo)
        layout.addLayout(level_row)
        self._level_combos.append(combo)
        
        cb = QCheckBox("Apply stopwords")
        cb.setChecked(self.use_stopwords)
        cb.toggled.connect(self._on_stopwords_toggled)
        layout.addWidget(cb)
        self._stopword_checks.append(cb)
        
        return group
    
    def _create_keyword_processing_group(self) -> QGroupBox:
        """Create keyword processing options."""
        group = QGroupBox("Keyword Processing")
        layout = QVBoxLayout(group)
        
        # Stopwords file
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Stopwords File:"))
        sw_edit = QLineEdit()
        sw_edit.setText(self.stopwords_file)
        sw_edit.setReadOnly(True)
        file_row.addWidget(sw_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_stopwords)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)
        
        # General stopwords count
        n_gen = len(self._general_stopwords)
        if n_gen > 0:
            lbl = QLabel(f"✓ General stopwords: {n_gen} terms (always applied)")
            lbl.setStyleSheet("color: green;")
            layout.addWidget(lbl)
        
        # Specific categories
        if self._specific_stopwords:
            layout.addWidget(QLabel(f"<b>Specific categories ({len(self._specific_stopwords)}):</b>"))
            
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setMaximumHeight(100)
            scroll.setFrameShape(QFrame.NoFrame)
            
            cat_widget = QWidget()
            cat_layout = QVBoxLayout(cat_widget)
            cat_layout.setContentsMargins(0, 0, 0, 0)
            cat_layout.setSpacing(2)
            
            for cat, words in sorted(self._specific_stopwords.items()):
                cb = QCheckBox(f"{cat} ({len(words)} terms)")
                cb.setChecked(cat in self.selected_categories)
                cb.toggled.connect(lambda checked, c=cat: self._on_category_toggled(c, checked))
                cat_layout.addWidget(cb)
                if cat not in self._category_checkboxes:
                    self._category_checkboxes[cat] = []
                self._category_checkboxes[cat].append(cb)
            
            cat_layout.addStretch()
            scroll.setWidget(cat_widget)
            layout.addWidget(scroll)
            
            btn_row = QHBoxLayout()
            sel_all = QPushButton("Select All")
            sel_all.clicked.connect(self._select_all_categories)
            btn_row.addWidget(sel_all)
            desel_all = QPushButton("Deselect All")
            desel_all.clicked.connect(self._deselect_all_categories)
            btn_row.addWidget(desel_all)
            btn_row.addStretch()
            layout.addLayout(btn_row)
        
        # User stopwords file
        layout.addWidget(QLabel("User stopwords file:"))
        user_row = QHBoxLayout()
        user_edit = QLineEdit()
        user_edit.setText(self.user_stopwords_file)
        user_edit.setPlaceholderText("Text file with one stopword per line...")
        user_edit.setReadOnly(True)
        user_row.addWidget(user_edit)
        self._user_sw_edits.append(user_edit)
        
        user_browse = QPushButton("Browse")
        user_browse.clicked.connect(self._browse_user_stopwords)
        user_row.addWidget(user_browse)
        user_clear = QPushButton("Clear")
        user_clear.clicked.connect(self._clear_user_stopwords)
        user_row.addWidget(user_clear)
        layout.addLayout(user_row)
        
        # Extra stopwords
        layout.addWidget(QLabel("Extra stopwords (comma-separated):"))
        extra_edit = QLineEdit()
        extra_edit.setText(self.extra_stopwords)
        extra_edit.setPlaceholderText("word1, word2, word3...")
        extra_edit.textChanged.connect(self._on_extra_changed)
        layout.addWidget(extra_edit)
        self._extra_stopwords_edits.append(extra_edit)
        
        # Synonyms
        layout.addWidget(QLabel("Synonyms (old=new):"))
        syn_edit = QPlainTextEdit()
        syn_edit.setPlainText(self.synonyms_text)
        syn_edit.setMaximumHeight(50)
        syn_edit.setPlaceholderText("machine learning=ML")
        syn_edit.textChanged.connect(self._on_synonyms_changed)
        layout.addWidget(syn_edit)
        self._synonyms_edits.append(syn_edit)
        
        return group
    
    def _on_level_changed(self, index):
        sender = self.sender()
        lvl = sender.itemData(index)
        self.preprocess_level = lvl
        for c in self._level_combos:
            if c != sender:
                c.blockSignals(True)
                c.setCurrentIndex(index)
                c.blockSignals(False)
    
    def _on_stopwords_toggled(self, checked):
        self.use_stopwords = checked
        sender = self.sender()
        for cb in self._stopword_checks:
            if cb != sender:
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)
    
    def _on_category_toggled(self, cat, checked):
        if checked and cat not in self.selected_categories:
            self.selected_categories = self.selected_categories + [cat]
        elif not checked and cat in self.selected_categories:
            self.selected_categories = [c for c in self.selected_categories if c != cat]
        for cb in self._category_checkboxes.get(cat, []):
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
    
    def _select_all_categories(self):
        self.selected_categories = list(self._specific_stopwords.keys())
        for cbs in self._category_checkboxes.values():
            for cb in cbs:
                cb.blockSignals(True)
                cb.setChecked(True)
                cb.blockSignals(False)
    
    def _deselect_all_categories(self):
        self.selected_categories = []
        for cbs in self._category_checkboxes.values():
            for cb in cbs:
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
    
    def _on_extra_changed(self):
        sender = self.sender()
        text = sender.text()
        self.extra_stopwords = text
        for e in self._extra_stopwords_edits:
            if e != sender:
                e.blockSignals(True)
                e.setText(text)
                e.blockSignals(False)
    
    def _on_synonyms_changed(self):
        sender = self.sender()
        text = sender.toPlainText()
        self.synonyms_text = text
        for e in self._synonyms_edits:
            if e != sender:
                e.blockSignals(True)
                e.setPlainText(text)
                e.blockSignals(False)
    
    def _browse_stopwords(self):
        path, _ = QFileDialog.getOpenFileName(self, "Stopwords File", "", "Excel (*.xlsx);;All (*)")
        if path:
            self.stopwords_file = path
    
    def _browse_user_stopwords(self):
        path, _ = QFileDialog.getOpenFileName(self, "User Stopwords", "", "Text (*.txt);;All (*)")
        if path:
            self.user_stopwords_file = path
            for e in self._user_sw_edits:
                e.setText(path)
            self._load_user_stopwords()
    
    def _clear_user_stopwords(self):
        self.user_stopwords_file = ""
        for e in self._user_sw_edits:
            e.clear()
        self._user_stopwords = []
    
    def _load_user_stopwords(self):
        if not self.user_stopwords_file or not os.path.exists(self.user_stopwords_file):
            self._user_stopwords = []
            return
        try:
            with open(self.user_stopwords_file, 'r', encoding='utf-8') as f:
                self._user_stopwords = [line.strip().lower() for line in f if line.strip()]
        except Exception:
            self._user_stopwords = []
    
    def _get_all_stopwords(self) -> List[str]:
        stopwords = set(self._general_stopwords)
        for cat in self.selected_categories:
            if cat in self._specific_stopwords:
                stopwords.update(self._specific_stopwords[cat])
        stopwords.update(self._user_stopwords)
        if self.extra_stopwords.strip():
            extra = [w.strip().lower() for w in self.extra_stopwords.split(",") if w.strip()]
            stopwords.update(extra)
        return list(stopwords)
    
    def _create_load_tab(self):
        """Create Load Data tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Sample datasets
        samples_group = QGroupBox("Sample Datasets")
        samples_layout = QVBoxLayout(samples_group)
        self.sample_buttons = QButtonGroup(self)
        for key, info in SAMPLE_DATASETS.items():
            text = info["name"]
            if info["records"]:
                text += f" ({info['records']})"
            rb = QRadioButton(text)
            rb.setProperty("sample_key", key)
            self.sample_buttons.addButton(rb)
            samples_layout.addWidget(rb)
            if key == self.selected_sample:
                rb.setChecked(True)
        self.sample_buttons.buttonClicked.connect(self._on_sample_selected)
        layout.addWidget(samples_group)
        
        # File selection
        file_group = QGroupBox("Load from File")
        file_layout = QGridLayout(file_group)
        file_layout.addWidget(QLabel("File:"), 0, 0)
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("No file selected...")
        file_layout.addWidget(self.file_edit, 0, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(browse_btn, 0, 2)
        
        file_layout.addWidget(QLabel("Database:"), 1, 0)
        self.db_combo = QComboBox()
        self.db_combo.addItems(list(DATABASE_OPTIONS.keys()))
        self.db_combo.setCurrentText(self.selected_database)
        self.db_combo.currentTextChanged.connect(lambda t: setattr(self, "selected_database", t))
        file_layout.addWidget(self.db_combo, 1, 1, 1, 2)
        layout.addWidget(file_group)
        
        layout.addWidget(self._create_preprocessing_group())
        layout.addWidget(self._create_keyword_processing_group())
        
        self.load_btn = QPushButton("📂 Load Dataset")
        self.load_btn.setMinimumHeight(40)
        self.load_btn.setStyleSheet("font-weight: bold;")
        self.load_btn.clicked.connect(self._load_data)
        layout.addWidget(self.load_btn)
        
        layout.addStretch()
        self.tabs.addTab(tab, "📂 Load Data")
    
    def _create_api_tab(self):
        """Create API tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Search Query
        search_group = QGroupBox("🔍 Search Query")
        search_layout = QGridLayout(search_group)
        
        search_layout.addWidget(QLabel("Search By:"), 0, 0)
        self.search_by_combo = QComboBox()
        self.search_by_combo.addItems(SEARCH_BY_OPTIONS)
        idx = self.api_search_by if isinstance(self.api_search_by, int) else 0
        self.search_by_combo.setCurrentIndex(idx)
        self.search_by_combo.currentIndexChanged.connect(self._on_search_by_changed)
        search_layout.addWidget(self.search_by_combo, 0, 1)
        
        search_layout.addWidget(QLabel("Query:"), 1, 0)
        self.query_edit = QLineEdit()
        self.query_edit.setText(self.api_query)
        self.query_edit.textChanged.connect(lambda t: setattr(self, "api_query", t))
        search_layout.addWidget(self.query_edit, 1, 1)
        
        self.query_hint = QLabel()
        self.query_hint.setStyleSheet("color: gray; font-size: 10px;")
        search_layout.addWidget(self.query_hint, 2, 0, 1, 2)
        self._update_query_hint()
        
        layout.addWidget(search_group)
        
        # Filters
        filters_group = QGroupBox("⚙️ Filters")
        filters_layout = QGridLayout(filters_group)
        
        filters_layout.addWidget(QLabel("Years:"), 0, 0)
        year_row = QHBoxLayout()
        self.year_from = QSpinBox()
        self.year_from.setRange(1900, 2030)
        self.year_from.setValue(self.api_year_from)
        self.year_from.valueChanged.connect(lambda v: setattr(self, "api_year_from", v))
        year_row.addWidget(self.year_from)
        year_row.addWidget(QLabel("to"))
        self.year_to = QSpinBox()
        self.year_to.setRange(1900, 2030)
        self.year_to.setValue(self.api_year_to)
        self.year_to.valueChanged.connect(lambda v: setattr(self, "api_year_to", v))
        year_row.addWidget(self.year_to)
        year_row.addStretch()
        filters_layout.addLayout(year_row, 0, 1)
        
        filters_layout.addWidget(QLabel("Doc Type:"), 1, 0)
        self.doc_type_combo = QComboBox()
        self.doc_type_combo.addItems(DOC_TYPE_OPTIONS)
        idx = self.api_doc_type if isinstance(self.api_doc_type, int) else 0
        self.doc_type_combo.setCurrentIndex(idx)
        self.doc_type_combo.currentIndexChanged.connect(lambda i: setattr(self, "api_doc_type", i))
        filters_layout.addWidget(self.doc_type_combo, 1, 1)
        
        filters_layout.addWidget(QLabel("Max Results:"), 2, 0)
        self.max_spin = QSpinBox()
        self.max_spin.setRange(10, 10000)
        self.max_spin.setValue(self.api_max_results)
        self.max_spin.valueChanged.connect(lambda v: setattr(self, "api_max_results", v))
        filters_layout.addWidget(self.max_spin, 2, 1)
        
        self.oa_check = QCheckBox("Open Access only")
        self.oa_check.setChecked(self.api_open_access)
        self.oa_check.toggled.connect(lambda c: setattr(self, "api_open_access", c))
        filters_layout.addWidget(self.oa_check, 3, 0, 1, 2)
        
        layout.addWidget(filters_group)
        
        # Email
        email_row = QHBoxLayout()
        email_row.addWidget(QLabel("Email:"))
        self.email_edit = QLineEdit()
        self.email_edit.setText(self.api_email)
        self.email_edit.setPlaceholderText("your@email.com (for polite pool)")
        self.email_edit.textChanged.connect(lambda t: setattr(self, "api_email", t))
        email_row.addWidget(self.email_edit)
        layout.addLayout(email_row)
        
        layout.addWidget(self._create_preprocessing_group())
        layout.addWidget(self._create_keyword_processing_group())
        
        # Fetch button
        self.fetch_btn = QPushButton("🌐 Fetch from OpenAlex")
        self.fetch_btn.setMinimumHeight(40)
        self.fetch_btn.setStyleSheet("font-weight: bold;")
        self.fetch_btn.clicked.connect(self._fetch_api)
        self.fetch_btn.setEnabled(HAS_REQUESTS)
        if not HAS_REQUESTS:
            self.fetch_btn.setToolTip("Install 'requests' package")
        layout.addWidget(self.fetch_btn)
        
        layout.addStretch()
        self.tabs.addTab(tab, "🌐 API")
    
    def _on_search_by_changed(self, idx):
        self.api_search_by = idx
        self._update_query_hint()
    
    def _update_query_hint(self):
        st = self.search_by_combo.currentText()
        hints = {
            "Keywords": 'e.g., "machine learning" AND neural',
            "Author": "e.g., John Smith or 0000-0002-1234-5678",
            "Institution": "e.g., Harvard University",
            "ISSN/Journal": "e.g., Nature or 0028-0836",
            "DOI List": "e.g., 10.1000/xyz, 10.1000/abc",
        }
        self.query_hint.setText(hints.get(st, ""))
    
    def _on_sample_selected(self, btn):
        key = btn.property("sample_key")
        self.selected_sample = key
        if key != "none":
            self.file_edit.clear()
            self._current_file = None
    
    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", 
            "All Supported (*.csv *.xlsx *.txt *.bib);;All (*)"
        )
        if path:
            self.file_edit.setText(path)
            self._current_file = path
            for btn in self.sample_buttons.buttons():
                if btn.property("sample_key") == "none":
                    btn.setChecked(True)
                    break
            self.selected_sample = "none"
    
    def _load_data(self):
        """Load from file."""
        self.Error.clear()
        self.Information.clear()
        
        if self.user_stopwords_file:
            self._load_user_stopwords()
        
        if self._current_file:
            filepath = self._current_file
            db = DATABASE_OPTIONS.get(self.selected_database, "")
        elif self.selected_sample != "none":
            if not HAS_BIBLIUM:
                self.Error.load_error("Biblium required for samples")
                return
            info = SAMPLE_DATASETS[self.selected_sample]
            filepath = os.path.join(BIBLIUM_DATA_DIR, info["filename"])
            db = info["db"]
        else:
            self.Error.load_error("No file selected")
            return
        
        if not os.path.exists(filepath):
            self.Error.load_error(f"File not found: {filepath}")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)
        self.status_label.setText("Loading...")
        self.load_btn.setEnabled(False)
        QApplication.processEvents()
        
        try:
            df = None
            if HAS_BIBLIUM:
                try:
                    bib = BiblioAnalysis(filepath, db=db if db else None)
                    df = bib.df
                    if self.preprocess_level >= 1:
                        try:
                            bib.add_countries()
                            bib.add_labels()
                        except:
                            pass
                    if self.preprocess_level >= 2 and self.use_stopwords:
                        try:
                            bib.apply_stopwords()
                        except:
                            pass
                    df = bib.df
                except Exception as e:
                    logger.warning(f"Biblium failed: {e}")
            
            if df is None:
                ext = Path(filepath).suffix.lower()
                if ext == ".csv":
                    df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
                elif ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(filepath)
                else:
                    df = pd.read_csv(filepath, sep="\t", encoding="utf-8", on_bad_lines="skip")
            
            self.progress_bar.setValue(70)
            df = self._apply_text_processing(df)
            
            self.progress_bar.setValue(90)
            self._data = df
            self._send_output()
            
            self.progress_bar.setValue(100)
            self.Information.loaded(len(df), len(df.columns))
            self.status_label.setText(f"✓ Loaded {len(df):,} records")
            
        except Exception as e:
            self.Error.load_error(str(e))
            self.status_label.setText(f"Error: {e}")
        
        finally:
            self.progress_bar.setVisible(False)
            self.load_btn.setEnabled(True)
    
    def _extract_countries_from_affiliations(self, df):
        """Extract countries from affiliations column (like biblium does).
        
        Creates 'Countries of Authors' column if it doesn't exist and 
        'Affiliations' column is present.
        """
        if df is None:
            return df
        
        # Check if Countries column already exists
        country_cols = ["Countries of Authors", "Countries", "Country", "authorships.countries"]
        for col in country_cols:
            if col in df.columns:
                # Already have a countries column
                return df
        
        # Check for Affiliations column
        aff_col = None
        for col in ["Affiliations", "Affiliation", "affiliations", "C1"]:
            if col in df.columns:
                aff_col = col
                break
        
        if aff_col is None:
            return df
        
        # Load country validation data
        valid_countries = set()
        if HAS_BIBLIUM:
            try:
                from biblium import utilsbib
                valid_countries = set(utilsbib._get_l_countries() or [])
            except:
                pass
        
        if not valid_countries:
            # Fallback country list
            valid_countries = set(COUNTRY_CODE_TO_NAME.values())
            # Add some common variations
            valid_countries.update([
                "USA", "UK", "United States", "United Kingdom", "England",
                "Scotland", "Wales", "China", "India", "Germany", "France",
                "Italy", "Spain", "Netherlands", "Belgium", "Switzerland",
                "Sweden", "Norway", "Denmark", "Finland", "Poland", "Russia",
                "Japan", "South Korea", "Australia", "Canada", "Brazil",
                "Mexico", "Argentina", "Chile", "Taiwan", "Hong Kong", 
                "Singapore", "Malaysia", "Thailand", "Indonesia", "Vietnam",
                "Egypt", "South Africa", "Nigeria", "Kenya", "Morocco",
                "Turkey", "Israel", "Saudi Arabia", "Iran", "Pakistan",
                "Ireland", "Austria", "Portugal", "Greece", "Czech Republic",
                "Hungary", "Romania", "Bulgaria", "Croatia", "Slovenia",
                "New Zealand", "Colombia", "Peru", "Venezuela",
            ])
        
        # Extract countries from affiliations
        unique_countries = []
        
        for affil in df[aff_col].fillna(""):
            affil_str = str(affil)
            # Split by semicolon (Scopus style)
            entries = [entry.strip() for entry in affil_str.split(";")]
            # Get last part after comma (usually country)
            countries_raw = [entry.split(",")[-1].strip() for entry in entries if "," in entry]
            # Validate and collect unique countries
            valid = []
            for c in countries_raw:
                # Clean the country name
                c_clean = c.strip().rstrip(".")
                # Check if it's a valid country
                if c_clean in valid_countries:
                    valid.append(c_clean)
                elif c_clean.upper() in COUNTRY_CODE_TO_NAME:
                    valid.append(COUNTRY_CODE_TO_NAME[c_clean.upper()])
            
            unique_list = sorted(set(valid))
            unique_countries.append("; ".join(unique_list))
        
        df = df.copy()
        df["Countries of Authors"] = unique_countries
        
        return df
    
    def _apply_text_processing(self, df):
        """Apply stopwords, synonyms, and extract countries."""
        if df is None:
            return df
        
        # First, extract countries from affiliations if needed
        df = self._extract_countries_from_affiliations(df)
        
        kw_col = None
        for col in ["Keywords", "Author Keywords", "keywords", "concepts", "topics"]:
            if col in df.columns:
                kw_col = col
                break
        
        if kw_col is None:
            return df
        
        sample = df[kw_col].dropna().iloc[0] if len(df[kw_col].dropna()) > 0 else ""
        sep = "|" if "|" in str(sample) else ";"
        
        if self.use_stopwords:
            stopwords = set(self._get_all_stopwords())
            if stopwords:
                def remove_sw(text):
                    if pd.isna(text):
                        return text
                    parts = [p.strip() for p in str(text).split(sep)]
                    filtered = [p for p in parts if p.lower() not in stopwords]
                    return sep.join(filtered)
                df[kw_col] = df[kw_col].apply(remove_sw)
        
        if self.synonyms_text.strip():
            replacements = {}
            for line in self.synonyms_text.strip().split("\n"):
                if "=" in line:
                    old, new = line.split("=", 1)
                    if old.strip() and new.strip():
                        replacements[old.strip().lower()] = new.strip()
            
            if replacements:
                def apply_syn(text):
                    if pd.isna(text):
                        return text
                    parts = [p.strip() for p in str(text).split(sep)]
                    replaced = [replacements.get(p.lower(), p) for p in parts]
                    return sep.join(replaced)
                df[kw_col] = df[kw_col].apply(apply_syn)
        
        return df
    
    def _fetch_api(self):
        """Fetch from OpenAlex API."""
        self.Error.clear()
        self.Information.clear()
        
        if not self.api_query.strip():
            self.Error.load_error("Enter a search query")
            return
        
        if not HAS_REQUESTS:
            self.Error.load_error("requests package required")
            return
        
        if self.user_stopwords_file:
            self._load_user_stopwords()
        
        # Cancel any existing worker
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(1000)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting fetch...")
        self.fetch_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        
        # Get doc type safely
        try:
            if isinstance(self.api_doc_type, int) and self.api_doc_type > 0:
                doc_type = DOC_TYPE_OPTIONS[self.api_doc_type]
            else:
                doc_type = None
        except:
            doc_type = None
        
        params = {
            "query": self.api_query,
            "search_type": self.search_by_combo.currentText(),
            "year_from": self.api_year_from,
            "year_to": self.api_year_to,
            "doc_type": doc_type,
            "max_results": self.api_max_results,
            "open_access": self.api_open_access,
            "email": self.api_email,
        }
        
        self._worker = FetchWorker(params)
        self._worker.progress.connect(self._on_fetch_progress, Qt.QueuedConnection)
        self._worker.finished.connect(self._on_fetch_finished, Qt.QueuedConnection)
        self._worker.start()
    
    def _on_fetch_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)
    
    def _on_fetch_finished(self, result, error):
        self.progress_bar.setVisible(False)
        self.fetch_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        
        if error:
            if error != "Cancelled":
                self.Error.load_error(error)
            self.status_label.setText(f"Error: {error}")
            return
        
        if result is None or (isinstance(result, pd.DataFrame) and result.empty):
            self.Error.empty_data()
            self.status_label.setText("No data found")
            return
        
        # Apply text processing
        result = self._apply_text_processing(result)
        
        self._data = result
        n_rows, n_cols = result.shape
        
        self._send_output()
        self.Information.loaded(n_rows, n_cols)
        self.status_label.setText(f"✓ Loaded {n_rows:,} records with {n_cols} columns")
    
    def _send_output(self):
        if self._data is None:
            self.Outputs.data.send(None)
            return
        try:
            table = self._df_to_table(self._data)
            self.Outputs.data.send(table)
        except Exception as e:
            self.Error.load_error(f"Conversion: {e}")
            self.Outputs.data.send(None)
    
    def _df_to_table(self, df):
        """Convert DataFrame to Orange Table."""
        attrs = []
        metas = []
        
        for col in df.columns:
            col_data = df[col]
            if pd.api.types.is_numeric_dtype(col_data.dtype):
                attrs.append(ContinuousVariable(str(col)))
            elif pd.api.types.is_bool_dtype(col_data.dtype):
                attrs.append(DiscreteVariable(str(col), values=["False", "True"]))
            elif col_data.nunique() <= 20 and col_data.nunique() > 0:
                values = [str(v) for v in sorted(col_data.dropna().unique())]
                if values:
                    attrs.append(DiscreteVariable(str(col), values=values))
                else:
                    metas.append(StringVariable(str(col)))
            else:
                metas.append(StringVariable(str(col)))
        
        domain = Domain(attrs, metas=metas)
        
        X = np.zeros((len(df), len(attrs)), dtype=float)
        M = np.zeros((len(df), len(metas)), dtype=object)
        
        for i, var in enumerate(attrs):
            col = df[var.name]
            if isinstance(var, ContinuousVariable):
                X[:, i] = pd.to_numeric(col, errors="coerce").fillna(np.nan).values
            elif isinstance(var, DiscreteVariable):
                mapping = {v: j for j, v in enumerate(var.values)}
                X[:, i] = col.apply(lambda x: mapping.get(str(x), np.nan)).values
        
        for i, var in enumerate(metas):
            M[:, i] = df[var.name].fillna("").astype(str).values
        
        return Table.from_numpy(domain, X, metas=M if metas else None)
    
    def onDeleteWidget(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(1000)
        super().onDeleteWidget()


if __name__ == "__main__":
    WidgetPreview(OWBibliographicData).run()
