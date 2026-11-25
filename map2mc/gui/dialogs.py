"""Dialogs: block palette editor and building list."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout, QHeaderView,
                               QLabel, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout)

from ..config import CLASSES, TREE_TYPES, default_palette

PAL_COLS = [("surface", "Oberfläche"), ("subsurface", "Untergrund"), ("sub_depth", "Tiefe"),
            ("biome", "Biom"), ("tree", "Baumart"), ("tree_density", "Baumdichte"), ("vegetation", "Pflanzen")]


class PaletteDialog(QDialog):
    """Edit the block assignment per terrain class.

    Blocks may be mixes: 'minecraft:stone*4,minecraft:andesite' (weights after *).
    """

    def __init__(self, palette: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Blockzuordnung bearbeiten")
        self.resize(1150, 560)
        self.palette = {k: dict(v) for k, v in palette.items()}
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Mischungen: <code>minecraft:stone*4,minecraft:andesite</code> (Gewicht nach *). "
                             "Blockzustände z. B. <code>minecraft:basalt[axis=y]</code>. "
                             "Bei Wasserklassen ist „Oberfläche“ der Gewässerboden."))
        self.table = QTableWidget(len(CLASSES), len(PAL_COLS) + 1)
        self.table.setHorizontalHeaderLabels(["Klasse"] + [c[1] for c in PAL_COLS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        for i, (key, label, color) in enumerate(CLASSES):
            it = QTableWidgetItem(label)
            it.setFlags(Qt.ItemIsEnabled)
            it.setBackground(QColor(*color))
            it.setForeground(QColor(0, 0, 0) if sum(color) > 350 else QColor(255, 255, 255))
            self.table.setItem(i, 0, it)
            self._fill_row(i, key)
        for c, w in enumerate([190, 330, 200, 50, 170, 90, 80, 70]):
            self.table.setColumnWidth(c, w)
        lay.addWidget(self.table)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        reset = QPushButton("Standard wiederherstellen")
        reset.clicked.connect(self._reset)
        bb.addButton(reset, QDialogButtonBox.ResetRole)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _fill_row(self, i, key):
        p = self.palette.get(key, {})
        for c, (field, _) in enumerate(PAL_COLS, start=1):
            if field == "tree":
                w = QComboBox()
                w.addItems(TREE_TYPES)
                w.setCurrentText(str(p.get("tree", "none")))
                self.table.setCellWidget(i, c, w)
            elif field == "sub_depth":
                w = QSpinBox()
                w.setRange(0, 16)
                w.setValue(int(p.get("sub_depth", 3)))
                self.table.setCellWidget(i, c, w)
            elif field in ("tree_density", "vegetation"):
                w = QDoubleSpinBox()
                w.setRange(0, 1)
                w.setSingleStep(0.05)
                w.setValue(float(p.get(field, 0)))
                self.table.setCellWidget(i, c, w)
            else:
                self.table.setItem(i, c, QTableWidgetItem(str(p.get(field, ""))))

    def _reset(self):
        self.palette = default_palette()
        for i, (key, *_r) in enumerate(CLASSES):
            self._fill_row(i, key)

    def _accept(self):
        for i, (key, *_r) in enumerate(CLASSES):
            p = self.palette.setdefault(key, {})
            for c, (field, _) in enumerate(PAL_COLS, start=1):
                w = self.table.cellWidget(i, c)
                if isinstance(w, QComboBox):
                    p[field] = w.currentText()
                elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                    p[field] = w.value()
                else:
                    p[field] = self.table.item(i, c).text().strip()
        self.accept()


class BuildingDialog(QDialog):
    """List of detected buildings. Heights are not in the map, so they are flagged as estimates."""
    goto = Signal(float, float)

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.prj = project
        self.setWindowTitle("Gebäude")
        self.resize(900, 600)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Gelb = Höhe geschätzt (nicht aus der Karte ablesbar). Doppelklick zeigt das Gebäude. "
                             "Typ „none“ entfernt ein Gebäude, „platform“ baut ein Deck (z. B. Schiffe/Stege)."))
        self.table = QTableWidget(len(project.buildings), 7)
        self.table.setHorizontalHeaderLabels(["ID", "Fläche (Blöcke²)", "Typ", "Stockwerke", "Dach-Block",
                                              "Wand-Block (leer = Standard)", "Name"])
        s2 = project.settings.blocks_per_pixel ** 2
        for r, b in enumerate(project.buildings):
            it = QTableWidgetItem(str(b.id))
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if b.estimated:
                it.setBackground(QColor(255, 220, 90))
            self.table.setItem(r, 0, it)
            a = QTableWidgetItem(f"{b.area_px * s2:.0f}")
            a.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(r, 1, a)
            kind = QComboBox()
            kind.addItems(["house", "platform", "none"])
            kind.setCurrentText(b.kind)
            self.table.setCellWidget(r, 2, kind)
            st = QSpinBox()
            st.setRange(1, 30)
            st.setValue(int(b.storeys))
            self.table.setCellWidget(r, 3, st)
            self.table.setItem(r, 4, QTableWidgetItem(b.roof_block))
            self.table.setItem(r, 5, QTableWidgetItem(b.wall_block))
            self.table.setItem(r, 6, QTableWidgetItem(b.name))
        self.table.resizeColumnsToContents()
        self.table.cellDoubleClicked.connect(self._goto)
        lay.addWidget(self.table)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _goto(self, r, _c):
        u, v = self.prj.buildings[r].centroid()
        self.goto.emit(u, v)

    def _save(self):
        for r, b in enumerate(self.prj.buildings):
            ch = {}
            kind = self.table.cellWidget(r, 2).currentText()
            st = self.table.cellWidget(r, 3).value()
            roof = self.table.item(r, 4).text().strip()
            wall = self.table.item(r, 5).text().strip()
            name = self.table.item(r, 6).text().strip()
            if kind != b.kind:
                ch["kind"] = kind
            if st != b.storeys:
                ch["storeys"] = st
            if roof != b.roof_block:
                ch["roof_block"] = roof
            if wall != b.wall_block:
                ch["wall_block"] = wall
            if name != b.name:
                ch["name"] = name
            if ch:
                self.prj.edit_building(b, **ch)
        self.accept()


class SingleBuildingDialog(QDialog):
    def __init__(self, b, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Gebäude {b.id}")
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Typ"))
        self.kind = QComboBox()
        self.kind.addItems(["house", "platform", "none"])
        self.kind.setCurrentText(b.kind)
        row.addWidget(self.kind)
        row.addWidget(QLabel("Stockwerke"))
        self.storeys = QSpinBox()
        self.storeys.setRange(1, 30)
        self.storeys.setValue(int(b.storeys))
        row.addWidget(self.storeys)
        lay.addLayout(row)
        lay.addWidget(QLabel(f"Dachfarbe aus Karte: RGB {tuple(b.roof_rgb)} → {b.roof_block}"
                             + ("\nHöhe ist geschätzt." if b.estimated else "")))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
