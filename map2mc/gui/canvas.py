"""Zoomable map view with paint tools (class brush, height brushes, eyedropper, building picker)."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
                               QGraphicsView)


def to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, 2)
    h, w, c = rgb.shape
    fmt = QImage.Format_RGBA8888 if c == 4 else QImage.Format_RGB888
    return QImage(rgb.data, w, h, w * c, fmt).copy()


class MapCanvas(QGraphicsView):
    """Emits stroke(u, v, first) while the left button paints, picked(u, v) for single clicks."""
    stroke = Signal(float, float, bool)
    stroke_end = Signal()
    picked = Signal(float, float)
    rect_drawn = Signal(float, float, float, float)
    hover = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.base = QGraphicsPixmapItem()
        self.overlay = QGraphicsPixmapItem()
        self.overlay.setOpacity(0.55)
        self.scene().addItem(self.base)
        self.scene().addItem(self.overlay)
        self.cursor_item = QGraphicsEllipseItem()
        self.cursor_item.setPen(QPen(QColor(255, 255, 255), 0))
        self.cursor_item.setZValue(10)
        self.scene().addItem(self.cursor_item)
        self.rect_item = QGraphicsRectItem()
        self.rect_item.setPen(QPen(QColor(255, 60, 60), 0))
        self.rect_item.setZValue(11)
        self.scene().addItem(self.rect_item)
        self._rect_start = None
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.mode = "view"          # view | paint | pick | rect
        self.brush = 6.0
        self._painting = False
        self._panning = None
        self.setBackgroundBrush(QColor(24, 26, 32))

    def set_image(self, rgb: np.ndarray, fit=False):
        self.base.setPixmap(QPixmap.fromImage(to_qimage(rgb)))
        # generous margin around the image so zooming/panning works at the edges too
        r = self.base.boundingRect()
        self.scene().setSceneRect(r.adjusted(-r.width(), -r.height(), r.width(), r.height()))
        if fit:
            self.fitInView(self.base, Qt.KeepAspectRatio)

    def set_overlay(self, rgba: np.ndarray | None):
        if rgba is None:
            self.overlay.setPixmap(QPixmap())
        else:
            self.overlay.setPixmap(QPixmap.fromImage(to_qimage(rgba)))

    def center_on_px(self, u, v, zoom=None):
        if zoom:
            self.resetTransform()
            self.scale(zoom, zoom)
        self.centerOn(QPointF(u, v))

    # --- events
    def wheelEvent(self, e):
        """Zoom towards the cursor.

        Qt's AnchorUnderMouse relies on the view's own mouse tracking, which our
        mouseMoveEvent replaces – so the anchor is done by hand: remember the
        scene point under the cursor, scale, then shift the view so that the
        same point is under the cursor again.
        """
        dy = e.angleDelta().y()
        if dy == 0:
            return
        f = 1.25 if dy > 0 else 0.8
        cur = self.transform().m11()
        if not (0.02 <= cur * f <= 80):          # sensible zoom limits
            return
        pos = e.position().toPoint()
        before = self.mapToScene(pos)
        self.scale(f, f)
        after = self.mapToScene(pos)
        center = self.mapToScene(self.viewport().rect().center())
        self.centerOn(center + (before - after))
        # centerOn rounds to whole scrollbar steps; correct the remaining offset so
        # the map point under the cursor stays put (no drift over many steps)
        p = self.mapFromScene(before)
        hb, vb = self.horizontalScrollBar(), self.verticalScrollBar()
        hb.setValue(hb.value() + (p.x() - pos.x()))
        vb.setValue(vb.value() + (p.y() - pos.y()))
        e.accept()

    def _uv(self, e):
        p = self.mapToScene(e.position().toPoint())
        return p.x(), p.y()

    def mousePressEvent(self, e):
        if e.button() in (Qt.MiddleButton, Qt.RightButton) or (e.button() == Qt.LeftButton and self.mode == "view"):
            self._panning = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() == Qt.LeftButton:
            u, v = self._uv(e)
            if self.mode == "paint":
                self._painting = True
                self.stroke.emit(u, v, True)
            elif self.mode == "pick":
                self.picked.emit(u, v)
            elif self.mode == "rect":
                self._rect_start = (u, v)
                self.rect_item.setRect(u, v, 0, 0)
                self.rect_item.setVisible(True)

    def mouseMoveEvent(self, e):
        u, v = self._uv(e)
        r = self.brush
        self.cursor_item.setRect(u - r, v - r, 2 * r, 2 * r)
        self.cursor_item.setVisible(self.mode == "paint")
        self.hover.emit(u, v)
        if self._panning is not None:
            d = e.position() - self._panning
            self._panning = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if self._painting:
            self.stroke.emit(u, v, False)
        if self._rect_start is not None:
            u0, v0 = self._rect_start
            self.rect_item.setRect(min(u0, u), min(v0, v), abs(u - u0), abs(v - v0))

    def mouseReleaseEvent(self, e):
        if self._panning is not None:
            self._panning = None
            self.setCursor(Qt.ArrowCursor)
        if self._painting:
            self._painting = False
            self.stroke_end.emit()
        if self._rect_start is not None:
            u0, v0 = self._rect_start
            u, v = self._uv(e)
            self._rect_start = None
            self.rect_item.setVisible(False)
            if abs(u - u0) >= 1 and abs(v - v0) >= 1:
                self.rect_drawn.emit(min(u0, u), min(v0, v), max(u0, u), max(v0, v))
