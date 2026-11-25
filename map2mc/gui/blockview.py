"""Textured 3D block view (Litematica/Voxy-like) built on QOpenGLWidget.

Two control modes:
* player (default, like Minecraft): click into the view to capture the mouse
  (Esc releases it), mouse = look around, W/A/S/D = move, Space / Shift = up /
  down (flying) or jump (walking), Ctrl = sprint, F = switch flying / walking,
  mouse wheel = flying speed, R = back to the start position.
* orbit: left drag = orbit, right/middle drag = pan, wheel = zoom,
  double click = reset camera.  Optional hologram mode draws the blocks
translucent and blue-tinted like a schematic preview; the layer limit hides
everything above a given height (look inside houses).
"""
from __future__ import annotations

import math

import numpy as np
import time

from PySide6.QtCore import QPoint, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QImage, QMatrix4x4, QSurfaceFormat, QVector3D
from PySide6.QtWidgets import QLabel
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture
from PySide6.QtOpenGLWidgets import QOpenGLWidget

GL_FLOAT = 0x1406
GL_TRIANGLES = 0x0004
GL_DEPTH_TEST = 0x0B71
GL_BLEND = 0x0BE2
GL_CULL_FACE = 0x0B44
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_COLOR_BUFFER_BIT = 0x4000
GL_DEPTH_BUFFER_BIT = 0x0100
GL_TRUE, GL_FALSE = 1, 0

VERT = """
#ifdef GL_ES
precision mediump float;
#endif
attribute vec3 aPos;
attribute vec2 aUv;
attribute vec3 aCol;
uniform mat4 uMvp;
varying vec2 vUv;
varying vec3 vCol;
varying float vY;
void main() {
    gl_Position = uMvp * vec4(aPos, 1.0);
    vUv = aUv;
    vCol = aCol;
    vY = aPos.y;
}
"""
FRAG = """
#ifdef GL_ES
precision mediump float;
#endif
uniform sampler2D uTex;
uniform float uAlpha;
uniform float uHolo;
uniform float uCutout;
uniform float uClipY;
varying vec2 vUv;
varying vec3 vCol;
varying float vY;
void main() {
    if (vY > uClipY) discard;
    vec4 t = texture2D(uTex, vUv);
    if (uCutout > 0.5 && t.a < 0.5) discard;
    vec3 c = t.rgb * vCol;
    if (uHolo > 0.5) c = c * 0.7 + vec3(0.05, 0.14, 0.30);
    float a = uAlpha;
    gl_FragColor = vec4(c, a);
}
"""


class BlockView(QOpenGLWidget):
    status = Signal(str)
    moved = Signal(float, float)        # camera target (x, z) in scene coordinates after a drag

    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        self.setFormat(fmt)
        self.setMinimumSize(300, 200)
        self.setFocusPolicy(Qt.StrongFocus)
        self._pending = None
        self._prog = None
        self._tex = None
        self._bufs = {}
        self._counts = {"opaque": 0, "water": 0}
        self.holo = False
        self.clip_y = 1e9
        self.center = np.zeros(3, np.float32)
        self.extent = 64.0
        self.world_extent = 64.0
        self.yaw, self.pitch, self.dist = -35.0, 35.0, 150.0
        self.pan = np.zeros(3, np.float32)
        self._last = None
        self.gl_error = ""
        # --- player mode
        self.mode = "player"
        self.eye = np.array([0.0, 100.0, 0.0])
        self.pyaw, self.ppitch = 0.0, -25.0
        self.walk = False
        self.vy = 0.0
        self.on_ground = False
        self.fly_speed = 12.0
        self.keys = set()
        self.captured = False
        self.world_origin = (0, 0)
        self._ground = None
        self._last_emit = np.zeros(3)
        self._t_last = time.perf_counter()
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self.hud = QLabel(self)
        self.hud.setStyleSheet("QLabel{color:white;background:rgba(0,0,0,110);padding:4px;font:9pt 'Consolas';}")
        self.hud.move(6, 6)
        self.cross = QLabel("+", self)
        self.cross.setStyleSheet("QLabel{color:white;font:bold 16pt 'Consolas';background:transparent;}")
        self.cross.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hud.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._update_hud()

    # ------------------------------------------------------------- data
    def set_scene(self, opaque: np.ndarray, water: np.ndarray, atlas: np.ndarray, center, focus_size: float,
                  world_size: float | None = None, keep_camera: bool = False, shift=(0.0, 0.0)):
        """center: camera target (x, y, z) in scene coordinates; focus_size: size of the
        detailed area (initial zoom); world_size: extent of everything (far plane).
        keep_camera + shift: the scene origin moved by shift (x, z) - keep looking at the same spot."""
        self.center = np.array(center, np.float32)
        self.extent = float(max(focus_size, 16))
        self.world_extent = float(max(world_size or 0, self.extent))
        self._pending = (np.ascontiguousarray(opaque, np.float32), np.ascontiguousarray(water, np.float32),
                         np.ascontiguousarray(atlas))
        if keep_camera:
            self.pan[0] -= shift[0]
            self.pan[2] -= shift[1]
            self.pan[1] = 0
            self.eye[0] -= shift[0]
            self.eye[2] -= shift[1]
            self._last_emit[0] -= shift[0]
            self._last_emit[2] -= shift[1]
            self.update()
        else:
            self.reset_camera()
            self.reset_player()

    def set_ground(self, top, water, ox, oz):
        """Surface heights for walking: top [z,x] world y of the highest block (scene frame:
        top[0,0] is at scene (ox, oz)); water [z,x] True where that block is water."""
        self._ground = (top, water, float(ox), float(oz))

    def ground_at(self, x, z):
        """Feet height (y) when standing at scene x, z (water: chest deep)."""
        if self._ground is None:
            return -1e9
        top, water, ox, oz = self._ground
        i, j = int(math.floor(z - oz)), int(math.floor(x - ox))
        if not (0 <= i < top.shape[0] and 0 <= j < top.shape[1]):
            return -1e9
        t = int(top[i, j])
        if t <= -32768:
            return -1e9
        return t + 1.0 - (1.2 if water[i, j] else 0.0)

    def reset_player(self):
        g = self.ground_at(0.0, self.extent * 0.55)
        base = g if g > -1e8 else float(self.center[1])
        self.eye = np.array([0.0, base + 24.0, self.extent * 0.55])
        self.pyaw, self.ppitch = 0.0, -28.0
        self.vy = 0.0
        self._last_emit = self.eye.copy()
        self._update_hud()
        self.update()

    def set_mode(self, mode):
        self.mode = mode
        if mode != "player":
            self._release()
        self._update_hud()
        self.update()

    def player_pos(self):
        return float(self.eye[0]), float(self.eye[2])

    def target(self):
        if self.mode == "player":
            return self.player_pos()
        t = self.center + self.pan
        return float(t[0]), float(t[2])

    def clear_scene(self):
        self._pending = (np.zeros((0, 8), np.float32), np.zeros((0, 8), np.float32), np.zeros((16, 16, 4), np.uint8))
        self.update()

    def reset_camera(self):
        self.yaw, self.pitch = -35.0, 35.0
        self.dist = self.extent * 1.25
        self.pan[:] = 0
        self.update()

    def set_hologram(self, on: bool):
        self.holo = bool(on)
        self.update()

    def set_clip(self, y: float):
        self.clip_y = float(y)
        self.update()

    # ------------------------------------------------------------- GL
    def initializeGL(self):
        try:
            self._prog = QOpenGLShaderProgram(self)
            ok = self._prog.addShaderFromSourceCode(QOpenGLShader.Vertex, VERT)
            ok &= self._prog.addShaderFromSourceCode(QOpenGLShader.Fragment, FRAG)
            self._prog.bindAttributeLocation("aPos", 0)
            self._prog.bindAttributeLocation("aUv", 1)
            self._prog.bindAttributeLocation("aCol", 2)
            ok &= self._prog.link()
            if not ok:
                raise RuntimeError(self._prog.log())
        except Exception as e:      # no usable OpenGL: keep the widget alive, show a message
            self.gl_error = str(e) or "OpenGL nicht verfügbar"
            self._prog = None
            self.status.emit("3D-Ansicht: " + self.gl_error)

    def _upload(self):
        opaque, water, atlas = self._pending
        self._pending = None
        for key, data in (("opaque", opaque), ("water", water)):
            buf = self._bufs.get(key)
            if buf is None:
                buf = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
                buf.create()
                self._bufs[key] = buf
            buf.bind()
            raw = data.tobytes()
            buf.allocate(raw, len(raw))
            buf.release()
            self._counts[key] = len(data)
        if self._tex is not None:
            self._tex.destroy()
        h, w = atlas.shape[:2]
        img = QImage(atlas.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        self._tex = QOpenGLTexture(img, QOpenGLTexture.DontGenerateMipMaps)
        self._tex.setMinificationFilter(QOpenGLTexture.Nearest)
        self._tex.setMagnificationFilter(QOpenGLTexture.Nearest)
        self._tex.setWrapMode(QOpenGLTexture.ClampToEdge)

    def _forward(self):
        y, p = math.radians(self.pyaw), math.radians(self.ppitch)
        return np.array([math.cos(p) * math.sin(y), math.sin(p), -math.cos(p) * math.cos(y)])

    def _mvp(self) -> QMatrix4x4:
        proj = QMatrix4x4()
        aspect = self.width() / max(1, self.height())
        if self.mode == "player":
            proj.perspective(70.0, aspect, 0.1, self.world_extent * 1.6 + 600)
            view = QMatrix4x4()
            tgt = self.eye + self._forward()
            view.lookAt(QVector3D(*map(float, self.eye)), QVector3D(*map(float, tgt)), QVector3D(0, 1, 0))
            return proj * view
        proj.perspective(55.0, aspect, max(0.5, self.dist * 0.01), self.dist * 4 + self.world_extent * 1.6)
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        tgt = self.center + self.pan
        eye = tgt + self.dist * np.array([math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)])
        view = QMatrix4x4()
        view.lookAt(QVector3D(*map(float, eye)), QVector3D(*map(float, tgt)), QVector3D(0, 1, 0))
        return proj * view

    def paintGL(self):
        f = self.context().functions()
        if self.holo:
            f.glClearColor(0.08, 0.09, 0.12, 1.0)
        else:
            f.glClearColor(0.62, 0.78, 0.98, 1.0)
        f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self._prog is None:
            return
        if self._pending is not None:
            self._upload()
        if self._tex is None or (self._counts["opaque"] + self._counts["water"]) == 0:
            return
        f.glEnable(GL_DEPTH_TEST)
        f.glDisable(GL_CULL_FACE)
        f.glEnable(GL_BLEND)
        f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        p = self._prog
        p.bind()
        p.setUniformValue("uMvp", self._mvp())
        p.setUniformValue1i("uTex", 0)
        p.setUniformValue1f("uHolo", 1.0 if self.holo else 0.0)
        p.setUniformValue1f("uClipY", float(self.clip_y))
        self._tex.bind(0)
        for key, cutout, alpha, depth_write in (("opaque", 1.0, 0.55 if self.holo else 1.0, True),
                                                ("water", 0.0, 0.45 if self.holo else 0.85, False)):
            n = self._counts[key]
            if not n:
                continue
            p.setUniformValue1f("uCutout", cutout)
            p.setUniformValue1f("uAlpha", alpha)
            f.glDepthMask(GL_TRUE if depth_write else GL_FALSE)
            buf = self._bufs[key]
            buf.bind()
            stride = 8 * 4
            for loc, off, size in ((0, 0, 3), (1, 12, 2), (2, 20, 3)):
                p.enableAttributeArray(loc)
                p.setAttributeBuffer(loc, GL_FLOAT, off, size, stride)
            f.glDrawArrays(GL_TRIANGLES, 0, n)
            for loc in (0, 1, 2):
                p.disableAttributeArray(loc)
            buf.release()
        f.glDepthMask(GL_TRUE)
        self._tex.release()
        p.release()

    # ------------------------------------------------------------- HUD
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.cross.adjustSize()
        self.cross.move(self.width() // 2 - self.cross.width() // 2, self.height() // 2 - self.cross.height() // 2)

    def _update_hud(self):
        if self.mode != "player":
            self.hud.setText("Orbit: linke Maus drehen · rechte Maus verschieben · Rad zoomen · Doppelklick zurücksetzen")
            self.cross.hide()
        else:
            X = self.eye[0] + self.world_origin[0]
            Z = self.eye[2] + self.world_origin[1]
            m = "Laufen" if self.walk else f"Fliegen ({self.fly_speed:.0f} Blöcke/s)"
            cap = "Esc: Maus freigeben" if self.captured else "Klick ins Bild: Maus fangen"
            self.hud.setText(f"X {X:.0f}  Y {self.eye[1]:.0f}  Z {Z:.0f}   {m}\n"
                             "WASD bewegen · Leertaste/Shift hoch/runter (Laufen: springen) · Strg sprinten\n"
                             f"F Fliegen/Laufen · Mausrad Tempo · R Start · {cap}")
            self.cross.setVisible(self.captured)
        self.hud.adjustSize()

    # ------------------------------------------------------------- player movement
    def _capture(self):
        self.captured = True
        self.setCursor(Qt.BlankCursor)
        self.setMouseTracking(True)
        self.grabMouse()
        QCursor.setPos(self.mapToGlobal(QPoint(self.width() // 2, self.height() // 2)))
        self._update_hud()

    def _release(self):
        if self.captured:
            self.releaseMouse()
        self.captured = False
        self.unsetCursor()
        self.setMouseTracking(False)
        self._update_hud()

    def keyPressEvent(self, e):
        if self.mode != "player" or e.isAutoRepeat():
            return super().keyPressEvent(e)
        k = e.key()
        if k == Qt.Key_Escape:
            self._release()
        elif k == Qt.Key_F:
            self.walk = not self.walk
            self.vy = 0.0
        elif k == Qt.Key_R:
            self.reset_player()
        else:
            self.keys.add(k)
        self._t_last = time.perf_counter()
        if not self._timer.isActive():
            self._timer.start()
        self._update_hud()

    def keyReleaseEvent(self, e):
        if e.isAutoRepeat():
            return
        self.keys.discard(e.key())

    def focusOutEvent(self, e):
        self.keys.clear()
        self._release()
        super().focusOutEvent(e)

    def _tick(self):
        now = time.perf_counter()
        dt = min(0.1, now - self._t_last)
        self._t_last = now
        self.step(dt)
        if not self.keys and (not self.walk or self.on_ground):
            self._timer.stop()
            self._maybe_emit(force=True)

    def step(self, dt):
        """Advance the player by dt seconds with the currently pressed keys."""
        K = self.keys
        y = math.radians(self.pyaw)
        fwd = np.array([math.sin(y), 0.0, -math.cos(y)])
        right = np.array([math.cos(y), 0.0, math.sin(y)])
        mv = fwd * ((Qt.Key_W in K) - (Qt.Key_S in K)) + right * ((Qt.Key_D in K) - (Qt.Key_A in K))
        n = np.linalg.norm(mv)
        if n > 0:
            mv /= n
        sprint = Qt.Key_Control in K
        if not self.walk:
            v = self.fly_speed * (3.0 if sprint else 1.0)
            up = (Qt.Key_Space in K) - (Qt.Key_Shift in K)
            self.eye += mv * v * dt
            self.eye[1] += up * v * dt
            g = self.ground_at(self.eye[0], self.eye[2])
            self.eye[1] = max(self.eye[1], g + 0.3)               # never fly into the ground
        else:
            v = 5.6 if sprint else 4.3
            feet = self.eye[1] - 1.62
            for axis in (0, 2):                                   # per axis -> slide along walls
                nxt = self.eye.copy()
                nxt[axis] += mv[axis] * v * dt
                g = self.ground_at(nxt[0], nxt[2])
                if g - feet <= 0.6:                               # steps up to slabs/paths; walls stop
                    self.eye = nxt
            if Qt.Key_Space in K and self.on_ground:
                self.vy = 8.4                                     # about 1.25 blocks high like in the game
                self.on_ground = False
            self.vy = max(-60.0, self.vy - 32.0 * dt)
            feet = self.eye[1] - 1.62 + self.vy * dt
            g = self.ground_at(self.eye[0], self.eye[2])
            if feet <= g:
                feet, self.vy, self.on_ground = g, 0.0, True
            else:
                self.on_ground = False
            self.eye[1] = feet + 1.62
        self._maybe_emit()
        self._update_hud()
        self.update()

    def _maybe_emit(self, force=False):
        d = float(np.hypot(*(self.eye - self._last_emit)[[0, 2]]))
        if d > self.extent * 0.3 or (force and d > 2):
            self._last_emit = self.eye.copy()
            self.moved.emit(*self.player_pos())

    # ------------------------------------------------------------- input
    def mousePressEvent(self, e):
        self.setFocus()
        if self.mode == "player":
            if not self.captured and e.button() == Qt.LeftButton:
                self._capture()
            return
        self._last = e.position()

    def mouseDoubleClickEvent(self, e):
        if self.mode != "player":
            self.reset_camera()

    def mouseMoveEvent(self, e):
        if self.mode == "player":
            if not self.captured:
                return
            c = QPoint(self.width() // 2, self.height() // 2)
            d = e.position().toPoint() - c
            if d.x() == 0 and d.y() == 0:
                return
            self.pyaw = (self.pyaw + d.x() * 0.15) % 360
            self.ppitch = max(-89.0, min(89.0, self.ppitch - d.y() * 0.15))
            QCursor.setPos(self.mapToGlobal(c))
            self.update()
            return
        if self._last is None:
            return
        d: QPointF = e.position() - self._last
        self._last = e.position()
        if e.buttons() & Qt.LeftButton:
            self.yaw -= d.x() * 0.4
            self.pitch = max(-89.0, min(89.0, self.pitch + d.y() * 0.4))
        elif e.buttons() & (Qt.RightButton | Qt.MiddleButton):
            y = math.radians(self.yaw)
            right = np.array([math.cos(y), 0, -math.sin(y)], np.float32)
            fwd = np.array([math.sin(y), 0, math.cos(y)], np.float32)
            k = self.dist * 0.0025
            self.pan += (-d.x() * right - d.y() * fwd) * k
        self.update()

    def mouseReleaseEvent(self, e):
        if self.mode != "player" and self._last is not None and e.button() in (Qt.RightButton, Qt.MiddleButton):
            self.moved.emit(*self.target())
        self._last = None

    def wheelEvent(self, e):
        dy = e.angleDelta().y()
        if not dy:
            return
        if self.mode == "player":
            self.fly_speed = float(np.clip(self.fly_speed * (1.25 if dy > 0 else 0.8), 2.0, 200.0))
            self._update_hud()
            return
        self.dist = float(np.clip(self.dist * (0.85 if dy > 0 else 1 / 0.85), 4.0,
                                  max(self.extent * 6, self.world_extent * 1.5)))
        self.update()
