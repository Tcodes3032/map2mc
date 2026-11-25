"""Drive every GUI function once and report Python errors (does not need a user).

    python tools/gui_smoke_test.py <project_dir>      (works on a COPY of the project)
"""
import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PySide6.QtCore import QPoint, QPointF, QSettings, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

errors = []
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def hook(t, v, tb):
    errors.append("".join(traceback.format_exception(t, v, tb)))


def main():
    src = sys.argv[1]
    tmp = tempfile.mkdtemp(prefix="map2mc_smoke_")
    prj_dir = os.path.join(tmp, "prj")
    shutil.copytree(src, prj_dir, ignore=shutil.ignore_patterns("worlds"))
    sys.excepthook = hook
    app = QApplication(sys.argv)
    QSettings("map2mc", "map2mc").setValue("last_project", "")
    lang = os.environ.get("MAP2MC_LANG")
    if lang:
        QSettings("map2mc", "map2mc").setValue("language", lang)
    # never block on modal dialogs
    QMessageBox.critical = staticmethod(lambda *a, **k: errors.append("critical dialog: " + str(a[2])[:500]))
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QDialog.exec = lambda self: (self.accept() if hasattr(self, "_accept") is False else self._accept(), 1)[1]
    from map2mc.gui.app import TOOLS, VIEWS, MainWindow
    print('>> create window', flush=True)
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    print('>> open project', flush=True)
    w.open_project(prj_dir)
    app.processEvents()
    steps = []

    def step(name, fn):
        print('>>', name, flush=True)
        n = len(errors)
        try:
            fn()
            app.processEvents()
        except Exception:  # noqa: BLE001
            errors.append(traceback.format_exc())
        steps.append((name, len(errors) == n))

    for v in VIEWS:
        step(f"view {v}", lambda v=v: w.view_box.setCurrentIndex(VIEWS.index(v)))
    step("overlay classes", lambda: w.overlay_classes.setChecked(True))
    step("show buildings", lambda: w.show_buildings.setChecked(True))

    def wheel(delta, x, y):
        pos = QPointF(x, y)
        ev = QWheelEvent(pos, w.canvas.mapToGlobal(pos.toPoint()).toPointF(), QPoint(), QPoint(0, delta),
                         Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        w.canvas.wheelEvent(ev)

    def zoom_test():
        w.canvas.set_image(w.prj.image, fit=True)
        app.processEvents()
        vp = w.canvas.viewport().rect()
        x, y = vp.width() * 0.75, vp.height() * 0.7
        before = w.canvas.mapToScene(QPoint(int(x), int(y)))
        for _ in range(4):
            wheel(120, x, y)
        p = w.canvas.mapFromScene(before)          # where is that map point on screen now?
        d = ((p.x() - x) ** 2 + (p.y() - y) ** 2) ** 0.5
        print(f"   zoom: map point under cursor moved {d:.1f} screen px after 4 zoom steps", flush=True)
        if d > 4.0:
            errors.append(f"zoom drifts: point under cursor moved {d:.1f} screen px")
        for _ in range(8):
            wheel(-120, x, y)
    step("zoom at cursor", zoom_test)

    h, wd = w.prj.image.shape[:2]
    for i, t in enumerate(TOOLS):
        def use(i=i, t=t):
            w.tool.setCurrentIndex(i)
            u, v = wd * 0.5, h * 0.5
            if w.canvas.mode == "paint":
                w.on_stroke(u, v, True)
                w.on_stroke(u + 3, v + 2, False)
                w.on_stroke(-50, -50, False)        # outside the image
                w.on_stroke_end()
            elif w.canvas.mode == "pick":
                w.on_pick(u, v)
                w.on_pick(-5, -5)
            elif w.canvas.mode == "rect":
                w.on_rect(u, v, u + 6, v + 4)
            w.on_hover(u, v)
            w.on_hover(-10, 5000)
        step(f"tool {t}", use)
    step("palette dialog", w.edit_palette)
    step("buildings dialog", w.edit_buildings)
    step("3d preview", lambda: (w.tabs.setCurrentIndex(1), w.rot.setValue(1), w.exag.setValue(40)))
    step("settings change", lambda: (w.fields["blocks_per_pixel"].setValue(2.5), w._update_size()))

    done = {"flag": False}

    def finish():
        app.quit()

    from PySide6.QtWidgets import QFileDialog
    lit_dir = os.path.join(tmp, "lit")
    os.makedirs(lit_dir, exist_ok=True)
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (os.path.join(lit_dir, "area.litematic"), ""))
    QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: os.path.join(lit_dir, "tiles"))
    # small tree pack (zip) built from a folder given as 2nd argument, if any
    pack_zip = None
    if len(sys.argv) > 2 and os.path.isdir(sys.argv[2]):
        import glob
        import zipfile
        pack_zip = os.path.join(tmp, "pack.zip")
        with zipfile.ZipFile(pack_zip, "w") as z:
            for f in sorted(glob.glob(os.path.join(sys.argv[2], "**", "*.schem"), recursive=True))[:300:10]:
                z.write(f, os.path.relpath(f, sys.argv[2]))
        QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (pack_zip, ""))
    def player_walk(bv):
        """Fly east with W+Ctrl (the detail area must follow), then land in walking mode."""
        from PySide6.QtCore import Qt as _Qt
        assert bv.mode == "player" and bv._ground is not None, "player mode / ground missing"
        bv.pyaw, bv.walk = 90.0, False
        bv.keys = {_Qt.Key_W, _Qt.Key_Control}
        x0 = bv.eye[0]
        for _ in range(30):
            bv.step(0.1)
        assert bv.eye[0] - x0 > 50, "player did not move"
        bv.keys = set()
        bv.walk = True
        for _ in range(80):
            bv.step(0.05)
        g = bv.ground_at(bv.eye[0], bv.eye[2])
        assert g > -1e8 and abs(bv.eye[1] - 1.62 - g) < 0.01 and bv.on_ground, "player did not land on the ground"
        bv.walk = False

    rush = {}

    def fast_fly(w):
        from PySide6.QtCore import Qt as _Qt
        from PySide6.QtWidgets import QMessageBox as _MB
        boxes = []
        orig = _MB.information
        _MB.information = staticmethod(lambda *a, **k: boxes.append(a))
        bv = w.blockview
        bv.walk, bv.pyaw = False, 270.0
        bv.keys = {_Qt.Key_W, _Qt.Key_Control}
        for _ in range(6):                       # each jump is far enough to request a new detail area
            for _ in range(5):
                bv.step(0.1)
        bv.keys = set()
        _MB.information = orig
        assert not boxes, f"busy message shown: {boxes}"
        assert w._b3_thread is not None, "no 3D load running"
        rush["target"] = (w.b3_x.value(), w.b3_z.value())

    def check_fast_fly(w):
        fx, fz = w._b3_info["focus"]
        tx, tz = rush["target"]
        assert abs(fx - tx) <= 2 and abs(fz - tz) <= 2, f"focus {fx},{fz} != newest request {tx},{tz}"

    # stage 2 (after generation): tree pack, 3D block view + .litematic export, one job after the other
    stage2 = [("tree pack import", w.import_tree_pack)] if pack_zip else []
    stage2 += [
        ("3d block view load", lambda: (w.tool.setCurrentIndex(TOOLS.index("3D-Ausschnitt wählen (Klick)")),
                                        w.on_pick(wd * 0.5, h * 0.5))),
        ("3d player walk (WASD) -> detail follows", lambda: player_walk(w.blockview)),
        ("3d fast flying: 6 reloads in a row, no 'busy'", lambda: fast_fly(w)),
        ("3d fast flying: newest position loaded", lambda: check_fast_fly(w)),
        ("hologram + layer clip", lambda: (w.b3_holo.setChecked(True), w.b3_clip.setValue(max(1, w.b3_clip.maximum() // 2)),
                                           w.blockview.grabFramebuffer().save(os.path.join(os.path.dirname(__file__), "..", "_out", "smoke_blockview.png")),
                                           w.b3_holo.setChecked(False), w.b3_clip.setValue(w.b3_clip.maximum()))),
        ("litematic area export", w.export_litematic_area),
        ("litematic tile export", w.export_litematic_tiles),
    ]

    def watch():
        if w.thread is not None or w._b3_thread is not None or w._b3_pending is not None:
            return
        if not done["flag"]:
            done["flag"] = True
            steps.append(("analysis + generation", True))
        if stage2:
            name, fn = stage2.pop(0)
            step(name, fn)
            return
        if not done.get("end"):
            done["end"] = True
            ok_area = os.path.exists(os.path.join(lit_dir, "area.litematic"))
            tiles = os.listdir(os.path.join(lit_dir, "tiles")) if os.path.isdir(os.path.join(lit_dir, "tiles")) else []
            print(f"   litematic area: {ok_area}, tiles: {len(tiles)}", flush=True)
            steps.append(("litematic files written", ok_area and len(tiles) > 0))
            if not (ok_area and tiles):
                errors.append("litematic export produced no files")
            QTimer.singleShot(100, finish)

    def run_jobs():
        w.run_analysis()
        def after_analysis():
            if w.thread is not None:
                QTimer.singleShot(300, after_analysis)
                return
            w.out_dir.setText(os.path.join(tmp, "world"))
            w.run_generate()
            t = QTimer(w)
            t.timeout.connect(watch)
            t.start(300)
        QTimer.singleShot(300, after_analysis)
    QTimer.singleShot(0, run_jobs)
    QTimer.singleShot(420000, finish)
    app.exec()
    ok = os.path.exists(os.path.join(tmp, "world", "level.dat"))
    for name, good in steps:
        print(("OK   " if good else "FAIL ") + name)
    print("world written:", ok)
    print(f"{len(errors)} error(s)")
    for e in errors:
        print("-" * 60)
        print(e)
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if errors or not ok else 0)


if __name__ == "__main__":
    main()
