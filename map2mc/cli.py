"""Command line interface.

    python -m map2mc gui
    python -m map2mc new   <image> <project_dir> [--scale 3]
    python -m map2mc analyse <project_dir>
    python -m map2mc generate <project_dir> <world_dir> [--regions 0,0 1,0] [--zip out.zip]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(prog="map2mc", description="Top-down map image -> Minecraft Java 1.21.11 world")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("gui")
    a = sub.add_parser("new")
    a.add_argument("image")
    a.add_argument("project")
    a.add_argument("--scale", type=float)
    b = sub.add_parser("analyse")
    b.add_argument("project")
    g = sub.add_parser("generate")
    g.add_argument("project")
    g.add_argument("world")
    g.add_argument("--regions", nargs="*", help="only these regions, e.g. 0,0 -1,0")
    g.add_argument("--zip")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    if args.cmd in (None, "gui"):
        from .gui.app import run
        return run()

    from .project import Project, export_zip

    def bar(f):
        n = int(f * 30)
        sys.stdout.write("\r[" + "#" * n + "-" * (30 - n) + f"] {f * 100:5.1f}%")
        sys.stdout.flush()

    if args.cmd == "new":
        prj = Project.create(args.image, args.project)
        if args.scale:
            prj.settings.blocks_per_pixel = args.scale
            prj.save()
        print(f"Projekt angelegt: {prj.dir} (Maßstab {prj.settings.blocks_per_pixel} Blöcke/Pixel)")
    elif args.cmd == "analyse":
        prj = Project(args.project)
        t = time.time()
        prj.run_classification(bar)
        print()
        prj.run_derive(bar, log=print)
        print(f"\nfertig in {time.time() - t:.1f} s")
    elif args.cmd == "generate":
        prj = Project(args.project)
        regions = [list(map(int, r.split(","))) for r in args.regions] if args.regions else None
        t = time.time()
        prj.generate(args.world, progress=bar, log=lambda s: print("\n" + s), only_regions=regions)
        if args.zip:
            export_zip(args.world, args.zip)
            print("ZIP:", args.zip)
        print(f"\nfertig in {time.time() - t:.1f} s")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
