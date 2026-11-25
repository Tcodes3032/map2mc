"""Validate a generated world with a real vanilla Minecraft server.

    python tools/validate_with_server.py <server.jar> <eula.txt> <world_dir> [--sample N] [--all-chunks]

* copies the world into a temporary server folder (the original is untouched),
* needs an eula.txt in which YOU have accepted the Minecraft EULA (eula=true) –
  this script never accepts it for you,
* force-loads chunks (all, or a sample) and fails on any chunk loading error,
* compares N random columns: the block our writer put at the top of the
  column must be exactly what the server reports (/execute if block).
"""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from map2mc import anvil  # noqa: E402

ERROR_PATTERNS = re.compile(r"(Couldn't load chunk|Failed to load|Recovering|corrupt|Exception|key missing|ERROR\])", re.I)


def top_block(chunk: dict, x: int, z: int):
    best = None
    for s in chunk["sections"]:
        bs = s["block_states"]
        pal = bs["palette"]
        if "data" in bs:
            bits = max(4, int(np.ceil(np.log2(len(pal)))))
            idx = anvil.unpack_indices(bs["data"], bits, 4096).reshape(16, 16, 16)
        else:
            idx = np.zeros((16, 16, 16), int)
        for y in range(16):
            e = pal[idx[y, z, x]]
            if e["Name"] != "minecraft:air":
                props = e.get("Properties")
                st = e["Name"] + ("[" + ",".join(f"{k}={v}" for k, v in props.items()) + "]" if props else "")
                best = (s["Y"].value * 16 + y, st)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("server_jar")
    ap.add_argument("eula")
    ap.add_argument("world")
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--all-chunks", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    if "eula=true" not in open(a.eula, encoding="utf-8").read():
        sys.exit("eula.txt does not contain eula=true – accept the Minecraft EULA yourself first.")

    regions = sorted(f for f in os.listdir(os.path.join(a.world, "region")) if f.endswith(".mca"))
    chunks = []
    for f in regions:
        _, rx, rz, _ = f.split(".")
        path = os.path.join(a.world, "region", f)
        with open(path, "rb") as fh:
            head = fh.read(4096)
        for i in range(1024):
            if head[4 * i:4 * i + 4] != b"\0\0\0\0":
                chunks.append((int(rx) * 32 + i % 32, int(rz) * 32 + i // 32, path, i % 32, i // 32))
    print(f"{len(regions)} regions, {len(chunks)} chunks")

    rnd = random.Random(7)
    checks = []
    for cx, cz, path, lx, lz in rnd.sample(chunks, min(a.sample, len(chunks))):
        x, z = rnd.randrange(16), rnd.randrange(16)
        y, st = top_block(anvil.read_region_chunk(path, lx, lz), x, z)
        checks.append((cx * 16 + x, y, cz * 16 + z, st))

    tmp = tempfile.mkdtemp(prefix="map2mc_validate_")
    try:
        shutil.copy(a.server_jar, os.path.join(tmp, "server.jar"))
        shutil.copy(a.eula, os.path.join(tmp, "eula.txt"))
        shutil.copytree(a.world, os.path.join(tmp, "world"))
        with open(os.path.join(tmp, "server.properties"), "w") as f:
            f.write("level-name=world\nonline-mode=false\nserver-port=25982\nview-distance=3\n"
                    "simulation-distance=3\nspawn-protection=0\nmax-players=1\n")
        proc = subprocess.Popen(["java", "-Xmx4G", "-jar", "server.jar", "--nogui"], cwd=tmp,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        lines: list[str] = []

        def reader():
            for ln in proc.stdout:
                lines.append(ln.rstrip())
        threading.Thread(target=reader, daemon=True).start()

        def send(cmd):
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()

        def wait_for(pattern, timeout=300):
            t = time.time()
            while time.time() - t < timeout:
                if any(re.search(pattern, ln) for ln in lines):
                    return True
                time.sleep(0.5)
            return False

        if not wait_for(r"Done \("):
            sys.exit("server did not start:\n" + "\n".join(lines[-40:]))
        start_errors = [ln for ln in lines if ERROR_PATTERNS.search(ln)]

        # force-load chunks in batches (forceload accepts at most 256 chunks per command)
        targets = chunks if a.all_chunks else rnd.sample(chunks, min(600, len(chunks)))
        t0 = time.time()
        never_loaded = []
        for i in range(0, len(targets), 200):
            pending = targets[i:i + 200]
            for cx, cz, *_ in pending:
                send(f"forceload add {cx * 16} {cz * 16}")
            # poll with "execute unless loaded" until every chunk of the batch is fully loaded
            for attempt in range(60):
                time.sleep(1.0)
                mark = f"BATCH_{i}_{attempt}"
                n0 = len(lines)
                for cx, cz, *_ in pending:
                    send(f"execute unless loaded {cx * 16} 0 {cz * 16} run say NOT_LOADED {cx} {cz}")
                send(f"say {mark}")
                wait_for(rf"\b{mark}\b", 60)
                miss = {tuple(map(int, ln.split("NOT_LOADED ")[1].split()[:2]))
                        for ln in lines[n0:] if "NOT_LOADED" in ln and "run say" not in ln}
                pending = [c for c in pending if (c[0], c[1]) in miss]
                if not pending:
                    break
            never_loaded += pending
            send("forceload remove all")
        print(f"force-loaded {len(targets) - len(never_loaded)}/{len(targets)} chunks "
              f"(verified with 'execute if loaded') in {time.time() - t0:.0f} s")
        if never_loaded:
            print("never loaded:", never_loaded[:20])

        for X, Y, Z, st in checks:
            send(f"forceload add {X} {Z}")
        time.sleep(8)
        for k, (X, Y, Z, st) in enumerate(checks):
            if "_leaves[" in st:
                # the server recomputes 'distance' of (persistent) leaves on neighbour updates
                st = re.sub(r"distance=\d+,?", "", st).replace(",]", "]")
            send(f"execute if block {X} {Y} {Z} {st} run say CHECK_OK {k}")
            send(f"execute unless block {X} {Y} {Z} {st} run say CHECK_FAIL {k} {X} {Y} {Z} {st}")
        send("say CHECKS_DONE")
        wait_for(r"CHECKS_DONE", 60)
        send("save-all flush")
        time.sleep(3)
        send("stop")
        proc.wait(timeout=120)

        ok = sum(1 for ln in lines if "CHECK_OK" in ln and "run say" not in ln)
        fails = [ln for ln in lines if "CHECK_FAIL" in ln and "run say" not in ln]
        errors = start_errors + [ln for ln in lines if ERROR_PATTERNS.search(ln) and ln not in start_errors]
        print(f"block checks: {ok}/{len(checks)} ok")
        for ln in fails:
            print("  ", ln)
        print(f"errors in server log: {len(errors)}")
        for ln in errors[:20]:
            print("  ", ln)
        sys.exit(0 if ok == len(checks) and not errors and not never_loaded else 1)
    finally:
        if a.keep:
            print("kept:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
