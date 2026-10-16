#!/usr/bin/env python3
"""Render a small PNG preview for every STEP file in the repo and keep a
gallery README next to them.

For each directory that contains .stp/.step files this script maintains:

    <dir>/previews/<part>.png     two shaded views of the part
    <dir>/previews/manifest.json  source hash + metadata per part
    <dir>/README.md               gallery between the marker comments below

A part is only re-rendered when its STEP file's SHA-256 changes, so unchanged
files produce no diff. Use --force to re-render everything.

Usage:
    python scripts/stp-previews/render.py [--force] [--check]

--check exits non-zero if any preview or README is missing or stale, without
writing anything. CI uses it on pull requests.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from PIL import Image  # noqa: E402

from OCP.BRep import BRep_Tool  # noqa: E402
from OCP.BRepMesh import BRepMesh_IncrementalMesh  # noqa: E402
from OCP.IFSelect import IFSelect_RetDone  # noqa: E402
from OCP.STEPControl import STEPControl_Reader  # noqa: E402
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED  # noqa: E402
from OCP.TopExp import TopExp_Explorer  # noqa: E402
from OCP.TopLoc import TopLoc_Location  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402

START = "<!-- stp-previews:start -->"
END = "<!-- stp-previews:end -->"
STEP_SUFFIXES = {".stp", ".step"}

# Two opposite viewpoints (elevation, azimuth). Part orientation in the STEP
# files is arbitrary, so the views are labelled neutrally rather than
# "front"/"back".
VIEWS = [("View A", 28, -55), ("View B", -28, 125)]
FIG_SIZE_IN = (8, 4)
DPI = 100
PALETTE_COLORS = 96
BASE_RGB = np.array([0.62, 0.68, 0.76])


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True, capture_output=True, text=True,
    )
    return Path(out.stdout.strip())


def tracked_step_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root,
        check=True, capture_output=True, text=True,
    )
    files = [root / p for p in out.stdout.split("\0") if p]
    return sorted(p for p in files if p.suffix.lower() in STEP_SUFFIXES)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_triangles(path: Path) -> np.ndarray:
    """Return an (N, 3, 3) array of triangles in millimetres."""
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"could not read {path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    # Relative deflection keeps the triangle count proportional to part size.
    BRepMesh_IncrementalMesh(shape, 0.02, False, 0.3, True)

    tris = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            nodes = np.array([
                (lambda p: (p.X(), p.Y(), p.Z()))(tri.Node(i).Transformed(trsf))
                for i in range(1, tri.NbNodes() + 1)
            ])
            reversed_face = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(i).Get()
                if reversed_face:
                    b, c = c, b
                tris.append(nodes[[a - 1, b - 1, c - 1]])
        exp.Next()
    if not tris:
        raise RuntimeError(f"no geometry in {path}")
    return np.asarray(tris)


def view_vector(elev: float, azim: float) -> np.ndarray:
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def shade(tris: np.ndarray, elev: float, azim: float) -> np.ndarray:
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths == 0, 1, lengths)
    # Headlight slightly above and to the left of the camera.
    light = view_vector(elev + 20, azim - 25)
    lambert = np.abs(normals @ light)  # two-sided: STEP normals are not always consistent
    intensity = 0.35 + 0.65 * lambert
    return np.clip(BASE_RGB[None, :] * intensity[:, None], 0, 1)


def render_png(tris: np.ndarray, title: str) -> bytes:
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    centre, span = (lo + hi) / 2, (hi - lo)
    half = span.max() / 2 * 1.02

    fig = plt.figure(figsize=FIG_SIZE_IN, dpi=DPI)
    for idx, (label, elev, azim) in enumerate(VIEWS, start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.add_collection3d(Poly3DCollection(
            tris, facecolors=shade(tris, elev, azim), edgecolors="none",
            linewidths=0, antialiased=False,
        ))
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
            setter(c - half, c + half)
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(label, fontsize=9, color="#555")
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)

    # Palette-quantise: shaded grey-blue renders compress to a few tens of KB.
    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    img = img.quantize(colors=PALETTE_COLORS, method=Image.Quantize.MEDIANCUT)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def describe(stem: str) -> tuple[str, str, str]:
    """Split 'Rear_Cover_501-0019_Rev-1' into ('Rear Cover', '501-0019', '1')."""
    m = re.match(r"^(?P<name>.+?)_(?P<pn>\d{3}-\d{4})_Rev-(?P<rev>[\w.]+)$", stem)
    if not m:
        return stem.replace("_", " "), "", ""
    return m["name"].replace("_", " "), m["pn"], m["rev"]


def gallery_markdown(entries: list[dict]) -> str:
    lines = [
        START,
        "<!-- Generated by scripts/stp-previews/render.py; edits inside these markers are overwritten. -->",
        "",
        "| Preview | Part | Part number | Rev | Size (mm) | Triangles |",
        "|---|---|---|---|---|---|",
    ]
    for e in entries:
        size = " x ".join(f"{v:.1f}" for v in e["size_mm"])
        lines.append(
            f"| ![{e['name']}](previews/{e['png']}) "
            f"| [{e['name']}]({e['file']}) | {e['part_number']} | {e['rev']} "
            f"| {size} | {e['triangles']:,} |"
        )
    lines += [
        "",
        "Previews are two shaded views from opposite corners. Part orientation "
        "comes from each STEP file's own coordinates.",
        END,
    ]
    return "\n".join(lines)


def updated_readme(existing: str | None, dirname: str, gallery: str) -> str:
    if existing is None:
        return f"# {dirname}\n\n{gallery}\n"
    if START in existing and END in existing:
        pre, rest = existing.split(START, 1)
        _, post = rest.split(END, 1)
        return pre + gallery + post
    return existing.rstrip("\n") + "\n\n## Part previews\n\n" + gallery + "\n"


def process_dir(directory: Path, step_files: list[Path], force: bool, check: bool) -> list[str]:
    """Return a list of human-readable problems (check mode) or changes (write mode)."""
    preview_dir = directory / "previews"
    manifest_path = preview_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    report, entries = [], []
    new_manifest = {}

    for step in step_files:
        digest = sha256(step)
        png_name = f"{step.stem}.png"
        png_path = preview_dir / png_name
        cached = manifest.get(step.name)
        fresh = cached and cached.get("sha256") == digest and png_path.exists()

        if fresh and not force:
            meta = cached
        elif check:
            report.append(f"stale preview: {png_path.relative_to(directory.parent)}")
            meta = cached or {"size_mm": [0, 0, 0], "triangles": 0}
        else:
            print(f"rendering {step.name} ...", flush=True)
            tris = load_triangles(step)
            pts = tris.reshape(-1, 3)
            size = (pts.max(0) - pts.min(0)).round(2).tolist()
            name, pn, rev = describe(step.stem)
            preview_dir.mkdir(exist_ok=True)
            png_path.write_bytes(render_png(tris, f"{name}  {pn}  Rev {rev}".strip()))
            meta = {"sha256": digest, "size_mm": size, "triangles": int(len(tris))}
            report.append(f"rendered {png_path} ({png_path.stat().st_size // 1024} KB)")

        new_manifest[step.name] = {k: meta[k] for k in ("sha256", "size_mm", "triangles") if k in meta}
        name, pn, rev = describe(step.stem)
        entries.append({
            "name": name, "part_number": pn, "rev": rev, "file": step.name,
            "png": png_name, "size_mm": meta["size_mm"], "triangles": meta["triangles"],
        })

    # Drop previews whose STEP file no longer exists.
    for old in set(manifest) - set(new_manifest):
        stale_png = preview_dir / f"{Path(old).stem}.png"
        if check:
            report.append(f"orphaned preview: {stale_png}")
        elif stale_png.exists():
            stale_png.unlink()
            report.append(f"removed {stale_png}")

    readme = directory / "README.md"
    current = readme.read_text() if readme.exists() else None
    wanted = updated_readme(current, directory.name, gallery_markdown(entries))
    if current != wanted:
        if check:
            report.append(f"stale gallery: {readme}")
        else:
            readme.write_text(wanted)
            report.append(f"updated {readme}")

    if not check:
        preview_dir.mkdir(exist_ok=True)
        text = json.dumps(new_manifest, indent=2, sort_keys=True) + "\n"
        if not manifest_path.exists() or manifest_path.read_text() != text:
            manifest_path.write_text(text)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-render every part")
    ap.add_argument("--check", action="store_true", help="report stale output, write nothing")
    args = ap.parse_args()

    root = repo_root()
    by_dir: dict[Path, list[Path]] = {}
    for f in tracked_step_files(root):
        by_dir.setdefault(f.parent, []).append(f)
    if not by_dir:
        print("no STEP files found")
        return 0

    report = []
    for directory, files in sorted(by_dir.items()):
        report += process_dir(directory, files, args.force, args.check)
    print("\n".join(report) or "all previews up to date")
    return 1 if args.check and report else 0


if __name__ == "__main__":
    sys.exit(main())
