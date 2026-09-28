"""Rasterize xBD-style pixel-coordinate WKT annotations into training masks."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from shapely.wkt import loads


DAMAGE_IDS = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4}


def polygon_mask(geometry, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    for polygon in polygons:
        if polygon.geom_type != "Polygon":
            raise ValueError(f"Expected Polygon or MultiPolygon, got {geometry.geom_type}")
        local = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(local, [np.rint(polygon.exterior.coords).astype(np.int32)], 1)
        for hole in polygon.interiors:
            cv2.fillPoly(local, [np.rint(hole.coords).astype(np.int32)], 0)
        mask |= local
    return mask


def process(pre_json, unclassified="no-damage", overwrite=False):
    root = pre_json.parent.parent
    post_json = pre_json.with_name(pre_json.name.replace("_pre_disaster", "_post_disaster"))
    pre_img = root / "images" / pre_json.with_suffix(".png").name
    post_img = pre_img.with_name(pre_img.name.replace("_pre_disaster", "_post_disaster"))
    imgs = [cv2.imread(str(p)) for p in (pre_img, post_img)]
    if any(a is None for a in imgs):
        raise OSError(f"Missing image pair for {pre_json}")
    if imgs[0].shape != imgs[1].shape:
        raise ValueError(f"Mismatched image pair: {pre_img}")
    shape = imgs[0].shape[:2]
    loc, dmg = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    pre = json.loads(pre_json.read_text())
    post = json.loads(post_json.read_text())
    for feature in pre["features"]["xy"]:
        loc[polygon_mask(loads(feature["wkt"]), shape) > 0] = 255
    unknown_count = 0
    for feature in post["features"]["xy"]:
        subtype = feature["properties"]["subtype"]
        if subtype == "un-classified":
            unknown_count += 1
            if unclassified == "error":
                raise ValueError(f"Unclassified annotation in {post_json}")
            subtype = "no-damage"
        if subtype not in DAMAGE_IDS:
            raise ValueError(f"Unknown damage label {subtype!r} in {post_json}")
        dmg[polygon_mask(loads(feature["wkt"]), shape) > 0] = DAMAGE_IDS[subtype]
    dest = root / "masks"
    paths = [dest / pre_img.name, dest / post_img.name]
    if not overwrite and any(p.exists() for p in paths):
        raise FileExistsError(f"Masks already exist for {pre_img.name}; use --overwrite intentionally")
    dest.mkdir(exist_ok=True)
    for path, mask in zip(paths, (loc, dmg)):
        if not cv2.imwrite(str(path), mask):
            raise OSError(f"Failed to write {path}")
    return unknown_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true", help="Discover nested event folders")
    parser.add_argument("--unclassified", choices=["no-damage", "error"], default="no-damage")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    files = []
    for root in args.roots:
        labels = sorted(root.rglob("labels")) if args.recursive else [root / "labels"]
        for folder in labels:
            files.extend(sorted(folder.glob("*_pre_disaster.json")))
    if not files:
        raise ValueError("No matching annotation pairs found")
    if len({p.resolve() for p in files}) != len(files):
        raise ValueError("Overlapping input roots")
    unclassified = sum(process(p, args.unclassified, args.overwrite) for p in files)
    print(json.dumps({"pairs": len(files), "unclassified_mapped_to_no_damage": unclassified}))


if __name__ == "__main__":
    main()
