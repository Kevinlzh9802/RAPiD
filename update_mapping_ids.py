#!/usr/bin/env python3
"""
Update all detections.json in subfolders by setting real_id from each subfolder's
id_mapping.csv (or id_mapping.json). Key is (file_name, temporal_id). Optionally re-render.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse
import csv
import json
from pathlib import Path

from batch_detect import list_subfolders, render_horizontal_bboxes


def load_mapping_csv(csv_path):
    """Load (file_name, temporal_id) -> real_id from CSV with columns file_name, temporal_id, real_id."""
    key_to_real = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row.get("file_name", "").strip()
            tid = int(row["temporal_id"])
            rid = int(row["real_id"])
            key_to_real[(fn, tid)] = rid
    return key_to_real


def load_mapping_json(json_path):
    """Load (file_name, temporal_id) -> real_id from JSON id_mapping (entries with file, temporal_id, real_id)."""
    with open(json_path, encoding="utf-8") as f:
        id_data = json.load(f)
    id_mapping_list = id_data.get("id_mapping", [])
    return {(m["file"], int(m["temporal_id"])): int(m["real_id"]) for m in id_mapping_list}


def main():
    parser = argparse.ArgumentParser(
        description="Update detections.json with real_id from id_mapping.csv (or .json); optionally re-render."
    )
    parser.add_argument("root_folder", type=str, help="Root folder containing subfolders with detections.")
    parser.add_argument(
        "--update-rendered",
        action="store_true",
        help="Re-render all images with real_id and overwrite existing rendered images.",
    )
    parser.add_argument("--detections-json", type=str, default="detections.json")
    parser.add_argument("--id-mapping-csv", type=str, default="id_mapping.csv")
    parser.add_argument("--id-mapping-json", type=str, default="id_mapping.json")
    parser.add_argument("--rendered-dir", type=str, default="rendered")
    args = parser.parse_args()

    root = Path(args.root_folder)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    subfolders = list_subfolders(root)
    if not subfolders:
        print(f"No subfolders found under {root}")
        return

    for subfolder in subfolders:
        csv_path = subfolder / args.id_mapping_csv
        json_path = subfolder / args.id_mapping_json
        detections_path = subfolder / args.detections_json

        if not detections_path.exists():
            print(f"  Skip {subfolder.name}: no {args.detections_json}")
            continue
        if csv_path.exists():
            file_tid_to_real = load_mapping_csv(csv_path)
        elif json_path.exists():
            file_tid_to_real = load_mapping_json(json_path)
        else:
            print(f"  Skip {subfolder.name}: no {args.id_mapping_csv} or {args.id_mapping_json}")
            continue

        with open(detections_path, encoding="utf-8") as f:
            det_data = json.load(f)

        for img_entry in det_data.get("images", []):
            file_name = img_entry.get("image", "")
            kept = []
            for b in img_entry.get("bboxes", []):
                tid = b.get("temporal_id")
                if tid is None:
                    kept.append(b)
                    continue
                key = (file_name, int(tid))
                if key not in file_tid_to_real:
                    print(f"  Warning: no mapping for temporal_id {tid} in file '{file_name}' (subfolder {subfolder.name})")
                    b["real_id"] = tid  # fallback
                    kept.append(b)
                    continue
                real_id = file_tid_to_real[key]
                if real_id == -1:
                    continue  # remove this bbox from detections
                b["real_id"] = real_id
                kept.append(b)
            img_entry["bboxes"] = kept

        with open(detections_path, "w", encoding="utf-8") as f:
            json.dump(det_data, f, indent=2, ensure_ascii=False)
        print(f"  Updated {detections_path}")

        if args.update_rendered:
            render_dir = subfolder / args.rendered_dir
            render_dir.mkdir(parents=True, exist_ok=True)
            for img_entry in det_data.get("images", []):
                img_name = img_entry.get("image")
                bboxes = img_entry.get("bboxes", [])
                img_path = subfolder / img_name
                if not img_path.exists():
                    print(f"    Skip render (missing): {img_name}")
                    continue
                stem = Path(img_name).stem
                ext = Path(img_name).suffix
                out_name = f"{stem}_det{ext}"
                out_path = render_dir / out_name
                render_horizontal_bboxes(img_path, bboxes, out_path, id_key="real_id")
            print(f"  Re-rendered {subfolder.name}/{args.rendered_dir}/")

    print("Done.")


if __name__ == "__main__":
    main()
