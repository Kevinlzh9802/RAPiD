#!/usr/bin/env python3
"""
Update all detections.json in subfolders by replacing temporal_id with real_id
from each subfolder's id_mapping.json. Optionally re-render all images with real_id.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse
import json
from pathlib import Path

from batch_detect import list_subfolders, render_horizontal_bboxes


def main():
    parser = argparse.ArgumentParser(
        description="Update detections.json with real_id from id_mapping.json; optionally re-render images."
    )
    parser.add_argument("root_folder", type=str, help="Root folder containing subfolders with detections.")
    parser.add_argument(
        "--update-rendered",
        action="store_true",
        help="Re-render all images with real_id and overwrite existing rendered images.",
    )
    parser.add_argument("--detections-json", type=str, default="detections.json")
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
        id_mapping_path = subfolder / args.id_mapping_json
        detections_path = subfolder / args.detections_json

        if not id_mapping_path.exists():
            print(f"  Skip {subfolder.name}: no {args.id_mapping_json}")
            continue
        if not detections_path.exists():
            print(f"  Skip {subfolder.name}: no {args.detections_json}")
            continue

        with open(id_mapping_path, encoding="utf-8") as f:
            id_data = json.load(f)
        id_mapping_list = id_data.get("id_mapping", [])
        temporal_to_real = {int(m["temporal_id"]): int(m["real_id"]) for m in id_mapping_list}

        with open(detections_path, encoding="utf-8") as f:
            det_data = json.load(f)

        for img_entry in det_data.get("images", []):
            for b in img_entry.get("bboxes", []):
                tid = b.get("temporal_id")
                if tid is not None and tid in temporal_to_real:
                    b["real_id"] = temporal_to_real[tid]
                elif tid is not None:
                    b["real_id"] = tid  # fallback if not in mapping

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
