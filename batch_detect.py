#!/usr/bin/env python3
"""
Batch detection over subfolders: run RAPiD on all images in each subfolder,
convert rotated bboxes to horizontal (axis-aligned), save one JSON per subfolder
and optionally render horizontal bboxes on images.
"""
# Prevent Qt/cv2 from loading GUI plugins when run as script
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse
import json
import numpy as np
from pathlib import Path

from PIL import Image
import cv2

from api import Detector


# -----------------------------------------------------------------------------
# Rotated bbox -> horizontal bbox (axis-aligned bounding box of the 4 corners)
# -----------------------------------------------------------------------------
def rotated_bbox_to_horizontal(x, y, w, h, angle_deg):
    """
    Convert one rotated bbox (center x,y, w, h, angle in degrees) to
    axis-aligned bbox in (x, y, w, h) with (x,y) = top-left corner.
    """
    c = np.cos(np.deg2rad(angle_deg))
    s = np.sin(np.deg2rad(angle_deg))
    R = np.array([[c, s], [-s, c]])
    # corners in local frame (center at origin)
    half = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    corners = (half @ R.T) + np.array([x, y])
    x_min = float(np.min(corners[:, 0]))
    x_max = float(np.max(corners[:, 0]))
    y_min = float(np.min(corners[:, 1]))
    y_max = float(np.max(corners[:, 1]))
    return x_min, y_min, (x_max - x_min), (y_max - y_min)


def detections_to_horizontal_bboxes(detections, temporal_id_start=0):
    """
    detections: tensor or ndarray, shape (N, 6) or (6,) with [x, y, w, h, angle, conf].
    temporal_id_start: first temporal_id to assign (for global numbering across images).
    Returns (list of dicts with temporal_id, next temporal_id).
    """
    if hasattr(detections, 'numpy'):
        detections = detections.numpy()
    if detections.size == 0:
        return [], temporal_id_start
    if detections.ndim == 1:
        detections = detections.reshape(1, -1)
    out = []
    tid = temporal_id_start
    for row in detections:
        x, y, w, h, a, conf = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])
        x_tl, y_tl, w_h, h_h = rotated_bbox_to_horizontal(x, y, w, h, a)
        out.append({
            "temporal_id": tid,
            "real_id": tid,
            "x": round(x_tl, 2), "y": round(y_tl, 2), "w": round(w_h, 2), "h": round(h_h, 2),
            "score": round(conf, 4),
        })
        tid += 1
    return out, tid


# -----------------------------------------------------------------------------
# Image listing and detection
# -----------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted([f.name for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])


def list_subfolders(root):
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def run_detection_on_folder(detector, folder_path, input_size=1024, conf_thres=0.3, test_aug=None):
    """
    Run detector on all images in folder_path. Yields (image_name, detections_tensor) per image.
    """
    folder_path = Path(folder_path)
    names = list_images(folder_path)
    for name in names:
        img_path = folder_path / name
        try:
            detections = detector.detect_one(
                img_path=str(img_path),
                return_img=False,
                input_size=input_size,
                conf_thres=conf_thres,
                test_aug=test_aug,
            )
            yield name, detections
        except Exception as e:
            print(f"  [skip] {name}: {e}")
            yield name, None


def render_horizontal_bboxes(image_path, bboxes, output_path, color=(0, 255, 0), thickness=2, id_key="temporal_id"):
    """
    Draw horizontal bboxes on image with id labels and save to output_path.
    image_path: path to original image (read as RGB for consistency).
    bboxes: list of {"x", "y", "w", "h", ...} with id_key (temporal_id or real_id).
    id_key: key to use for the label text ("temporal_id" or "real_id").
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, img.shape[0] / 800)
    font_thick = max(1, int(img.shape[0] / 400))
    for b in bboxes:
        x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        label = str(b.get(id_key, b.get("temporal_id", b.get("real_id", ""))))
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), color, -1)
        cv2.putText(img, label, (x + 2, y - 2), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), img_bgr)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch detect and save horizontal bboxes per subfolder.")
    parser.add_argument("root_folder", type=str, help="Root folder containing one subfolder per sequence.")
    parser.add_argument("--render", action="store_true", help="Render horizontal bboxes on images and save in subfolder.")
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--conf-thres", type=float, default=0.3)
    parser.add_argument("--weights", type=str, default="./weights/pL1_MWHB1024_Mar11_4000.ckpt")
    parser.add_argument("--no-cuda", action="store_true", help="Use CPU.")
    parser.add_argument("--output-json", type=str, default="detections.json", help="JSON filename inside each subfolder.")
    parser.add_argument("--id-mapping-json", type=str, default="id_mapping.json", help="Dummy JSON with files and temporal_id->real_id mapping.")
    args = parser.parse_args()

    root = Path(args.root_folder)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    print("Initializing detector...")
    detector = Detector(
        model_name="rapid",
        weights_path=args.weights,
        use_cuda=not args.no_cuda,
    )

    subfolders = list_subfolders(root)
    if not subfolders:
        print(f"No subfolders found under {root}")
        return

    print(f"Found {len(subfolders)} subfolder(s).")

    for subfolder in subfolders:
        print(f"\nProcessing: {subfolder.name}")
        images_data = []
        id_mapping_list = []  # [{"temporal_id": int, "real_id": int}, ...]
        next_temporal_id = 0
        file_names = []
        render_dir = subfolder / "rendered" if args.render else None
        if render_dir is not None:
            render_dir.mkdir(parents=True, exist_ok=True)

        for img_name, detections in run_detection_on_folder(
            detector, subfolder, input_size=args.input_size, conf_thres=args.conf_thres
        ):
            if detections is None:
                bboxes_h = []
                next_tid = next_temporal_id
            else:
                bboxes_h, next_tid = detections_to_horizontal_bboxes(detections, temporal_id_start=next_temporal_id)

            for b in bboxes_h:
                id_mapping_list.append({"temporal_id": b["temporal_id"], "real_id": b["temporal_id"]})

            images_data.append({
                "image": img_name,
                "bboxes": bboxes_h,
            })
            file_names.append(img_name)

            if args.render and render_dir is not None:
                stem = Path(img_name).stem
                ext = Path(img_name).suffix
                out_name = f"{stem}_det{ext}"
                out_path = render_dir / out_name
                render_horizontal_bboxes(subfolder / img_name, bboxes_h, out_path, id_key="temporal_id")

            next_temporal_id = next_tid

        json_path = subfolder / args.output_json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"images": images_data}, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {json_path} ({len(images_data)} images)")

        id_mapping_path = subfolder / args.id_mapping_json
        id_mapping_data = {"files": file_names, "id_mapping": id_mapping_list}
        with open(id_mapping_path, "w", encoding="utf-8") as f:
            json.dump(id_mapping_data, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {id_mapping_path} ({len(id_mapping_list)} ids)")

    print("\nDone.")


if __name__ == "__main__":
    main()
