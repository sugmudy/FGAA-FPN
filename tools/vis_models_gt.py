#!/usr/bin/env python
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector

from mmrotate.core import poly2obb_np, rbbox_overlaps


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize OrientedRCNN, FGAMFPN and GT on remote sensing images."
        ))
    parser.add_argument(
        "--img",
        type=str,
        default=None,
        help="Single image path. If set, --img-dir is ignored.")
    parser.add_argument(
        "--img-dir",
        type=str,
        default="data/split_DOTA/val/images",
        help="Directory containing images.")
    parser.add_argument(
        "--ann-dir",
        type=str,
        default="data/split_DOTA/val/labelTxt",
        help="Directory containing DOTA labelTxt annotations.")
    parser.add_argument(
        "--config-rcnn",
        type=str,
        default="configs/oriented_rcnn/BVAM.py",
        help="Config for OrientedRCNN.")
    parser.add_argument(
        "--checkpoint-rcnn",
        type=str,
        default="work_dirs/OrientedRCNN.pth",
        help="Checkpoint for OrientedRCNN.")
    parser.add_argument(
        "--config-fgam",
        type=str,
        default="configs/oriented_rcnn/FGAM.py",
        help="Config for FGAMFPN OrientedRCNN.")
    parser.add_argument(
        "--checkpoint-fgam",
        type=str,
        default="work_dirs/FGAMFPN.pth",
        help="Checkpoint for FGAMFPN.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="work_dirs/vis_results",
        help="Output directory for visualized images.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Score threshold for predictions.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Randomly sample this many images to process. 0 means all.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used for sampling images when --max-images > 0.")
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.2,
        help="IoU threshold to match predictions with GT (for FN/FP analysis).")
    parser.add_argument(
        "--hide-fp",
        action="store_true",
        help="Only draw correct detections (green) and missed GT (blue); do not draw false positives (red).")
    parser.add_argument(
        "--angle-version",
        type=str,
        default="le90",
        help="Angle version for GT conversion: le90/le135/oc.")
    parser.add_argument(
        "--tag-rcnn",
        type=str,
        default="orientedrcnn",
        help="Suffix tag for OrientedRCNN outputs.")
    parser.add_argument(
        "--tag-fgam",
        type=str,
        default="fgamfpn",
        help="Suffix tag for FGAMFPN outputs.")
    parser.add_argument(
        "--tag-gt",
        type=str,
        default="gt",
        help="Suffix tag for GT outputs.")
    parser.add_argument(
        "--allow-missing-gt",
        action="store_true",
        help="Skip GT visualization if annotation is missing.")
    return parser.parse_args()


def collect_images(img, img_dir):
    if img:
        return [Path(img)]
    img_root = Path(img_dir)
    images = []
    for ext in IMG_EXTS:
        images.extend(img_root.glob(f"*{ext}"))
        images.extend(img_root.glob(f"*{ext.upper()}"))
    return sorted(images)


def load_gt_bboxes(ann_path, angle_version):
    """Load GT from DOTA labelTxt.

    Returns:
        (gt_bboxes, gt_names) or None if ann missing.
    """
    bboxes = []
    names = []
    if not ann_path.exists():
        return None
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            poly = np.array(parts[:8], dtype=np.float32)
            try:
                obb = poly2obb_np(poly, angle_version)
            except Exception:
                continue
            if obb is None:
                continue
            cls_name = parts[8]
            bboxes.append(obb)
            names.append(cls_name)
    if not bboxes:
        return (np.zeros((0, 5), dtype=np.float32), [])
    return (np.array(bboxes, dtype=np.float32), names)


def _rbboxes_to_polys(rbboxes: np.ndarray) -> np.ndarray:
    """Convert rotated boxes (cx,cy,w,h,angle) -> polygons (N,4,2)."""
    if rbboxes.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    polys = []
    for bbox in rbboxes:
        xc, yc, w, h, ag = bbox[:5]
        wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
        hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
        p1 = (xc - wx - hx, yc - wy - hy)
        p2 = (xc + wx - hx, yc + wy - hy)
        p3 = (xc + wx + hx, yc + wy + hy)
        p4 = (xc - wx + hx, yc - wy + hy)
        polys.append([p1, p2, p3, p4])
    return np.asarray(polys, dtype=np.float32)


def _draw_rbboxes(img_bgr: np.ndarray,
                  rbboxes: np.ndarray,
                  color_bgr,
                  thickness: int = 2) -> np.ndarray:
    if rbboxes.size == 0:
        return img_bgr
    polys = _rbboxes_to_polys(rbboxes).round().astype(np.int32)
    for poly in polys:
        cv2.polylines(img_bgr, [poly], isClosed=True, color=color_bgr,
                      thickness=thickness)
    return img_bgr


def _parse_det_result(result):
    """Return (bboxes[N,5], scores[N], labels[N])."""
    if isinstance(result, tuple):
        bbox_result = result[0]
    else:
        bbox_result = result
    if bbox_result is None:
        return (np.zeros((0, 5), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    bboxes_list = []
    scores_list = []
    labels_list = []
    for cls_id, dets in enumerate(bbox_result):
        if dets is None or len(dets) == 0:
            continue
        dets = np.asarray(dets, dtype=np.float32)
        if dets.ndim != 2 or dets.shape[1] < 6:
            continue
        bboxes_list.append(dets[:, :5])
        scores_list.append(dets[:, 5])
        labels_list.append(np.full((dets.shape[0],), cls_id, dtype=np.int64))
    if not bboxes_list:
        return (np.zeros((0, 5), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    bboxes = np.concatenate(bboxes_list, axis=0)
    scores = np.concatenate(scores_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return bboxes, scores, labels


def _match_and_split(pred_boxes: np.ndarray,
                     pred_scores: np.ndarray,
                     pred_labels: np.ndarray,
                     gt_boxes: np.ndarray,
                     gt_labels: np.ndarray,
                     iou_thr: float):
    """Split preds into correct(green) and wrong(red), and GT into missed(blue)."""
    n_pred = pred_boxes.shape[0]
    n_gt = gt_boxes.shape[0]

    pred_correct = np.zeros((n_pred,), dtype=bool)
    pred_wrong = np.zeros((n_pred,), dtype=bool)
    gt_matched_correct = np.zeros((n_gt,), dtype=bool)
    gt_matched_wrong = np.zeros((n_gt,), dtype=bool)

    if n_pred == 0:
        gt_missed = np.ones((n_gt,), dtype=bool)
        return pred_correct, pred_wrong, gt_missed
    if n_gt == 0:
        pred_wrong[:] = True
        return pred_correct, pred_wrong, np.zeros((0,), dtype=bool)

    # IoU matrix (N, M)
    ious = rbbox_overlaps(
        torch.from_numpy(pred_boxes.astype(np.float32)),
        torch.from_numpy(gt_boxes.astype(np.float32)),
    ).cpu().numpy()

    order = np.argsort(-pred_scores)

    # Pass 1: match correct class first.
    for pi in order:
        if pred_correct[pi]:
            continue
        same_cls = (gt_labels == pred_labels[pi])
        candidates = np.where(same_cls & (~gt_matched_correct))[0]
        if candidates.size == 0:
            continue
        best_j = candidates[np.argmax(ious[pi, candidates])]
        if ious[pi, best_j] >= iou_thr:
            pred_correct[pi] = True
            gt_matched_correct[best_j] = True

    # Pass 2: match remaining GT by IoU (wrong class) to highlight label errors.
    for pi in order:
        if pred_correct[pi] or pred_wrong[pi]:
            continue
        candidates = np.where((~gt_matched_correct) & (~gt_matched_wrong))[0]
        if candidates.size == 0:
            break
        best_j = candidates[np.argmax(ious[pi, candidates])]
        if ious[pi, best_j] >= iou_thr:
            pred_wrong[pi] = True
            gt_matched_wrong[best_j] = True

    # Remaining preds are false positives -> wrong (red)
    pred_wrong |= (~pred_correct)
    gt_missed = ~(gt_matched_correct | gt_matched_wrong)
    return pred_correct, pred_wrong, gt_missed


def visualize_gt(img_path, ann_path, out_path, angle_version):
    gt = load_gt_bboxes(ann_path, angle_version)
    if gt is None:
        return False
    gt_bboxes, _gt_names = gt
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    img = _draw_rbboxes(img, gt_bboxes, color_bgr=(0, 255, 0))
    cv2.imwrite(str(out_path), img)
    return True


def visualize_pred_with_errors(model,
                               img_path,
                               ann_path,
                               out_path,
                               score_thr: float,
                               angle_version: str,
                               match_iou: float,
                               hide_fp: bool):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    gt = load_gt_bboxes(ann_path, angle_version)
    if gt is None:
        # No GT: just draw predictions in green.
        result = inference_detector(model, str(img_path))
        pred_boxes, pred_scores, _pred_labels = _parse_det_result(result)
        keep = pred_scores >= float(score_thr)
        pred_boxes = pred_boxes[keep]
        img = _draw_rbboxes(img, pred_boxes, color_bgr=(0, 255, 0))
        cv2.imwrite(str(out_path), img)
        return False

    gt_boxes, gt_names = gt
    class_map = {name: i for i, name in enumerate(getattr(model, "CLASSES", []) or [])}
    if not class_map:
        raise RuntimeError(
            "model.CLASSES is empty; cannot map GT class names for matching. "
            "Check your config metainfo / checkpoint meta.")
    gt_labels = np.array([class_map.get(n, -1) for n in gt_names], dtype=np.int64)
    valid = gt_labels >= 0
    gt_boxes = gt_boxes[valid]
    gt_labels = gt_labels[valid]

    result = inference_detector(model, str(img_path))
    pred_boxes, pred_scores, pred_labels = _parse_det_result(result)
    keep = pred_scores >= float(score_thr)
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]

    pred_correct, pred_wrong, gt_missed = _match_and_split(
        pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, match_iou)

    # Green: correct detections
    img = _draw_rbboxes(img, pred_boxes[pred_correct], color_bgr=(0, 255, 0))
    # Red: wrong detections (false positives or wrong class)
    if not hide_fp:
        img = _draw_rbboxes(
            img,
            pred_boxes[pred_wrong & (~pred_correct)],
            color_bgr=(0, 0, 255),
        )
    # Blue: missed GT objects
    img = _draw_rbboxes(img, gt_boxes[gt_missed], color_bgr=(0, 0, 255))

    cv2.imwrite(str(out_path), img)
    return True


def main():
    args = parse_args()

    images = collect_images(args.img, args.img_dir)
    if not images:
        raise FileNotFoundError("No images found. Check --img or --img-dir.")
    if args.max_images and args.max_images > 0 and len(images) > args.max_images:
        rng = random.Random(args.seed)
        images = sorted(rng.sample(images, k=args.max_images))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_rcnn = init_detector(args.config_rcnn, args.checkpoint_rcnn,
                               device=args.device)
    model_fgam = init_detector(args.config_fgam, args.checkpoint_fgam,
                               device=args.device)

    for img_path in images:
        stem = img_path.stem
        out_rcnn = out_dir / f"{stem}_{args.tag_rcnn}.png"
        out_fgam = out_dir / f"{stem}_{args.tag_fgam}.png"
        out_gt = out_dir / f"{stem}_{args.tag_gt}.png"

        ann_path = Path(args.ann_dir) / f"{stem}.txt"
        ok_pred1 = visualize_pred_with_errors(
            model_rcnn,
            img_path,
            ann_path,
            out_rcnn,
            args.score_thr,
            args.angle_version,
            args.match_iou,
            args.hide_fp,
        )
        ok_pred2 = visualize_pred_with_errors(
            model_fgam,
            img_path,
            ann_path,
            out_fgam,
            args.score_thr,
            args.angle_version,
            args.match_iou,
            args.hide_fp,
        )
        ok = visualize_gt(img_path, ann_path, out_gt, args.angle_version)
        if not ok and not args.allow_missing_gt:
            raise FileNotFoundError(
                f"Missing GT annotation: {ann_path}")

        print(f"Done: {img_path.name}")


if __name__ == "__main__":
    main()
