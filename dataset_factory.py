"""
无人机汽车检测数据集自动化工具（核心生成逻辑）

用法（本地）
  python dataset_factory.py
  python dataset_factory.py --config config.yaml
  python dataset_factory.py --config config.yaml --output_dir ./out_workdir

EXE 打包（Windows PowerShell，示例模板）
  1) 安装 pyinstaller（只用于打包环境）
     pip install pyinstaller
  2) 构建单文件 EXE，并把 config.yaml 一并打包：
     pyinstaller -F --name drone_dataset_factory --add-data "config.yaml;." dataset_factory.py
  3) 运行：
     drone_dataset_factory --config config.yaml

说明
- 网页端 web_tool.py 会复用本文件的核心逻辑生成数据集并打包 zip。
- 该文件不做任何阻塞式 input()，避免服务端卡死/假死。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


ProgressCb = Callable[[str, int, int, str], None]


def _is_finite_number(x: Any) -> bool:
    try:
        xf = float(x)
        return math.isfinite(xf)
    except Exception:
        return False


def _ensure_odd_pos_int(v: Any, default: int = 3) -> int:
    try:
        iv = int(v)
        if iv <= 0:
            return default
        if iv % 2 == 0:
            iv += 1
        return iv
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_stem(p: Path) -> str:
    # 保证 label 文件名是安全的纯 ascii 字符（仅替换少量不安全字符）
    s = p.stem
    for ch in [" ", "\\", "/", ":", "*", "?", "\"", "<", ">", "|"]:
        s = s.replace(ch, "_")
    return s


class ConfigLoader:
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError("config.yaml must be a YAML mapping/dict")
        return cfg

    @staticmethod
    def load_and_validate(config_path: str) -> Dict[str, Any]:
        cfg = ConfigLoader.load(config_path)
        cfg = ConfigLoader._apply_defaults(cfg)
        ConfigLoader._validate_schema(cfg)
        cfg = ConfigLoader._normalize_ratios(cfg)
        return cfg

    @staticmethod
    def _apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {
            "video_path": "drone_car_video.mp4",
            "frame_interval": 10,
            "max_frames": -1,
            "skip_blurry": True,
            "blur_threshold": 100.0,
            "morph_kernel": 3,
            "min_box_size": 20,
            "min_norm": 0.0,
            "max_norm": 1.0,
            "coord_eps": 0.01,
            "class_id": 0,
            "names": ["car"],
            "train_ratio": 0.7,
            "val_ratio": 0.2,
            "test_ratio": 0.1,
            "split_seed": 42,
            "allow_empty_labels": True,
            "zip_output": False,
            "output_base_dir": ".",
            "lower_red1": [0, 150, 100],
            "upper_red1": [10, 255, 255],
            "lower_red2": [170, 150, 100],
            "upper_red2": [180, 255, 255],
            # SFT：检测模型训练数据（从 YOLO labels 派生 jsonl）
            "sft_enable": True,
            "sft_include_test": False,
            "sft_empty_answer": "none",
            "sft_answer_sep": ";",
            "sft_output_dir": "sft",
        }
        out = dict(defaults)
        out.update(cfg)
        return out

    @staticmethod
    def _validate_hsv_range(v: Any, name: str) -> List[int]:
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise ValueError(f"{name} must be a list of 3 ints [H,S,V], got: {v}")
        out: List[int] = []
        for i, item in enumerate(v):
            if not _is_finite_number(item):
                raise ValueError(f"{name}[{i}] must be a number, got: {item}")
            out.append(int(item))
        return out

    @staticmethod
    def _validate_schema(cfg: Dict[str, Any]) -> None:
        if "video_path" not in cfg:
            raise ValueError("Missing config field: video_path")

        fi = int(cfg["frame_interval"])
        if fi < 1:
            raise ValueError("frame_interval must be >= 1")
        cfg["frame_interval"] = fi

        mf = int(cfg.get("max_frames", -1))
        if mf < -1:
            raise ValueError("max_frames must be -1 or >= 1")
        cfg["max_frames"] = mf

        cfg["skip_blurry"] = bool(cfg["skip_blurry"])
        cfg["blur_threshold"] = float(cfg["blur_threshold"])

        cfg["morph_kernel"] = _ensure_odd_pos_int(cfg.get("morph_kernel", 3), default=3)

        cfg["min_box_size"] = int(cfg["min_box_size"])
        if cfg["min_box_size"] < 1:
            raise ValueError("min_box_size must be >= 1")

        cfg["min_norm"] = float(cfg["min_norm"])
        cfg["max_norm"] = float(cfg["max_norm"])
        if not (0.0 <= cfg["min_norm"] <= cfg["max_norm"] <= 1.0):
            raise ValueError("min_norm/max_norm must satisfy 0<=min_norm<=max_norm<=1")

        cfg["coord_eps"] = float(cfg.get("coord_eps", 0.01))
        if cfg["coord_eps"] < 0:
            raise ValueError("coord_eps must be >= 0")

        cfg["class_id"] = int(cfg.get("class_id", 0))
        if cfg["class_id"] < 0:
            raise ValueError("class_id must be >= 0")

        names = cfg.get("names", ["car"])
        if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
            raise ValueError("names must be a list of strings")
        if len(names) == 0:
            raise ValueError("names must not be empty")

        cfg["train_ratio"] = float(cfg["train_ratio"])
        cfg["val_ratio"] = float(cfg["val_ratio"])
        cfg["test_ratio"] = float(cfg["test_ratio"])
        if min(cfg["train_ratio"], cfg["val_ratio"], cfg["test_ratio"]) < 0:
            raise ValueError("train/val/test_ratio must be >= 0")

        cfg["split_seed"] = int(cfg["split_seed"])
        cfg["allow_empty_labels"] = bool(cfg["allow_empty_labels"])
        cfg["zip_output"] = bool(cfg["zip_output"])
        cfg["output_base_dir"] = str(cfg.get("output_base_dir", "."))

        # SFT
        cfg["sft_enable"] = bool(cfg.get("sft_enable", True))
        cfg["sft_include_test"] = bool(cfg.get("sft_include_test", False))
        cfg["sft_empty_answer"] = str(cfg.get("sft_empty_answer", "none"))
        cfg["sft_answer_sep"] = str(cfg.get("sft_answer_sep", ";"))
        cfg["sft_output_dir"] = str(cfg.get("sft_output_dir", "sft"))

        # HSV
        cfg["lower_red1"] = ConfigLoader._validate_hsv_range(cfg["lower_red1"], "lower_red1")
        cfg["upper_red1"] = ConfigLoader._validate_hsv_range(cfg["upper_red1"], "upper_red1")
        cfg["lower_red2"] = ConfigLoader._validate_hsv_range(cfg["lower_red2"], "lower_red2")
        cfg["upper_red2"] = ConfigLoader._validate_hsv_range(cfg["upper_red2"], "upper_red2")

        # H range guard (OpenCV HSV: H 0..179)
        for nm in ["lower_red1", "upper_red1", "lower_red2", "upper_red2"]:
            h, s, v = cfg[nm]
            cfg[nm] = [_clamp(int(h), 0, 179), _clamp(int(s), 0, 255), _clamp(int(v), 0, 255)]

    @staticmethod
    def _normalize_ratios(cfg: Dict[str, Any]) -> Dict[str, Any]:
        s = cfg["train_ratio"] + cfg["val_ratio"] + cfg["test_ratio"]
        if s <= 0:
            raise ValueError("train_ratio + val_ratio + test_ratio must be > 0")
        # 自动归一化到 1，避免用户手滑导致直接失败
        if abs(s - 1.0) > 1e-6:
            cfg["train_ratio"] = cfg["train_ratio"] / s
            cfg["val_ratio"] = cfg["val_ratio"] / s
            cfg["test_ratio"] = cfg["test_ratio"] / s
            cfg["_ratio_normalized"] = True
        else:
            cfg["_ratio_normalized"] = False
        return cfg


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class RedBoxDetector:
    @staticmethod
    def detect(frame_bgr: np.ndarray, cfg: Dict[str, Any]) -> Tuple[List[YoloBox], Dict[str, Any]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return [], {"error": "empty_frame"}
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return [], {"error": "invalid_frame_shape"}

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        lower1 = np.array(cfg["lower_red1"], dtype=np.uint8)
        upper1 = np.array(cfg["upper_red1"], dtype=np.uint8)
        lower2 = np.array(cfg["lower_red2"], dtype=np.uint8)
        upper2 = np.array(cfg["upper_red2"], dtype=np.uint8)

        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

        k = int(cfg["morph_kernel"])
        kernel = np.ones((k, k), np.uint8)
        # 先做开运算去噪，再做闭运算补洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[YoloBox] = []
        st: Dict[str, Any] = {"contours": len(contours), "discard_small": 0, "discard_coord": 0, "boxes": 0}

        min_box_size = int(cfg["min_box_size"])
        min_norm = float(cfg["min_norm"])
        max_norm = float(cfg["max_norm"])
        coord_eps = float(cfg["coord_eps"])
        class_id = int(cfg["class_id"])

        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            if ww < min_box_size or hh < min_box_size:
                st["discard_small"] += 1
                continue

            x_center = (x + ww / 2.0) / float(w)
            y_center = (y + hh / 2.0) / float(h)
            w_norm = ww / float(w)
            h_norm = hh / float(h)

            # 坐标数值检查
            if not all(_is_finite_number(v) for v in [x_center, y_center, w_norm, h_norm]):
                st["discard_coord"] += 1
                continue

            # 允许微量越界并 clamp，明显越界直接丢弃
            if (x_center < -coord_eps or x_center > 1.0 + coord_eps or
                y_center < -coord_eps or y_center > 1.0 + coord_eps or
                w_norm <= 0.0 or h_norm <= 0.0 or
                w_norm < min_norm - coord_eps or w_norm > max_norm + coord_eps or
                h_norm < min_norm - coord_eps or h_norm > max_norm + coord_eps):
                st["discard_coord"] += 1
                continue

            x_center = _clamp(x_center, 0.0, 1.0)
            y_center = _clamp(y_center, 0.0, 1.0)
            w_norm = _clamp(w_norm, min_norm, 1.0)
            h_norm = _clamp(h_norm, min_norm, 1.0)

            if w_norm <= 0.0 or h_norm <= 0.0:
                st["discard_coord"] += 1
                continue

            boxes.append(YoloBox(class_id, x_center, y_center, w_norm, h_norm))

        st["boxes"] = len(boxes)
        return boxes, st


def detect_red_boxes_bgr(frame_bgr: np.ndarray, cfg: Dict[str, Any]) -> List[YoloBox]:
    boxes, _ = RedBoxDetector.detect(frame_bgr, cfg)
    return boxes


def zip_dir(source_dir: Path, zip_path: Path) -> Path:
    source_dir = source_dir.resolve()
    zip_path = zip_path.resolve()
    if zip_path.exists():
        zip_path.unlink()
    archive_base = str(zip_path.with_suffix(""))
    # make_archive 自动生成 .zip
    shutil.make_archive(archive_base, "zip", root_dir=str(source_dir))
    return zip_path


class DatasetFactory:
    def __init__(self, cfg: Dict[str, Any], output_base_dir: Optional[str] = None):
        self.cfg = cfg
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(output_base_dir) if output_base_dir else Path(cfg.get("output_base_dir", "."))
        self.output_dir = base_dir / f"drone_dataset_{self.timestamp}"
        self.raw_images_dir = self.output_dir / "images" / "all"
        self.raw_labels_dir = self.output_dir / "labels" / "all"
        self.split_images_dir = self.output_dir / "images"
        self.split_labels_dir = self.output_dir / "labels"
        self.stats_dir = self.output_dir / "stats"
        self._init_dirs()

        self.stats: Dict[str, Any] = {
            "started_at": self.timestamp,
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "video_path": cfg.get("video_path"),
            "frames_total": 0,
            "frames_sampled": 0,
            "frames_saved": 0,
            "frames_skipped_blurry": 0,
            "label_files_total": 0,
            "boxes_total": 0,
            "boxes_discarded_by_detection": 0,
            "empty_label_files": 0,
            "invalid_label_lines_removed": 0,
            "invalid_label_lines_total": 0,
            "split": {},
            "sft": {
                "enabled": bool(cfg.get("sft_enable", True)),
                "empty_answer": cfg.get("sft_empty_answer", "none"),
                "answer_sep": cfg.get("sft_answer_sep", ";"),
                "splits": {},
                "samples_total": 0,
                "empty_targets": 0,
                "boxes_total": 0,
            },
            "quality": {
                "splits": {},
                "overall": {},
            },
            "ratio_normalized": bool(cfg.get("_ratio_normalized", False)),
        }

    def _init_dirs(self) -> None:
        splits = ["train", "val", "test"]
        for d in [self.raw_images_dir, self.raw_labels_dir, self.stats_dir]:
            d.mkdir(parents=True, exist_ok=True)
        for s in splits:
            (self.split_images_dir / s).mkdir(parents=True, exist_ok=True)
            (self.split_labels_dir / s).mkdir(parents=True, exist_ok=True)

    def _is_blurry(self, frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        var = float(lap.var())
        return var < float(self.cfg["blur_threshold"])

    def extract_frames(self, video_path: str, progress_cb: Optional[ProgressCb] = None) -> List[Path]:
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video not found: {video_file}")

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_file}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fi = int(self.cfg["frame_interval"])
        max_frames = int(self.cfg["max_frames"])

        frame_idx = 0
        saved_count = 0
        sampled_count = 0
        blurry_count = 0
        frame_paths: List[Path] = []

        last_report_frame = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.stats["frames_total"] += 1

            if fi >= 1 and (frame_idx % fi == 0):
                sampled_count += 1
                if bool(self.cfg["skip_blurry"]):
                    try:
                        if self._is_blurry(frame):
                            blurry_count += 1
                            frame_idx += 1
                            if max_frames > 0 and saved_count >= max_frames:
                                break
                            continue
                    except Exception:
                        # 模糊检测失败时保守地保留该帧，避免漏检导致训练集太小
                        pass

                # 保存帧
                stem = f"frame_{frame_idx:08d}"
                img_path = self.raw_images_dir / f"{_safe_stem(Path(stem))}.jpg"
                try:
                    ok = cv2.imwrite(str(img_path), frame)
                    if not ok:
                        raise RuntimeError("cv2.imwrite failed")
                    frame_paths.append(img_path)
                    saved_count += 1
                except Exception:
                    # 保存失败跳过该帧
                    pass

                if max_frames > 0 and saved_count >= max_frames:
                    break

            frame_idx += 1

            if progress_cb is not None:
                # 控制回调频率，避免 UI 过载
                if total > 0:
                    if frame_idx == 0 or frame_idx - last_report_frame >= max(5, total // 200):
                        progress_cb("extract", frame_idx, total, "Extracting frames")
                        last_report_frame = frame_idx
                else:
                    if frame_idx - last_report_frame >= 50:
                        progress_cb("extract", frame_idx, max(frame_idx, 1), "Extracting frames")
                        last_report_frame = frame_idx

        cap.release()

        self.stats["frames_sampled"] = sampled_count
        self.stats["frames_saved"] = saved_count
        self.stats["frames_skipped_blurry"] = blurry_count

        if saved_count <= 0:
            raise RuntimeError("No frames extracted. Check frame_interval/skip_blurry/blur_threshold.")

        return frame_paths

    def generate_labels(self, image_paths: List[Path], progress_cb: Optional[ProgressCb] = None) -> None:
        boxes_total = 0
        empty_labels = 0
        label_files_total = 0
        invalid_discard = 0

        for i, img_path in enumerate(image_paths):
            stem = _safe_stem(img_path)
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                continue

            label_fp = self.raw_labels_dir / f"{stem}.txt"
            try:
                boxes, ds = RedBoxDetector.detect(img, self.cfg)
                # boxes_discarded：基于检测输出的 discard 估计
                invalid_discard += int(ds.get("discard_small", 0)) + int(ds.get("discard_coord", 0))

                if len(boxes) == 0:
                    empty_labels += 1

                with open(label_fp, "w", encoding="utf-8") as f:
                    if len(boxes) > 0 or self.cfg["allow_empty_labels"]:
                        for b in boxes:
                            # YOLO: cls x y w h
                            line = f"{b.class_id} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}"
                            f.write(line + "\n")

                boxes_total += len(boxes)
                label_files_total += 1
            except Exception:
                # 单帧失败不让全流程崩
                try:
                    with open(label_fp, "w", encoding="utf-8") as f:
                        pass
                    label_files_total += 1
                except Exception:
                    pass

            if progress_cb is not None:
                progress_cb("label", i + 1, len(image_paths), "Generating labels")

        self.stats["label_files_total"] = label_files_total
        self.stats["boxes_total"] = boxes_total
        self.stats["empty_label_files"] = empty_labels
        self.stats["boxes_discarded_by_detection"] = invalid_discard

    def validate_labels(self, progress_cb: Optional[ProgressCb] = None) -> None:
        label_files = sorted(self.raw_labels_dir.glob("*.txt"))
        invalid_total = 0
        removed = 0

        for i, lf in enumerate(label_files):
            kept: List[str] = []
            try:
                with open(lf, "r", encoding="utf-8") as f:
                    lines = [x.strip() for x in f.read().splitlines() if x.strip() != ""]
            except Exception:
                lines = []

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    invalid_total += 1
                    removed += 1
                    continue
                if not all(_is_finite_number(p) for p in parts):
                    invalid_total += 1
                    removed += 1
                    continue
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:])

                if cls < 0:
                    invalid_total += 1
                    removed += 1
                    continue

                # 强制范围 + 允许微量越界并 clamp
                coord_eps = float(self.cfg["coord_eps"])
                ok = True
                if x < -coord_eps or x > 1.0 + coord_eps:
                    ok = False
                if y < -coord_eps or y > 1.0 + coord_eps:
                    ok = False
                if w <= 0.0 or h <= 0.0:
                    ok = False
                if w < float(self.cfg["min_norm"]) - coord_eps or w > 1.0 + coord_eps:
                    ok = False
                if h < float(self.cfg["min_norm"]) - coord_eps or h > 1.0 + coord_eps:
                    ok = False

                if not ok:
                    invalid_total += 1
                    removed += 1
                    continue

                x = _clamp(x, 0.0, 1.0)
                y = _clamp(y, 0.0, 1.0)
                w = _clamp(w, float(self.cfg["min_norm"]), 1.0)
                h = _clamp(h, float(self.cfg["min_norm"]), 1.0)

                kept.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            # 覆写（即使 kept 为空也保留空文件）
            try:
                with open(lf, "w", encoding="utf-8") as f:
                    if len(kept) > 0:
                        f.write("\n".join(kept) + "\n")
            except Exception:
                pass

            if progress_cb is not None:
                progress_cb("validate", i + 1, len(label_files), "Validating labels")

        self.stats["invalid_label_lines_total"] = invalid_total
        self.stats["invalid_label_lines_removed"] = removed

        if self.stats["label_files_total"] <= 0 or len(label_files) <= 0:
            raise RuntimeError("No label files found to validate.")

    def split_dataset(self, image_paths: List[Path], progress_cb: Optional[ProgressCb] = None) -> None:
        splits = ["train", "val", "test"]
        train_ratio = float(self.cfg["train_ratio"])
        val_ratio = float(self.cfg["val_ratio"])

        rng = random.Random(int(self.cfg["split_seed"]))
        paths = list(image_paths)
        rng.shuffle(paths)

        n = len(paths)
        train_n = int(n * train_ratio)
        val_n = int(n * val_ratio)
        if train_n + val_n > n:
            val_n = max(0, n - train_n)
        test_n = n - train_n - val_n

        split_map = {
            "train": paths[:train_n],
            "val": paths[train_n:train_n + val_n],
            "test": paths[train_n + val_n:],
        }

        for s in splits:
            self.stats["split"][s] = {"images": 0, "labels": 0}

        # 复制到最终目录
        for s in splits:
            for i, img_path in enumerate(split_map[s]):
                stem = _safe_stem(img_path)
                src_img = img_path
                src_label = self.raw_labels_dir / f"{stem}.txt"
                dst_img = self.split_images_dir / s / src_img.name
                dst_label = self.split_labels_dir / s / f"{stem}.txt"
                try:
                    shutil.copy2(str(src_img), str(dst_img))
                    # label 可能不存在（理论上不会），因此容错
                    if src_label.exists():
                        shutil.copy2(str(src_label), str(dst_label))
                    else:
                        with open(dst_label, "w", encoding="utf-8") as f:
                            pass
                    self.stats["split"][s]["images"] += 1
                    self.stats["split"][s]["labels"] += 1
                except Exception:
                    continue

                if progress_cb is not None:
                    denom = max(1, len(split_map[s]))
                    progress_cb("split", i + 1, denom, f"Splitting to {s}")

    def compute_quality_dashboard(self, progress_cb: Optional[ProgressCb] = None) -> None:
        """
        生成“数据质量看板”统计（无须真值 GT）
        输出：
        - empty_rate（空目标帧占比）
        - boxes_per_image 分布
        - area_norm = w*h 分布
        - aspect_sym = max(w/h, h/w) 分布（归一化长宽比，对称）
        """

        def area_bin_label(area: float) -> str:
            # area: normalized area, in (0,1]
            edges = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
            labels = [
                "<1e-4",
                "1e-4~5e-4",
                "5e-4~1e-3",
                "1e-3~5e-3",
                "5e-3~1e-2",
                "1e-2~5e-2",
                "5e-2~1e-1",
                "1e-1~5e-1",
                ">=5e-1",
            ]
            for i in range(len(edges) - 1):
                lo = edges[i]
                hi = edges[i + 1]
                if i == len(edges) - 2:
                    if area >= lo and area <= hi:
                        return labels[i]
                if area >= lo and area < hi:
                    return labels[i]
            return ">=5e-1"

        def aspect_bin_label(a: float) -> str:
            # aspect_sym >= 1
            edges = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 50.0]
            labels = ["1~1.2", "1.2~1.5", "1.5~2", "2~3", "3~5", "5~8", "8~15", "15~50", ">=50"]
            a = float(a)
            if a >= 50.0:
                return ">=50"
            for i in range(len(edges) - 1):
                lo = edges[i]
                hi = edges[i + 1]
                if a >= lo and a < hi:
                    return labels[i]
            if a >= edges[-1]:
                return labels[-2]
            return labels[0]

        def boxes_hist_bin(cnt: int) -> str:
            if cnt <= 0:
                return "0"
            if cnt == 1:
                return "1"
            if cnt == 2:
                return "2"
            if cnt == 3:
                return "3"
            if cnt == 4:
                return "4"
            if cnt == 5:
                return "5"
            if 6 <= cnt <= 10:
                return "6~10"
            if 11 <= cnt <= 20:
                return "11~20"
            return "21+"

        def parse_label_lines(label_fp: Path) -> List[Tuple[float, float]]:
            """
            返回 [(w_norm, h_norm), ...]
            坐标在 write/validate 阶段已保证归一化合法，这里仍做容错 clamp
            """
            out: List[Tuple[float, float]] = []
            try:
                with open(label_fp, "r", encoding="utf-8") as f:
                    for raw in f.read().splitlines():
                        line = raw.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            continue
                        # cls = int(float(parts[0]))
                        w = float(parts[3])
                        h = float(parts[4])
                        if not all(_is_finite_number(v) for v in [w, h]):
                            continue
                        w = _clamp(w, 0.0, 1.0)
                        h = _clamp(h, 0.0, 1.0)
                        if w <= 0.0 or h <= 0.0:
                            continue
                        out.append((w, h))
            except Exception:
                pass
            return out

        overall_images = 0
        overall_empty = 0
        overall_boxes = 0
        overall_area_hist: Dict[str, int] = {}
        overall_aspect_hist: Dict[str, int] = {}
        overall_boxes_hist: Dict[str, int] = {}

        splits = ["train", "val", "test"]
        for s in splits:
            img_dir = self.split_images_dir / s
            label_dir = self.split_labels_dir / s
            images = sorted(img_dir.glob("*.jpg"))
            total_images = len(images)
            empty_images = 0

            area_hist: Dict[str, int] = {}
            aspect_hist: Dict[str, int] = {}
            boxes_hist: Dict[str, int] = {}
            boxes_total = 0

            for i, img_path in enumerate(images):
                stem = _safe_stem(img_path)
                label_fp = label_dir / f"{stem}.txt"
                wh_list = parse_label_lines(label_fp)
                cnt = len(wh_list)
                boxes_total += cnt
                if cnt == 0:
                    empty_images += 1

                # boxes per image
                bbin = boxes_hist_bin(cnt)
                boxes_hist[bbin] = boxes_hist.get(bbin, 0) + 1

                # area & aspect
                for (w, h) in wh_list:
                    area = w * h
                    abin = area_bin_label(area)
                    area_hist[abin] = area_hist.get(abin, 0) + 1

                    eps = 1e-9
                    aspect = max(w / (h + eps), h / (w + eps))
                    sbin = aspect_bin_label(aspect)
                    aspect_hist[sbin] = aspect_hist.get(sbin, 0) + 1

                if progress_cb is not None:
                    denom = max(1, total_images)
                    progress_cb("quality", i + 1, denom, f"Quality stats ({s})")

            overall_images += total_images
            overall_empty += empty_images
            overall_boxes += boxes_total

            # merge overall hists
            for k, v in area_hist.items():
                overall_area_hist[k] = overall_area_hist.get(k, 0) + v
            for k, v in aspect_hist.items():
                overall_aspect_hist[k] = overall_aspect_hist.get(k, 0) + v
            for k, v in boxes_hist.items():
                overall_boxes_hist[k] = overall_boxes_hist.get(k, 0) + v

            empty_rate = float(empty_images) / float(total_images) if total_images > 0 else 0.0
            self.stats["quality"]["splits"][s] = {
                "images": total_images,
                "empty_images": empty_images,
                "empty_rate": empty_rate,
                "boxes_total": boxes_total,
                "boxes_per_image_hist": boxes_hist,
                "area_hist": area_hist,
                "aspect_hist": aspect_hist,
            }

        overall_empty_rate = float(overall_empty) / float(overall_images) if overall_images > 0 else 0.0
        self.stats["quality"]["overall"] = {
            "images": overall_images,
            "empty_images": overall_empty,
            "empty_rate": overall_empty_rate,
            "boxes_total": overall_boxes,
            "boxes_per_image_hist": overall_boxes_hist,
            "area_hist": overall_area_hist,
            "aspect_hist": overall_aspect_hist,
        }

    def export_sft(self, progress_cb: Optional[ProgressCb] = None) -> None:
        """
        从 YOLO labels 派生检测模型的 SFT jsonl 数据：
        每行包含：image(路径)、target(结构化 boxes)、prompt/answer
        选择 A：保留空目标帧（answer 使用 sft_empty_answer）
        """
        if not bool(self.cfg.get("sft_enable", True)):
            return

        sft_dir = self.output_dir / str(self.cfg.get("sft_output_dir", "sft"))
        sft_dir.mkdir(parents=True, exist_ok=True)

        splits = ["train", "val"]
        if bool(self.cfg.get("sft_include_test", False)):
            splits.append("test")

        prompt = (
            "请根据图中标注的红色框目标，输出所有目标的边界框。"
            "坐标为归一化(x_center,y_center,width,height)。"
            "每个目标输出：class_id x_center y_center width height；空目标输出：none。"
        )

        empty_answer = str(self.cfg.get("sft_empty_answer", "none"))
        sep = str(self.cfg.get("sft_answer_sep", ";"))

        total_samples = 0
        total_empty = 0
        total_boxes = 0

        for si, s in enumerate(splits):
            img_dir = self.split_images_dir / s
            label_dir = self.split_labels_dir / s
            jsonl_path = sft_dir / f"sft_{s}.jsonl"

            samples = 0
            empty_targets = 0
            boxes_total = 0

            # 为了稳定性：按文件名排序
            images = sorted(img_dir.glob("*.jpg"))
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for i, img_path in enumerate(images):
                    stem = _safe_stem(img_path)
                    label_path = label_dir / f"{stem}.txt"

                    boxes: List[Dict[str, Any]] = []
                    if label_path.exists():
                        try:
                            with open(label_path, "r", encoding="utf-8") as lf:
                                for line in lf.read().splitlines():
                                    line = line.strip()
                                    if not line:
                                        continue
                                    parts = line.split()
                                    if len(parts) != 5:
                                        continue
                                    cls = int(float(parts[0]))
                                    x = float(parts[1])
                                    y = float(parts[2])
                                    w = float(parts[3])
                                    h = float(parts[4])
                                    # clamp 到 YOLO 合法范围（以防万一）
                                    x = _clamp(x, 0.0, 1.0)
                                    y = _clamp(y, 0.0, 1.0)
                                    w = _clamp(w, float(self.cfg["min_norm"]), 1.0)
                                    h = _clamp(h, float(self.cfg["min_norm"]), 1.0)
                                    if w <= 0.0 or h <= 0.0:
                                        continue
                                    boxes.append(
                                        {
                                            "class_id": cls,
                                            "x_center": x,
                                            "y_center": y,
                                            "width": w,
                                            "height": h,
                                        }
                                    )
                        except Exception:
                            # 单个文件失败不让全流程崩
                            boxes = []

                    if len(boxes) == 0:
                        empty_targets += 1
                        answer = empty_answer
                    else:
                        box_strs: List[str] = []
                        for b in boxes:
                            box_strs.append(
                                f"{int(b['class_id'])} {b['x_center']:.6f} {b['y_center']:.6f} {b['width']:.6f} {b['height']:.6f}"
                            )
                        answer = sep.join(box_strs)

                    image_rel = f"images/{s}/{img_path.name}"
                    sample = {
                        "image": image_rel,
                        "task": "object_detection",
                        "target": boxes,
                        "prompt": prompt,
                        "answer": answer,
                    }
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

                    samples += 1
                    boxes_total += len(boxes)

                    if progress_cb is not None:
                        denom = max(1, len(images))
                        # 让 UI 显示得更像一个 stage
                        progress_cb("sft", i + 1, denom, f"Exporting SFT ({s})")

            self.stats["sft"]["splits"][s] = {
                "samples": samples,
                "empty_targets": empty_targets,
                "boxes_total": boxes_total,
                "jsonl": str(Path(self.cfg.get("sft_output_dir", "sft")) / f"sft_{s}.jsonl"),
            }
            total_samples += samples
            total_empty += empty_targets
            total_boxes += boxes_total

        self.stats["sft"]["samples_total"] = total_samples
        self.stats["sft"]["empty_targets"] = total_empty
        self.stats["sft"]["boxes_total"] = total_boxes

    def write_data_yaml(self) -> None:
        # YOLOv5/8 常见格式
        names = self.cfg.get("names", ["car"])
        data_yaml = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": names,
        }
        out = self.output_dir / "data.yaml"
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    def write_stats(self) -> None:
        # 统计文件
        stats_json = self.stats_dir / "stats.json"
        stats_txt = self.stats_dir / "stats.txt"

        try:
            with open(stats_json, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # 可读文本
        try:
            s = self.stats
            lines = [
                f"Dataset output: {self.output_dir}",
                f"Video: {s.get('video_path')}",
                f"Frames total: {s.get('frames_total')}",
                f"Frames sampled: {s.get('frames_sampled')}",
                f"Frames saved: {s.get('frames_saved')}",
                f"Skipped blurry: {s.get('frames_skipped_blurry')}",
                f"Label files total: {s.get('label_files_total')}",
                f"Boxes total: {s.get('boxes_total')}",
                f"Empty label files: {s.get('empty_label_files')}",
                f"Invalid label lines removed: {s.get('invalid_label_lines_removed')}",
                f"Split: {json.dumps(s.get('split', {}), ensure_ascii=False)}",
                f"SFT enabled: {s.get('sft', {}).get('enabled', False)}",
                f"SFT samples total: {s.get('sft', {}).get('samples_total', 0)}",
                f"SFT empty targets: {s.get('sft', {}).get('empty_targets', 0)}",
                f"SFT boxes total: {s.get('sft', {}).get('boxes_total', 0)}",
            ]
            with open(stats_txt, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass

    def run(self, video_path: Optional[str] = None, progress_cb: Optional[ProgressCb] = None) -> Tuple[Path, Optional[Path]]:
        if video_path is None:
            video_path = self.cfg["video_path"]

        try:
            frame_paths = self.extract_frames(video_path, progress_cb=progress_cb)
            self.generate_labels(frame_paths, progress_cb=progress_cb)
            self.validate_labels(progress_cb=progress_cb)
            self.split_dataset(frame_paths, progress_cb=progress_cb)
            self.export_sft(progress_cb=progress_cb)
            self.compute_quality_dashboard(progress_cb=progress_cb)
            self.write_data_yaml()
            self.write_stats()
            zip_path: Optional[Path] = None
            if bool(self.cfg.get("zip_output", False)):
                zip_path = self.output_dir.with_suffix(".zip")
                zip_dir(self.output_dir, zip_path)
            return self.output_dir, zip_path
        except Exception as e:
            # 写 run.log 在外部由 main 处理；这里抛出让调用端捕获
            raise e


def generate_dataset(
    cfg: Dict[str, Any],
    video_path: Optional[str] = None,
    output_base_dir: Optional[str] = None,
    progress_cb: Optional[ProgressCb] = None,
) -> Dict[str, Any]:
    # 网页端可能会把 cfg 直接作为 dict 传进来，因此也需要执行同样的 schema 校验。
    cfg2 = ConfigLoader._apply_defaults(dict(cfg))
    ConfigLoader._validate_schema(cfg2)
    cfg2 = ConfigLoader._normalize_ratios(cfg2)

    factory = DatasetFactory(cfg2, output_base_dir=output_base_dir)
    out_dir, zip_path = factory.run(video_path=video_path, progress_cb=progress_cb)
    return {"output_dir": out_dir, "zip_path": zip_path, "stats": factory.stats}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Drone red-box to YOLO dataset generator")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output_dir", default=None, help="Output base directory (parent folder)")
    args = parser.parse_args(argv)

    config_path = args.config
    try:
        cfg = ConfigLoader.load_and_validate(config_path)

        # 根据命令行覆盖输出根目录
        output_base_dir = args.output_dir if args.output_dir else None

        # 写日志：放在输出目录无法提前确定时，把日志放到当前目录
        log_path = Path(".") / "run.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().isoformat()}] Start generation. config={config_path}\n")

        result = generate_dataset(cfg, video_path=cfg["video_path"], output_base_dir=output_base_dir, progress_cb=None)
        out_dir = result["output_dir"]
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] Success. output_dir={out_dir}\n")

        print(f"[OK] Dataset generated: {out_dir}")
        if result.get("zip_path") is not None:
            print(f"[OK] Zip: {result['zip_path']}")
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        log_path = Path(".") / "run.log"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now().isoformat()}] ERROR: {e}\n{tb}\n")
        except Exception:
            pass
        print(f"[ERROR] Failed: {e}")
        # 打到 stderr 以便上层脚本捕获
        print(tb, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())