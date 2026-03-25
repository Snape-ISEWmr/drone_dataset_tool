"""
Streamlit 网页端：上传-调参-预览-生成-下载 zip

启动（本地）：
  streamlit run web_tool.py

启动（公网部署/容器常用）：
  streamlit run web_tool.py --server.address 0.0.0.0 --server.port 8501
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from dataset_factory import ConfigLoader, detect_red_boxes_bgr, generate_dataset, zip_dir


def _round_floats(obj: Any, ndigits: int = 2) -> Any:
    """Recursively round float values for UI display."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(x, ndigits) for x in obj]
    return obj


def _resolve_default_config_path() -> str:
    return str(Path(__file__).resolve().with_name("config.yaml"))


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def make_previews(
    video_path: str,
    cfg: Dict[str, Any],
    frame_interval: int,
    max_previews: int = 6,
) -> List[Tuple[Image.Image, str]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    previews: List[Tuple[Image.Image, str]] = []
    frame_idx = 0
    sampled_seen = 0

    # Laplacian blur check（复用 cfg 规则；检测算法复用 dataset_factory 的红框检测）
    def is_blurry(frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        var = float(lap.var())
        return var < float(cfg["blur_threshold"])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_interval >= 1 and (frame_idx % frame_interval == 0):
                sampled_seen += 1
                if bool(cfg["skip_blurry"]):
                    try:
                        if is_blurry(frame):
                            frame_idx += 1
                            continue
                    except Exception:
                        pass

                boxes = detect_red_boxes_bgr(frame, cfg)
                overlay = frame.copy()
                h, w = overlay.shape[:2]

                for b in boxes:
                    x1 = int((b.x_center - b.width / 2.0) * w)
                    y1 = int((b.y_center - b.height / 2.0) * h)
                    x2 = int((b.x_center + b.width / 2.0) * w)
                    y2 = int((b.y_center + b.height / 2.0) * h)
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(0, min(w - 1, x2))
                    y2 = max(0, min(h - 1, y2))
                    # 可视化使用绿色描边，方便用户在原始红框背景上区分叠加结果
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

                tag = f"sampled={sampled_seen}, frame={frame_idx}, boxes={len(boxes)}"
                previews.append((_bgr_to_pil(overlay), tag))
                if len(previews) >= max_previews:
                    break

            frame_idx += 1
    finally:
        cap.release()

    return previews


st.set_page_config(page_title="无人机数据集工具", page_icon="🚁")
st.title("无人机汽车检测数据集工具（红框->YOLO）")

default_cfg_path = _resolve_default_config_path()
try:
    base_cfg = ConfigLoader.load_and_validate(default_cfg_path)
except Exception as e:
    st.error(f"读取默认 config.yaml 失败：{e}")
    st.stop()

uploaded = st.file_uploader("上传视频（MP4）", type=["mp4"])

if uploaded is not None:
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    t.write(uploaded.read())
    t.flush()

    st.video(uploaded)
    st.divider()

    st.sidebar.title("使用说明 & 参数")
    st.sidebar.markdown("1. 上传 `mp4` 视频\n2. 先点 `运行少量预览` 确认绿色框贴合\n3. 再点 `生成数据集` 导出 YOLO 数据集并下载 zip")
    st.sidebar.divider()

    st.sidebar.subheader("抽帧与模糊过滤")
    frame_interval = st.sidebar.number_input(
        "frame_interval（>=1，越大越快）",
        min_value=1,
        step=1,
        value=int(base_cfg.get("frame_interval", 10)),
        help="抽帧间隔：每隔多少帧抽取 1 帧。越大越快，但数据多样性会下降。",
    )
    skip_blurry = st.sidebar.checkbox(
        "跳过模糊帧（Laplacian 方差过滤）",
        value=bool(base_cfg.get("skip_blurry", True)),
        help="开启后：如果画面过于模糊（纹理少），该帧会被跳过，减少低质量样本。",
    )
    blur_threshold = st.sidebar.number_input(
        "blur_threshold（Laplacian 方差阈值）",
        min_value=0.0,
        step=1.0,
        value=float(base_cfg.get("blur_threshold", 100.0)),
        help="模糊判定阈值。值越大越严格（更容易跳过帧），值越小越宽松（更少跳过）。",
    )
    min_box_size = st.sidebar.number_input(
        "min_box_size（最小框像素）",
        min_value=1,
        step=1,
        value=int(base_cfg.get("min_box_size", 20)),
        help="小于该像素阈值的候选红框会被过滤掉：值越大越“严格”（可能漏检），值越小越“宽松”（可能噪声多）。",
    )

    st.sidebar.subheader("数据划分比例")
    val_ratio = st.sidebar.slider(
        "val_ratio",
        min_value=0.0,
        max_value=1.0,
        value=float(base_cfg.get("val_ratio", 0.2)),
        step=0.01,
        help="验证集比例：用于评估训练效果。越大验证越稳定，但训练数据会减少。",
    )
    test_ratio = st.sidebar.slider(
        "test_ratio",
        min_value=0.0,
        max_value=1.0,
        value=float(base_cfg.get("test_ratio", 0.1)),
        step=0.01,
        help="测试集比例：用于最终评估（或留作对比）。太大可能减少训练样本。",
    )
    # train_ratio 由其余决定，避免出现总和过大导致 train/val/test 为负
    if val_ratio + test_ratio >= 1.0:
        st.sidebar.warning("`val_ratio + test_ratio >= 1.0`：将自动按程序归一化/修正，建议减小比例。")
    train_ratio = 1.0 - val_ratio - test_ratio

    with st.sidebar.expander("高级：HSV 阈值（红色检测）", expanded=False):
        lower_red1 = st.sidebar.text_input(
            "lower_red1 [H,S,V]",
            value=str(base_cfg.get("lower_red1", [0, 150, 100])),
            help="HSV 红色范围的下界（第一段）。通常不需要改，除非预览漏检/误检很多。",
        )
        upper_red1 = st.sidebar.text_input(
            "upper_red1 [H,S,V]",
            value=str(base_cfg.get("upper_red1", [10, 255, 255])),
            help="HSV 红色范围的上界（第一段）。",
        )
        lower_red2 = st.sidebar.text_input(
            "lower_red2 [H,S,V]",
            value=str(base_cfg.get("lower_red2", [170, 150, 100])),
            help="HSV 红色范围的下界（第二段，OpenCV 的红色通常分成两段）。",
        )
        upper_red2 = st.sidebar.text_input(
            "upper_red2 [H,S,V]",
            value=str(base_cfg.get("upper_red2", [180, 255, 255])),
            help="HSV 红色范围的上界（第二段）。",
        )

    st.sidebar.subheader("输出设置")
    zip_output = st.sidebar.checkbox("生成后打包 zip（用于下载）", value=True, help="生成完成后将输出目录打包为 zip，便于直接下载。")
    sft_enable = st.sidebar.checkbox(
        "同时生成 SFT(jsonl)（检测训练数据）",
        value=bool(base_cfg.get("sft_enable", True)),
        help="开启后会在输出目录额外生成 SFT jsonl（从 YOLO labels 派生）。",
    )
    sft_include_test = st.sidebar.checkbox(
        "包含 test split 到 SFT",
        value=bool(base_cfg.get("sft_include_test", False)),
        help="如果只训练 train/val，可以保持关闭以减少数据量。",
    )
    max_frames = st.sidebar.number_input(
        "max_frames（-1 表示不限制）",
        value=int(base_cfg.get("max_frames", -1)),
        step=1,
        help="限制最大抽帧数量（用于快速验证）。-1 表示不限制。",
    )

    st.sidebar.caption("提示：若预览中误检很多/漏检很多，请先调 HSV 阈值或调大 `min_box_size` 再预览。")

    preview_clicked = st.sidebar.button(
        "运行少量预览",
        help="只抽取少量帧进行快速检测预览。会使用当前的 frame_interval / skip_blurry / blur_threshold / HSV / min_box_size 参数。",
    )
    generate_clicked = st.sidebar.button(
        "生成数据集",
        help="扫描整段视频，生成 YOLO 数据集并（可选）导出 SFT(jsonl)与 zip 下载包；同样会使用当前的 min_box_size / HSV / 模糊过滤参数。",
    )
    st.sidebar.divider()

    # 预览（主界面）
    st.subheader("预览检测效果（少量抽帧）")
    st.caption("提示：预览只会抽取少量帧来验证红色框检测效果；正式生成会扫描整段视频并导出 YOLO 数据集。")
    if preview_clicked:
        try:
            cfg = dict(base_cfg)
            cfg.update(
                {
                    "skip_blurry": skip_blurry,
                    "blur_threshold": float(blur_threshold),
                    "frame_interval": int(frame_interval),
                    "min_box_size": int(min_box_size),
                    "train_ratio": float(max(0.0, train_ratio)),
                    "val_ratio": float(val_ratio),
                    "test_ratio": float(test_ratio),
                }
            )

            def parse_hsv(s: str) -> List[int]:
                s2 = s.strip()
                # 允许用户输入如 [0, 150, 100]
                v = ast.literal_eval(s2)
                if not isinstance(v, (list, tuple)) or len(v) != 3:
                    raise ValueError("HSV must be [H,S,V]")
                return [int(v[0]), int(v[1]), int(v[2])]

            cfg["lower_red1"] = parse_hsv(lower_red1)
            cfg["upper_red1"] = parse_hsv(upper_red1)
            cfg["lower_red2"] = parse_hsv(lower_red2)
            cfg["upper_red2"] = parse_hsv(upper_red2)

            previews = make_previews(
                video_path=t.name,
                cfg=cfg,
                frame_interval=int(frame_interval),
                max_previews=6,
            )
            if not previews:
                st.warning("未获取到可预览样本（可能跳过了所有模糊帧）。")
            else:
                st.success(f"预览完成：展示 {len(previews)} 张样本。")
                cols = st.columns(len(previews))
                for idx, (pil_img, tag) in enumerate(previews):
                    with cols[idx]:
                        st.image(pil_img, caption=tag)
        except Exception as e:
            st.error(f"预览失败：{e}")

    # 生成完整数据集
    st.subheader("生成完整数据集")
    st.caption("说明：生成会进行抽帧、模糊过滤、红色框检测、YOLO 标签生成、标签质量校验与 train/val/test 划分。若开启 zip，会自动打包下载。")
    if generate_clicked:
        progress = st.progress(0)
        status = st.empty()
        out_zip_path: Optional[Path] = None

        # session_state 中避免重复生成：如果你想优化成“生成一次多次下载”，可以继续扩展
        work_dir = Path(tempfile.mkdtemp(prefix="drone_dataset_"))
        try:
            cfg = dict(base_cfg)
            cfg.update(
                {
                    "skip_blurry": skip_blurry,
                    "blur_threshold": float(blur_threshold),
                    "min_box_size": int(min_box_size),
                    "frame_interval": int(frame_interval),
                    "train_ratio": float(max(0.0, train_ratio)),
                    "val_ratio": float(val_ratio),
                    "test_ratio": float(test_ratio),
                    "zip_output": bool(zip_output),
                    "max_frames": int(max_frames),
                    "video_path": t.name,
                    "sft_enable": bool(sft_enable),
                    "sft_include_test": bool(sft_include_test),
                }
            )

            def parse_hsv(s: str) -> List[int]:
                v = ast.literal_eval(s.strip())
                return [int(v[0]), int(v[1]), int(v[2])]

            cfg["lower_red1"] = parse_hsv(lower_red1)
            cfg["upper_red1"] = parse_hsv(upper_red1)
            cfg["lower_red2"] = parse_hsv(lower_red2)
            cfg["upper_red2"] = parse_hsv(upper_red2)

            stage_ranges = {
                "extract": (0.0, 0.45),
                "label": (0.45, 0.65),
                "validate": (0.65, 0.82),
                "split": (0.82, 0.95),
                "zip": (0.95, 1.0),
            }

            def progress_cb(stage: str, current: int, total: int, message: str) -> None:
                if stage in stage_ranges:
                    lo, hi = stage_ranges[stage]
                    frac = 0.0
                    if total > 0:
                        frac = max(0.0, min(1.0, float(current) / float(total)))
                    value = lo + frac * (hi - lo)
                else:
                    value = 0.0
                progress.progress(min(1.0, max(0.0, float(value))))
                status.write(f"{stage}: {message} ({current}/{total})")

            # 生成：核心算法复用 dataset_factory
            result = generate_dataset(
                cfg=cfg,
                video_path=t.name,
                output_base_dir=str(work_dir),
                progress_cb=progress_cb,
            )

            out_dir = result["output_dir"]
            stats = result.get("stats", {})
            stats_r = _round_floats(stats, 2) if isinstance(stats, dict) else stats

            # 如果配置未开启 zip_output，但用户勾选了下载 zip，需要补一次打包
            zip_path = result.get("zip_path")
            if bool(zip_output) and zip_path is None:
                zip_path = out_dir.rstrip("\\/") + ".zip"
                out_zip_path = Path(zip_path)
                zip_dir(Path(out_dir), out_zip_path)
            if out_zip_path is None and zip_path is not None:
                out_zip_path = Path(zip_path)

            if zip_output and out_zip_path is not None and out_zip_path.exists():
                status.write("打包完成，准备下载...")
                progress.progress(1.0)
                st.success("数据集生成完成！")
                st.caption("输出目录在服务器侧为临时目录，建议以下载的 zip 为准。")
                st.subheader("统计摘要")
                st.json(
                    {
                        "frames_total": stats_r.get("frames_total"),
                        "frames_saved": stats_r.get("frames_saved"),
                        "frames_skipped_blurry": stats_r.get("frames_skipped_blurry"),
                        "label_files_total": stats_r.get("label_files_total"),
                        "boxes_total": stats_r.get("boxes_total"),
                        "boxes_discarded_by_detection": stats_r.get("boxes_discarded_by_detection"),
                        "empty_label_files": stats_r.get("empty_label_files"),
                        "invalid_label_lines_removed": stats_r.get("invalid_label_lines_removed"),
                        "split": stats_r.get("split"),
                        "sft": stats_r.get("sft"),
                    }
                )

                quality = stats_r.get("quality") or {}
                if quality.get("splits"):
                    st.subheader("数据质量看板（自动量化）")
                    q_splits = quality.get("splits", {})
                    empty_rate_chart = {}
                    for s in ["train", "val", "test"]:
                        if s in q_splits:
                            empty_rate_chart[s] = [q_splits[s].get("empty_rate", 0.0)]
                    if empty_rate_chart:
                        # 为空/越界时仍可绘制，输入为 {split: rate}（纯标量）
                        st.bar_chart({k: float(v[0]) for k, v in empty_rate_chart.items()}, height=120)

                    # 可解释的质量提示（无真值 GT 情况下的“代理诊断”）
                    invalid_removed = stats_r.get("invalid_label_lines_removed", 0) or 0
                    invalid_total = stats_r.get("invalid_label_lines_total", 0) or 0
                    if invalid_total > 0 and float(invalid_removed) / float(invalid_total) > 0.02:
                        st.warning(f"标签质量提示：发现较多非法标签行（removed={invalid_removed}, total={invalid_total}）。建议检查 HSV 阈值/过滤阈值。")

                    # 空目标比例提示
                    for s in ["train", "val", "test"]:
                        if s in q_splits:
                            er = float(q_splits[s].get("empty_rate", 0.0) or 0.0)
                            if er >= 0.7:
                                st.warning(f"{s} 空目标比例偏高：empty_rate={er:.2f}。可能是阈值过严或红框检测漏检。")

                    tabs = st.tabs(["train", "val", "test"])
                    order_boxes = ["0", "1", "2", "3", "4", "5", "6~10", "11~20", "21+"]
                    order_area = [
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
                    order_aspect = ["1~1.2", "1.2~1.5", "1.5~2", "2~3", "3~5", "5~8", "8~15", "15~50", ">=50"]

                    for tab, split_name in zip(tabs, ["train", "val", "test"]):
                        with tab:
                            if split_name not in q_splits:
                                st.info(f"{split_name} split 不存在（可能样本少或配置未生成）")
                                continue
                            qs = q_splits[split_name]
                            st.write(
                                f"{split_name}：images={qs.get('images')} empty_rate={float(qs.get('empty_rate', 0.0)):.2f} boxes_total={qs.get('boxes_total')}"
                            )

                            boxes_hist = qs.get("boxes_per_image_hist", {})
                            if boxes_hist:
                                data_boxes = {k: int(boxes_hist.get(k, 0)) for k in order_boxes if int(boxes_hist.get(k, 0)) > 0}
                                if data_boxes:
                                    st.bar_chart(data_boxes, height=160)

                            area_hist = qs.get("area_hist", {})
                            if area_hist:
                                data_area = {k: int(area_hist.get(k, 0)) for k in order_area if int(area_hist.get(k, 0)) > 0}
                                if data_area:
                                    st.bar_chart(data_area, height=180)

                            aspect_hist = qs.get("aspect_hist", {})
                            if aspect_hist:
                                data_aspect = {k: int(aspect_hist.get(k, 0)) for k in order_aspect if int(aspect_hist.get(k, 0)) > 0}
                                if data_aspect:
                                    st.bar_chart(data_aspect, height=180)

                with open(out_zip_path, "rb") as f:
                    st.download_button(
                        label="下载数据集 zip",
                        data=f.read(),
                        file_name=out_zip_path.name,
                        mime="application/zip",
                        help="下载包含 images/labels/data.yaml/stats/sft(optional) 的数据集 zip 包。",
                    )
            else:
                st.success("数据集生成完成（未生成 zip，请在配置中开启 zip_output）")
                st.subheader("统计摘要")
                st.json(
                    {
                        "frames_total": stats_r.get("frames_total"),
                        "frames_saved": stats_r.get("frames_saved"),
                        "frames_skipped_blurry": stats_r.get("frames_skipped_blurry"),
                        "label_files_total": stats_r.get("label_files_total"),
                        "boxes_total": stats_r.get("boxes_total"),
                        "boxes_discarded_by_detection": stats_r.get("boxes_discarded_by_detection"),
                        "empty_label_files": stats_r.get("empty_label_files"),
                        "invalid_label_lines_removed": stats_r.get("invalid_label_lines_removed"),
                        "split": stats_r.get("split"),
                        "sft": stats_r.get("sft"),
                    }
                )
                quality = stats_r.get("quality") or {}
                if quality.get("splits"):
                    st.subheader("数据质量看板（自动量化）")
                    q_splits = quality.get("splits", {})
                    empty_rate_chart = {}
                    for s in ["train", "val", "test"]:
                        if s in q_splits:
                            empty_rate_chart[s] = [q_splits[s].get("empty_rate", 0.0)]
                    if empty_rate_chart:
                        st.bar_chart({k: float(v[0]) for k, v in empty_rate_chart.items()}, height=120)
                    invalid_removed = stats_r.get("invalid_label_lines_removed", 0) or 0
                    invalid_total = stats_r.get("invalid_label_lines_total", 0) or 0
                    if invalid_total > 0 and float(invalid_removed) / float(invalid_total) > 0.02:
                        st.warning(f"标签质量提示：发现较多非法标签行（removed={invalid_removed}, total={invalid_total}）。建议检查 HSV 阈值/过滤阈值。")
                    for s in ["train", "val", "test"]:
                        if s in q_splits:
                            er = float(q_splits[s].get("empty_rate", 0.0) or 0.0)
                            if er >= 0.7:
                                st.warning(f"{s} 空目标比例偏高：empty_rate={er:.2f}。可能是阈值过严或红框检测漏检。")
                    tabs = st.tabs(["train", "val", "test"])
                    order_boxes = ["0", "1", "2", "3", "4", "5", "6~10", "11~20", "21+"]
                    order_area = [
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
                    order_aspect = ["1~1.2", "1.2~1.5", "1.5~2", "2~3", "3~5", "5~8", "8~15", "15~50", ">=50"]

                    for tab, split_name in zip(tabs, ["train", "val", "test"]):
                        with tab:
                            if split_name not in q_splits:
                                st.info(f"{split_name} split 不存在（可能样本少或配置未生成）")
                                continue
                            qs = q_splits[split_name]
                            st.write(
                                f"{split_name}：images={qs.get('images')} empty_rate={float(qs.get('empty_rate', 0.0)):.2f} boxes_total={qs.get('boxes_total')}"
                            )
                            boxes_hist = qs.get("boxes_per_image_hist", {})
                            if boxes_hist:
                                data_boxes = {k: int(boxes_hist.get(k, 0)) for k in order_boxes if int(boxes_hist.get(k, 0)) > 0}
                                if data_boxes:
                                    st.bar_chart(data_boxes, height=160)
                            area_hist = qs.get("area_hist", {})
                            if area_hist:
                                data_area = {k: int(area_hist.get(k, 0)) for k in order_area if int(area_hist.get(k, 0)) > 0}
                                if data_area:
                                    st.bar_chart(data_area, height=180)
                            aspect_hist = qs.get("aspect_hist", {})
                            if aspect_hist:
                                data_aspect = {k: int(aspect_hist.get(k, 0)) for k in order_aspect if int(aspect_hist.get(k, 0)) > 0}
                                if data_aspect:
                                    st.bar_chart(data_aspect, height=180)

                if out_dir is not None:
                    st.code(f"服务器侧输出目录：{out_dir}")

        except Exception as e:
            st.error(f"生成失败：{e}")
        finally:
            try:
                progress.progress(1.0)
            except Exception:
                pass