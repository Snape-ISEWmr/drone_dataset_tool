# 无人机汽车检测数据集自动化工具

将“无人机航拍视频（红色标注框）”自动抽帧并识别红框，生成 **YOLO 目标检测数据集**（图片 + 标签 + `data.yaml` + 统计报告），支持：
- 本地 CLI 运行
- Streamlit 网页端运行（可公网部署）
- 依据约定可打包为 Windows EXE

---

## 1. 项目结构
- `dataset_factory.py`：核心生成逻辑（抽帧、红框检测、YOLO 标签生成、标签校验、划分、导出 `data.yaml` 和统计）
- `web_tool.py`：Streamlit 网页端（上传视频、调参、预览、生成、下载 zip）
- `config.yaml`：默认配置参数（可被网页端参数覆盖）
- `requirements.txt`：依赖锁定
- `runtime.txt`：Python 版本锁定（`3.10.8`）

---

## 2. 本地运行（CLI）

1) 安装依赖：
```bash
pip install -r requirements.txt
```

2) 直接运行（读取 `config.yaml`）：
```bash
python dataset_factory.py --config config.yaml
```

3) 指定输出父目录：
```bash
python dataset_factory.py --config config.yaml --output_dir ./out_workdir
```

---

## 3. 网页运行（Streamlit）

```bash
streamlit run web_tool.py --server.address 0.0.0.0 --server.port 8501
```

在页面里：
- 上传 `mp4` 视频
- 调整抽帧/模糊过滤/HSV 阈值等参数
- 可先点“运行少量预览”看绿色框检测效果
- 再点“生成数据集”，完成后下载 zip

---

## 4. 输出目录与产物

生成目录形如：
`drone_dataset_YYYYMMDD_HHMMSS/`

其中包含：
- `images/train|val|test/*.jpg`
- `labels/train|val|test/*.txt`（YOLO 格式：`cls x_center y_center w h`，全部为归一化数值）
- `data.yaml`
- `stats/stats.json`、`stats/stats.txt`
- `sft/`（SFT 检测训练数据，按 split 输出 `sft_train.jsonl` / `sft_val.jsonl` / 可选 `sft_test.jsonl`）
- （可选）整目录 zip 包用于下载

---

## 5. 参数说明（config.yaml）
- `frame_interval`：抽帧间隔（>= 1）
- `skip_blurry` / `blur_threshold`：Laplacian 方差判定模糊帧
- `lower_red1/upper_red1/lower_red2/upper_red2`：红色 HSV 阈值（OpenCV HSV：H 0..179）
- `min_box_size`：最小框尺寸（像素）
- `train_ratio/val_ratio/test_ratio`：划分比例（总和会在程序内自动归一化）
- `zip_output`：是否自动打包 zip（用于网页端下载）
- `sft_enable`：是否从 YOLO labels 派生 SFT(jsonl)
- `sft_include_test`：是否把 test split 也导出到 SFT
- `sft_empty_answer`：空目标帧在 SFT 中的 answer（你选择了保留空目标，因此这些帧会出现在 jsonl 中）

---

## 6. EXE 打包（Windows）

`dataset_factory.py` 顶部包含了 pyinstaller 的命令模板示例，建议按需修改输出文件名：
```powershell
pyinstaller -F --name drone_dataset_factory --add-data "config.yaml;." dataset_factory.py
```

