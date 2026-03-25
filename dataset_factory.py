import cv2
import numpy as np
import os
import json
import yaml
import shutil
import sys
from datetime import datetime
from pathlib import Path

class ConfigLoader:
    @staticmethod
    def load(config_path="config.yaml"):
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在")
            input("回车退出")
            sys.exit(1)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg

class DatasetFactory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"drone_dataset_{self.timestamp}")
        self._init_dirs()

    def _init_dirs(self):
        dirs = ["images","labels","train/images","train/labels","val/images","val/labels","test/images","test/labels","stats"]
        for d in dirs:
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)

    def extract_frames(self):
        cap = cv2.VideoCapture(self.cfg["video_path"])
        frame_count = saved_count = 0
        frame_paths = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_count % self.cfg["frame_interval"] == 0:
                fp = self.output_dir / "images" / f"frame_{saved_count:06d}.jpg"
                cv2.imwrite(str(fp), frame)
                frame_paths.append(fp)
                saved_count +=1
            frame_count +=1
        cap.release()
        print(f"✅ 抽帧完成：{saved_count} 张")
        return frame_paths

    def parse_red_boxes(self, img_path):
        img = cv2.imread(str(img_path))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(cv2.inRange(hsv, np.array(self.cfg["lower_red1"]), np.array(self.cfg["upper_red1"])),
                              cv2.inRange(hsv, np.array(self.cfg["lower_red2"]), np.array(self.cfg["upper_red2"])))
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h,w = img.shape[:2]
        boxes = []
        for c in cnts:
            x,y,ww,hh = cv2.boundingRect(c)
            if ww<20 or hh<20: continue
            xc = (x+ww/2)/w
            yc = (y+hh/2)/h
            boxes.append([0,xc,yc,ww/w,hh/h])
        return boxes

    def generate_labels(self, fps):
        for fp in fps:
            boxes = self.parse_red_boxes(fp)
            with open(self.output_dir/"labels"/f"{fp.stem}.txt","w") as f:
                f.write("\n".join([" ".join(map(str,b)) for b in boxes]))
        print("✅ 标签生成完成")

    def split_dataset(self, fps):
        np.random.shuffle(fps)
        n = len(fps)
        train = fps[:int(n*0.7)]
        val = fps[int(n*0.7):int(n*0.9)]
        test = fps[int(n*0.9):]
        for d,ps in {"train":train,"val":val,"test":test}.items():
            for p in ps:
                shutil.copy(p, self.output_dir/d/"images")
                shutil.copy(self.output_dir/"labels"/f"{p.stem}.txt", self.output_dir/d/"labels")
        print("✅ 数据集划分完成")

    def run(self):
        print("🚀 开始生成数据集")
        fps = self.extract_frames()
        self.generate_labels(fps)
        self.split_dataset(fps)
        print(f"🎉 完成！输出目录：{self.output_dir}")
        input("回车退出")

if __name__ == "__main__":
    cfg = ConfigLoader.load()
    DatasetFactory(cfg).run()