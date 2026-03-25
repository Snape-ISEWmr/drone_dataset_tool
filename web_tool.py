import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path

st.set_page_config(page_title="无人机数据集工具", page_icon="🚁")
st.title("🚁 无人机汽车检测数据集工具")

uploaded = st.file_uploader("上传视频",["mp4"])
if uploaded:
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    t.write(uploaded.read())
    st.video(uploaded)

    if st.button("生成数据集"):
        with st.spinner("生成中..."):
            out = Path(tempfile.mkdtemp())
            cap = cv2.VideoCapture(t.name)
            cnt=0
            while cap.isOpened():
                ret,f = cap.read()
                if not ret: break
                if cnt%10==0:
                    cv2.imwrite(str(out/f"frame_{cnt:06d}.jpg"),f)
                cnt+=1
            cap.release()
            st.success("✅ 数据集生成完成！")
            st.balloons()