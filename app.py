import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import shutil
import uuid
import random
from ultralytics import YOLO
from PIL import Image


model = YOLO("new_best.pt")  


annotated_dir = "detected"
zoomed_dir = "zoomed_detected"


def clear_output_folders():
    for folder in [annotated_dir, zoomed_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)  # delete old folder
        os.makedirs(folder, exist_ok=True)

st.title(" PPE Detection App")
st.write("Upload an image to detect PPE items, persons, and zoom on detections.")


conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   
    clear_output_folders()

    
    suffix = os.path.splitext(uploaded_file.name)[-1]
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tfile.write(uploaded_file.read())
    image_path = tfile.name

    
    results = model.predict(source=image_path, conf=conf_thresh)

  
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    names = model.names

    
    color_map = {i: [random.randint(0, 255) for _ in range(3)] for i in set(classes)}

    
    annotated_img = img_rgb.copy()
    zoomed_images = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(classes[i])
        conf = float(confidences[i])
        cls_name = names[cls_id]

        
        color = color_map[cls_id]
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(annotated_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

       
        cropped = img_rgb[y1:y2, x1:x2]
        if cropped.size > 0:
            zoomed = cv2.resize(cropped, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            zoomed_images.append((cls_name, zoomed))

            
            crop_filename = f"{cls_name}_{i+1}.jpg"
            cv2.imwrite(os.path.join(zoomed_dir, crop_filename), cv2.cvtColor(zoomed, cv2.COLOR_RGB2BGR))

    
    annotated_filename = f"{uuid.uuid4().hex}_detected.jpg"
    annotated_path = os.path.join(annotated_dir, annotated_filename)
    cv2.imwrite(annotated_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

    
    st.subheader(" Detected Image with Boxes")
    st.image(annotated_img, channels="RGB", use_container_width=True)

    
    with open(annotated_path, "rb") as f:
        st.download_button("⬇ Download Annotated Image", f, file_name=annotated_filename)

    
    if zoomed_images:
        st.subheader(" Zoomed Detections")
        cols = st.columns(3)
        for idx, (cls_name, zoomed) in enumerate(zoomed_images):
            with cols[idx % 3]:
                st.image(zoomed, caption=cls_name, channels="RGB")
    else:
        st.info("No detections found.")

    st.success(f" Annotated image saved in: {annotated_dir}/")
    st.success(f" Zoomed crops saved in: {zoomed_dir}/")
