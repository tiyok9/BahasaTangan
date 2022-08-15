import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input as Resnet50_preprocess


st.header("Indentifikasi Bahasa Isyarat Pada Tangan Manusia")
model = tf.keras.models.load_model("/content/gdrive/MyDrive/ColabNotebooks/model_resnet.hdf5")
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

map_dict = {0: 'A',
            1: 'B',
            2: 'C',
            3: 'D',
            4: 'E',
            5: 'F',
            6: 'G',
            7: 'H',
            8: 'I',
            9: 'J'}


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = Resnet50_preprocess(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Klik Untuk Prediksi")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Tangan Ini Menunjukan Huruf {}".format(map_dict [prediction]))