import streamlit as st
from PIL import Image
import numpy as np
import torch
from UNetutils import Denoise

st.title("Teeth X-Ray Denoising App")
st.write("Upload a noisy x-ray image, and let's make it clear!")
uploaded_file = st.file_uploader("Choose a noisy image...", type=["jpg", "jpeg", "png"])
weights_pth = r"D:\All NITK\Clubs\IEEE\Envision '24\ImageDenoising\models\unet_weights_2.pth"

def denoise_image(input_image_pth):
    output_image = Denoise(input_image_pth, weights_pth)
    return output_image

if uploaded_file is not None:
    # read image
    input_image = Image.open(uploaded_file).convert('L')
    
    # display noisy image
    st.subheader("Noisy Image")
    st.image(input_image, caption='Noisy Image', use_column_width=True)

    # denoise image
    output_image = denoise_image(uploaded_file)
    output_image_pil = Image.fromarray(output_image)

    # display denoised image
    st.subheader("Denoised Image")
    st.image(output_image_pil, caption='Denoised Image', use_column_width=True)