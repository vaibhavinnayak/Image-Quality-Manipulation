import streamlit as st
from PIL import Image
import numpy as np
import torch
from ModelUtils import Enhance

st.title("Image Super Resolution App")
st.write("Hotel? Trivago. Low Quality Image? Super Resolution.")

uploaded_file = st.file_uploader("Choose a Low Quality Image...", type=["jpg", "jpeg", "png"])

weights_pth = r"c:\codings\Envision\SuperResWeights.pth"

def enhance_image(input_image_pth):
    output_image = Enhance(input_image_pth, weights_pth)
    return output_image

if uploaded_file is not None:
    # read image
    input_image = Image.open(uploaded_file).convert('RGB')
    
    # display noisy image
    st.subheader("Low Quality Image")
    st.image(input_image, caption='Low Quality Image', width=256)

    # denoise image
    output_image = enhance_image(uploaded_file)

    output_image_normalized = (output_image - output_image.min()) / (output_image.max() - output_image.min())
    output_image_uint8 = (output_image_normalized * 255).astype(np.uint8)

    # Convert the output image to a PIL Image
    output_image_pil = Image.fromarray(np.transpose((output_image_uint8),(1,2,0)))

    # Display denoised image
    st.subheader("Enhanced Image")
    st.image(output_image_pil, caption='Enhanced Image', width=256)