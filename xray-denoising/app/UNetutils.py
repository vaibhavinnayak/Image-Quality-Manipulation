from UNetModel import Autoencoder
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from skimage import exposure
# import cv2

def Denoise(img_pth, model_weights_pth):
    model = Autoencoder()
    model.load_state_dict(torch.load(model_weights_pth, map_location=torch.device('cpu')))
    # set the model to evaluation mode
    model.eval()

    # load and preprocess the input image
    img = Image.open(img_pth).convert('L')  # convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((512, 736)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)  # add batch dimension
    img_tensor
    # perform inference
    with torch.no_grad():
        output_tensor = model(img_tensor)

    output_img = (output_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
    # contrast stretching
    p2, p98 = np.percentile(output_img, (2, 98))
    output_img_contrast_stretched = exposure.rescale_intensity(output_img, in_range=(p2, p98))

    return output_img_contrast_stretched
    