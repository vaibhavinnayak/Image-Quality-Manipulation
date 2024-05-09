from UNET import UNET
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2
from PIL import Image

# import cv2

def Enhance(img_pth, model_weights_pth):

    model = UNET()
    model.load_state_dict(torch.load(model_weights_pth, map_location=torch.device('cpu')))
    # set the model to evaluation mode
    model.eval()

    # load and preprocess the input image
    img = Image.open(img_pth).convert('RGB')  # loading as RGB

    transform = v2.Compose([    #applying test transforms
                    v2.Resize((256,256)),
                    v2.ToTensor(),   
                    v2.ToDtype(torch.float32)
                ])
    
    
    img_tensor = transform(img).unsqueeze(0)  # add batch dimension

    # perform inference
    with torch.no_grad():
        output_tensor = model(img_tensor)

    output_img = (output_tensor.squeeze().cpu().numpy())
    

    return output_img
    