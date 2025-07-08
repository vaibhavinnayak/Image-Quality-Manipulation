# Image Quality Manipulation using Deep Autoencoders

## Mentors

- Aakarsh Bansal
- Abhishek Srinivas
- Raajan Wankhade

## Mentees

- Ananya A. K
- Raunak Nayak
- Sai Akhilesh Donga
- Sanga Balanarsimha
- Sarth Shah
- Utkarsh Shukla
- Vaibhavi Nagaraja Nayak
- Vanshika Mittal
- Vedh Adla

## Aim

- Perform denoising of images and image super resolution using deep autoencoders.
- To create a simple frontend(Streamlit) to deploy the model.

## Introduction and Overview

In our project, we use the capabilities of autoencoder architectures to enhance image quality. By using super-resolution and noise removal techniques, our project aims to tackle two of the most important problems with Image Quality. Deep autoencoders unveil intricate details within images, making them significant tools in applications requiring image quality preservation and restoration such as medical diagnostics, surveillance, and satellite imagery.

During the course of the project, we were able to gain knowledge in the fields of Machine Learning, Deep Learning, Convolutional Neural Networks. We also completed Kaggle tasks during the learning phase of the project.

## Technologies used

1. Python
1. Streamlit
1. Pytorch


## Datasets

We made use of 2 datasets that were publicly available on Kaggle for performing denoising and super resolution. For the denoising of images, we used a dataset on Kaggle that contained 120 black-white images of Teeth X-Ray. For the super resolution component, we used a dataset on Kaggle that had 685 low and corresponding high resolution images. The links for the datasets are provided below:

[Super Resolution Dataset](https://www.kaggle.com/datasets/adityachandrasekhar/image-super-resolution)

[Denoising Dataset](https://www.kaggle.com/datasets/parthplc/medical-image-dataset)


## Model and Architecture

### 1. Denoising of Images

As our dataset had noise free images of Teeth X-Ray, we first added either Gaussian/ Uniform noise to the X-Ray images of teeth.

We then used an architecture that would model a UNET here. Our architecture consisted of 2 encoder layers followed by 2 decoder layers and we also employed the use of skip connections for garnering global context.

We then used Cross Entropy Loss as the loss function, Adam as the optimiser, the number of epochs to be 200 and the learning rate to be around 1e-5. We were able to converge to a loss of around 0.000290.

### 2. Super Resolution of Images

Again, we tried to model an architecture similar to UNET here. We however, used a more complex architecture as compared to the denoising model with 5 encoder layers, also making use of skip connections.

We also tried 2 different types of loss - VGG Loss and MSE Loss. The model was trained for first 10 epochs using a combination of VGG Loss annd MSE loss to ensure semantic accuracy, and then using only MSE Loss for 10 more. Adam optimiser with learning rate of 0.0002 was used, the loss converged to about 0.001.

## Streamlit Interface

1. We were also able to create a simple user interface using Streamlit. After training the model for both tasks, we saved their weights, which allows us to use the model for any image without having to train again.
1. For both tasks, we have a simple frontend where the user can choose to insert an image for either denoising or super resolution. The frontend would then display the new image - after denoising on increasing the resolution.

## Conclusion

We were able to build 2 models that could successfully perform denoising and super resolution, in addition to building a Streamlit model. During the course of this project, we also understood the basics of Deep Learning and Convolutional Neural Networks.

## Running the Streamlit App

1. Ensure that you have installed Git on your system.
You can check the installation using: 

```
git --version
```

2. Install streamlit
```
pip install streamlit
```
3. To run the app, please follow the given instructions:

    - Clone the repository onto your local system
    ```
    git clone https://github.com/vaibhavinnayak/Image-Quality-Manipulation.git 
    ```
    - After this, for super resolution:
    ```
    cd autoencoder-image-quality-manipulation/super-resolution/app
    ```
    - For denoising,
    ```
    cd autoencoder-image-quality-manipulation/xray-denoising/app
    ```
    - Ensure that path to model weights is set correctly in the ```weights_pth ``` variable inside ```app.py```
    - After this:
    ```
    streamlit run app.py
    ```


