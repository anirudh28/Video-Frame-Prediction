# Video Frame Prediction

## Problem Statement

To predict the 22nd frame using the 11 frames and perform segmentation masking on it.

## Project Overview

We divide the task into three parts:

1. **Segmentation Model Training:**
    - In the root folder, run:
        ```bash
        python segformer.py
        ```
    - Train the segmentation model for 20 epochs.
    - The model is saved in `/outputs/simvp` as `segformer.pth`.
      
2. **Frame Prediction Model Training:**
    - Navigate to the `SimVP` folder.
    - Set the `predict` argument as `False` in `main.py` during training.
    - Adjust argument values as needed or change defaults in `main.py`.
    - Train the model for 25 epochs with a learning rate of 0.0001.
    - Run the following command in your terminal for training:
        ```bash
        python main.py
        ```
    - After training, the model is saved in `/outputs/simvp` as `checkpoint.pth`.

3. **22nd Frame Prediction and Segmentation:**
    - Set the `predict` argument as `True` in `main.py` for prediction. 
    - In the `SimVP` folder, run:
        ```bash
        python main.py
        ```   
    - This saves the last frames in `/outputs/simvp/predictions`.
    - The frames are then used to predict segmentations, and the results are saved as `pred_masks.npy` in the "outputs" folder.
  
   
