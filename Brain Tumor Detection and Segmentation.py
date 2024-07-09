#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[ ]:





# In[ ]:


import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# Load the classification model
try:
    model = tf.keras.models.load_model('modelFineT.h5')
except Exception as e:
    print(f"Error loading model: {e}")

# Define class names for classification
class_names = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Define a custom loss function for Vgg19 UNet model

epsilon = 1e-5
smooth = 1

def tversky(y_true, y_pred):
    y_true_pos = K.cast(K.flatten(y_true), dtype='float32')
    y_pred_pos = K.cast(K.flatten(y_pred), dtype='float32')
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def focal_tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

try:
    model_seg = tf.keras.models.load_model("seg_model.h5", custom_objects={"focal_tversky": focal_tversky, "tversky": tversky, "tversky_loss": tversky_loss})
except Exception as e:
    print(f"Error loading segmentation model: {e}")

# Define the classification prediction function
def img_pred(img, model):
    try:
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # Resize the image
        img_resized = cv2.resize(opencv_image, (150, 150))
        # Reshape the image to match the input shape of the model
        img_reshaped = img_resized.reshape(1, 150, 150, 3)
        # Predict the class
        prediction = model.predict(img_reshaped)
        predicted_class = np.argmax(prediction, axis=1)[0]
        # Map the prediction to class names
        return class_names[predicted_class]
    except Exception as e:
        return f"Error during prediction: {e}"

# Define the segmentation prediction function
def prediction(image, model):
    try:
        # Convert PIL image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Resize the image
        img_resized = cv2.resize(opencv_image, (256, 256))
        img_preprocessed = np.array(img_resized, dtype=np.float64)
        
        # Standardize the image
        img_preprocessed -= img_preprocessed.mean()
        img_preprocessed /= img_preprocessed.std()

        # Expand dimensions to create a batch of 1 image
        X = np.expand_dims(img_preprocessed, axis=0)

        # Make prediction of mask
        predict = model.predict(X)

        # Check if the predicted mask contains any tumor
        if predict.round().astype(int).sum() == 0:
            has_mask = False
            img_with_mask = np.copy(img_resized)
        else:
            has_mask = True
            img_with_mask = np.copy(img_resized)
            mask_resized = cv2.resize(predict.squeeze().round().astype(np.uint8), (img_resized.shape[1], img_resized.shape[0]))
            img_with_mask[mask_resized == 1] = [255, 0, 0]  # Overlay red color where mask is predicted

        return Image.fromarray(img_with_mask.astype(np.uint8)), has_mask

    except Exception as e:
        print(f"Prediction error: {e}")
        return None, False

# Gradio interface function for segmentation
def gradio_interface(image, model_seg):
    try:
        # Perform prediction
        img_with_mask, has_mask = prediction(image, model_seg)

        # Prepare outputs for Gradio
        if has_mask:
            return img_with_mask
        else:
            return Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))

    except Exception as e:
        print(f"Error in Gradio interface function: {e}")
        return None

# Combined Gradio interface function for classification and segmentation
def combined_gradio_interface(img, model, model_seg):
    # Perform classification
    classification_result = img_pred(img, model)
    
    if classification_result == 'no_tumor':
        # If no tumor is predicted, return a message and no image
        return classification_result, "This is no tumor"
    
    # Perform segmentation if there is a tumor
    segmentation_result = gradio_interface(img, model_seg)
    
    return classification_result, segmentation_result

# Define the Gradio interface
interface = gr.Interface(
    fn=lambda img: combined_gradio_interface(img, model, model_seg),
    inputs=gr.Image(type="pil", label="Upload an MRI scan"),
    outputs=[gr.Textbox(label="Predicted Classification"), gr.Image(label="MRI with Predicted Mask")],
    title="Brain Tumor Classification and Segmentation",
    description="Upload an MRI scan to classify the type of brain tumor and get the segmentation mask."
)

# Launch the Gradio interface
interface.launch(share=True)


# In[ ]:





# In[ ]:




