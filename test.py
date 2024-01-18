import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import cv2

def select_image():
    # Set up the tkinter file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Open the file dialog
    return file_path


def preprocess_image(image_path, size=(300, 300)):
    # Load and preprocess the image
    img = image.load_img(image_path, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = cv2.resize(img_array, size)

    # Apply edge detection
    edges = cv2.Canny(img_array.astype('uint8'), 100, 200)

    # Extract the waveform
    # This is a naive implementation and may need to be adapted
    waveform = np.mean(edges, axis=0)

    # Normalize the waveform
    waveform = waveform / np.max(waveform)

    # Reshape for the model input
    waveform = np.expand_dims(waveform, axis=0)  # Add batch dimension
    waveform = np.expand_dims(waveform, axis=2)  # Add channel dimension
    return waveform

image_path = select_image()

# Load and preprocess the image
if image_path:  # If a file is selected
    image = preprocess_image(image_path)

    # Load the saved model
    model = load_model('E:/PHD Projects/Project 1/CNN_RNN_model.h5')

    # Predict using your model
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    
    print(prediction, predicted_class)  # Add batch dimension

    if image.ndim > 1:
        image = np.squeeze(image)

    plt.figure(figsize=(10, 4))
    plt.plot(image)
    plt.title('1D ECG Signal Extracted from Image')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
else:
    print("No file selected.")

