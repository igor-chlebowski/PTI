import tensorflow as tf
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2

class MajorPredictor:
    model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'branchtex1.keras'))

    def __init__(self, file):
        self.file = file
        self.bitmap = self.process_image(file)
        print("Bitmap shape:", self.bitmap.shape)

    @classmethod
    def process_image(self, image):
        img = Image.open(image).convert('L')  # Convert to grayscale

        img = img.resize((100, 200))

        img_array = np.array(img) 

        thresh = 128
        img_array = (img_array > thresh).astype(np.uint8)

        # cv2.imshow('Bitmap', img_array*255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # img_array =  img_array.reshape(-1, 100, 200, 1)  # Reshape to (1, height, width, channels)
        img_array = np.expand_dims(img_array, axis=(0, -1)) 

        return img_array


    def predict(self):
        # Classes = [0: text, 1: complex expression, 2: brackets, 3: fraction]

        print("Predicting...")
        predicted = self.model.predict(self.bitmap)
        predicted_class = np.argmax(predicted, axis=1)[0]
        classes = ['text', 'complex expression', 'brackets', 'fraction']
        return classes[predicted_class]
    

def main():
    file_path = 'dane/brackets.png' # input("Enter the path to the image file: ")
    if not os.path.exists(file_path):
        print("File does not exist.")
        return
    predictor = MajorPredictor(file_path)
    prediction = predictor.predict()
    print("Prediction:", prediction)

if __name__ == "__main__":
    main()