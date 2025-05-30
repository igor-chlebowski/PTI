#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_digit.py

Skrypt do predykcji pojedynczej cyfry z pliku PNG (czarno-biały).

Użycie:
    python predict_digit.py /pełna/ścieżka/do/obrazu.png
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

def load_model(model_path=""):
    model_path = os.path.join(os.path.dirname(__file__), 'digit_recognition_model.h5')

    print(f"Wczytywanie modelu z: {model_path}")
    """Wczytuje wytrenowany model z pliku .h5"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    # obsługa różnych wersji Pillow:
    resample = getattr(Image, 'Resampling', Image).LANCZOS
    img = img.resize((28, 28), resample)
    arr = np.array(img).astype('float32')
    arr = 255.0 - arr
    arr = arr / 255.0
    arr = (arr >= 0.5).astype('float32')
    return arr.reshape(1, 28, 28, 1)




def show_two_windows(orig_path, processed_arr):
    """
    Wyświetla oryginał i obraz po preprocessingu w dwóch oddzielnych oknach.
    
    orig_path: ścieżka do pliku (np. .png)
    processed_arr: wynik preprocess_image (shape (1,28,28,1), float32 [0,1])
    """
    # 1. Wczytaj oryginalny obraz w odcieniach szarości
    orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise FileNotFoundError(f"Nie można wczytać pliku: {orig_path}")

    # 2. Przygotuj przetworzony obraz 28×28 jako uint8 [0–255]
    proc = (processed_arr.reshape(28,28) * 255).astype(np.uint8)

    # 3. Wyświetl oryginał
    cv2.namedWindow('Oryginał', cv2.WINDOW_AUTOSIZE)  # okno w rozmiarze obrazu
    cv2.imshow('Oryginał', orig)

    # 4. Wyświetl przetworzony (28×28) – też bez skalowania
    cv2.namedWindow('Po preprocessingu', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Po preprocessingu', proc)

    # 5. Czekaj na klawisz, potem zamknij oba okna
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def predict(img_path, model_path='digit_recognition_model.h5'):
    model = load_model(model_path)
    img_arr = preprocess_image(img_path)
    preds = model.predict(img_arr)
    digit = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return digit, confidence



if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Użycie: python predict_digit.py /pełna/ścieżka/do/obrazu.png")
        sys.exit(1)

    image_path = sys.argv[1]
    img_arr = preprocess_image(image_path)
    show_two_windows(image_path, img_arr)
    try:
        digit, conf = predict(image_path)
        print(f"Rozpoznana cyfra: {digit} (pewność: {conf:.2%})")
    except Exception as e:
        print("Błąd podczas predykcji:", e)
        sys.exit(2)
