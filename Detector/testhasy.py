#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_symbol.py

Skrypt do predykcji pojedynczego symbolu z pliku PNG (czarno-biały, 32x32).

Użycie:
    python predict_symbol.py /pełna/ścieżka/do/obrazu.png
"""

import sys
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import pandas as pd

def load_model(model_path='hasy_recognition_model.h5'):
    """Wczytuje wytrenowany model z pliku .h5"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")
    return tf.keras.models.load_model(model_path)

def load_symbols(symbols_path='HASY/symbols.csv'):
    """Wczytuje listę symboli z pliku CSV"""
    if not os.path.exists(symbols_path):
        raise FileNotFoundError(f"Nie znaleziono pliku symboli: {symbols_path}")
    tab = pd.read_csv(symbols_path, header=None)
    return tab[0].tolist()

def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    resample = getattr(Image, 'Resampling', Image).LANCZOS
    img = img.resize((32, 32), resample)
    arr = np.array(img).astype('float32') / 255.0
    return arr.reshape(1, 32, 32, 1)

def show_two_windows(orig_path, processed_arr):
    orig = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise FileNotFoundError(f"Nie można wczytać pliku: {orig_path}")
    proc = (processed_arr.reshape(32,32) * 255).astype(np.uint8)
    cv2.namedWindow('Oryginał', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Oryginał', orig)
    cv2.namedWindow('Po preprocessingu', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Po preprocessingu', proc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict(img_path, model_path='hasy_recognition_model.h5', symbols_path='HASY/symbols.csv'):
    model = load_model(model_path)
    symbols = load_symbols(symbols_path)
    img_arr = preprocess_image(img_path)
    preds = model.predict(img_arr)
    idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    symbol = symbols[idx] if idx < len(symbols) else f"(idx {idx})"
    return symbol, confidence

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Użycie: python predict_symbol.py /pełna/ścieżka/do/obrazu.png")
        sys.exit(1)

    image_path = sys.argv[1]
    img_arr = preprocess_image(image_path)
    show_two_windows(image_path, img_arr)
    try:
        symbol, conf = predict(image_path)
        print(f"Rozpoznany symbol: {symbol} (pewność: {conf:.2%})")
    except Exception as e:
        print("Błąd podczas predykcji:", e)
        sys.exit(2)
