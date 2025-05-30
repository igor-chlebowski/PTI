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

def remap_symbol(symbol_id, symbols_path=os.path.dirname(__file__) + '/HASY/symbols.csv'):

    if not os.path.exists(symbols_path):
        raise FileNotFoundError(f"Nie znaleziono pliku symboli: {symbols_path}")

    # Wczytujemy CSV z nagłówkami: symbol_id, latex, training_samples, test_samples
    df = pd.read_csv(symbols_path)

    # Filtrujemy wiersz po symbol_id
    matched = df.loc[df['symbol_id'] == symbol_id, 'latex']

    if matched.empty:
        raise KeyError(f"Brak symbolu o id={symbol_id} w pliku {symbols_path}")

    print("Znaleziono symbol:", matched.iloc[0], " dla id=", symbol_id)
    # matched.iloc[0] to nasz ciąg LaTeX
    return matched.iloc[0]


def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    resample = getattr(Image, 'Resampling', Image).LANCZOS
    img = img.resize((32, 32), resample)
    arr = np.array(img).astype('float32') / 255.0
    return arr.reshape(1, 32, 32, 1)



def predict(
        img_path, 
        model_path=os.path.join(os.path.dirname(__file__), 'hasy_recognition_model.h5'), 
        symbols_path=os.path.join(os.path.dirname(__file__),  'HASY/symbols.csv')
    ):
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
    
    try:
        symbol, conf = predict(image_path)
        znak = remap_symbol(int(symbol), os.path.join(os.path.dirname(__file__),  'HASY/symbols.csv'))
        print(f"Rozpoznany symbol: {znak} (pewność: {conf:.2%})")
    except Exception as e:
        print("Błąd podczas predykcji:", e)
        sys.exit(2)
