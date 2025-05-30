import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Konfiguracja ---
IMAGE_PATH = 'test.jpg' # Używamy obrazu, który załadowałeś
# IMAGE_PATH = 'sciezka/do/twojego/obrazu.jpg'

# --- Konfiguracja dla findContours ---
MIN_CONTOUR_AREA = 10 # Minimalna powierzchnia konturu, aby uznać go za symbol (do dostrojenia!)

# --- Konfiguracja dla Zamknięcia Morfologicznego ---
# Chcemy połączyć elementy w pionie (np. kreski '=')
# Szerokość kernela może być mała, wysokość powinna być nieco większa niż przerwa.
# Rozmiary kernela (width, height) - do eksperymentowania!
KERNEL_WIDTH = 3
KERNEL_HEIGHT = 5 # Zwiększ tę wartość, jeśli '=' nadal jest rozdzielony
CLOSING_ITERATIONS = 1 # Liczba powtórzeń operacji zamknięcia
PADDING_PIXELS = 9
# --- Główna funkcja przetwarzająca ---
def prepare_image_for_contours(image_path):
    # ... (kod wczytywania, konwersji do szarości, progowania adaptacyjnego - bez zmian) ...
    if not os.path.exists(image_path):
        print(f"Błąd: Plik obrazu nie został znaleziony pod ścieżką: {image_path}")
        return None, None

    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if original_image is None:
        print(f"Błąd: Nie można wczytać obrazu ze ścieżki: {image_path}")
        return None, None

    print("Obraz wczytany pomyślnie.")

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    print("Konwersja do skali szarości zakończona.")

    # Opcjonalne rozmycie
    # gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    print("Stosowanie progowania adaptacyjnego...")
    block_size = 25
    constant_c = 27
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, constant_c
    )
    print(f"Zakończono progowanie adaptacyjne (blockSize={block_size}, C={constant_c}).")
    # --- NOWY KROK: Zamknięcie Morfologiczne ---
    print(f"Stosowanie Zamknięcia Morfologicznego (kernel: {KERNEL_WIDTH}x{KERNEL_HEIGHT}, iter: {CLOSING_ITERATIONS})...")
    # Stworzenie kernela - prostokątny, wyższy niż szerszy, aby łączyć w pionie
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_WIDTH, KERNEL_HEIGHT))

    # Zastosowanie operacji zamknięcia
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=CLOSING_ITERATIONS)
    print("Zakończono Zamknięcie Morfologiczne.")
    # -----------------------------------------

    # Zwracamy teraz również obraz po zamknięciu do ewentualnej wizualizacji
    return original_image, binary_image, closed_image

# --- Funkcja do znajdowania i rysowania konturów ---
def find_and_draw_contours(original_image, image_to_find_contours_on, min_area_threshold, padding):
    """
    Znajduje kontury, filtruje je, oblicza prostokąty z paddingiem
    i rysuje je na kopii oryginalnego obrazu.
    """
    contours, hierarchy = cv2.findContours(image_to_find_contours_on, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Znaleziono {len(contours)} potencjalnych konturów (po operacjach morfologicznych).")

    output_image = original_image.copy()
    img_h, img_w = original_image.shape[:2] # Pobranie wymiarów obrazu do sprawdzania granic
    found_padded_bounding_boxes = [] # Lista na współrzędne Z PADDINGIEM
    valid_contours_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area_threshold:
            valid_contours_count += 1
            # Oblicz ciasny prostokąt ograniczający
            (x, y, w, h) = cv2.boundingRect(contour)

            # --- NOWY KROK: Obliczanie współrzędnych z paddingiem ---
            # Odjęcie paddingu od współrzędnych startowych (x, y)
            # Dodanie paddingu do wymiarów (w, h) - po 2*padding na wymiar
            x_pad = x - padding
            y_pad = y - padding
            w_pad = w + (2 * padding)
            h_pad = h + (2 * padding)

            # Sprawdzenie i korekta granic, aby nie wyjść poza obraz
            x_pad = max(0, x_pad) # x nie może być mniejsze niż 0
            y_pad = max(0, y_pad) # y nie może być mniejsze niż 0
            # Sprawdź, czy prawy dolny róg nie wychodzi poza obraz
            # Jeśli tak, dostosuj szerokość/wysokość
            if x_pad + w_pad > img_w:
                w_pad = img_w - x_pad # Szerokość = od x_pad do krawędzi obrazu
            if y_pad + h_pad > img_h:
                h_pad = img_h - y_pad # Wysokość = od y_pad do krawędzi obrazu
            # --------------------------------------------------------

            # Zapisz współrzędne Z PADDINGIEM
            found_padded_bounding_boxes.append((x_pad, y_pad, w_pad, h_pad))

            # Narysuj prostokąt Z PADDINGIEM
            cv2.rectangle(output_image, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), (0, 255, 0), 2)

            # --- Miejsce na dalsze kroki ---
            # Wycięcie ROI Z PADDINGIEM z OBRAZU BINARNEGO (lub szarego)
            # roi_padded = image_to_find_contours_on[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            # Teraz 'roi_padded' ma margines i jest gotowy do rozpoznawania
            # print(f"  - Symbol w ({x_pad},{y_pad}) wymiary {w_pad}x{h_pad} (z paddingiem)")

    print(f"Znaleziono {valid_contours_count} konturów spełniających próg pola > {min_area_threshold}.")
    return output_image, found_padded_bounding_boxes

def save_rois(image_source_for_roi, boxes, output_folder):
    """
    Wycina fragmenty obrazu (ROI) na podstawie podanych prostokątów,
    dopasowuje każde ROI do kwadratu przez dołożenie białych pikseli
    i zapisuje je do wskazanego folderu.
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"Przygotowano folder wyjściowy: {output_folder}")

    saved_count = 0
    for i, (x, y, w, h) in enumerate(boxes):
        # Wycięcie ROI z obrazu źródłowego
        roi = image_source_for_roi[y:y+h, x:x+w]
        if roi.size == 0:
            print(f"  - Pominięto pusty ROI dla boxa: {(x, y, w, h)}")
            continue

        # --- TU: dopasowanie do kwadratu ---
        # ustalamy nowy rozmiar = dłuższy bok
        max_side = max(w, h)

        # liczymy ile pikseli dołożyć z każdej strony
        dw = max_side - w
        dh = max_side - h
        pad_left   = dw // 2
        pad_right  = dw - pad_left
        pad_top    = dh // 2
        pad_bottom = dh - pad_top

        # biała ramka w przestrzeni BGR
        white = [255, 255, 255]
        roi_square = cv2.copyMakeBorder(
            roi,
            top=pad_top, bottom=pad_bottom,
            left=pad_left, right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=white
        )
        # ----------------------------------

        filename = f"{x}_{y}_{w}_{h}.png"
        full_path = os.path.join(output_folder, filename)
        try:
            cv2.imwrite(full_path, roi_square)
            saved_count += 1
        except Exception as e:
            print(f"  - BŁĄD podczas zapisywania ROI {filename}: {e}")

    print(f"Zapisano {saved_count} plików ROI w folderze '{output_folder}'.")

# --- Główny blok wykonawczy ---
if __name__ == "__main__":
    # Teraz funkcja zwraca 3 obrazy
    original, binary_after_thresh, closed_binary = prepare_image_for_contours(IMAGE_PATH)
    if original is not None and closed_binary is not None:
        # Przekazujemy obraz PO ZAMKNIĘCIU i wartość PADDINGU
        image_with_boxes, padded_boxes = find_and_draw_contours(
            original, closed_binary, MIN_CONTOUR_AREA, PADDING_PIXELS
        )
        print(f"Współrzędne znalezionych prostokątów (x, y, szerokość, wysokość): {padded_boxes}")
        
        save_rois(original, padded_boxes, "symbole")
        # Wyświetlanie wyników - teraz 3 obrazy dla lepszego wglądu
        print("Wyświetlanie obrazów...")
        plt.figure(figsize=(15, 5)) # Szersze okno

        plt.subplot(1, 3, 1) # Pierwszy obraz
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Oryginalny Obraz')
        plt.axis('off')

        plt.subplot(1, 3, 2) # Drugi obraz - po zamknięciu
        plt.imshow(closed_binary, cmap='gray')
        plt.title(f'Obraz po Zamknięciu Morf.\n(Kernel {KERNEL_WIDTH}x{KERNEL_HEIGHT})')
        plt.axis('off')

        plt.subplot(1, 3, 3) # Trzeci obraz - wynik końcowy
        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title(f'Wykryte Symbole (Kontury > {MIN_CONTOUR_AREA}px)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        print("Zakończono. Sprawdź, czy znak '=' jest teraz w jednym prostokącie.")
        print("Jeśli nie, eksperymentuj z KERNEL_HEIGHT i CLOSING_ITERATIONS.")