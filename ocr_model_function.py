# ocr_engine.py

import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['en'])

def extract_text_from_image_bytes(image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Step 1: Add padding (50 pixels)
    padded = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Step 2: Resize (zoom with padding)
    resized = cv2.resize(padded, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # Step 3: Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Step 4: Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=30)

    # Step 5: Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Step 6: Morphological Opening
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    results = reader.readtext(processed)
    return "\n".join([text for (_, text, _) in results])
