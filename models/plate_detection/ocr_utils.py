import cv2
import pytesseract
import numpy as np
import os

plate_cascade = None
try:
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_russian_plate_number.xml')
    if os.path.exists(cascade_path):
        plate_cascade = cv2.CascadeClassifier(cascade_path)
except Exception:
    pass

def preprocess_for_ocr(crop):
    # Resize to make text clearer for tesseract
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to remove noise but keep edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return binary

def extract_plate_text(frame, x, y, w, h):
    # Restrict to lower 60% of the vehicle where plates usually are, to reduce false positives
    y_offset = int(h * 0.4)
    vehicle_crop = frame[y + y_offset : y + h, x : x + w]
    
    if vehicle_crop.size == 0:
        return "", None
        
    best_text = ""
    best_rect = None
    plates_crops = []

    # 1. Try Haar Cascade first if available
    if plate_cascade is not None:
        gray_veh = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray_veh, scaleFactor=1.1, minNeighbors=3, minSize=(20, 10))
        for (px, py, pw, ph) in plates:
            plate_crop = vehicle_crop[py:py+ph, px:px+pw]
            plates_crops.append((plate_crop, (x + px, y + y_offset + py, pw, ph)))
            
    # 2. Fallback to Morphological Edge Detection if no plate found
    if len(plates_crops) == 0:
        gray_veh = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray_veh, cv2.MORPH_BLACKHAT, rectKern)
        
        # Sobel X to find vertical edges
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = 255 * ((gradX - minVal) / (maxVal - minVal + 1e-6))
        gradX = gradX.astype("uint8")
        
        # Smooth and threshold
        gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        _, thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Find contours
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
        for c in cnts:
            (cx, cy, cw, ch) = cv2.boundingRect(c)
            aspectRatio = cw / float(ch)
            # Plates are usually rectangular with aspect ratio between 2 and 5
            if 2.0 <= aspectRatio <= 5.5 and cw > 30 and ch > 10:
                plates_crops.append((vehicle_crop[cy:cy+ch, cx:cx+cw], (x + cx, y + y_offset + cy, cw, ch)))

    # 3. Perform OCR on all potential plate crops
    for crop, rect in plates_crops:
        if crop.size == 0: continue
        processed = preprocess_for_ocr(crop)
        # PSM 8: Treat the image as a single word
        try:
            text = pytesseract.image_to_string(processed, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        except Exception as e:
            print("Tesseract OCR error:", e)
            text = ""
        text = "".join([c for c in text if c.isalnum()])
        if len(text) > len(best_text) and len(text) >= 4:
            best_text = text
            best_rect = rect
            
    return best_text, best_rect
